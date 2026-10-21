import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .variables import NormStat, SpectralStat

def l2_normalise(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    return x / (jnp.linalg.norm(x) + eps)

class SpectralNormLinear(nnx.Module):
    def __init__(self, d_in, d_out, c: float, rngs: nnx.Rngs):
        # Bounded L <= c
        self.c = c

        # Linear network component
        self.network = nnx.Linear(d_in, d_out, rngs=rngs)

        # Direction vector for power iteration
        u_init = jax.random.normal(rngs.params(), (d_out,))
        self.u = SpectralStat(l2_normalise(u_init))
        self.v = SpectralStat(l2_normalise(self.network.kernel.value @ u_init))

        # Largest singular value (define as variable to allow updates when jit)
        self.sigma = SpectralStat(jnp.ones(()))

        # Warm start u so the first forward pass already has a good sigma estimate
        self.update_uv(self.network.kernel.value, n_iter=20)

    def update_uv(self, W: jax.Array, n_iter: int) -> None:
        # Extract direction vector
        u = self.u.value

        # Stop gradient flow through W here
        W = jax.lax.stop_gradient(W)

        # Compute transformed unit vector
        for _ in range(n_iter):
            v = l2_normalise(W @ u)
            u = l2_normalise(v @ W)

        # Save data
        self.u.value = u
        self.v.value = v

    def set_exact_uv(self) -> None:
        # Exact top singular vectors from one SVD, so sigma = v^T W u is the true spectral norm (use before saving)
        U, _, Vt = jnp.linalg.svd(self.network.kernel.value, full_matrices=False)
        self.v.value = U[:, 0]
        self.u.value = Vt[0]

    def spectral_norm(self, W: jax.Array) -> jax.Array:
        u = jax.lax.stop_gradient(self.u.value)
        v = jax.lax.stop_gradient(self.v.value)
        return v @ W @ u
    
    def __call__(self, x: jax.Array, update_spectral_norm: bool = True) -> jax.Array:
        # Extract weight matrix (d_in, d_out)
        W = self.network.kernel.value

        # Use power iteration to update spectral norm
        if update_spectral_norm:
            self.update_uv(W, n_iter=1)

        # Compute spectral norm from current W
        sigma = self.spectral_norm(W)
        self.sigma.value = jax.lax.stop_gradient(sigma)

        # Rescale only when sigma > c (scale < 1 means the constraint is binding)
        if self.c == float("inf"):
            W_sn = W
        else:
            scale = jnp.minimum(1.0, self.c / jnp.maximum(sigma, 1e-12))
            W_sn = W * scale
        y = x @ W_sn

        # Add bias if used
        if self.network.bias is not None:
            y = y + self.network.bias.value
        return y