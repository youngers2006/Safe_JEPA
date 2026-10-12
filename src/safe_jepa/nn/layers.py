import jax
import jax.numpy as jnp
import flax.nnx as nnx

class NormStat(nnx.Variable):
    """Non-trainable standardisation constant"""
    pass

class SpectralStat(nnx.Variable):
    """Power-iteration state: derived from W, not learned, not data-dependent."""
    pass

class SpectralNormLinear(nnx.Module):
    def __init__(self, d_in, d_out, c: float, rngs: nnx.Rngs):
        # Bounded L <= c
        self.c = c

        # Linear network component
        self.network = nnx.Linear(d_in, d_out, rngs=rngs)

        # Largest singular value (define as variable to allow updates when jit)
        self.sigma = SpectralStat(jnp.ones(()))

        # Direction vector for power iteration
        u_init = jax.random.normal(rngs.params(), (d_out,))
        u_init = u_init / (jnp.linalg.norm(u_init) + 1e-8)
        self.u = SpectralStat(u_init)

    def power_iteration(self, W: jax.Array) -> None:
        # Extract direction vector
        u_val = self.u.value

        # Compute transformed unit vector
        v = W @ u_val
        v = v / (jnp.linalg.norm(v) + 1e-8)

        # Recompute rotated direction vector
        u_val = v @ W
        u_val = u_val / (jnp.linalg.norm(u_val) + 1e-8)

        # Use Raleigh quotient to obtain largest singular value
        sigma_val = jnp.dot(u_val, v @ W)

        # Stop gradients to make sure neither variable is updated but GD
        self.u.value = jax.lax.stop_gradient(u_val)
        self.sigma.value = jax.lax.stop_gradient(sigma_val)

    def __call__(self, x: jax.Array, update_spectral_norm: bool = True) -> jax.Array:
        # Extract weight matrix (d_in, d_out)
        W = self.network.kernel.value

        # Use power iteration to update spectral norm
        if update_spectral_norm:
            self.power_iteration(W)

        # Apply projection function to rescale weights
        # If scale < 1 the constraint isnt binding so the normalisation isnt needed
        scale = jnp.minimum(1.0, self.c / jnp.maximum(self.sigma.value, 1e-12))
        W_sn = W * scale
        y = x @ W_sn

        # Add bias if used
        if self.network.bias is not None:
            y = y + self.network.bias.value
        return y