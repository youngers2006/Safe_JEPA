import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Import files
from Networks import SpectralNormLinear

class Q_safety_critic(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        self.layers = []
        
        current_dim = d_in
        
        for h in hidden_features:
            self.layers.append(
                SpectralNormLinear(current_dim, h, rngs=rngs)
            )
            self.layers.append(
                nnx.LayerNorm(
                    h, use_scale=False, use_bias=False, epsilon=1e-5, rngs=rngs
                )
            )
            current_dim = h
        self.output_layer = SpectralNormLinear(current_dim, d_out, rngs=rngs)
        
    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        x = jnp.concatenate([z, u], axis=-1)

        for l in range(0, len(self.layers), 2):
            linear_layer = self.layers[l]
            norm_layer = self.layers[l+1]

            x = linear_layer(x, update_spectral_norm)
            x = norm_layer(x)
            x = nnx.silu(x)
        
        return self.output_layer(x, update_spectral_norm)