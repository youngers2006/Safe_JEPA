import jax
import jax.numpy as jnp
import flax.nnx as nnx

from nn.layers import SpectralNormLinear, NormStat, SpectralStat
from nn.stems import ConvStem, MLPStem

class Encoder(nnx.Module):
    def __init__(self, cfg, mu, sigma, rngs: nnx.Rngs):
        self.mu = NormStat(jnp.asarray(mu, dtype=jnp.float32))
        self.sigma = NormStat(jnp.asarray(sigma,  dtype=jnp.float32))

        if cfg["obs_type"] == "vision":
            self.stem = ConvStem(cfg["in_channels"], rngs=rngs)
            feature_dim = cfg["flattened_dim"]
        elif cfg["obs_type"] == "state":
            self.stem = MLPStem(cfg["obs_dim"], cfg["d_hidden"], rngs=rngs)
            feature_dim = cfg["d_hidden"]
        else:
            raise ValueError(f"unknown obs_type: {cfg['obs_type']}")

        # Processing Network
        self.linear_proj = nnx.Linear(feature_dim, cfg["d_latent"], rngs=rngs)
        self.layer_norm = nnx.LayerNorm(cfg["d_latent"], use_scale=True, use_bias=True, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Standardisation constants
        mu = jax.lax.stop_gradient(self.mu)
        sigma = jax.lax.stop_gradient(self.sigma)

        # Standardise observation
        x = (x - mu) / jnp.maximum(sigma, 1e-6)
        
        # Run stem network
        x = self.stem(x)
        
        # Project and normalize
        x = self.linear_proj(x)
        x = self.layer_norm(x)
        return nnx.tanh(x)