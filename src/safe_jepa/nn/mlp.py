from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from .layers import SpectralNormLinear

MAXMIN_DELTA = 0.05

def smooth_maxmin(x: jax.Array) -> jax.Array:
    a, b = jnp.split(x, 2, axis=-1)
    s, r = a + b, jnp.sqrt((a - b) ** 2 + MAXMIN_DELTA ** 2)
    return jnp.concatenate([(s + r) / 2, (s - r) / 2], axis=-1)

def per_layer_bound(network_bound: float, n_hidden: int) -> float:
    return network_bound ** (1 / (n_hidden + 1))

class DynamicsPredictor(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        lipschitz_bound = cfg["lipschitz_bound"]
        self.hidden_features = hidden_features
        temp_layers = []
        current_dim = d_in

        if lipschitz_bound < 1.0:
            raise ValueError(f"Residual dynamics need lipschitz_bound >= 1, got {lipschitz_bound}")
        layer_bound = per_layer_bound(lipschitz_bound - 1.0, len(hidden_features))
        
        for h in hidden_features:
            temp_layers.append(
                SpectralNormLinear(current_dim, h, layer_bound, rngs=rngs)
            )
            current_dim = h
        self.layers = temp_layers
        self.output_layer = SpectralNormLinear(current_dim, d_out, layer_bound, rngs=rngs)
        
    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        x = jnp.concatenate([z, u], axis=-1)

        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            x = linear_layer(x, update_spectral_norm)
            x = smooth_maxmin(x)
        
        return z + self.output_layer(x, update_spectral_norm)

class ValueNet(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        lipschitz_bound = cfg["lipschitz_bound"]
        layer_bound = per_layer_bound(lipschitz_bound, len(hidden_features))
        self.hidden_features = hidden_features
        temp_layers = []
            
        current_dim = d_in
        
        for h in hidden_features:
            temp_layers.append(
                    SpectralNormLinear(current_dim, h, layer_bound, rngs=rngs)
            )
            current_dim = h
        self.layers = temp_layers
        self.output_layer = SpectralNormLinear(current_dim, d_out, layer_bound, rngs=rngs)
             
    def __call__(self, z: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            z = linear_layer(z, update_spectral_norm)
            z = smooth_maxmin(z)
        
        return self.output_layer(z, update_spectral_norm)

class ValueQNet(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        temp_layers = []

        current_dim = d_in
        for h in hidden_features:
            temp_layers.append(
                nnx.Linear(current_dim, h, rngs=rngs)
            )
            current_dim = h
        self.layers = temp_layers
        self.output_layer = nnx.Linear(current_dim, d_out, rngs=rngs)

    def __call__(self, z: jax.Array, u: jax.Array):
        x = jnp.concatenate([z, u], axis=-1)
        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            x = linear_layer(x)
            x = smooth_maxmin(x)

        return self.output_layer(x)

class RewardPredictor(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        lipschitz_bound = cfg["lipschitz_bound"]
        layer_bound = per_layer_bound(lipschitz_bound, len(hidden_features))
        self.hidden_features = hidden_features
        temp_layers = []
        
        current_dim = d_in
        
        for h in hidden_features:
            temp_layers.append(
                SpectralNormLinear(current_dim, h, layer_bound, rngs=rngs)
            )
            current_dim = h
        self.layers = temp_layers
        self.output_layer = SpectralNormLinear(current_dim, d_out, layer_bound, rngs=rngs)

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        x = jnp.concatenate([z, u], axis=-1)

        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            x = linear_layer(x, update_spectral_norm)
            x = smooth_maxmin(x)
        
        return self.output_layer(x, update_spectral_norm)

class MarginBarrier(nnx.Module):
    def __init__(
            self, 
            dynamics: DynamicsPredictor, 
            margin_fn: nnx.Module, 
            eps,
            scale,
            alpha: float
        ):
        self.dynamics = dynamics
        self.margin_fn = margin_fn
        self.eps = tuple(np.atleast_1d(eps).tolist())
        self.scale = tuple(scale)
        self.alpha = alpha

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        eps, scale = jnp.asarray(self.eps), jnp.asarray(self.scale)
        h_now = (self.margin_fn(z, update_spectral_norm=False) - eps) / scale
        h_next = (self.margin_fn(self.dynamics(z, u, update_spectral_norm=False), update_spectral_norm=False) - eps) / scale
        return jnp.max((1.0 - self.alpha) * h_now - h_next, axis=-1, keepdims=True)

class MarginNet(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        lipschitz_bound = cfg["lipschitz_bound"]
        layer_bound = per_layer_bound(lipschitz_bound, len(hidden_features))
        self.hidden_features = hidden_features
        temp_layers = []
            
        current_dim = d_in
        
        for h in hidden_features:
            temp_layers.append(
                    SpectralNormLinear(current_dim, h, layer_bound, rngs=rngs)
            )
            current_dim = h
        self.layers = temp_layers
        self.output_layer = SpectralNormLinear(current_dim, d_out, layer_bound, rngs=rngs)
             
    def __call__(self, z: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            z = linear_layer(z, update_spectral_norm)
            z = smooth_maxmin(z)
        
        return self.output_layer(z, update_spectral_norm)