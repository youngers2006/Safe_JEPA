from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .layers import SpectralNormLinear

SILU_LIPSCHITZ = 1.09984

def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))

def per_layer_bound(network_bound: float, n_hidden: int) -> float:
    return (network_bound / SILU_LIPSCHITZ ** n_hidden) ** (1 / (n_hidden + 1))

class DynamicsPredictor(nnx.Module):
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
            x = nnx.silu(x)
        
        return self.output_layer(x, update_spectral_norm)

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
            z = nnx.silu(z)
        
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
            x = nnx.silu(x)

        return self.output_layer(x)

class RewardPredictor(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        lipschitz_bound = cfg["lipschitz_bound"]

        # Two-hot head (DreamerV3, TD-MPC2): the reward as a distribution over num_bins bins evenly spaced in symlog
        # space between vmin and vmax, trained by cross-entropy, so rare large rewards do not drag every prediction
        self.two_hot = cfg.get("two_hot", False)
        if self.two_hot:
            self.vmin, self.vmax, self.num_bins = cfg["vmin"], cfg["vmax"], cfg["num_bins"]
            d_out = self.num_bins
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

    def outputs(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        # The network's output: the reward itself, or with the two-hot head the logits over the bins
        x = jnp.concatenate([z, u], axis=-1)

        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            x = linear_layer(x, update_spectral_norm)
            x = nnx.silu(x)
        
        return self.output_layer(x, update_spectral_norm)

    def bins(self) -> jax.Array:
        return jnp.linspace(self.vmin, self.vmax, self.num_bins)

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        out = self.outputs(z, u, update_spectral_norm)
        if not self.two_hot:
            return out

        # Expected reward under the predicted distribution
        return jnp.sum(jax.nn.softmax(out, axis=-1) * symexp(self.bins()), axis=-1, keepdims=True)

    def loss(self, z: jax.Array, u: jax.Array, reward: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        # Per-row training loss: squared error, or with the two-hot head the cross-entropy against the reward's two-hot
        # encoding (its symlog split between the two nearest bins in proportion to how close it is to each)
        out = self.outputs(z, u, update_spectral_norm)
        if not self.two_hot:
            return (out.squeeze(axis=-1) - reward) ** 2

        bins = self.bins()
        y = jnp.clip(symlog(reward), self.vmin, self.vmax)
        lo = jnp.clip(jnp.searchsorted(bins, y, side="right") - 1, 0, self.num_bins - 2)
        frac = (y - bins[lo]) / (bins[lo + 1] - bins[lo])
        target = (jax.nn.one_hot(lo, self.num_bins) * (1.0 - frac)[..., None]
                  + jax.nn.one_hot(lo + 1, self.num_bins) * frac[..., None])
        return -jnp.sum(target * jax.nn.log_softmax(out, axis=-1), axis=-1)

class MarginBarrier(nnx.Module):
    def __init__(self, dynamics: DynamicsPredictor, margin_fn: nnx.Module, eps: float, alpha: float):
        self.dynamics = dynamics
        self.margin_fn = margin_fn
        self.eps = eps
        self.alpha = alpha

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        h_now = self.margin_fn(z, update_spectral_norm=False) - self.eps
        h_next = self.margin_fn(self.dynamics(z, u, update_spectral_norm=False), update_spectral_norm=False) - self.eps
        return (1.0 - self.alpha) * h_now - h_next

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
            z = nnx.silu(z)
        
        return self.output_layer(z, update_spectral_norm)