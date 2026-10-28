from __future__ import annotations
import math
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .layers import SpectralNormLinear

SILU_LIPSCHITZ = 1.09984

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

class GoalPredictor(RewardPredictor):
    # Logit of reaching the goal on a step, from (z, u): the reward head's network, built from GoalParams
    pass

class ExpectedReward(nnx.Module):
    # The planners' reward with the goal head: the dense reward plus the goal bonus times the probability of reaching
    # the goal on the step. Training weights goal rows by goal_weight, which raises the optimal logit by
    # log(goal_weight), so the probability takes it off
    def __init__(self, reward_fn: RewardPredictor, goal_fn: GoalPredictor, goal_reward: float, goal_weight: float):
        self.reward_fn = reward_fn
        self.goal_fn = goal_fn
        self.goal_reward = goal_reward
        self.log_goal_weight = math.log(goal_weight)

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        p_goal = jax.nn.sigmoid(self.goal_fn(z, u, update_spectral_norm=False) - self.log_goal_weight)
        return self.reward_fn(z, u, update_spectral_norm=False) + self.goal_reward * p_goal
