from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .layers import SpectralNormLinear

# Lipschitz constant of SiLU: max |silu'(x)| = 1.0998393 at x ~ 2.3994, rounded up so the bound stays guaranteed
SILU_LIPSCHITZ = 1.09984

def per_layer_bound(network_bound: float, n_hidden: int) -> float:
    # Lipschitz constants multiply through the network: n_hidden + 1 linear layers each bounded by c with a SiLU
    # between each pair gives L <= c^(n_hidden + 1) * SILU_LIPSCHITZ^n_hidden, so solve for c with L = network_bound
    return (network_bound / SILU_LIPSCHITZ ** n_hidden) ** (1 / (n_hidden + 1))

class DynamicsPredictor(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        layer_bound = per_layer_bound(cfg["lipschitz_bound"], len(hidden_features))
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
        layer_bound = per_layer_bound(cfg["lipschitz_bound"], len(hidden_features))
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
        layer_bound = per_layer_bound(cfg["lipschitz_bound"], len(hidden_features))
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

class QSafetyCritic(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        # Unpack
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        layer_bound = per_layer_bound(cfg["lipschitz_bound"], len(hidden_features))

        # Networks layers
        self.hidden_features = hidden_features
        temp_layers = []
        current_dim = d_in

        # Create network
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

    def compute_targets(
        self,
        target_Q: QSafetyCritic,
        next_z_target: jax.Array,     # (B, d_z)
        sampled_actions: jax.Array,   # (B*K, d_u)
        safety_cost: jax.Array,       # (B,)
        terminal: jax.Array,          # (B,)
        discount: float,
        Q_minima_samples: int
    ) -> jax.Array:
        # Get batch and z size
        B, d_z = next_z_target.shape

        # Extend z target in action sample axis to allow computation
        # Must be flattened (B, K, d_z) -> (B*K, d_z)
        z_q = jnp.repeat(
            next_z_target[:, None, :], Q_minima_samples, axis=1
        ).reshape(B * Q_minima_samples, d_z)

        # Target critic selects the minimising action. Approximate this by sampling 
        target_next_q = target_Q(
            z_q, sampled_actions, update_spectral_norm=False
        ).squeeze(axis=-1).reshape(B, Q_minima_samples)
        target_q_min = jnp.min(target_next_q, axis=-1) # (B,)

        # Compute Bellman recursion target
        return jax.lax.stop_gradient(
            jnp.clip(safety_cost + discount * (1.0 - safety_cost) * (1.0 - terminal) * target_q_min, 0.0, 1.0)
        )

    def compute_loss(
            self, 
            z: jax.Array, # (B, d_z)
            action: jax.Array, # (B, d_u)
            q_target: jax.Array, # (B, )
            sampled_actions: jax.Array,
            action_bounds: tuple[float, float],
            cql_alpha: float,
            Q_minima_samples: int = 64
        ):
        # Obtain batch size and the number of members of each batch is seen per ensemble member
        batch_size = z.shape[0]

        # Safety critic bellman target formulation y = I(c_t) + gamma * (1 - c_t) * (1 - d_t) * min_u_Q_next
        # =================================================================================
        # q_risk_id: (B, ). Score safety of actions taken
        q_risk_id = self(
            z, action, update_spectral_norm=True
        ).squeeze(axis=-1)

        # Compute the bellman recursion error
        bellman_risk_loss = jnp.mean((q_risk_id - q_target) ** 2)
        # =================================================================================

        # CQL q loss, pushes up ood actions up
        # =================================================================================
        # expand z dims in action sample dims
        z_expanded = jnp.repeat(
            jnp.expand_dims(z, axis=1), Q_minima_samples, axis=1
        ).reshape(-1, z.shape[-1])

        # Calculate safety prediction at each action sample 
        q_risk_sampled = self(
            z_expanded, sampled_actions, update_spectral_norm=False
        ).squeeze(axis=-1).reshape(batch_size, Q_minima_samples)

        d_u = sampled_actions.shape[-1]
        log_vol = d_u * jnp.log(action_bounds[1] - action_bounds[0])
        q_risk_ood = jax.nn.logsumexp(-q_risk_sampled, axis=-1) - jnp.log(Q_minima_samples) + log_vol

        # Get CQL loss, pushes seen actions down and unseen actions up
        cql_risk_loss = jnp.mean(
            q_risk_ood + q_risk_id
        )
        # =================================================================================

        # total safety Q loss for each ensemble member then combine into a single loss
        loss_s = bellman_risk_loss + (cql_alpha * cql_risk_loss)
        return loss_s, {
            "bellman_recursion_term":     bellman_risk_loss,
            "cql_term":     cql_risk_loss,
            "q_prob_data":  jnp.mean(q_risk_id),
            "q_prob_ood":   jnp.mean(q_risk_sampled),
            "q_pred_min":   jnp.min(q_risk_id),
            "q_pred_max":   jnp.max(q_risk_id)
        }