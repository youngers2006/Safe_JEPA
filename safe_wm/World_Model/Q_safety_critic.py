import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Import files
from World_Model.Networks import SpectralNormLinear

class QSafetyCritic(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        temp_layers = []
        
        current_dim = d_in
        
        for h in hidden_features:
            temp_layers.append(
                SpectralNormLinear(current_dim, h, rngs=rngs)
            )
            temp_layers.append(
                nnx.LayerNorm(
                    h, use_scale=False, use_bias=False, epsilon=1e-5, rngs=rngs
                )
            )
            current_dim = h
        self.layers = nnx.List(temp_layers)
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

class SafetyCriticEnsemble(nnx.Module):
    def __init__(self, ensemble_size: int, d_in: int, hidden_features: tuple[int, ...], d_out: int, *, rngs: nnx.Rngs):

        self.ensemble_size = ensemble_size

        VectorisedEnsemble = nnx.vmap(
            QSafetyCritic,
            in_axes=(None, None), # z, u
            out_axes=0,
            axis_size=ensemble_size
        )

        self.critic_ensemble = VectorisedEnsemble(
            d_in, hidden_features, d_out, rngs=rngs
        )

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        return self.critic_ensemble(z, u, update_spectral_norm)

    def get_moments(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        Q_vals = self(z, u, update_spectral_norm)
        mu = jnp.mean(Q_vals, axis=0)
        var = jnp.var(Q_vals, axis=0)
        return mu, var

    def compute_loss(
            self, 
            target_ensemble: "SafetyCriticEnsemble", 
            z: jax.Array, 
            next_z_target: jax.Array, 
            action: jax.Array, 
            safety_cost: jax.Array, 
            done: jax.Array,
            discount: float, 
            cql_alpha: float, 
            action_bounds: tuple[float, float], 
            rng_key: nnx.Rngs,
            Q_minima_samples: int = 64
        ):
        # Safety critic bellman target formulation y = I(c_t) + gamma * (1 - c_t) * min_u_Q_next
        q_risk_vals = self(z, action, update_spectral_norm=True).squeeze()

        # Compute MSE loss
        batch_size = z.shape[0]
        random_key = rng_key.default()
        sampled_actions = jax.random.uniform(
            random_key, 
            shape=(batch_size, Q_minima_samples, action.shape[-1]), 
            minval=action_bounds[0], 
            maxval=action_bounds[1]
        ).reshape(-1, action.shape[-1])

        z_q = jnp.repeat(jnp.expand_dims(next_z_target, axis=1), Q_minima_samples, axis=1).reshape(-1, next_z_target.shape[-1])
        next_q = self(
            z_q, sampled_actions, update_spectral_norm=False
        ).squeeze().reshape(self.ensemble_size, batch_size, Q_minima_samples)
        mean_next_q = jnp.mean(next_q, axis=0)
        best_action_indices = jnp.argmin(mean_next_q, axis=-1)

        target_next_q = target_ensemble(
            z_q, sampled_actions, update_spectral_norm=False
        ).squeeze().reshape(self.ensemble_size, batch_size, Q_minima_samples)

        indices_expanded = jnp.broadcast_to(
            best_action_indices[None, :, None],
            (self.ensemble_size, batch_size, 1)
        )

        selected_target_q = jnp.take_along_axis(target_next_q, indices_expanded, axis=-1).squeeze(axis=-1) 
        pessimistic_q = jnp.max(selected_target_q, axis=0)

        q_target = jax.lax.stop_gradient(safety_cost + discount * (1.0 - done) * pessimistic_q)
        loss_q_risk_mse = jnp.mean((q_risk_vals - q_target) ** 2, axis=1)

        # CQL q loss, pushes up ood actions up
        z_expanded = jnp.repeat(
            jnp.expand_dims(z, axis=1), Q_minima_samples, axis=1
        ).reshape(-1, z.shape[-1])
        q_risk_ood = self(
            z_expanded, sampled_actions, update_spectral_norm=True
        ).squeeze().reshape(self.ensemble_size, batch_size, Q_minima_samples)
        mean_q_risk_ood = jnp.mean(q_risk_ood, axis=-1)
        cql_risk_loss = jnp.mean(q_risk_vals - mean_q_risk_ood, axis=1)

        # total safety Q loss
        loss_s = loss_q_risk_mse + (cql_alpha * cql_risk_loss)
        return jnp.sum(loss_s)