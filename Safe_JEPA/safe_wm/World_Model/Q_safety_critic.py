import jax
import jax.numpy as jnp
import flax.nnx as nnx

# Import files
from .Networks import SpectralNormLinear

class QSafetyCritic(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, lipschitz_bound: float, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        temp_layers = []
        
        current_dim = d_in
        
        for h in hidden_features:
            temp_layers.append(
                SpectralNormLinear(current_dim, h, lipschitz_bound, rngs=rngs)
            )
            current_dim = h
        self.layers = temp_layers
        self.output_layer = SpectralNormLinear(current_dim, d_out, lipschitz_bound, rngs=rngs)
        
    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        x = jnp.concatenate([z, u], axis=-1)

        for l in range(0, len(self.layers)):
            linear_layer = self.layers[l]

            x = linear_layer(x, update_spectral_norm)
            x = nnx.silu(x)
        
        return self.output_layer(x, update_spectral_norm)

class SafetyCriticEnsemble(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        # Unpack
        ensemble_size = cfg["ensemble_size"]
        d_in = cfg["d_in"]
        hidden_features = cfg["hidden_features"]
        d_out = cfg["d_out"]
        lipschitz_bound = cfg["lipschitz_bound"]

        # Save enemble size
        self.ensemble_size = ensemble_size

        # ensemble maker function
        @nnx.split_rngs(splits=ensemble_size)
        @nnx.vmap
        def make_critic(member_rngs: nnx.Rngs):
            return QSafetyCritic(
                d_in, hidden_features, d_out, lipschitz_bound, rngs=member_rngs
            )

        self.critic_ensemble = make_critic(rngs)

    @nnx.vmap(in_axes=(0, None, None, None), out_axes=0)
    def forward_pass(model, z_in, u_in, update_sn):
        return model(z_in, u_in, update_sn)

    def __call__(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        @nnx.vmap(in_axes=(0, None, None), out_axes=0)
        def forward(model, input_z, input_u):
            return model(input_z, input_u, update_spectral_norm)
        return forward(self.critic_ensemble, z, u)

    def get_moments(self, z: jax.Array, u: jax.Array, update_spectral_norm: bool = False) -> tuple[jax.Array, jax.Array]:
        Q_vals = self(z, u, update_spectral_norm)
        mu = jnp.mean(Q_vals, axis=0)
        var = jnp.var(Q_vals, axis=0)
        return mu, jnp.sqrt(var + 1e-6)

    def compute_targets(
        self,
        target_ensemble: "SafetyCriticEnsemble",
        next_z_target: jax.Array,     # (B, d_z)
        sampled_actions: jax.Array,   # (B*K, d_u)
        safety_cost: jax.Array,       # (B,)
        terminal: jax.Array,          # (B,)
        discount: float,
        Q_minima_samples: int
    ) -> jax.Array:
        batch_size = next_z_target.shape[0]
        E = self.ensemble_size

        z_q = jnp.repeat(
            next_z_target[:, None, :], Q_minima_samples, axis=1
        ).reshape(-1, next_z_target.shape[-1])

        # Online critic selects the minimising action. Index only -- argmin
        # is not differentiable, so nothing flows back through this.
        next_q = self(
            z_q, sampled_actions, update_spectral_norm=False
        ).squeeze(axis=-1).reshape(E, batch_size, Q_minima_samples)
        best_idx = jnp.argmin(jnp.mean(next_q, axis=0), axis=-1)

        # Target critic evaluates it.
        target_next_q = target_ensemble(
            z_q, sampled_actions, update_spectral_norm=False
        ).squeeze(axis=-1).reshape(E, batch_size, Q_minima_samples)

        selected = jnp.take_along_axis(
            target_next_q,
            jnp.broadcast_to(best_idx[None, :, None], (E, batch_size, 1)),
            axis=-1,
        ).squeeze(axis=-1)

        c, t = safety_cost[None, :], terminal[None, :]
        return jax.lax.stop_gradient(c + discount * (1.0 - c) * (1.0 - t) * selected)

    def compute_loss(
            self, 
            z: jax.Array, # (Batch, d_z)
            action: jax.Array, # (Batch, d_u)
            q_target: jax.Array,
            sampled_actions: jax.Array,
            cql_alpha: float,
            Q_minima_samples: int = 64
        ):
        batch_size = z.shape[0]

        # Safety critic bellman target formulation y = I(c_t) + gamma * (1 - c_t) * (1 - d_t) * min_u_Q_next
        # q_risk: (M, B). Score safety of actions taken
        q_risk_vals = self(
            z, action, update_spectral_norm=True
        ).squeeze(axis=-1)

        loss_q_risk_mse = jnp.mean((q_risk_vals - q_target) ** 2, axis=1)

        # CQL q loss, pushes up ood actions up
        # expand z dims in action sample dims
        z_expanded = jnp.repeat(
            jnp.expand_dims(z, axis=1), Q_minima_samples, axis=1
        ).reshape(-1, z.shape[-1])

        # Calculate safety prediction at each action sample around state
        q_risk_ood = self(
            z_expanded, sampled_actions, update_spectral_norm=False
        ).squeeze(axis=-1).reshape(self.ensemble_size, batch_size, Q_minima_samples)

        # Obtain mean risk prediction across the set of samples
        mean_q_risk_ood = jnp.mean(q_risk_ood, axis=-1) # (Ensemble, Batch)

        # Get CQL loss, pushes seen actions down and unseen actions up
        cql_risk_loss = jnp.mean(q_risk_vals - mean_q_risk_ood, axis=1)

        # total safety Q loss for each ensemble member then combine into a single loss
        loss_s = loss_q_risk_mse + (cql_alpha * cql_risk_loss) # (Ensemble,)
        return jnp.mean(loss_s)