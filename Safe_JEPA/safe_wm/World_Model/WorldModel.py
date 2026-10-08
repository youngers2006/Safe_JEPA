import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from functools import partial

# import modules
from .Networks import ValueNet, ValueQNet, Encoder, DynamicsPredictor, RewardPredictor, SpectralStat
from .Q_safety_critic import SafetyCriticEnsemble, QSafetyCritic

class TrainableBundle(nnx.Module):
    def __init__(self, encoder, dynamics, value_fn, value_Q, safety_critic, reward_fn):
        self.encoder = encoder
        self.dynamics = dynamics
        self.value_fn = value_fn
        self.value_Q = value_Q
        self.safety_critic = safety_critic
        self.reward_fn = reward_fn

class TargetBundle(nnx.Module):
    def __init__(self, encoder, value_Q, safety_critic):
        self.encoder = encoder
        self.value_Q = value_Q
        self.safety_critic = safety_critic

def expectile_regression_loss(u, tau):
    return jnp.where(u > 0, tau, 1 - tau) * (u **2)

class WorldModel(nnx.Module):
    def __init__(
            self, 
            cfg,
            obs_mean,
            obs_std,
            *, 
            rngs: nnx.Rngs
        ):
        # Load params
        # ==============================================================
        self.lambda_dyn = cfg["lambda_dyn"]
        self.lambda_v = cfg["lambda_v"]
        self.lambda_r = cfg["lambda_r"]
        self.lambda_s = cfg["lambda_s"]
        self.lambda_var = cfg["lambda_var"]
        self.lambda_cov = cfg["lambda_cov"]
        self.tau = cfg["tau"]

        lr = cfg["lr"]
        self.discount = cfg["discount"]
        self.gamma = cfg["gamma"] # VicReg variance threshold
        self.cql_alpha = cfg["alpha"] # CQL penalty weight
        self.ER_tau = cfg["ER_tau"] # Expectile regression weighting
        # ==============================================================

        self.encoder = Encoder(
            cfg=cfg["EncoderParams"],
            mu=obs_mean,
            sigma=obs_std,
            rngs=rngs
        )

        self.target_encoder = Encoder(
            cfg=cfg["EncoderParams"],
            mu=obs_mean,
            sigma=obs_std,
            rngs=rngs
        )

        self.value_Q = ValueQNet(
            cfg=cfg["ValueQParams"],
            rngs=rngs
        )

        self.target_value_Q = ValueQNet(
            cfg=cfg["ValueQParams"],
            rngs=rngs
        )

        self.safety_critic = SafetyCriticEnsemble(
            cfg=cfg["SafetyCriticParams"],
            rngs=rngs      
        )
        
        self.target_safety_critic = SafetyCriticEnsemble(
            cfg=cfg["SafetyCriticParams"],
            rngs=rngs
        )

        nnx.update(self.target_encoder, nnx.state(self.encoder, nnx.Param))
        nnx.update(self.target_value_Q, nnx.state(self.value_Q, nnx.Param))
        nnx.update(self.target_value_Q, nnx.state(self.value_Q, SpectralStat))
        nnx.update(self.target_safety_critic, nnx.state(self.safety_critic, nnx.Param))
        nnx.update(self.target_safety_critic, nnx.state(self.safety_critic, SpectralStat))

        self.dynamics = DynamicsPredictor(
            cfg=cfg["DynamicsParams"],
            rngs=rngs
        )

        self.reward_fn = RewardPredictor(
            cfg=cfg["RewardParams"],
            rngs=rngs
        )

        self.value_fn = ValueNet(
            cfg=cfg["ValueParams"],
            rngs=rngs
        )

        self.trainable_nodes = TrainableBundle(
            self.encoder, 
            self.dynamics, 
            self.value_fn,
            self.value_Q,
            self.safety_critic, 
            self.reward_fn
        )
        self.target_nodes = TargetBundle(
            self.target_encoder, 
            self.target_value_Q, 
            self.target_safety_critic
        )
        self.optimiser = nnx.Optimizer(
            self.trainable_nodes, optax.adam(learning_rate=lr), wrt=nnx.Param
        )

    @nnx.jit
    def update_target_encoder(self, tau: float = 0.01) -> None:
        # Extract both param sets
        online_params = nnx.state(self.encoder, nnx.Param)
        target_params = nnx.state(self.target_encoder, nnx.Param)

        # Use moving average to update target encoder
        new_target_params = optax.incremental_update(
            new_tensors=online_params,
            old_tensors=target_params,
            step_size=tau
        )

        # Update the target encoder state
        nnx.update(self.target_encoder, new_target_params)

    @nnx.jit
    def update_target_value_Q(self, tau: float = 0.01) -> None:
        # Extract both param sets
        online_params = nnx.state(self.value_Q, nnx.Param)
        target_params = nnx.state(self.target_value_Q, nnx.Param)

        # Use moving average to update target encoder
        new_target_params = optax.incremental_update(
            new_tensors=online_params,
            old_tensors=target_params,
            step_size=tau
        )

        # Update the target encoder state
        nnx.update(self.target_value_Q, new_target_params)
        nnx.update(self.target_value_Q, nnx.state(self.value_Q, SpectralStat))

    @nnx.jit
    def update_target_safety_critic(self, tau: float = 0.01) -> None:
        # Extract both param sets
        online_params = nnx.state(self.safety_critic, nnx.Param)
        target_params = nnx.state(self.target_safety_critic, nnx.Param)

        # Use moving average to update target encoder
        new_target_params = optax.incremental_update(
            new_tensors=online_params,
            old_tensors=target_params,
            step_size=tau
        )

        # Update the target encoder state
        nnx.update(self.target_safety_critic, new_target_params)
        nnx.update(self.target_safety_critic, nnx.state(self.safety_critic, SpectralStat))

    @nnx.jit
    def update_target_networks(self, tau_vals:tuple[float, ...]) -> None:
        self.update_target_encoder(tau_vals[0])
        self.update_target_safety_critic(tau_vals[1])
        self.update_target_value_Q(tau_vals[2])

    @partial(nnx.jit, static_argnames=("Q_minima_samples", "action_bounds"))
    def train_step(
        self,
        obs: jax.Array, 
        next_obs: jax.Array, 
        action: jax.Array, 
        reward: jax.Array, 
        safety_cost: jax.Array, 
        terminal: jax.Array,
        key: jax.Array,
        ens_mask: jax.Array,
        Q_minima_samples: int = 64,
        action_bounds: tuple[float, float] = (-1.0, 1.0)
    ) -> dict:
        # Create z targets
        # ===============================================================
        next_z_target = self.target_encoder(next_obs)
        next_z_target = jax.lax.stop_gradient(
            next_z_target
        )
        # ===============================================================

        # Create value target
        # ===============================================================
        next_value_Q_target = jax.lax.stop_gradient(
            reward + self.discount * (1.0 - terminal) * self.value_fn(next_z_target).squeeze(axis=-1)
        )
        # ===============================================================

        # Create safety targets
        # ===============================================================
        sampled_actions = jax.random.uniform(
            key,
            shape=(obs.shape[0], Q_minima_samples, action.shape[-1]),
            minval=action_bounds[0], maxval=action_bounds[1],
        ).reshape(-1, action.shape[-1])

        q_target = self.safety_critic.compute_targets(
            self.target_safety_critic, next_z_target, sampled_actions,
            safety_cost, terminal, self.discount, Q_minima_samples,
        )
        # ===============================================================

        def loss_fn(trainable_partition: TrainableBundle, target_partition: TargetBundle) -> dict:
            # Extract trainable networks
            enc = trainable_partition.encoder
            dyn = trainable_partition.dynamics
            val_fn = trainable_partition.value_fn
            val_fn_Q = trainable_partition.value_Q
            safety_Q = trainable_partition.safety_critic
            rew_fn = trainable_partition.reward_fn

            # Extract target networks
            target_val_fn_Q = target_partition.value_Q

            # Encoder and Dynamics Loss
            # ===============================================================
            # Encode observations to latent space
            z = enc(obs)

            # Get next z prediction
            next_z = dyn(z, action, update_spectral_norm=True)

            # Get latent loss
            loss_z = jnp.mean((next_z - next_z_target) ** 2)
            # ===============================================================

            # Reward Loss
            # ===============================================================
            r_pred = rew_fn(z, action, update_spectral_norm=True).squeeze(axis=-1)
            loss_r = jnp.mean((r_pred - reward) ** 2)
            # ===============================================================

            # Value Loss
            # ===============================================================
            # Get value prediction
            v_pred = val_fn(z, update_spectral_norm=True).squeeze(axis=-1)

            # Get value loss
            next_value_target = jax.lax.stop_gradient(
                target_val_fn_Q(z, action)
            ).squeeze(axis=-1)
            
            loss_v_v = jnp.mean(
                expectile_regression_loss(next_value_target - v_pred, self.ER_tau)
            )

            q_pred = val_fn_Q(z, action).squeeze(axis=-1)
            loss_v_q = jnp.mean((next_value_Q_target - q_pred) ** 2)

            loss_v = loss_v_q + loss_v_v
            # ===============================================================

            # VicReg Loss
            # ===============================================================
            # Obtain covariace matrix (d, d)
            # Note rowvar=False is because the row dimension is what we want to get the cov of
            cov_mat = jnp.cov(z, rowvar=False)
            d = cov_mat.shape[0]

            # Calculate vireg variance loss with digonal terms
            std = jnp.sqrt(jnp.diagonal(cov_mat) + 1e-4)
            loss_var = jnp.mean(jnp.maximum(0.0, self.gamma - std))

            # Calculate vicreg covariance loss with off diagonal terms
            off_diag = cov_mat - jnp.diag(jnp.diagonal(cov_mat))
            loss_cov = jnp.sum(off_diag ** 2) / d

            # Compute vicreg loss
            loss_vicreg = self.lambda_var * loss_var + self.lambda_cov * loss_cov
            # ===============================================================

            # Compute safety constraint loss
            # ===============================================================
            loss_s, q_aux = safety_Q.compute_loss(
                z,
                action,
                q_target,
                sampled_actions,
                action_bounds,
                ens_mask,
                self.cql_alpha,
                Q_minima_samples
            )
            # ===============================================================

            # Total world model loss
            total_loss = (self.lambda_dyn * loss_z + self.lambda_v * loss_v + 
                          loss_vicreg + self.lambda_s * loss_s + self.lambda_r * loss_r)

            # Record Metrics
            metrics = {
                "loss_total": total_loss,
                "loss_dyn": loss_z,
                "loss_v": loss_v,
                "loss_safety": loss_s,
                "loss_var": loss_var,
                "loss_cov": loss_cov,
                "loss_r": loss_r,
                **q_aux,
                "q_target_min": jnp.min(q_target),
                "q_target_max": jnp.max(q_target),
                "sigma_dyn_0":  dyn.layers[0].sigma.value
            }
            return total_loss, metrics

        # Calculate losses and 
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grad = grad_fn(self.trainable_nodes)
        self.optimiser.update(grad)

        # Update target networks
        self.update_target_networks(self.tau)
        return metrics