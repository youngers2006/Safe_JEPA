import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# import modules
from World_Model.Networks import ValueNet, Encoder, DynamicsPredictor, RewardPredictor, SpectralStat
from World_Model.Q_safety_critic import SafetyCriticEnsemble, QSafetyCritic

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
        self.rngs = rngs
        # ==============================================================

        self.encoder = Encoder(
            cfg=cfg["EncoderParams"],
            mu=obs_mean,
            std=obs_std,
            rngs=rngs
        )

        self.target_encoder = Encoder(
            cfg=cfg["EncoderParams"],
            mu=obs_mean,
            std=obs_std,
            rngs=rngs
        )

        self.value_fn = ValueNet(
            cfg=cfg["ValueParams"],
            rngs=rngs
        )

        self.target_value_fn = ValueNet(
            cfg=cfg["ValueParams"],
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
        nnx.update(self.target_value_fn, nnx.state(self.value_fn, nnx.Param))
        nnx.update(self.target_value_fn, nnx.state(self.value_fn, SpectralStat))
        nnx.update(self.target_safety_critic, nnx.state(self.safety_critic, nnx.Param))
        nnx.update(self.target_safety_critic, nnx.state(self.safety_critic, SpectralStat))

        self.dynamics = DynamicsPredictor(
            cfg=cfg["DynamicsParams"],
            rngs=rngs
        )

        self.reward_fn = RewardPredictor(
            cfg=cfg["DynamicsParams"],
            rngs=rngs
        )

        self.trainable_nodes = nnx.List(
            [self.encoder, self.dynamics, self.value_fn, self.safety_critic, self.reward_fn]
        )
        self.target_nodes = nnx.List(
            [self.target_encoder, self.target_value_fn, self.target_safety_critic]
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
    def update_target_value_fn(self, tau: float = 0.01) -> None:
        # Extract both param sets
        online_params = nnx.state(self.value_fn, nnx.Param)
        target_params = nnx.state(self.target_value_fn, nnx.Param)

        # Use moving average to update target encoder
        new_target_params = optax.incremental_update(
            new_tensors=online_params,
            old_tensors=target_params,
            step_size=tau
        )

        # Update the target encoder state
        nnx.update(self.target_value_fn, new_target_params)
        nnx.update(self.target_value_fn, nnx.state(self.value_fn, SpectralStat))

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
        self.update_target_value_fn(tau_vals[2])

    @nnx.jit
    def train_step(
        self,
        obs: jax.Array, 
        next_obs: jax.Array, 
        action: jax.Array, 
        reward: jax.Array, 
        safety_cost: jax.Array, 
        terminal: jax.Array, 
        Q_minima_samples: int = 64,
        action_bounds: tuple[float, float] = (-1.0, 1.0)
    ) -> dict:
        def loss_fn(trainable_partition: nnx.List) -> dict:
            # Extract trainable networks
            enc, dyn, val_fn, safety_Q, rew_fn = trainable_partition

            # Encoder and Dynamics Loss
            # ===============================================================
            # Encode observations to latent space
            z = enc(obs)

            # create z targets
            next_z_target = self.target_encoder(next_obs)
            next_z_target = jax.lax.stop_gradient(next_z_target)

            # Get next z prediction
            next_z = dyn(z, action, update_spectral_norm=True)

            # Get latent loss
            loss_z = jnp.mean((next_z - next_z_target) ** 2)
            # ===============================================================

            # Reward Loss
            # ===============================================================
            r_pred = rew_fn(z, action, update_spectral_norm=True).squeeze()
            loss_r = jnp.mean((r_pred - reward) ** 2)
            # ===============================================================

            # Value Loss
            # ===============================================================
            next_v_target = self.target_value_fn(next_z_target, update_spectral_norm=False).squeeze()
            target_v = reward + self.discount * (1.0 - terminal) * next_v_target

            # Get value prediction
            v_pred = val_fn(z, update_spectral_norm=True).squeeze()

            # Get value loss
            loss_v = jnp.mean((v_pred - target_v) ** 2)
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
            loss_s = safety_Q.compute_loss(
                self.target_safety_critic,
                z,
                next_z_target,
                action,
                safety_cost,
                terminal,
                self.discount,
                self.cql_alpha,
                action_bounds,
                self.rngs,
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
                "loss_r": loss_r
            }
            return total_loss, metrics

        # Calculate losses and 
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grad = grad_fn(self.trainable_nodes)
        self.optimiser.update(grad)

        # Update target networks
        self.update_target_networks(self.tau)
        return metrics