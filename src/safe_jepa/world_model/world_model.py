import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from functools import partial

# import modules
from ..nn.mlp import ValueNet, ValueQNet, DynamicsPredictor, RewardPredictor, MarginNet
from ..nn.variables import SpectralStat
from .encoder import Encoder

class TrainableBundle(nnx.Module):
    def __init__(self, encoder, dynamics, value_fn, value_Q, margin_fn, reward_fn):
        self.encoder = encoder
        self.dynamics = dynamics
        self.value_fn = value_fn
        self.value_Q = value_Q
        self.margin_fn = margin_fn
        self.reward_fn = reward_fn

class TargetBundle(nnx.Module):
    def __init__(self, encoder, value_Q):
        self.encoder = encoder
        self.value_Q = value_Q

def expectile_regression_loss(u, tau):
    return jnp.where(u > 0, tau, 1 - tau) * (u **2)

def masked_mean(x, mask):
    return jnp.sum(mask * x) / jnp.maximum(jnp.sum(mask), 1.0)

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
        self.lambda_dyn = cfg["lambda_dyn"]
        self.lambda_v = cfg["lambda_v"]
        self.lambda_r = cfg["lambda_r"]
        self.lambda_m = cfg["lambda_m"]
        self.lambda_var = cfg["lambda_var"]
        self.lambda_cov = cfg["lambda_cov"]
        self.tau = cfg["tau"]

        lr = cfg["lr"]
        self.discount = cfg["discount"]
        self.gamma = cfg["gamma"] # VicReg variance threshold
        self.ER_tau = cfg["ER_tau"] # Expectile regression weighting
        self.rho = cfg["rho"] # Weight rho^t on losses at rollout step t

        self.margin_scale = tuple(c["scale"] for c in cfg["margin_constraints"])
        self.margin_edge_weight = cfg["margin_edge_weight"]
        self.margin_edge_scale = cfg["margin_edge_scale"]

        self.dense_reward = cfg["dense_reward"]
        self.goal_threshold = cfg["goal_threshold"]
        self.goal_reward = cfg["goal_reward"]

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

        nnx.update(self.target_encoder, nnx.state(self.encoder, nnx.Param))
        nnx.update(self.target_value_Q, nnx.state(self.value_Q, nnx.Param))
        nnx.update(self.target_value_Q, nnx.state(self.value_Q, SpectralStat))

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

        self.margin_fn = MarginNet(
            cfg=cfg["MarginParams"],
            rngs=rngs
        )

        self.trainable_nodes = TrainableBundle(
            self.encoder, 
            self.dynamics, 
            self.value_fn,
            self.value_Q,
            self.margin_fn, 
            self.reward_fn
        )
        self.target_nodes = TargetBundle(
            self.target_encoder, 
            self.target_value_Q
        )
        self.optimiser = nnx.Optimizer(
            self.trainable_nodes, 
            optax.chain((optax.clip_by_global_norm(20.0)), optax.adam(learning_rate=lr)), 
            wrt=nnx.Param
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
    def update_target_networks(self, tau_vals:tuple[float, ...]) -> None:
        self.update_target_encoder(tau_vals[0])
        self.update_target_value_Q(tau_vals[1])

    @partial(nnx.jit, static_argnames="update")
    def train_step(
        self,
        obs: jax.Array, # (B, H, obs_dim)
        next_obs: jax.Array, # (B, H, obs_dim)
        action: jax.Array, # (B, H, d_u)
        reward: jax.Array, # (B, H)
        margin: jax.Array, # (B, H, K) hazard margin at next_obs
        terminal: jax.Array, # (B, H)
        mask: jax.Array, # (B, H)
        update: bool = True
    ) -> dict:
        # B windows of H consecutive transitions
        B, H, d_u = action.shape

        # Create z targets
        next_z_target = self.target_encoder(
            next_obs.reshape((B * H,) + next_obs.shape[2:])
        ).reshape(B, H, -1)
        next_z_target = jax.lax.stop_gradient(
            next_z_target
        )

        # Create value target
        next_value_Q_target = jax.lax.stop_gradient(
            reward + self.discount * (1.0 - terminal) * self.value_fn(next_z_target).squeeze(axis=-1)
        ) # (B, H)

        def loss_fn(trainable_partition: TrainableBundle, target_partition: TargetBundle) -> dict:
            # Extract trainable networks
            enc = trainable_partition.encoder
            dyn = trainable_partition.dynamics
            val_fn = trainable_partition.value_fn
            val_fn_Q = trainable_partition.value_Q
            margin_fn = trainable_partition.margin_fn
            rew_fn = trainable_partition.reward_fn

            # Extract target networks
            target_val_fn_Q = target_partition.value_Q

            # Encoder and Dynamics Loss
            # Encode observations to latent space
            z0 = enc(obs[:, 0])
            z = z0

            total_loss = 0
            w_sum = 0.0
            step_metrics = {}
            for t in range(H):
                a, m = action[:, t], mask[:, t]
                sn = (t == 0) and update # spectral norm update done once per train step

                # Get next z prediction
                next_z = dyn(z, a, update_spectral_norm=sn)
                loss_z = masked_mean(jnp.mean((next_z - next_z_target[:, t]) ** 2, axis=-1), m)

                # Reward Loss
                r_target = reward[:, t]
                if self.dense_reward:
                    r_target = r_target - self.goal_reward * (r_target > self.goal_threshold)
                r_pred = rew_fn(z, a, update_spectral_norm=sn).squeeze(axis=-1)
                loss_r = masked_mean((r_pred - r_target) ** 2, m)

                # Value Loss
                # Get value prediction
                v_pred = val_fn(z, update_spectral_norm=sn).squeeze(axis=-1)

                # Get value loss
                next_value_target = jax.lax.stop_gradient(
                    target_val_fn_Q(z, a)
                ).squeeze(axis=-1)
                
                loss_v_v = masked_mean(
                    expectile_regression_loss(next_value_target - v_pred, self.ER_tau), m
                )

                q_pred = val_fn_Q(z, a).squeeze(axis=-1)
                loss_v_q = masked_mean((next_value_Q_target[:, t] - q_pred) ** 2, m)

                loss_v = loss_v_q + loss_v_v

                # Compute margin constraint loss
                m_pred = margin_fn(next_z, update_spectral_norm=sn) # (B, K)
                scale = jnp.asarray(self.margin_scale)
                w_m = m[:, None] * (1.0 + self.margin_edge_weight * jnp.exp(-jnp.maximum(margin[:, t], 0.0) / (self.margin_edge_scale * scale)))
                loss_m = jnp.sum(w_m * ((m_pred - margin[:, t]) / scale) ** 2) / jnp.maximum(jnp.sum(w_m), 1e-8)
                
                # Total world model loss
                total_loss = total_loss + self.rho ** t * (self.lambda_dyn * loss_z + self.lambda_v * loss_v + 
                            self.lambda_m * loss_m + self.lambda_r * loss_r)
                w_sum = w_sum + self.rho ** t

                if t == 0:
                    loss_z_0, loss_v_0, loss_m_0, loss_r_0 = loss_z, loss_v, loss_m, loss_r
                step_metrics[f"loss_dyn_t{t}"] = loss_z

                z = next_z

            # VicReg Loss on the encoder output
            # Obtain covariace matrix (d, d)
            # Note rowvar=False is because the row dimension is what we want to get the cov of
            cov_mat = jnp.cov(z0, rowvar=False)
            d = cov_mat.shape[0]

            # Calculate vireg variance loss with digonal terms
            std = jnp.sqrt(jnp.diagonal(cov_mat) + 1e-4)
            loss_var = jnp.mean(jnp.maximum(0.0, self.gamma - std))

            # Calculate vicreg covariance loss with off diagonal terms
            off_diag = cov_mat - jnp.diag(jnp.diagonal(cov_mat))
            loss_cov = jnp.sum(off_diag ** 2) / d

            # Compute vicreg loss
            loss_vicreg = self.lambda_var * loss_var + self.lambda_cov * loss_cov

            # add vicreg loss to total loss
            total_loss = total_loss / w_sum + loss_vicreg

            # Record Metrics
            metrics = {
                "loss_total": total_loss,
                "loss_dyn": loss_z_0,
                "loss_v": loss_v_0,
                "loss_margin": loss_m_0,
                "loss_var": loss_var,
                "loss_cov": loss_cov,
                "loss_r": loss_r_0,
                **step_metrics,
                "sigma_dyn_0":  dyn.layers[0].sigma.value
            }
            return total_loss, metrics

        if not update:
            return loss_fn(self.trainable_nodes, self.target_nodes)[1]

        # Calculate losses and 
        grad_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=True)
        (loss, metrics), grad = grad_fn(self.trainable_nodes, self.target_nodes)
        self.optimiser.update(grad)

        # Update target networks
        self.update_target_networks(self.tau)
        return metrics