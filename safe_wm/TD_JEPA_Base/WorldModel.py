import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# import modules
from Networks import SuccessorFeatures, ContextEncoder, DynamicsPredictor

class WorldModel(nnx.Module):
    def __init__(self, d_in_obs: int, image_size: int, d_latent: int, d_action: int, horizon: int, lr: float, *, rngs: nnx.Rngs):
        d = image_size
        d = (d - 3) // 2 + 1  # Conv1: Kernel 3, Stride 2
        d = d - 3 + 1         # Conv2: Kernel 3, Stride 1
        d = d - 3 + 1         # Conv3: Kernel 3, Stride 1
        d = d - 3 + 1         # Conv4: Kernel 3, Stride 1
        flattened_dim = d * d * 32

        self.encoder = ContextEncoder(
            in_channels=d_in_obs,
            flattened_dim=flattened_dim,
            d_latent=d_latent,
            rngs=rngs
        )

        self.target_encoder = ContextEncoder(
            in_channels=d_in_obs,
            flattened_dim=flattened_dim,
            d_latent=d_latent,
            rngs=rngs
        )
        nnx.update(self.target_encoder, nnx.state(self.encoder, nnx.Param))

        self.successor = SuccessorFeatures(
            d_in=d_latent + d_action,
            hidden_features=(256, 256),
            d_out=1,
            rngs=rngs
        )

        self.dynamics = DynamicsPredictor(
            d_in=d_latent + d_action,
            hidden_features=(256, 256),
            d_out=d_latent,
            rngs=rngs
        )

        self.trainable_nodes = (self.encoder, self.dynamics, self.successor)
        self.optimiser = nnx.Optimizer(self.trainable_nodes, optax.adam(learning_rate=lr), wrt=nnx.Param)

        self.horizon = horizon

    @nnx.jit
    def update_target_network(self, tau: float = 0.01) -> None:
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
    def train_step(self, batch: dict, lambda_z: float = 1.0, lambda_v: float = 1.0):
        # batch['obs']: (Batch, Horizon+1, 64, 64, Channels)
        # batch['actions']: (Batch, Horizon, d_action)
        # batch['td_targets']: (Batch, Horizon)
        obs = batch['obs']
        actions = batch['actions']
        td_targets = batch['td_targets']

        def loss_fn(trainable_partition):
            enc, dyn, succ = trainable_partition
            total_loss = 0.0

            z_hat = enc(obs[:, 0])

            for k in range(self.horizon):
                z_target = self.target_encoder(obs[:, k+1])
                z_target = jax.lax.stop_gradient(z_target)

                z_hat = dyn(z_hat, actions[:, k], update_spectral_norm=True)

                v_pred = succ(z_hat, actions[:, k], update_spectral_norm=True)

                loss_z = jnp.mean((z_hat - z_target) ** 2)
                loss_v = jnp.mean((v_pred.squeeze() - td_targets[:, k]) ** 2)

                total_loss += (lambda_z * loss_z) + (lambda_v * loss_v)

            return total_loss / self.horizon

        grad_fn = nnx.value_and_grad(loss_fn, wrt=nnx.Param)
        loss, grad = grad_fn(self.trainable_nodes)
        self.optimiser.update(grad)
        return loss