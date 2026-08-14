import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# import modules
from Networks import ValueNet, Encoder, DynamicsPredictor, RewardPredictor
from Q_safety_critic import Q_safety_critic

class WorldModel(nnx.Module):
    def __init__(
            self, 
            d_in_obs: int, 
            image_size: int, 
            d_latent: int, 
            d_action: int, 
            lr: float, 
            gamma: float = 1.0, 
            discount: float = 0.99, 
            alpha: float = 1.0,
            lambda_dyn: float = 1.0,
            lambda_v: float = 0.1,
            lambda_r: float = 1.0,
            lambda_s: float = 1.0,
            lambda_var: float = 10.0,
            *, 
            rngs: nnx.Rngs
        ):
        d = image_size
        d = (d - 3) // 2 + 1  # Conv1: Kernel 3, Stride 2
        d = d - 3 + 1         # Conv2: Kernel 3, Stride 1
        d = d - 3 + 1         # Conv3: Kernel 3, Stride 1
        d = d - 3 + 1         # Conv4: Kernel 3, Stride 1
        flattened_dim = d * d * 32

        self.encoder = Encoder(
            in_channels=d_in_obs,
            flattened_dim=flattened_dim,
            d_latent=d_latent,
            rngs=rngs
        )

        self.value_fn = ValueNet(
            d_in=d_latent,
            hidden_features=(256, 256),
            d_out=1,
            rngs=rngs
        )

        self.target_encoder = Encoder(
            in_channels=d_in_obs,
            flattened_dim=flattened_dim,
            d_latent=d_latent,
            rngs=rngs
        )

        self.target_value_fn = ValueNet(
            d_in=d_latent,
            hidden_features=(256, 256),
            d_out=1,
            rngs=rngs
        )

        self.safety_critic = Q_safety_critic(
            d_in=d_latent + d_action,
            hidden_features=(256, 256),
            d_out=1,
            rngs=rngs
                    
        )
        
        self.target_safety_critic = Q_safety_critic(
            d_in=d_latent + d_action,
            hidden_features=(256, 256),
            d_out=1,
            rngs=rngs
                    
        )

        nnx.update(self.target_encoder, nnx.state(self.encoder, nnx.Param))
        nnx.update(self.target_value_fn, nnx.state(self.value_fn, nnx.Param))
        nnx.update(self.target_safety_critic, nnx.state(self.safety_critic, nnx.Param))

        self.dynamics = DynamicsPredictor(
            d_in=d_latent + d_action,
            hidden_features=(256, 256),
            d_out=d_latent,
            rngs=rngs
        )

        self.reward_fn = RewardPredictor(
            d_in=d_latent + d_action,
            hidden_features=(256, 256),
            d_out=d_latent,
            rngs=rngs
        )

        self.trainable_nodes = (self.encoder, self.dynamics, self.value_fn, self.safety_critic, self.reward_fn)
        self.target_nodes = (self.target_encoder, self.target_value_fn, self.target_safety_critic)
        self.optimiser = nnx.Optimizer(self.trainable_nodes, optax.adam(learning_rate=lr), wrt=nnx.Param)

        self.lambda_dyn = lambda_dyn
        self.lambda_v = lambda_v
        self.lambda_r = lambda_r
        self.lambda_s = lambda_s
        self.lambda_var = lambda_var

        self.discount = discount
        self.gamma = gamma # VicReg variance threshold
        self.cql_alpha = alpha # CQL penalty weight
        self.rngs = rngs

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

    @nnx.jit
    def update_target_networks(self, tau_vals:tuple[float, ...]) -> None:
        self.update_target_encoder(tau_vals[0])
        self.update_target_safety_critic(tau_vals[1])
        self.update_target_value_fn(tau_vals[2])

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

    @nnx.jit
    def train_step(
        self, 
        obs: jax.Array, 
        next_obs: jax.Array, 
        action: jax.Array, 
        reward: jax.Array, 
        safety_cost: jax.Array, 
        done: jax.Array, 
        Q_minima_samples: int = 64,
        action_bounds: tuple[float, float] = (-1.0, 1.0)
    ) -> jax.Array:
        def loss_fn(trainable_partition: tuple[jax.Array, ...]) -> dict:
            # Extract trainable networks
            enc, dyn, val_fn, safety_Q, rew_fn = trainable_partition

            # Encode observations to latent space
            z = enc(obs)

            # create z targets
            next_z_target = self.target_encoder(next_obs)
            next_z_target = jax.lax.stop_gradient(next_z_target)

            # Get next z prediction
            next_z = dyn(z, action, update_spectral_norm=True)

            # Get latent loss
            loss_z = jnp.mean((next_z - next_z_target) ** 2)

            # Get reward loss
            r_pred = rew_fn(z, action, update_spectral_norm=True).squeeze()
            loss_r = jnp.mean((r_pred - reward) ** 2)

            # Get next value bellman target
            next_v_target = self.target_value_fn(next_z_target, update_spectral_norm=False).squeeze()
            target_v = reward + self.discount * (1.0 - done) * next_v_target

            # Get value prediction
            v_pred = val_fn(z, update_spectral_norm=True).squeeze()

            # Get value loss
            loss_v = jnp.mean((v_pred - target_v) ** 2)

            # Get std.dev of data in batch and ensure it remains above threshold to prevent collapse
            std = jnp.sqrt(jnp.var(z, axis=0) + 1e-4)
            loss_vicreg = jnp.mean(jax.nn.relu(self.gamma - std))

            loss_s = safety_Q.compute_loss(
                self.target_safety_critic,
                z,
                next_z_target,
                action,
                safety_cost,
                done,
                self.discount,
                self.cql_alpha,
                action_bounds,
                rng_key,
                Q_minima_samples
            )

            # Total world model loss
            total_loss = (self.lambda_dyn * loss_z + self.lambda_v * loss_v + 
                          self.lambda_var * loss_vicreg + self.lambda_s * loss_s + self.lambda_r * loss_r)

            metrics = {
                "loss_total": total_loss,
                "loss_dyn": loss_z,
                "loss_v": loss_v,
                "loss_safety": loss_s,
                "loss_var": loss_vicreg,
                "loss_r": loss_r
            }
            return total_loss, metrics

        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, wrt=nnx.Param)
        (loss, metrics), grad = grad_fn(self.trainable_nodes)
        self.optimiser.update(grad)
        return metrics