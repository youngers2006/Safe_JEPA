import jax
import jax.numpy as jnp
import flax.nnx as nnx

class ConvStem(nnx.Module):
    def __init__(self, in_channels: int, rngs: nnx.Rngs):
        # Channels: in_channels -> 32 -> 32 -> 32 -> 32 
        # Standard DrQ-V2 CNN setup, uses Valid padding
        self.conv1 = nnx.Conv(in_channels, 32, kernel_size=(3, 3), strides=(2, 2), padding='VALID', rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), padding='VALID', rngs=rngs)
        self.conv3 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), padding='VALID', rngs=rngs)
        self.conv4 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), padding='VALID', rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Run Convnet
        x = nnx.silu(self.conv1(x))
        x = nnx.silu(self.conv2(x))
        x = nnx.silu(self.conv3(x))
        x = nnx.silu(self.conv4(x))
        
        # Flatten the spatial dimension
        batch_size = x.shape[0]
        return x.reshape((batch_size, -1))

class MLPStem(nnx.Module):
    def __init__(self, obs_dim: int, d_hidden: int, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(obs_dim, d_hidden, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(d_hidden, use_scale=True, use_bias=True, epsilon=1e-5, rngs=rngs)
        self.layer2 = nnx.Linear(d_hidden, d_hidden, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(d_hidden, use_scale=True, use_bias=True, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.silu(
            self.layer_norm1(
                self.layer1(x)
            )
        )
        x = nnx.silu(
            self.layer_norm2(
                self.layer2(x)
            )
        )
        return x