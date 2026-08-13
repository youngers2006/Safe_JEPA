import jax
import jax.numpy as jnp
import flax.nnx as nnx

class Encoder(nnx.Module):
    def __init__(self, in_channels: int, flattened_dim: int, d_latent: int, *, rngs: nnx.Rngs):
        # Channels: in_channels -> 32 -> 32 -> 32 -> 32 
        # Standard DrQ-V2 CNN setup, uses Valid padding
        self.conv1 = nnx.Conv(in_channels, 32, kernel_size=(3, 3), strides=(2, 2), rngs=rngs)
        self.conv2 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), rngs=rngs)
        self.conv3 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), rngs=rngs)
        self.conv4 = nnx.Conv(32, 32, kernel_size=(3, 3), strides=(1, 1), rngs=rngs)

        # Processing Network
        self.linear_proj = nnx.Linear(flattened_dim, d_latent, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(d_latent, use_scale=True, use_bias=True, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Run Convnet
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.conv4(x))
        
        # Flatten the spatial dimension
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        
        # Project and normalize
        x = self.linear_proj(x)
        x = self.layer_norm(x)
        return nnx.tanh(x)

class SpectralNormLinear(nnx.Module):
    def __init__(self, d_in, d_out, *, rngs):
        self.network = nnx.Linear(d_in, d_out, rngs=rngs)

        # Largest singular value (define as variable to allow updates when jit)
        self.sigma = nnx.Variable(jnp.ones(()))

        # Direction vector for power iteration
        u_init = jax.random.normal(rngs.params(), (d_out,))
        u_init = u_init / (jnp.linalg.norm(u_init) + 1e-8)
        self.u = nnx.Variable(u_init)

    def power_iteration(self, W: jax.Array) -> None:
        # Extract direction vector
        u_val = self.u.value

        # Compute transformed unit vector
        v = W @ u_val
        v = v / (jnp.linalg.norm(v) + 1e-8)

        # Recompute rotated direction vector
        u_val = v @ W
        u_val = u_val / (jnp.linalg.norm(u_val) + 1e-8)

        # Use Raleigh quotient to obtain largest singular value
        sigma_val = jnp.dot(u_val, v @ W)

        # Stop gradients to make sure neither variable is updated but GD
        self.u.value = jax.lax.stop_gradient(u_val)
        self.sigma.value = jax.lax.stop_gradient(sigma_val)

    def __call__(self, x: jax.Array, update_spectral_norm: bool = True) -> jax.Array:
        # Extract weight matrix (d_in, d_out)
        W = self.network.kernel.value

        # Use power iteration to update spectral norm
        if update_spectral_norm:
            self.power_iteration(W)

        # Normalise weight matrix
        W_sn = W / self.sigma.value
        y = x @ W_sn

        # Add bias if used
        if self.network.bias is not None:
            y = y + self.network.bias.value
        return y
    
class DynamicsPredictor(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        self.layers = []
        
        current_dim = d_in
        
        for h in hidden_features:
            self.layers.append(
                SpectralNormLinear(current_dim, h, rngs=rngs)
            )
            self.layers.append(
                nnx.LayerNorm(
                    h, use_scale=False, use_bias=False, epsilon=1e-5, rngs=rngs
                )
            )
            current_dim = h
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

class ValueNet(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, *, rngs: nnx.Rngs):
        self.hidden_features = hidden_features
        self.layers = []
            
        current_dim = d_in
        
        for h in hidden_features:
            self.layers.append(
                    SpectralNormLinear(current_dim, h, rngs=rngs)
            )
            self.layers.append(
                nnx.LayerNorm(
                    h, use_scale=False, use_bias=False, epsilon=1e-5, rngs=rngs
                )
            )
            current_dim = h
        self.output_layer = SpectralNormLinear(current_dim, d_out, rngs=rngs)
             
    def __call__(self, z: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        for l in range(0, len(self.layers), 2):
            linear_layer = self.layers[l]
            norm_layer = self.layers[l+1]

            z = linear_layer(z, update_spectral_norm)
            z = norm_layer(z)
            z = nnx.silu(z)
        
        return self.output_layer(z, update_spectral_norm)