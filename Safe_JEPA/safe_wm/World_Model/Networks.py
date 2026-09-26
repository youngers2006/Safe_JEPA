import jax
import jax.numpy as jnp
import flax.nnx as nnx

class Encoder(nnx.Module):
    def __init__(self, cfg, rngs: nnx.Rngs):
        if cfg.obs_type == "vision":
            self.stem = ConvStem(cfg.in_channels, rngs=rngs)
            feature_dim = cfg.flattened_dim
        elif cfg.obs_type == "state":
            self.stem = MLPStem(cfg.obs_dim, cfg.d_hidden, rngs=rngs)
            feature_dim = cfg.d_hidden
        else:
            raise ValueError(f"unknown obs_type: {cfg.obs_type}")

        # Processing Network
        self.linear_proj = nnx.Linear(feature_dim, cfg.d_latent, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(cfg.d_latent, use_scale=True, use_bias=True, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Run stem network
        x = self.stem(x)
        
        # Project and normalize
        x = self.linear_proj(x)
        x = self.layer_norm(x)
        return nnx.tanh(x)

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

class SpectralStat(nnx.Variable):
    """Power-iteration state: derived from W, not learned, not data-dependent."""
    pass

class SpectralNormLinear(nnx.Module):
    def __init__(self, d_in, d_out, rngs):
        self.network = nnx.Linear(d_in, d_out, rngs=rngs)

        # Largest singular value (define as variable to allow updates when jit)
        self.sigma = SpectralStat(jnp.ones(()))

        # Direction vector for power iteration
        u_init = jax.random.normal(rngs.params(), (d_out,))
        u_init = u_init / (jnp.linalg.norm(u_init) + 1e-8)
        self.u = SpectralStat(u_init)

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
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, rngs: nnx.Rngs):
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

class ValueNet(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, rngs: nnx.Rngs):
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
             
    def __call__(self, z: jax.Array, update_spectral_norm: bool = False) -> jax.Array:
        for l in range(0, len(self.layers), 2):
            linear_layer = self.layers[l]
            norm_layer = self.layers[l+1]

            z = linear_layer(z, update_spectral_norm)
            z = norm_layer(z)
            z = nnx.silu(z)
        
        return self.output_layer(z, update_spectral_norm)

class RewardPredictor(nnx.Module):
    def __init__(self, d_in: int, hidden_features: tuple[int, ...], d_out: int, rngs: nnx.Rngs):
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