import jax
import jax.numpy as jnp
import flax.nnx as nnx

class NormStat(nnx.Variable):
    """Non-trainable standardisation constant"""
    pass

class SpectralStat(nnx.Variable):
    """Power-iteration state: derived from W, not learned, not data-dependent."""
    pass