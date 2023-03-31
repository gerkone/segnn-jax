import jax.numpy as jnp

__conf = {
    "gradient_normalization": "element",  # "element" or "path"
    "path_normalization": "element",  # "element" or "path"
    "default_dtype": jnp.float32,
    "o3_layer": "new",  # "new" or "legacy"
}


def config(key):
    return __conf[key]
