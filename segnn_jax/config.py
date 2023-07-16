import jax.numpy as jnp

__conf = {
    "gradient_normalization": "element",  # "element" or "path"
    "path_normalization": "element",  # "element" or "path"
    "default_dtype": jnp.float32,
    "o3_layer": "tpl",  # "tpl" (tp + Linear) or "fctp" (FullyConnected) or "scn" (SCN)
}


def config(key):
    return __conf[key]
