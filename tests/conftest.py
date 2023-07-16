import os

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import pytest

from segnn_jax import SteerableGraphsTuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@pytest.fixture
def dummy_graph():
    def _rand_graph(n_graphs: int = 1, attr_irreps: str = "1x0e + 1x1o"):
        attr_irreps = e3nn.Irreps(attr_irreps)
        return SteerableGraphsTuple(
            graph=jraph.GraphsTuple(
                nodes=e3nn.IrrepsArray("1x1o", jnp.ones((n_graphs * 5, 3))),
                edges=None,
                senders=jnp.zeros((10 * n_graphs,), dtype=jnp.int32),
                receivers=jnp.zeros((10 * n_graphs,), dtype=jnp.int32),
                n_node=jnp.array([5] * n_graphs),
                n_edge=jnp.array([10] * n_graphs),
                globals=None,
            ),
            additional_message_features=None,
            edge_attributes=None,
            node_attributes=e3nn.IrrepsArray(
                attr_irreps, jnp.ones((n_graphs * 5, attr_irreps.dim))
            ),
        )

    return _rand_graph


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)
