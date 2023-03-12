import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp
import jraph
import pytest
from e3nn_jax.util import assert_equivariant

from segnn_jax import SEGNN, SteerableGraphsTuple, weight_balanced_irreps


@pytest.fixture
def rand_graph():
    def _rand_graph(n_graphs: int = 1):
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
            node_attributes=e3nn.IrrepsArray("1x0e+1x1o", jnp.ones((n_graphs * 5, 4))),
        )

    return _rand_graph


@pytest.mark.parametrize("task", ["graph", "node"])
@pytest.mark.parametrize("norm", ["none", "instance"])
def test_equivariance(key, rand_graph, norm, task):
    import segnn_jax.config
    segnn_jax.config.set_config("o3_layer", "new")

    segnn = lambda x: SEGNN(
        hidden_irreps=weight_balanced_irreps(8, e3nn.Irreps.spherical_harmonics(1)),
        output_irreps=e3nn.Irreps("1x1o"),
        num_layers=1,
        task=task,
        norm=norm,
    )(x)
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    graph = rand_graph()
    params, segnn_state = segnn.init(key, graph)

    def wrapper(x):
        st_graph = graph._replace(
            graph=graph.graph._replace(nodes=x),
            node_attributes=e3nn.spherical_harmonics("1x0e+1x1o", x, normalize=True),
        )
        y, _ = segnn.apply(params, segnn_state, st_graph)
        return e3nn.IrrepsArray("1x1o", y)

    assert_equivariant(wrapper, key, args_in=(e3nn.normal("1x1o", key, (5,)),))


@pytest.mark.parametrize("task", ["graph", "node"])
@pytest.mark.parametrize("norm", ["none", "instance"])
def test_equivariance_legacy(key, rand_graph, norm, task):
    import segnn_jax.config
    segnn_jax.config.set_config("o3_layer", "legacy")

    segnn = lambda x: SEGNN(
        hidden_irreps=weight_balanced_irreps(8, e3nn.Irreps.spherical_harmonics(1)),
        output_irreps=e3nn.Irreps("1x1o"),
        num_layers=1,
        task=task,
        norm=norm,
    )(x)
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    graph = rand_graph()
    params, segnn_state = segnn.init(key, graph)

    def wrapper(x):
        st_graph = graph._replace(
            graph=graph.graph._replace(nodes=x),
            node_attributes=e3nn.spherical_harmonics("1x0e+1x1o", x, normalize=True),
        )
        y, _ = segnn.apply(params, segnn_state, st_graph)
        return e3nn.IrrepsArray("1x1o", y)

    assert_equivariant(wrapper, key, args_in=(e3nn.normal("1x1o", key, (5,)),))
