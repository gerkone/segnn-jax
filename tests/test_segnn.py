import e3nn_jax as e3nn
import haiku as hk
import pytest
from e3nn_jax.util import assert_equivariant

from segnn_jax import SEGNN, weight_balanced_irreps


@pytest.mark.parametrize("task", ["graph", "node"])
@pytest.mark.parametrize("norm", ["none", "instance"])
def test_equivariance(key, dummy_graph, norm, task):
    import segnn_jax.blocks

    segnn_jax.blocks.O3Layer = segnn_jax.blocks.O3TensorProduct

    segnn = lambda x: SEGNN(
        hidden_irreps=weight_balanced_irreps(8, e3nn.Irreps.spherical_harmonics(1)),
        output_irreps=e3nn.Irreps("1x1o"),
        num_layers=1,
        task=task,
        norm=norm,
    )(x)
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    graph = dummy_graph()
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
def test_equivariance_legacy(key, dummy_graph, norm, task):
    import segnn_jax.blocks

    segnn_jax.blocks.O3Layer = segnn_jax.blocks.O3TensorProductLegacy

    segnn = lambda x: SEGNN(
        hidden_irreps=weight_balanced_irreps(8, e3nn.Irreps.spherical_harmonics(1)),
        output_irreps=e3nn.Irreps("1x1o"),
        num_layers=1,
        task=task,
        norm=norm,
    )(x)
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    graph = dummy_graph()
    params, segnn_state = segnn.init(key, graph)

    def wrapper(x):
        st_graph = graph._replace(
            graph=graph.graph._replace(nodes=x),
            node_attributes=e3nn.spherical_harmonics("1x0e+1x1o", x, normalize=True),
        )
        y, _ = segnn.apply(params, segnn_state, st_graph)
        return e3nn.IrrepsArray("1x1o", y)

    assert_equivariant(wrapper, key, args_in=(e3nn.normal("1x1o", key, (5,)),))


if __name__ == "__main__":
    pytest.main()
