import e3nn_jax as e3nn
import haiku as hk
import pytest
from e3nn_jax.utils import assert_equivariant

from segnn_jax import (
    SEGNN,
    O3TensorProduct,
    O3TensorProductFC,
    O3TensorProductSCN,
    weight_balanced_irreps,
)


@pytest.mark.parametrize("task", ["graph", "node"])
@pytest.mark.parametrize("norm", ["none", "instance"])
@pytest.mark.parametrize(
    "O3Layer", [O3TensorProduct, O3TensorProductFC, O3TensorProductSCN]
)
def test_segnn_equivariance(key, dummy_graph, task, norm, O3Layer):
    scn = O3Layer == O3TensorProductSCN

    hidden_irreps = weight_balanced_irreps(
        8, e3nn.Irreps.spherical_harmonics(1), use_sh=not scn
    )

    def segnn(x):
        return SEGNN(
            hidden_irreps=hidden_irreps,
            output_irreps=e3nn.Irreps("1x1o"),
            num_layers=1,
            task=task,
            norm=norm,
            o3_layer=O3Layer,
        )(x)

    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    if scn:
        attr_irreps = e3nn.Irreps("1x1o")
    else:
        attr_irreps = e3nn.Irreps("1x0e+1x1o")

    graph = dummy_graph(attr_irreps=attr_irreps)
    params, segnn_state = segnn.init(key, graph)

    def wrapper(x):
        if scn:
            attrs = e3nn.IrrepsArray(attr_irreps, x.array)
        else:
            attrs = e3nn.spherical_harmonics(attr_irreps, x, normalize=True)
        st_graph = graph._replace(
            graph=graph.graph._replace(nodes=x),
            node_attributes=attrs,
        )
        y, _ = segnn.apply(params, segnn_state, st_graph)
        return e3nn.IrrepsArray("1x1o", y)

    assert_equivariant(wrapper, key, e3nn.normal("1x1o", key, (5,)))


if __name__ == "__main__":
    pytest.main()
