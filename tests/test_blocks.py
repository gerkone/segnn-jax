import e3nn_jax as e3nn
import haiku as hk
import pytest
from e3nn_jax.utils import assert_equivariant

from segnn_jax import (
    O3TensorProduct,
    O3TensorProductFC,
    O3TensorProductGate,
    O3TensorProductSCN,
)


@pytest.mark.parametrize("biases", [False, True])
@pytest.mark.parametrize(
    "O3Layer", [O3TensorProduct, O3TensorProductFC, O3TensorProductSCN]
)
def test_linear_layers(key, biases, O3Layer):
    def f(x1, x2):
        return O3Layer("1x1o", biases=biases)(x1, x2)

    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    def wrapper(x1, x2):
        return f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        e3nn.normal("1x1o", key, (5,)),
        e3nn.normal("1x1o", key, (5,)),
    )


@pytest.mark.parametrize("biases", [False, True])
@pytest.mark.parametrize(
    "O3Layer", [O3TensorProduct, O3TensorProductFC, O3TensorProductSCN]
)
def test_gated_layers(key, biases, O3Layer):
    def f(x1, x2):
        return O3TensorProductGate("1x1o", biases=biases, o3_layer=O3Layer)(x1, x2)

    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    def wrapper(x1, x2):
        return f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        e3nn.normal("1x1o", key, (5,)),
        e3nn.normal("1x1o", key, (5,)),
    )


if __name__ == "__main__":
    pytest.main()
