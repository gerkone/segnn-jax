import e3nn_jax as e3nn
import haiku as hk
import pytest
from e3nn_jax.utils import assert_equivariant

from segnn_jax import O3TensorProduct, O3TensorProductFC, O3TensorProductGate


@pytest.mark.parametrize("biases", [False, True])
def test_linear(key, biases):
    def f(x1, x2):
        return O3TensorProduct("1x1o", biases=biases)(x1, x2)

    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    def wrapper(x1, x2):
        return f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


@pytest.mark.parametrize("biases", [False, True])
def test_linear_fully_connected(key, biases):
    def f(x1, x2):
        return O3TensorProductFC("1x1o", biases=biases)(x1, x2)

    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    def wrapper(x1, x2):
        return f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


@pytest.mark.parametrize("biases", [False, True])
def test_gated(key, biases):
    def f(x1, x2):
        return O3TensorProductGate("1x1o", biases=biases, o3_layer=O3TensorProduct)(
            x1, x2
        )

    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    def wrapper(x1, x2):
        return f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


@pytest.mark.parametrize("biases", [False, True])
def test_linear_legacy(key, biases):
    def f(x1, x2):
        return O3TensorProductFC("1x1o", biases=biases, o3_layer=O3TensorProductFC)(
            x1, x2
        )

    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    def wrapper(x1, x2):
        return f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


if __name__ == "__main__":
    pytest.main()
