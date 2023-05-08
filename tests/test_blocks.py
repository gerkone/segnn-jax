import e3nn_jax as e3nn
import haiku as hk
import pytest
from e3nn_jax.util import assert_equivariant

from segnn_jax import O3TensorProduct, O3TensorProductGate, O3TensorProductLegacy


@pytest.mark.parametrize("biases", [False, True])
def test_linear(key, biases):
    f = lambda x1, x2: O3TensorProduct("1x1o", biases=biases)(x1, x2)
    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    wrapper = lambda x1, x2: f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


@pytest.mark.parametrize("biases", [False, True])
def test_gated(key, biases):
    import segnn_jax.blocks

    segnn_jax.blocks.O3Layer = segnn_jax.blocks.O3TensorProduct

    f = lambda x1, x2: O3TensorProductGate("1x1o", biases=biases)(x1, x2)
    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    wrapper = lambda x1, x2: f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


@pytest.mark.parametrize("biases", [False, True])
def test_linear_legacy(key, biases):
    f = lambda x1, x2: O3TensorProductLegacy("1x1o", biases=biases)(x1, x2)
    f = hk.without_apply_rng(hk.transform(f))

    v = e3nn.normal("1x1o", key, (5,))
    params = f.init(key, v, v)

    wrapper = lambda x1, x2: f.apply(params, x1, x2)

    assert_equivariant(
        wrapper,
        key,
        args_in=(e3nn.normal("1x1o", key, (5,)), e3nn.normal("1x1o", key, (5,))),
    )


if __name__ == "__main__":
    pytest.main()
