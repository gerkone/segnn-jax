from .blocks import O3TensorProduct, O3TensorProductGate
from .graph_utils import SteerableGraphsTuple
from .irreps_computer import balanced_irreps, weight_balanced_irreps
from .segnn import SEGNN, SEGNNLayer

__all__ = [
    "SEGNN",
    "SEGNNLayer",
    "O3TensorProduct",
    "O3TensorProductGate",
    "weight_balanced_irreps",
    "balanced_irreps",
    "SteerableGraphsTuple",
]

__version__ = "0.3"
