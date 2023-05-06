from typing import Callable, Tuple

from torch.utils.data import DataLoader

from .nbody.utils import setup_nbody_data
from .qm9.utils import setup_qm9_data
from .train import train

__all__ = ["setup_data", "train"]


__setup_conf = {
    "qm9": setup_qm9_data,
    "charged": setup_nbody_data,
    "gravity": setup_nbody_data,
}


def setup_data(args) -> Tuple[DataLoader, DataLoader, DataLoader, Callable, Callable]:
    assert args.dataset in [
        "qm9",
        "charged",
        "gravity",
    ], f"Unknown dataset {args.dataset}"
    setup_fn = __setup_conf[args.dataset]
    return setup_fn(args)
