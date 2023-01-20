import os.path as osp
import pathlib
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union

import numpy as np

DATA_DIR = "data"


class BaseDataset(ABC):
    """Abstract n-body dataset class."""

    def __init__(
        self,
        data_type,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=5,
        normalize=False,
    ):
        self.partition = partition
        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        self.suffix += f"_{data_type}{n_bodies}_initvel1"
        self.data_type = data_type
        self.max_samples = int(max_samples)
        self.normalize = normalize

        self.data = None

    def get_n_nodes(self):
        return self.data[0].shape[2]

    def _get_partition_frames(self) -> Tuple[int, int]:
        if self.dataset_name == "default":
            frame_0, frame_target = 6, 8
        elif self.dataset_name == "small":
            frame_0, frame_target = 30, 40
        elif self.dataset_name == "small_out_dist":
            frame_0, frame_target = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return frame_0, frame_target

    def __len__(self) -> int:
        return len(self.data[0])

    def _load(self) -> Tuple[np.ndarray, ...]:
        filepath = pathlib.Path(__file__).parent.resolve()

        loc = np.load(osp.join(filepath, DATA_DIR, "loc_" + self.suffix + ".npy"))
        vel = np.load(osp.join(filepath, DATA_DIR, "vel_" + self.suffix + ".npy"))
        edges = np.load(osp.join(filepath, DATA_DIR, "edges_" + self.suffix + ".npy"))
        q = np.load(osp.join(filepath, DATA_DIR, "q_" + self.suffix + ".npy"))

        return loc, vel, edges, q

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        std = x.std(axis=0)
        x = x - x.mean(axis=0)
        return np.divide(x, std, out=x, where=std != 0)

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, *args) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError


class ChargedDataset(BaseDataset):
    """N-body charged dataset class."""

    def __init__(
        self,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=5,
        normalize=False,
    ):
        super().__init__(
            "charged", partition, max_samples, dataset_name, n_bodies, normalize
        )
        self.data, self.edges = self.load()

    def preprocess(self, *args) -> Tuple[np.ndarray, ...]:
        # swap n_nodes - n_features dimensions
        loc, vel, edges, charges = args
        loc, vel = np.transpose(loc, (0, 1, 3, 2)), np.transpose(vel, (0, 1, 3, 2))
        n_nodes = loc.shape[2]
        loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0 : self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # swap n_nodes - batch_size and add nf dimension
        edge_attr = np.array(edge_attr).T
        edge_attr = np.expand_dims(edge_attr, 2)

        if self.normalize:
            loc = self._normalize(loc)
            vel = self._normalize(vel)
            charges = self._normalize(charges)

        return loc, vel, edge_attr, edges, charges

    def load(self):
        loc, vel, edges, q = self._load()

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, q)
        return (loc, vel, edge_attr, charges), edges

    def __getitem__(self, i: Union[Sequence, int]) -> Tuple[np.ndarray, ...]:
        frame_0, frame_target = self._get_partition_frames()

        loc, vel, edge_attr, charges = self.data

        loc, vel, edge_attr, charges, target_loc = (
            loc[i, frame_0],
            vel[i, frame_0],
            edge_attr[i],
            charges[i],
            loc[i, frame_target],
        )

        if not isinstance(i, int):
            # flatten batch and nodes dimensions
            loc = loc.reshape(-1, *loc.shape[2:])
            vel = vel.reshape(-1, *vel.shape[2:])
            edge_attr = edge_attr.reshape(-1, *edge_attr.shape[2:])
            charges = charges.reshape(-1, *charges.shape[2:])
            target_loc = target_loc.reshape(-1, *target_loc.shape[2:])

        return loc, vel, edge_attr, charges, target_loc


class GravityDataset(BaseDataset):
    """N-body gravity dataset class."""

    def __init__(
        self,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=100,
        neighbours=6,
        target="pos",
        normalize=False,
    ):
        super().__init__(
            "gravity", partition, max_samples, dataset_name, n_bodies, normalize
        )
        assert target in ["pos", "force"]
        self.neighbours = int(neighbours)
        self.target = target
        self.data = self.load()

    def preprocess(self, *args) -> Tuple[np.ndarray, ...]:
        loc, vel, force, mass = args
        # NOTE this was in the original paper but does not look right
        # loc = np.transpose(loc, (0, 1, 3, 2))
        # vel = np.transpose(vel, (0, 1, 3, 2))
        # force = np.transpose(force, (0, 1, 3, 2))
        loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        force = force[0 : self.max_samples, :, :, :]

        if self.normalize:
            loc = self._normalize(loc)
            vel = self._normalize(vel)
            force = self._normalize(force)
            mass = self._normalize(mass)

        return loc, vel, force, mass

    def load(self):
        loc, vel, edges, q = self._load()

        self.num_nodes = loc.shape[-1]

        loc, vel, force, mass = self.preprocess(loc, vel, edges, q)
        return (loc, vel, force, mass)

    def __getitem__(self, i: Union[Sequence, int]) -> Tuple[np.ndarray, ...]:
        frame_0, frame_target = self._get_partition_frames()

        loc, vel, force, mass = self.data
        if self.target == "pos":
            y = loc[i, frame_target]
        elif self.target == "force":
            y = force[i, frame_target]
        loc, vel, force, mass = (loc[i, frame_0], vel[i, frame_0], force[i], mass[i])

        if not isinstance(i, int):
            # flatten batch and nodes dimensions
            loc = loc.reshape(-1, *loc.shape[2:])
            vel = vel.reshape(-1, *vel.shape[2:])
            force = force.reshape(-1, *force.shape[2:])
            mass = mass.reshape(-1, *mass.shape[2:])
            y = y.reshape(-1, *y.shape[2:])

        return loc, vel, force, mass, y
