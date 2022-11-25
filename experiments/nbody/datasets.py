import os.path as osp
import pathlib
from abc import ABC
from typing import Sequence, Tuple, Union

import jax.numpy as jnp

DATA_DIR = "data"


class BaseDataset(ABC):
    def __init__(
        self,
        data_type,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=5,
    ):
        self.partition = partition
        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name

        self.suffix += f"_{data_type}{n_bodies}_initvel1"

        self.max_samples = int(max_samples)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].shape[2]

    def get_partition_frames(self) -> Tuple[int, int]:
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

    def _load(self):
        filepath = pathlib.Path(__file__).parent.resolve()

        loc = jnp.load(osp.join(filepath, DATA_DIR, "loc_" + self.suffix + ".npy"))
        vel = jnp.load(osp.join(filepath, DATA_DIR, "vel_" + self.suffix + ".npy"))
        edges = jnp.load(osp.join(filepath, DATA_DIR, "edges_" + self.suffix + ".npy"))
        q = jnp.load(osp.join(filepath, DATA_DIR, "q_" + self.suffix + ".npy"))

        return loc, vel, edges, q

    def load(self):
        raise NotImplementedError

    def preprocess(self) -> Tuple[jnp.ndarray, ...]:
        raise NotImplementedError


class ChargedDataset(BaseDataset):
    def __init__(
        self, partition="train", max_samples=1e8, dataset_name="small", n_bodies=5
    ):
        super().__init__("charged", partition, max_samples, dataset_name, n_bodies)
        self.data, self.edges = self.load()

    def preprocess(self, loc, vel, edges, charges) -> Tuple[jnp.ndarray, ...]:
        # swap n_nodes - n_features dimensions
        loc, vel = jnp.transpose(loc, (0, 1, 3, 2)), jnp.transpose(vel, (0, 1, 3, 2))
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
        edge_attr = jnp.array(edge_attr).T
        edge_attr = jnp.expand_dims(edge_attr, 2)

        return loc, vel, edge_attr, edges, charges

    def load(self):
        loc, vel, edges, q = self._load()

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, q)
        return (loc, vel, edge_attr, charges), edges

    def __getitem__(self, i: Union[Sequence, int]) -> Tuple[jnp.ndarray, ...]:
        frame_0, frame_target = self.get_partition_frames()

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

    def get_edges(self, batch_size, n_nodes):
        edges = [jnp.array(self.edges[0]), jnp.array(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [jnp.concatenate(rows), jnp.concatenate(cols)]
        return edges


class GravityDataset(BaseDataset):
    def __init__(
        self,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=100,
        neighbours=6,
        target="pos",
    ):
        super().__init__("gravity", partition, max_samples, dataset_name, n_bodies)
        assert target in ["pos", "force"]
        self.neighbours = int(neighbours)
        self.target = target
        self.data = self.load()

    def preprocess(self, loc, vel, force, mass):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # NOTE this was in the original paper but does not look right
        # loc = jnp.transpose(loc, (0, 1, 3, 2))
        # vel = jnp.transpose(vel, (0, 1, 3, 2))
        # force = jnp.transpose(force, (0, 1, 3, 2))
        loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        force = force[0 : self.max_samples, :, :, :]

        return loc, vel, force, mass

    def load(self):
        loc, vel, edges, q = self._load()

        self.num_nodes = loc.shape[-1]

        loc, vel, force, mass = self.preprocess(loc, vel, edges, q)
        return (loc, vel, force, mass)

    def __getitem__(self, i: Union[Sequence, int]) -> Tuple[jnp.ndarray, ...]:
        frame_0, frame_target = self.get_partition_frames()

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
