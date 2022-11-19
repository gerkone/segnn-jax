import os.path as osp
import pathlib
from typing import Sequence, Tuple, Union

import jax.numpy as jnp


class NBodyDataset:
    """
    NBodyDataset

    """

    def __init__(self, partition="train", max_samples=1e8, dataset_name="nbody_small"):
        self.partition = partition
        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += "_charged5_initvel1"
        elif dataset_name in ("nbody_small", "nbody_small_out_dist"):
            self.suffix += "_charged5_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data, self.edges = self.load()

    def load(self):
        dir_ = pathlib.Path(__file__).parent.absolute()
        loc = jnp.load(osp.join(dir_, "dataset", "loc_" + self.suffix + ".npy"))
        vel = jnp.load(osp.join(dir_, "dataset", "vel_" + self.suffix + ".npy"))
        edges = jnp.load(osp.join(dir_, "dataset", "edges_" + self.suffix + ".npy"))
        charges = jnp.load(osp.join(dir_, "dataset", "charges_" + self.suffix + ".npy"))

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
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

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].shape[2]

    def __getitem__(self, i: Union[Sequence, int]) -> Tuple[jnp.ndarray, ...]:
        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges, loc_T = (
            loc[i, frame_0],
            vel[i, frame_0],
            edge_attr[i],
            charges[i],
            loc[i, frame_T],
        )

        if not isinstance(i, int):
            # flatten batch and nodes dimensions
            loc = loc.reshape(-1, *loc.shape[2:])
            vel = vel.reshape(-1, *vel.shape[2:])
            edge_attr = edge_attr.reshape(-1, *edge_attr.shape[2:])
            charges = charges.reshape(-1, *charges.shape[2:])
            loc_T = loc.reshape(-1, *loc_T.shape[2:])

        return loc, vel, edge_attr, charges, loc_T

    def __len__(self):
        return len(self.data[0])

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


if __name__ == "__main__":
    NBodyDataset()
