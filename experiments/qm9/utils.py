from typing import Callable, List, Optional, Tuple, Union

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph
from torch.utils.data._utils import pin_memory
from torch.utils.data.dataloader import (
    _BaseDataLoaderIter,
    _SingleProcessDataLoaderIter,
)
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from experiments.qm9.dataset import QM9
from segnn import SteerableGraphsTuple


def QM9GraphTransform(
    node_features_irreps: e3nn.Irreps,
    edge_features_irreps: e3nn.Irreps,
    lmax_attributes: int,
    max_batch_nodes: int,
    max_batch_edges: int,
) -> Callable:
    """
    Build a function that converts torch DataBatch into SteerableGraphsTuple.

    Mostly a quick fix out of lazyness. Rewriting QM9 in jax is not trivial.
    """
    attribute_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)

    def _to_steerable_graph(data: Data) -> Tuple[SteerableGraphsTuple, jnp.array]:
        graph = jraph.GraphsTuple(
            nodes=e3nn.IrrepsArray(node_features_irreps, jnp.array(data.x)),
            edges=None,
            senders=jnp.array(data.edge_index[0]),
            receivers=jnp.array(data.edge_index[1]),
            n_node=jnp.diff(jnp.array(data.ptr)),
            # n_edge is not used anywhere by segnn, but is neded for padding
            n_edge=jnp.array([jnp.array(data.edge_index[1]).shape[0]]),
            globals=None,
        )
        # pad for jax static shapes
        node_attr_pad = ((0, max_batch_nodes - jnp.sum(graph.n_node) + 1), (0, 0))
        edge_attr_pad = ((0, max_batch_edges - jnp.sum(graph.n_edge) + 1), (0, 0))
        graph = jraph.pad_with_graphs(
            graph,
            n_node=max_batch_nodes + 1,
            n_edge=max_batch_edges + 1,
            n_graph=graph.n_node.shape[0] + 1,
        )
        st_graph = SteerableGraphsTuple(
            graph=graph,
            node_attributes=e3nn.IrrepsArray(
                attribute_irreps, jnp.pad(jnp.array(data.node_attr), node_attr_pad)
            ),
            edge_attributes=e3nn.IrrepsArray(
                attribute_irreps, jnp.pad(jnp.array(data.edge_attr), edge_attr_pad)
            ),
            additional_message_features=e3nn.IrrepsArray(
                edge_features_irreps,
                jnp.pad(jnp.array(data.additional_message_features), edge_attr_pad),
            ),
        )
        # pad targets
        target = jnp.append(jnp.array(data.y), 0)
        return st_graph, target

    return _to_steerable_graph


class TakeUntilIterator(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        self._max_nodes = loader.max_batch_nodes
        self._max_edges = loader.max_batch_edges

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        while True:
            data_ = self._dataset_fetcher.fetch(index)  # may raise StopIteration
            if (
                data_.x.shape[0] > self._max_nodes
                or data_.edge_index.shape[1] > self._max_edges
            ):
                break
            data = data_
            # TODO better condition and index steps
            index.extend(self._next_index())
        if self._pin_memory:
            data = pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class TakeUntilDataLoader(DataLoader):
    """
    Bad implementation of a dataloader that takes until a certain size.

    Mostly here reduce the amount of padding a bit. Note that this can introduce a bias
    towards larger molecules (as they fill the batches faster and count more towards
    the loss), but speeds up the training so it's the default.
    """

    def __init__(
        self,
        max_batch_nodes: int,
        max_batch_edges: int,
        dataset: Union[Dataset, List[BaseData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            dataset, batch_size, shuffle, follow_batch, exclude_keys, **kwargs
        )
        self.max_batch_nodes = max_batch_nodes
        self.max_batch_edges = max_batch_edges

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        if self.num_workers == 0:
            return TakeUntilIterator(self)
        else:
            raise NotImplementedError


def setup_qm9_data(args) -> Tuple[DataLoader, DataLoader, DataLoader, Callable]:
    dataset_train = QM9(
        "datasets",
        args.target,
        args.radius,
        partition="train",
        lmax_attr=args.lmax_attributes,
        feature_type=args.feature_type,
    )
    dataset_val = QM9(
        "datasets",
        args.target,
        args.radius,
        partition="valid",
        lmax_attr=args.lmax_attributes,
        feature_type=args.feature_type,
    )
    dataset_test = QM9(
        "datasets",
        args.target,
        args.radius,
        partition="test",
        lmax_attr=args.lmax_attributes,
        feature_type=args.feature_type,
    )

    max_batch_nodes = int(
        max(
            sum(d.top_n_nodes(args.batch_size))
            for d in [dataset_train, dataset_val, dataset_test]
        )
    )

    max_batch_edges = int(
        max(
            sum(d.top_n_edges(args.batch_size))
            for d in [dataset_train, dataset_val, dataset_test]
        )
    )

    # load data
    # NOTE replace with normal DataLoader if slower training is ok
    loader_train = TakeUntilDataLoader(
        max_batch_nodes,
        max_batch_edges,
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    loader_val = TakeUntilDataLoader(
        max_batch_nodes,
        max_batch_edges,
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    loader_test = TakeUntilDataLoader(
        max_batch_nodes,
        max_batch_edges,
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )

    to_graphs_tuple = QM9GraphTransform(
        args.node_irreps,
        args.additional_message_irreps,
        args.lmax_attributes,
        max_batch_nodes=max_batch_nodes,
        max_batch_edges=max_batch_edges,
    )
    return loader_train, loader_val, loader_test, to_graphs_tuple
