import argparse
from math import floor
from typing import Any, Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import optax
from e3nn_jax import Irreps, IrrepsArray, spherical_harmonics
from jraph import segment_mean

from experiments.nbody.dataset_nbody import NBodyDataset
from segnn import SEGNN, SteerableGraphsTuple, weight_balanced_irreps

key = jax.random.PRNGKey(0)

time_exp_dic = {"time": 0, "counter": 0}


def O3Transform(
    node_features_irreps: Irreps, edge_features_irreps: Irreps, lmax_attr: int
) -> Callable:
    attribute_irreps = Irreps.spherical_harmonics(lmax_attr)

    def _o3_transform(
        graph: SteerableGraphsTuple,
        loc: jnp.ndarray,
        vel: jnp.ndarray,
        charges: jnp.ndarray,
    ) -> SteerableGraphsTuple:

        prod_charges = charges[graph.senders] * charges[graph.receivers]
        rel_pos = loc[graph.senders] - loc[graph.receivers]
        edge_dist = jnp.sqrt(jnp.power(rel_pos, 2).sum(1, keepdims=True))
        # NOTE additional_message_features is the same as edges in this implementation
        edges = IrrepsArray(
            edge_features_irreps,
            jnp.concatenate((edge_dist, prod_charges), axis=-1),
        )

        vel_abs = jnp.sqrt(jnp.power(vel, 2).sum(1, keepdims=True))
        mean_pos = loc.mean(1, keepdims=True)

        nodes = IrrepsArray(
            node_features_irreps,
            jnp.concatenate((loc - mean_pos, vel, vel_abs), axis=1),
        )

        edge_attributes = spherical_harmonics(
            attribute_irreps, rel_pos, normalize=True, normalization="integral"
        )
        vel_embedding = spherical_harmonics(
            attribute_irreps, vel, normalize=True, normalization="integral"
        )
        # scatter edge attributes
        sum_n_node = tree.tree_leaves(nodes)[0].shape[0]
        node_attributes = (
            tree.tree_map(
                lambda e: segment_mean(e, graph.receivers, sum_n_node),
                edge_attributes,
            )
            + vel_embedding
        )

        return SteerableGraphsTuple(
            nodes=nodes,
            edges=edges,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
            senders=graph.senders,
            receivers=graph.receivers,
            n_node=graph.n_node,
            n_edge=graph.n_edge,
            globals=graph.globals,
        )

    return _o3_transform


class NbodyGraphDataloader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        lmax_attr: int,
        drop_last: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.lmax_attr = lmax_attr
        self._n_nodes = self.dataset.get_n_nodes()
        self._drop_last = drop_last

        self._transform = transform

    def __iter__(self):
        i = 0
        edge_indices = self.dataset.get_edges(self.batch_size, self._n_nodes)
        senders, receivers = edge_indices[0], edge_indices[1]
        while i < len(self.dataset):
            cur_batch = min(self.batch_size, len(self.dataset) - i)
            if cur_batch < self.batch_size and self._drop_last:
                break
            if cur_batch < self.batch_size:
                # recompute edges for truncated batch
                edge_indices = self.dataset.get_edges(cur_batch, self._n_nodes)
                senders, receivers = edge_indices[0], edge_indices[1]
            loc, vel, _, charges, targets = self.dataset[i : (i + cur_batch)]
            graph = SteerableGraphsTuple(
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([self._n_nodes] * cur_batch),
                n_edge=jnp.array([len(senders)] * cur_batch),
            )
            graph = self._transform(graph, loc, vel, charges)
            i += cur_batch
            yield graph, targets

    def __len__(self) -> int:
        return floor(self.n_batches) if self._drop_last else round(self.n_batches)

    @property
    def n_batches(self) -> float:
        return len(self.dataset) / self.batch_size


def train(segnn: hk.Transformed, args):
    # load data
    o3_transform = O3Transform(args.node_irreps, args.edge_irreps, args.lmax_attr)

    dataset_train = NBodyDataset(
        partition="train",
        dataset_name=args.nbody_name,
        max_samples=args.max_samples,
    )
    dataset_val = NBodyDataset(
        partition="val",
        dataset_name=args.nbody_name,
    )
    dataset_test = NBodyDataset(
        partition="test",
        dataset_name=args.nbody_name,
    )
    loader_train = NbodyGraphDataloader(
        dataset_train,
        args.batch_size,
        args.lmax_attr,
        drop_last=False,
        transform=o3_transform,
    )
    loader_val = NbodyGraphDataloader(
        dataset_val,
        args.batch_size,
        args.lmax_attr,
        drop_last=True,
        transform=o3_transform,
    )
    loader_test = NbodyGraphDataloader(
        dataset_test,
        args.batch_size,
        args.lmax_attr,
        drop_last=True,
        transform=o3_transform,
    )

    params = segnn.init(key, next(iter(loader_train))[0])
    opt_init, opt_update = optax.adam(learning_rate=args.lr)

    @jax.jit
    def predict(params: hk.Params, graph: SteerableGraphsTuple) -> jnp.ndarray:
        return segnn.apply(params, graph)

    @jax.jit
    def mse(
        params: hk.Params, graph: SteerableGraphsTuple, target: jnp.ndarray
    ) -> float:
        out = predict(params, graph)
        return (jnp.square(out - target)).mean()

    @jax.jit
    def update(
        params: hk.Params, graph: SteerableGraphsTuple, target: jnp.ndarray, opt_state
    ) -> Tuple[float, hk.Params, Any]:
        loss, grads = jax.value_and_grad(mse)(params, graph, target)
        updates, opt_state = opt_update(grads, opt_state)
        return loss, optax.apply_updates(params, updates), opt_state

    opt_state = opt_init(params)

    for e in range(0, args.epochs):
        train_loss = 0
        val_loss = 0
        for graph, target in loader_train:
            loss, params, opt_state = update(params, graph, target, opt_state)
            train_loss += loss
        for graph, target in loader_val:
            val_loss += mse(params, graph, target)
        train_loss /= loader_train.n_batches
        val_loss /= loader_val.n_batches
        print(
            "Epoch {: <3} - training = {:.4f}, validation = {:.4f}".format(
                e + 1, train_loss, val_loss
            )
        )

    test_loss = 0
    for graph, target in loader_test:
        test_loss += mse(params, graph, target)
    test_loss /= loader_test.n_batches
    print("Training done. Test loss = {:.4f}".format(test_loss))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Run parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size. Does not scale with number of gpus.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="weight decay")

    # Data parameters
    parser.add_argument("--dataset", type=str, default="qm9", help="Data set")
    parser.add_argument(
        "--root", type=str, default="datasets", help="Data set location"
    )

    # Nbody parameters:
    parser.add_argument(
        "--target", type=str, default="pos", help="Target value [pos, force]"
    )
    parser.add_argument(
        "--nbody_name",
        type=str,
        default="nbody_small",
        help="Name of nbody data [nbody, nbody_small]",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3000,
        help="Maximum number of samples in nbody dataset",
    )

    # Model parameters
    parser.add_argument(
        "--hidden_features", type=int, default=128, help="max degree of hidden rep"
    )
    parser.add_argument(
        "--lmax_h", type=int, default=2, help="max degree of hidden rep"
    )
    parser.add_argument(
        "--lmax_attr",
        type=int,
        default=3,
        help="max degree of geometric attribute embedding",
    )
    parser.add_argument(
        "--subspace_type",
        type=str,
        default="weightbalanced",
        help="How to divide spherical harmonic subspaces",
    )
    parser.add_argument(
        "--layers", type=int, default=7, help="Number of message passing layers"
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="instance",
        help="Normalisation type [instance, batch]",
    )
    parser.add_argument(
        "--pool", type=str, default="avg", help="Pooling type type [avg, sum]"
    )
    args = parser.parse_args()

    args.node_irreps = Irreps("2x1o + 1x0e")
    args.edge_irreps = Irreps("2x0e")

    args.edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
    args.node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)

    # Create hidden irreps
    hidden_irreps = weight_balanced_irreps(
        args.hidden_features,
        args.node_attr_irreps,
        use_sh=True,
        lmax=args.lmax_h,
    )

    segnn = SEGNN(
        hidden_irreps=hidden_irreps,
        output_irreps=Irreps("1x1o"),
        num_layers=args.layers,
        task="node",
    )
    segnn = hk.without_apply_rng(hk.transform(segnn))

    train(segnn, args)
