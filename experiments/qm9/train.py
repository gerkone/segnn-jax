import argparse
import time
from typing import Any, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from torch_geometric.loader import DataLoader

import wandb
from experiments.qm9.dataset import QM9
from segnn import SEGNN, SteerableGraphsTuple, weight_balanced_irreps

from .utils import QM9GraphTransform

key = jax.random.PRNGKey(0)


def train(segnn: hk.Transformed, dataset_train, dataset_val, dataset_test, args):
    # load data
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    to_graphs_tuple = QM9GraphTransform(
        args.node_irreps,
        args.edge_irreps,
        args.lmax_attributes,
        max_batch_nodes=int(
            max(
                [
                    sum(d.top_n_nodes(args.batch_size))
                    for d in [dataset_train, dataset_val, dataset_test]
                ]
            )
            * 0.7
        ),
        max_batch_edges=int(
            max(
                [
                    sum(d.top_n_edges(args.batch_size))
                    for d in [dataset_train, dataset_val, dataset_test]
                ]
            )
            * 0.7
        ),
    )

    print("Jitting...")
    init_graph, _ = to_graphs_tuple(next(iter(loader_train)))
    params, segnn_state = segnn.init(key, init_graph)
    opt_init, opt_update = optax.adamw(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )

    @jax.jit
    def predict(
        params: hk.Params, state: hk.State, graph: SteerableGraphsTuple
    ) -> Tuple[jnp.ndarray, hk.State]:
        return segnn.apply(params, state, graph)

    @jax.jit
    def mae(
        params: hk.Params,
        state: hk.State,
        graph: SteerableGraphsTuple,
        target: jnp.ndarray,
    ) -> float:
        pred, _ = predict(params, state, graph)
        return (jnp.abs(pred - target)).mean()

    @jax.jit
    def update(
        params: hk.Params,
        state: hk.State,
        graph: SteerableGraphsTuple,
        target: jnp.ndarray,
        opt_state,
    ) -> Tuple[float, hk.Params, Any]:
        loss, grads = jax.value_and_grad(mae)(params, state, graph, target)
        updates, opt_state = opt_update(grads, opt_state, params)
        return loss, optax.apply_updates(params, updates), opt_state

    opt_state = opt_init(params)
    avg_time = []

    for e in range(args.epochs):
        train_loss = 0
        train_start = time.perf_counter_ns()
        for data in loader_train:
            graph, target = to_graphs_tuple(data)
            loss, params, opt_state = update(
                params, segnn_state, graph, target, opt_state
            )
            train_loss += loss
        train_time = (time.perf_counter_ns() - train_start) / 1e6 / len(loader_train)
        train_loss /= len(loader_train)
        wandb_logs = {"train_loss": train_loss, "update_time": train_time}
        print(
            "[Epoch {:>4}] training loss {:.6f}, update time {:.3f}ms".format(
                e + 1, train_loss, train_time
            ),
            end="",
        )
        if e % args.val_freq == 0:
            val_loss = 0
            eval_start = time.perf_counter_ns()
            for data in loader_val:
                graph, target = to_graphs_tuple(data)
                val_loss += jax.lax.stop_gradient(
                    mae(params, segnn_state, graph, target)
                )
            eval_time = (time.perf_counter_ns() - eval_start) / 1e6 / len(loader_val)
            avg_time.append(eval_time)
            val_loss /= len(loader_val)
            wandb_logs.update({"val_loss": val_loss, "eval_time": eval_time})
            print(
                " - validation loss {:.6f}, eval time {:.3f}ms ({} graph batch)".format(
                    val_loss, eval_time, args.batch_size
                ),
                end="",
            )
        print()
        if args.wandb:
            wandb.log(wandb_logs)

    test_loss = 0
    for data in loader_test:
        graph, target = to_graphs_tuple(data)
        test_loss += jax.lax.stop_gradient(mae(params, segnn_state, graph, target))
    test_loss /= len(loader_test)
    avg_time = sum(avg_time) / len(avg_time)
    if args.wandb:
        wandb.log({"test_loss": test_loss, "avg_eval_time": avg_time})
    print(
        "Training done. Test loss {:.6f} - "
        "eval time {:.3f}ms (average, {} graph batch)".format(
            test_loss, avg_time, args.batch_size
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-12, help="Weight decay"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["qm9", "charged", "gravity"],
        help="Dataset name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Maximum number of samples in nbody dataset",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=10,
        help="Evaluation frequency (number of epochs)",
    )

    # gravity parameters
    parser.add_argument(
        "--target",
        type=str,
        default="pos",
        help="Target. e.g. pos, force (gravity), alpha (qm9)",
    )

    parser.add_argument(
        "--feature-type",
        type=str,
        default="one_hot",
        choices=["one_hot", "cormorant"],
        help="Type of input feature",
    )

    # Model parameters
    parser.add_argument(
        "--units", type=int, default=128, help="Number of values in the hidden layers"
    )
    parser.add_argument(
        "--lmax-hidden",
        type=int,
        default=2,
        help="Max degree of hidden representations.",
    )
    parser.add_argument(
        "--lmax-attributes",
        type=int,
        default=3,
        help="Max degree of geometric attribute embedding",
    )
    parser.add_argument(
        "--layers", type=int, default=7, help="Number of message passing layers"
    )
    parser.add_argument(
        "--blocks", type=int, default=2, help="Number of layers in steerable MLPs."
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="batch",
        choices=["instance", "batch"],
        help="Normalisation type",
    )

    # wandb parameters
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Activate weights and biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="segnn",
        help="Weights and biases project",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        help="Weights and biases entity",
    )

    args = parser.parse_args()

    if args.feature_type == "one_hot":
        args.node_irreps = e3nn.Irreps("5x0e")
    elif args.feature_type == "cormorant":
        args.node_irreps = e3nn.Irreps("15x0e")
    elif args.feature_type == "gilmer":
        args.node_irreps = e3nn.Irreps("11x0e")

    output_irreps = e3nn.Irreps("1x0e")

    args.edge_irreps = e3nn.Irreps("1x0e")

    # connect to wandb
    if args.wandb:
        wandb_name = "_".join(
            [
                args.wandb_project,
                args.dataset,
                args.target,
                str(int(time.time())),
            ]
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=args,
            entity=args.wandb_entity,
        )

    # Create hidden irreps
    hidden_irreps = weight_balanced_irreps(
        scalar_units=args.units,
        # attribute irreps
        irreps_right=e3nn.Irreps.spherical_harmonics(args.lmax_attributes),
        use_sh=True,
        lmax=args.lmax_hidden,
    )

    # build model
    segnn = SEGNN(
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        num_layers=args.layers,
        task="graph",
        pool="avg",
        blocks_per_layer=args.blocks,
        norm=args.norm,
    )
    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    dataset_train = QM9(
        "datasets",
        args.target,
        2,
        "train",
        args.lmax_attributes,
        feature_type=args.feature_type,
    )
    dataset_val = QM9(
        "datasets",
        args.target,
        2,
        "valid",
        args.lmax_attributes,
        feature_type=args.feature_type,
    )
    dataset_test = QM9(
        "datasets",
        args.target,
        2,
        "test",
        args.lmax_attributes,
        feature_type=args.feature_type,
    )

    train(segnn, dataset_train, dataset_val, dataset_test, args)
