import argparse
import time
from functools import partial

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import wandb

from experiments import setup_data, train
from segnn_jax import SEGNN, weight_balanced_irreps

key = jax.random.PRNGKey(1337)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (number of graphs).",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--lr-scheduling",
        action="store_true",
        help="Use learning rate scheduling",
    )
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

    # nbody parameters
    parser.add_argument(
        "--target",
        type=str,
        default="pos",
        help="Target. e.g. pos, force (gravity), alpha (qm9)",
    )
    parser.add_argument(
        "--neighbours",
        type=int,
        default=20,
        help="Number of connected nearest neighbours",
    )
    parser.add_argument(
        "--n-bodies",
        type=int,
        default=5,
        help="Number of bodies in the dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="small",
        choices=["small", "default", "small_out_dist"],
        help="Name of nbody data partition: default (200 steps), small (1000 steps)",
    )

    # qm9 parameters
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Radius (Angstrom) between which atoms to add links.",
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        default="one_hot",
        choices=["one_hot", "cormorant", "gilmer"],
        help="Type of input feature",
    )

    # Model parameters
    parser.add_argument(
        "--units", type=int, default=64, help="Number of values in the hidden layers"
    )
    parser.add_argument(
        "--lmax-hidden",
        type=int,
        default=1,
        help="Max degree of hidden representations.",
    )
    parser.add_argument(
        "--lmax-attributes",
        type=int,
        default=1,
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
        default="none",
        choices=["instance", "batch", "none"],
        help="Normalisation type",
    )
    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Use double precision in model",
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

    # if specified set jax in double precision
    jax.config.update("jax_enable_x64", args.double_precision)

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

    # feature representations
    if args.dataset == "qm9":
        args.task = "graph"
        if args.feature_type == "one_hot":
            args.node_irreps = e3nn.Irreps("5x0e")
        elif args.feature_type == "cormorant":
            args.node_irreps = e3nn.Irreps("15x0e")
        elif args.feature_type == "gilmer":
            args.node_irreps = e3nn.Irreps("11x0e")
        args.output_irreps = e3nn.Irreps("1x0e")
        args.additional_message_irreps = e3nn.Irreps("1x0e")
    elif args.dataset in ["charged", "gravity"]:
        args.task = "node"
        args.node_irreps = e3nn.Irreps("2x1o + 1x0e")
        args.output_irreps = e3nn.Irreps("1x1o")
        args.additional_message_irreps = e3nn.Irreps("2x0e")

    # Create hidden irreps
    hidden_irreps = weight_balanced_irreps(
        scalar_units=args.units,
        # attribute irreps
        irreps_right=e3nn.Irreps.spherical_harmonics(args.lmax_attributes),
        use_sh=True,
        lmax=args.lmax_hidden,
    )

    # build model
    def segnn(x):
        return SEGNN(
            hidden_irreps=hidden_irreps,
            output_irreps=args.output_irreps,
            num_layers=args.layers,
            task=args.task,
            pool="avg",
            blocks_per_layer=args.blocks,
            norm=args.norm,
        )(x)

    segnn = hk.without_apply_rng(hk.transform_with_state(segnn))

    loader_train, loader_val, loader_test, graph_transform, eval_trn = setup_data(args)

    if args.dataset == "qm9":
        from experiments.train import loss_fn

        def _mae(p, t):
            return jnp.abs(p - t)

        train_loss = partial(loss_fn, criterion=_mae, task=args.task)
        eval_loss = partial(loss_fn, criterion=_mae, eval_trn=eval_trn, task=args.task)
    if args.dataset in ["charged", "gravity"]:
        from experiments.train import loss_fn

        def _mse(p, t):
            return jnp.power(p - t, 2)

        train_loss = partial(loss_fn, criterion=_mse, do_mask=False)
        eval_loss = partial(loss_fn, criterion=_mse, do_mask=False)

    train(
        key,
        segnn,
        loader_train,
        loader_val,
        loader_test,
        train_loss,
        eval_loss,
        graph_transform,
        args,
    )
