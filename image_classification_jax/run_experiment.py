"""
Simple image classification script for training on CIFAR-10, CIFAR-100, ImageNet, or
Imagenette dataset. Supports ResNetTiny, ResNet18, ResNet50, ResNet101, ResNet152, and
ViT models.
"""

import os
import random
from functools import partial
from pprint import pprint
from typing import Optional, Any, NamedTuple
import wandb
import numpy as np

import jax
import jax.numpy as jnp
from flax.traverse_util import _get_params_dict, flatten_dict, _sorted_items
from jax import pmap
import flax
from flax import core
import optax
from optax.contrib._schedule_free import schedule_free_eval_params
import tensorflow_datasets as tfds
import tensorflow as tf
from psgd_jax import hessian_helper

from image_classification_jax.utils.imagenet_pipeline import (
    create_split,
    _add_tpu_host_options,
    split_batch,
)
from image_classification_jax.models.ViT import Transformer
from image_classification_jax.models.resnet import (
    ResNetTiny,
    ResNet18,
    ResNet50,
    ResNet101,
    ResNet152,
)
from image_classification_jax.utils.tf_preprocessing_tools import CifarPreprocess
from image_classification_jax.utils.training_utils import (
    to_full,
    to_half,
    z_loss,
    sync_batch_stats,
)


wandb.require("core")
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
jax.config.update("jax_default_matmul_precision", "default")


class TrainState(NamedTuple):
    step: jax.Array
    params: core.FrozenDict[str, Any]
    batch_stats: Optional[core.FrozenDict[str, Any]]
    opt_state: optax.OptState


def run_experiment(
    log_to_wandb: bool = True,
    wandb_entity: str = "",
    wandb_project: str = "image_classification_jax",
    wandb_config_update: Optional[dict] = None,
    global_seed: int = 100,
    dataset: str = "cifar10",
    batch_size: int = 256,
    n_epochs: int = 150,
    optimizer: optax.GradientTransformation = optax.adamw(1e-3),
    compute_in_bfloat16: bool = False,
    l2_regularization: float = 0.0,
    randomize_l2_reg: bool = False,
    apply_z_loss: bool = True,
    model_type: str = "resnet18",
    n_layers: int = 12,
    enc_dim: int = 768,
    n_heads: int = 12,
    n_empty_registers: int = 0,
    dropout_rate: float = 0.0,
    using_schedule_free: bool = False,
    psgd_calc_hessian: bool = False,
    psgd_precond_update_prob: float = 1.0,
):
    """Run an image classification experiment.

    Args:
        log_to_wandb: bool, whether to log to wandb.
        wandb_entity: str, wandb entity.
        wandb_project: str, wandb project.
        wandb_config_update: dict, additional config to add to wandb.init() call.
        global_seed: int, random seed.
        dataset: str, 'cifar10', 'cifar100', 'imagenet', 'imagenette'.
        batch_size: int, batch size.
        n_epochs: int, number of epochs.
        optimizer: optax.GradientTransformation, optimizer.
        compute_in_bfloat16: bool, whether to compute in bfloat16.
        l2_regularization: float, l2 regularization.
        randomize_l2_reg: bool, randomize l2 regularization (l2 reg * random uniform).
        apply_z_loss: bool, apply palm style z-loss, recommended.
        model_type: str, 'resnettiny', 'resnet18', 'resnet50', 'resnet101',
            'resnet152', 'vit'.
        n_layers: int, number of transformer layers.
        enc_dim: int, transformer encoder dimension.
        n_heads: int, number of transformer heads.
        n_empty_registers: int, number of empty registers for ViT, see
            https://arxiv.org/abs/2309.16588.
        dropout_rate: float, dropout rate for ViT.
        using_schedule_free: bool, whether the optimizer is wrapped in schedule-free.
            If True, evaluates params at `x`.
        psgd_calc_hessian: bool, If optimizer is PSGD, set this to True to calculate
            and pass in the hessian.
        psgd_precond_update_prob: float, If optimizer is PSGD, probability of
            calculating hessian and updating the preconditioner when
            `psgd_calc_hessian` is True.
    """
    # take a look at the devices and see if we're on CPU, GPU, or TPU
    devices = jax.local_devices()
    print(f"JAX Devices: {devices}")
    platform = devices[0].platform

    # set seeds
    rng = jax.random.PRNGKey(global_seed)  # jax uses explicit seed handling
    tf.random.set_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)

    # dataset
    if dataset == "imagenette":
        dataset_name = "imagenette/full-size-v2"
        n_classes = 10
    elif dataset == "imagenet":
        dataset_name = "imagenet2012"
        n_classes = 1000
    elif dataset == "cifar10":
        dataset_name = "cifar10"
        n_classes = 10
    elif dataset == "cifar100":
        dataset_name = "cifar100"
        n_classes = 100
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}, must be one of "
            f"'imagenet', 'imagenette', 'cifar10', 'cifar100'"
        )

    # wandb setup
    if log_to_wandb:
        if not wandb_entity:
            print(
                "WARNING: No wandb entity provided, running without logging to wandb."
            )
            log_to_wandb = False
        else:
            if not os.environ["WANDB_API_KEY"]:
                raise ValueError(
                    "No WANDB_API_KEY found in environment variables, see readme "
                    "for instructions on setting wandb API key."
                )
            wandb.login(key=os.environ["WANDB_API_KEY"])
            config = locals()
            if wandb_config_update is not None:
                config.update(wandb_config_update)
            wandb.init(entity=wandb_entity, project=wandb_project, config=config)

    def get_datasets():
        """Download and prepare tensorflow datasets."""
        ds_builder = tfds.builder(dataset_name)
        print("Downloading and preparing dataset.", flush=True)
        ds_builder.download_and_prepare()

        if dataset in ["imagenette", "imagenet"]:
            print("Using imagenet style data pipeline.")
            train_ds = create_split(
                ds_builder,
                batch_size,
                train=True,
                platform=platform,
                dtype=tf.float32,
                shuffle_buffer_size=250 if dataset == "imagenette" else 2000,
                prefetch=4,
            )
            test_ds = create_split(
                ds_builder,
                batch_size,
                train=False,
                platform=platform,
                dtype=tf.float32,
                shuffle_buffer_size=250 if dataset == "imagenette" else 2000,
                prefetch=4,
            )
        else:
            print("Using cifar style data pipeline.")
            train_ds = ds_builder.as_dataset(split="train", shuffle_files=True)
            test_ds = ds_builder.as_dataset(split="test", shuffle_files=True)

            if platform == "tpu":
                train_ds = _add_tpu_host_options(train_ds)
                test_ds = _add_tpu_host_options(test_ds)

            train_ds = (
                train_ds.repeat()
                .shuffle(2000)
                .map(
                    CifarPreprocess(True, dataset),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .batch(
                    batch_size,
                    drop_remainder=True,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .map(
                    split_batch,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .prefetch(8)
                .as_numpy_iterator()
            )
            test_ds = (
                test_ds.repeat()
                .shuffle(2000)
                .map(
                    CifarPreprocess(False, dataset),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .batch(
                    batch_size,
                    drop_remainder=True,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .map(
                    split_batch,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .prefetch(4)
                .as_numpy_iterator()
            )

        if platform in ["gpu", "tpu"]:
            # prefetch 1 (instead of 0) on TPU in case snag with
            # JAX async dispatch in train loop
            train_ds = flax.jax_utils.prefetch_to_device(
                train_ds, 2 if platform == "gpu" else 1
            )
            test_ds = flax.jax_utils.prefetch_to_device(
                test_ds, 2 if platform == "gpu" else 1
            )
        return train_ds, test_ds

    # download datasets and create data iterators
    train_ds, test_ds = get_datasets()
    if dataset == "imagenette":
        train_ds_size = 9469
    elif dataset == "imagenet":
        train_ds_size = 1281167
    else:
        train_ds_size = 50000
    steps_per_epoch = train_ds_size // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    n_steps = steps_per_epoch * n_epochs
    print(f"Total train steps: {n_steps}")
    print(f"Total epochs: {n_epochs}")

    # create model
    if "resnet" in model_type:
        print("Creating ResNet model.")
        if dataset in ["imagenet", "imagenette"]:
            fl_kernel_size, fl_stride, fl_pool = 7, 2, True
        else:
            fl_kernel_size, fl_stride, fl_pool = 3, 1, False
        if model_type == "resnettiny":
            rn_class = ResNetTiny
        elif model_type == "resnet18":
            rn_class = ResNet18
        elif model_type == "resnet50":
            rn_class = ResNet50
        elif model_type == "resnet101":
            rn_class = ResNet101
        elif model_type == "resnet152":
            rn_class = ResNet152
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model = rn_class(
            num_classes=n_classes,
            first_layer_kernel_size=fl_kernel_size,
            first_layer_stride=fl_stride,
            first_layer_max_pool=fl_pool,
        )
    elif model_type == "vit":
        print("Creating ViT model.")
        model = Transformer(
            n_layers=n_layers,
            enc_dim=enc_dim,
            n_heads=n_heads,
            n_empty_registers=n_empty_registers,
            dropout_rate=dropout_rate,
            output_dim=n_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    def loss_fn(params, batch_stats, rng, images, labels):
        """Computes loss for a single batch.

        Args:
            params: dict, model parameters.
            batch_stats: dict, batch statistics.
            rng: PRNGKey
            images: jnp.ndarray, batch of images.
            labels: jnp.ndarray, batch of labels.

        Returns:
            loss: float, mean loss.
            aux: tuple of new model state and logits.
        """
        # optionally carry out calculations in bfloat16
        if compute_in_bfloat16:
            params = to_half(params)
            images = to_half(images)

        rng, subkey = jax.random.split(rng)
        if "resnet" in model_type:
            logits, new_model_state = model.apply(
                {"params": params, "batch_stats": batch_stats},
                images,
                rngs={"dropout": subkey},
                mutable=["batch_stats"],
                is_training=True,
            )
        else:
            logits = model.apply(
                {"params": params}, images, rngs={"dropout": subkey}, is_training=True
            )
            new_model_state = {"batch_stats": batch_stats}
        if compute_in_bfloat16:
            assert logits.dtype == jnp.bfloat16
        # back to float32 for loss calculation
        logits = to_full(logits)
        one_hot = jax.nn.one_hot(labels, n_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
        orig_loss = loss

        # z-loss, https://arxiv.org/pdf/2204.02311
        if apply_z_loss:
            loss += z_loss(logits).mean() * 1e-4

        if l2_regularization > 0:
            to_l2 = []
            for key, value in _sorted_items(flatten_dict(_get_params_dict(params))):
                path = "/" + "/".join(key)
                if "kernel" in path:
                    to_l2.append(jnp.linalg.norm(value))
            l2_loss = jnp.linalg.norm(jnp.array(to_l2))

            if randomize_l2_reg:
                rng, subkey = jax.random.split(rng)
                multiplier = jax.random.uniform(
                    subkey, dtype=jnp.float32, minval=0.0, maxval=2.0
                )
                l2_loss *= multiplier

            loss += l2_regularization * l2_loss

        return loss, (new_model_state, logits, orig_loss)

    @partial(pmap, axis_name="batch", donate_argnums=(1,))
    def train_step(rng, state, batch):
        """Applies an update to parameters and returns new state.

        Args:
            rng: PRNGKey, random number generator.
            state: TrainState, current state.
            batch: dict, batch of data.

        Returns:
            rng: PRNGKey, random number generator.
            new_state: TrainState, new state.
            loss: float, mean loss.
            accuracy: float, mean accuracy.
            grad_norm: float, mean gradient
        """
        if psgd_calc_hessian:
            rng, subkey1, subkey2 = jax.random.split(rng, 3)

            # use psgd hessian helper to calc hvp and pass into psgd
            subkey1 = jax.lax.all_gather(subkey1, "batch")
            # same key on all devices for random vector and precond update prob
            subkey1 = subkey1[0]
            (_, aux), grads, hvp, vector, update_precond = hessian_helper(
                subkey1,
                state.step,
                loss_fn,
                state.params,
                loss_fn_extra_args=(
                    state.batch_stats,
                    subkey2,
                    batch["image"],
                    batch["label"],
                ),
                has_aux=True,
                preconditioner_update_probability=psgd_precond_update_prob,
            )

            grads = jax.lax.pmean(grads, axis_name="batch")
            hvp = jax.lax.pmean(hvp, axis_name="batch")

            updates, new_opt_state = optimizer.update(
                grads,
                state.opt_state,
                state.params,
                Hvp=hvp,
                vector=vector,
                update_preconditioner=update_precond,
            )
        else:
            rng, subkey = jax.random.split(rng)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, state.batch_stats, subkey, batch["image"], batch["label"]
            )
            # mean gradients across devices
            grads = jax.lax.pmean(grads, axis_name="batch")

            updates, new_opt_state = optimizer.update(
                grads, state.opt_state, state.params
            )

        new_model_state, logits, loss = aux
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch["label"])

        # apply updates to model params
        new_params = optax.apply_updates(state.params, updates)

        # create new state
        new_state = state._replace(
            step=state.step + 1,
            params=new_params,
            batch_stats=new_model_state["batch_stats"],
            opt_state=new_opt_state,
        )

        # mean stats across devices
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        # grad norm metric
        grad_norm = optax.global_norm(grads)

        return rng, new_state, loss, accuracy, grad_norm

    @partial(pmap, axis_name="batch")
    def inference(state, batch):
        """Computes gradients, loss and accuracy for a single batch."""

        variables = {
            "params": (
                schedule_free_eval_params(state.opt_state, state.params)
                if using_schedule_free
                else state.params
            )
        }
        if "resnet" in model_type:
            variables["batch_stats"] = state.batch_stats
        images, labels = batch["image"], batch["label"]

        logits = model.apply(variables, images, is_training=False)
        one_hot = jax.nn.one_hot(labels, n_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch["label"])

        # mean stats across devices
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        return loss, accuracy

    @pmap
    def create_train_state(rng):
        """Creates initial `TrainState`.

        Decorated with `pmap` so train state is automatically replicated across devices.
        """
        image_size = 224 if dataset in ["imagenet", "imagenette"] else 32
        dummy_image = jnp.ones([1, image_size, image_size, 3])  # batch size 1 for init
        variables = model.init(rng, dummy_image, is_training=False)

        opt_state = optimizer.init(variables["params"])

        print("Network params:")
        pprint(
            jax.tree.map(lambda x: x.shape, variables["params"]),
            width=120,
            compact=True,
        )

        # create initial train state
        state = TrainState(
            step=jnp.zeros([], jnp.int32),
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            opt_state=opt_state,
        )
        return state

    print("Creating train state.")
    state = create_train_state(jax.device_put_replicated(rng, jax.local_devices()))
    rng = jax.random.split(rng, len(jax.local_devices()))  # split rng for pmap
    print("Train state created.")

    # print number of parameters
    total_params = jax.tree.map(
        lambda x: jnp.prod(jnp.array(x.shape)),
        jax.tree.map(lambda x: x[0], state.params),
    )
    total_params = sum(jax.tree.leaves(total_params))
    print(f"Total number of parameters: {total_params}")

    # test inference real quick
    _ = inference(state, next(test_ds))

    print(f"Training for {n_steps} steps...")
    all_test_losses = []
    all_test_accs = []
    train_losses = []
    train_accuracies = []
    grad_norms = []
    for e in range(n_epochs):
        for i in range(steps_per_epoch):
            rng, state, train_loss, train_accuracy, grad_norm = train_step(
                rng, state, next(train_ds)
            )
            train_losses.append(train_loss[0].item())
            train_accuracies.append(train_accuracy[0].item())
            grad_norms.append(grad_norm[0].item())

            if state.step[0].item() % 100 == 0 or (
                i == steps_per_epoch - 1 and e == n_epochs - 1
            ):
                # sync batch stats before evaluation
                if state.batch_stats is not None:
                    state = sync_batch_stats(state)
                test_losses = []
                test_accuracies = []
                for j in range(10):
                    test_loss, test_accuracy = inference(state, next(test_ds))
                    test_losses.append(test_loss[0].item())
                    test_accuracies.append(test_accuracy[0].item())
                mean_test_loss = np.mean(test_losses)
                all_test_losses.append(mean_test_loss)
                mean_test_acc = np.mean(test_accuracies)
                all_test_accs.append(mean_test_acc)
                mean_grad_norm = np.mean(grad_norms)
                single_params = jax.tree.map(lambda x: x[0], state.params)
                params_norm = optax.global_norm(single_params)

                to_log = {
                    "epoch": e,
                    "train_loss": np.mean(train_losses),
                    "train_accuracy": np.mean(train_accuracies) * 100,
                    "test_loss": mean_test_loss,
                    "test_accuracy": mean_test_acc * 100,
                    "grad_norm": mean_grad_norm,
                    "params_norm": params_norm,
                }
                if log_to_wandb:
                    wandb.log(to_log, step=state.step[0].item())
                    wandb.summary["min_loss"] = min(all_test_losses)
                    wandb.summary["max_accuracy"] = max(all_test_accs) * 100
                if state.step[0].item() % 1000 == 0 or (
                    i == steps_per_epoch - 1 and e == n_epochs - 1
                ):
                    print(
                        "step:% 3d, epoch: % 3d, train_loss: %.4f, "
                        "train_accuracy: %.2f, test_loss: %.4f, "
                        "test_accuracy: %.2f, grad_norm: %.2f, params_norm: %.2f"
                        % (
                            state.step[0].item(),
                            e,
                            to_log["train_loss"],
                            to_log["train_accuracy"],
                            to_log["test_loss"],
                            to_log["test_accuracy"],
                            to_log["grad_norm"],
                            to_log["params_norm"],
                        )
                    )

                train_losses = []
                train_accuracies = []

    print(f"Min loss: {min(all_test_losses):.4f}")
    print(f"Max accuracy: {max(all_test_accs) * 100:.2f}%")

    if log_to_wandb:
        wandb.finish()
