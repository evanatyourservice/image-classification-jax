# image-classification-jax

Run image classification experiments in JAX with ViT, resnet, cifar10, cifar100, imagenette, and imagenet.

Meant to be simple but good quality. Includes:
- ViT with qk normalization, swiglu, empty registers
- Palm style z-loss (https://arxiv.org/pdf/2204.02311)
- ability to use schedule-free from `optax.contrib`
- datasets currently implemented include cifar10, cifar100, imagenette, and imagenet

Currently no model sharding, only data parallelism (automatically splits batch `batch_size/n_devices`).


## Installation

```bash
pip install image-classification-jax
```


## Usage

Set your wandb key either in your python script or through command line:
```bash
export WANDB_API_KEY=<your_key>
```

Use `run_experiment` to run an experiment. Here's how you could run an experiment with
PSGD affine optimizer wrapped with schedule-free:

```python
import optax
from image_classification_jax.run_experiment import run_experiment
from psgd_jax.affine import affine

base_lr = 0.001
warmup = 256
lr = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.0, base_lr, warmup),
        optax.constant_schedule(base_lr),
    ],
    boundaries=[warmup],
)

psgd_opt = optax.chain(
    optax.clip_by_global_norm(1.0),
    affine(
        lr,
        preconditioner_update_probability=1.0,
        b1=0.0,
        weight_decay=0.0,
        max_size_triangular=0,
        max_skew_triangular=0,
        precond_init_scale=1.0,
    ),
)

optimizer = optax.contrib.schedule_free(psgd_opt, learning_rate=lr, b1=0.95)

run_experiment(
    log_to_wandb=True,
    wandb_entity="",
    wandb_project="image_classification_jax",
    wandb_config_update={  # extra logging info for wandb
        "optimizer": "psgd_affine",
        "lr": base_lr,
        "warmup": warmup,
        "b1": 0.95,
        "schedule_free": True,
    },
    global_seed=100,
    dataset="cifar10",
    batch_size=64,
    n_epochs=10,
    optimizer=optimizer,
    compute_in_bfloat16=False,
    apply_z_loss=True,
    model_type="vit",
    n_layers=4,
    enc_dim=64,
    n_heads=4,
    n_empty_registers=0,
    dropout_rate=0.0,
    using_schedule_free=True,  # set to True if optimizer wrapped with schedule_free
)
```
