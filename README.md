# image-classification-jax

Run image classification experiments in JAX with ViT, resnet, cifar10, cifar100, imagenette, and imagenet.

Meant to be simple but good quality. Includes:
- ViT with qk normalization, swiglu, empty registers
- Palm style z-loss
- ability to use schedule-free from `optax.contrib`
- ability to use PSGD optimizers from `psgd-jax` with hessian calc
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

Use `run_experiment` to run an experiment. The following example uses the `xmat` 
optimizer from `psgd-jax` wrapped in schedule-free.

```python
import optax
from image_classification_jax.run_experiment import run_experiment
from psgd_jax.xmat import xmat

lr = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.0, 0.01, 256),
        optax.constant_schedule(0.01),
    ],
    boundaries=[256],
)

optimizer = optax.contrib.schedule_free(xmat(lr, b1=0.0), learning_rate=lr, b1=0.95)

run_experiment(
    log_to_wandb=True,
    wandb_entity="",
    wandb_project="image_classification_jax",
    wandb_config_update={
        "optimizer": "psgd_xmat",
        "schedule_free": True,
        "learning_rate": 0.01,
        "warmup_steps": 256,
        "b1": 0.95,
    },
    global_seed=100,
    dataset="cifar10",
    batch_size=64,
    n_epochs=10,
    optimizer=optimizer,
    compute_in_bfloat16=False,
    l2_regularization=1e-4,
    randomize_l2_reg=False,
    apply_z_loss=True,
    model_type="vit",
    n_layers=12,
    enc_dim=768,
    n_heads=12,
    n_empty_registers=0,
    dropout_rate=0.0,
    using_schedule_free=True,
    psgd_calc_hessian=True,
    psgd_precond_update_prob=0.1,
)
```