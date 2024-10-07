import optax
from run_experiment import run_experiment


if __name__ == "__main__":
    optimizer = optax.contrib.schedule_free_adamw(warmup_steps=256, b1=0.95)

    run_experiment(
        log_to_wandb=True,
        wandb_entity="",
        wandb_project="image_classification_jax",
        wandb_config_update={  # extra logging info for wandb
            "optimizer": "adamw",
            "lr": 0.0025,
            "warmup": 256,
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
