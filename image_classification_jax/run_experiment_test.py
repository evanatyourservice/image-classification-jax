import optax
from image_classification_jax.run_experiment import run_experiment
from psgd_jax.xmat import xmat


if __name__ == "__main__":
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
        l2_regularization=1e-5,
        randomize_l2_reg=False,
        apply_z_loss=True,
        model_type="vit",
        n_layers=4,
        enc_dim=64,
        n_heads=4,
        n_empty_registers=0,
        dropout_rate=0.0,
        using_schedule_free=True,
        psgd_calc_hessian=True,
        psgd_precond_update_prob=0.1,
    )
