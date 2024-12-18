import optax
from image_classification_jax.run_experiment import run_experiment
from psgd_jax.affine import affine


if __name__ == "__main__":
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
        l2_regularization=0.0001,
        randomize_l2_reg=False,
        apply_z_loss=True,
        model_type="vit",
        n_layers=4,
        enc_dim=64,
        n_heads=4,
        n_empty_registers=0,
        dropout_rate=0.0,
        using_schedule_free=True,  # set to True if optimizer wrapped with schedule_free
        psgd_calc_hessian=False,  # set to True if using PSGD and want to calc and pass in hessian
        psgd_precond_update_prob=1.0,
    )
