from image_classification_jax import run_experiment


if __name__ == "__main__":
    run_experiment(
        log_to_wandb=True,
        wandb_project="image_classification_jax",
        wandb_config_update=None,
        global_seed=100,
        dataset="imagenet",
        imagenet_gcs_path="gs://diffdata/imagenet",
        batch_size=256,
        n_epochs=50,
        compute_in_bfloat16=True,
        l2_regularization=0.0,
        randomize_l2_reg=False,
        apply_z_loss=False,
        model_type="vit",
        n_layers=24,
        enc_dim=768,
        n_heads=12,
        n_kv_heads=4,
        using_schedule_free=False,
        psgd_calc_hessian=False,
    )
