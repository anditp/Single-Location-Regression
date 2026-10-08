def get_configs():
    return [
        {
            "mode": "erf",
            "config": {
                "name": "on_manifold_original",
                "M": 5,
                "N": 5,
                "d": 20,
                "ell": 100,
                "gamma": 1.0,
                "epsilon": 0.1,
                "num_iterations": 100000,
                "batch_size": 256,
                "lr": 1e-4,
                "log_interval": 100,
                "device": "cpu",
                "lambda_schedule": True,
                "schedule_switch_iter": 20000,
                "lambda_increment": 1e-4,
                "use_penalty": False,
                "penalty_kv_weight": 10.0,
                "penalty_mean_weight": 1.0,
                "init_on_manifold": True,
                "lambda_r": 5e-2,
                "lambda_c": 5e-2,
                "lambda_r0": 5e-2,
                "lambda_c0": 5e-2,
            },
        }
    ]
