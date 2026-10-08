def get_configs():
    return [
        {
            "mode": "erf",
            "config": {
                "name": "not_manifold_original_N_50_ell_100",
                "M": 5,
                "N": 50,
                "d": 20,
                "ell": 10,
                "gamma": 1.0,
                "epsilon": 0.1,
                "num_iterations": 200000,
                "batch_size": 16,
                "lr": 1e-3,
                "log_interval": 100,
                "device": "cpu",
                "lambda_schedule": True,
                "schedule_switch_iter": 5000,
                "lambda_increment": 1e-4,
                "use_penalty": True,
                "penalty_kv_weight": 100.0,
                "penalty_mean_weight": 10.0,
                "init_on_manifold": False,
                "lambda_r": 1e-2,
                "lambda_c": 1e-2,
                "lambda_r0": 1e-1,
                "lambda_c0": 1e-1,
            },
        }
    ]
