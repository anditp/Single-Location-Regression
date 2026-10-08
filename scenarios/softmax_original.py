def get_configs():
    return [
        {
            "mode": "softmax",
            "config": {
                "name": "softmax_original",
                "M": 5,
                "N": 5,
                "d": 20,
                "ell": 100,
                "gamma": 1.0,
                "epsilon": 0.1,
                "num_iterations": 100000,
                "batch_size": 256,
                "lr": 1e-3,
                "log_interval": 100,
                "device": "cpu",
                "lambda_schedule": True,
                "schedule_switch_iter": 20000,
                "lambda_increment": 1e-4,
                "use_penalty": True,
                "penalty_kv_weight": 10.0,
                "penalty_mean_weight": 1.0,
                "init_on_manifold": False,
                "lambda_r": 1e-1,
                "lambda_c": 1e-1,
                "lambda_r0": 1e-1,
                "lambda_c0": 1e-1,
            },
        }
    ]
