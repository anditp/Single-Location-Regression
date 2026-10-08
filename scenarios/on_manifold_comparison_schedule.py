def get_configs():
    lambdas = [1e-2, 3e-2, 5e-2, 7e-2]
    configs = []
    for lam in lambdas:
        tag = f"{lam:.0e}".replace("e-0", "e-")
        configs.append(
            {
                "mode": "erf",
                "config": {
                    "name": f"on_manifold_comparison_schedule_{tag}",
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
                    "use_penalty": False,
                    "penalty_kv_weight": 10.0,
                    "penalty_mean_weight": 1.0,
                    "init_on_manifold": True,
                    "lambda_r": lam,
                    "lambda_c": lam,
                    "lambda_r0": 0.1,
                    "lambda_c0": 0.1,
                },
            }
        )
    return configs
