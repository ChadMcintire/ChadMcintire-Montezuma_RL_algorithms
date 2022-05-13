import argparse

def get_params():
    parser = argparse.ArgumentParser(
    description="Variable parameters based on the configuration of the machine or user's choice")

    parser.add_argument("--n_workers", default=2, type=int, help="Number of parallel environments.")
    parser.add_argument("--interval", default=50, type=int, help="The interval specifies how often different parameters should be saved and printed,"
    " counted by iterations.")

    parser.add_argument("--do_test", action="store_true", help="The flag determines whether to train the agent or play with it.")

    parser.add_argument("--render", action="store_true", help="The flag determines whether to render each agent or not.")

    #parser.add_argument("--train_from_scratch", action="store_false", help="The flag determines whether to train from scratch or continue previous tries.")
    #parser.add_argument('--no-feature', dest='feature', action='store_false')
    do_test_parser = parser.add_mutually_exclusive_group(required=False)
    do_test_parser.add_argument('--train_from_scratch', dest='train_from_scratch', action='store_true')
    do_test_parser.add_argument('--no-train_from_scratch', dest='train_from_scratch', action='store_false')
    parser.set_defaults(train_from_scratch=True)

    parser_params = parser.parse_args()

    """
    Parmeters based on the "Exploration by Random Network Distillation" paper.
    https://arxiv.org/abs/1810.12894    
    """        

                      #stops finding new rooms after this point in rollouts
                      #"total_rollouts_per_env": int(30e3),
    default_params = {"env_name": "MontezumaRevengeNoFrameskip-v4",
                      "state_shape": (4, 84, 84),
                      "obs_shape": (1, 84, 84),
                      "total_rollouts_per_env": int(30e3),
                      "max_frames_per_episode": 4500, #
                      "rollout_length": 128,
                      "n_epochs": 4,
                      "n_mini_batch": 4,
                      "lr": 1e-4,
                      "ext_gamma": 0.999,
                      "int_gamma": 0.99,
                      "lambda": 0.95,
                      "ext_adv_coeff": 2,
                      "int_adv_coeff": 1,
                      "ent_coeff": 0.001, 
                      "clip_range": 0.1,
                      "pre_normalization_steps": 50
                      }

    # endregion
    total_params = {**vars(parser_params), **default_params}
    print("params:", total_params)
    return total_params
