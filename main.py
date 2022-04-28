from Common.config import get_params
import gym

if __name__ == '__main__':
    config = get_params()
    
    #set up environment
    test_env = gym.make(config["env_name"])

    #add the size of the action space to the configuration dictionary
    config.update({"n_actions": test_env.action_space.n})
    
    #close popup window for test environment
    test_env.close()

    #set batch size based on the config values for rollout length, number of workers, and the number of minibatches 
    config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})

    #not sure what the predictor_proportion does
    config.update({"predictor_proportion": 32 / config["n_workers"]})

