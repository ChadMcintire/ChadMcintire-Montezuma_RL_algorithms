from Common.config import get_params
import gym

if __name__ == '__main__':
    config = get_params()
    
    test_env = gym.make(config["env_name"])

