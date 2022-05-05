from Common.config import get_params
import gym
from Brain.brain import Brain
from Common.logger import Logger
from Common.runner import Worker

def run_workers(worker, conn):
    worker.step(conn)

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
    
    brain = Brain(**config)

    logger = Logger(brain, **config)

    if not config["do_test"]:
        if not config["train_from_scratch"]:
            checkpoint = logger.load_weights()
            brain.set_from_checkpoint(checkpoint)
            running_ext_reward = checkpoint["running_reward"]
            init_iteration = checkpoint["iteration"]
            episode = checkpoint["episode"]
            visited_rooms = checkpoint["visited_rooms"]
            logger.running_ext_reward = running_ext_reward
            logger.episode = episode
            logger.visted_rooms = visited_rooms

        else:
            init_iteration = 0
            running_ext_reward = 0
            episode = 0 
            visited_rooms = set([1])

        workers = [Worker(i, **config) for i in range(config["n_workers"])]

        parents = []
        for worker in workers:
            parent_conn, child_conn = Pipe()
            p = Process(target=run_workers, args=(worker, child_conn,))
            p.daemon = True
            parents.append(parent_conn)
            p.start()

        if config["train_from_scratch"]:
            print("---Pre_normalization started.---")
            states = []
            total_pre_normalization_steps = config["roll_length"] * config["pre_normalization_steps"]
            actions = np.random.randint(0, config["n_actions"], (total_pre_normalization_steps, config["n_workers"]))
            for t in range(total_pre_normalization_steps):

                for worker_id, parent in enumerate(parents):
                    parent.recv() #only collect next_states for normalization

                for parent, a in zip(parents, actions[t]):
                    parent.send(a)

                for parent in parents:
                    s_, *_ = parent.recv()
                    states.append(s_[-1, ...].reshape(1, 84, 84))

                if len(states) % (config["n_workers"] * config["rollout_length"]) == 0:
                    brain.state_rms.update(np.stack(states))
                    states = []

            print("---Pre_normalization is done.---")
        
        rollout_base_shape = config["n_workers"], config["rollout_length"]

        init_states = np.zeros(rollout_base_shape + config["state_shape"], dtype=np.uint8)

