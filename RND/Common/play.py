from torch import device
import os
from Common.utils import *
import time


class Play:
    def __init__(self, env, agent, checkpoint, max_episode=1):
        self.env = make_atari(env, 4500, sticky_action=False)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_from_checkpoint(checkpoint)
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists("Results"):
            os.mkdir("Results")
        self.VideoWriter = cv2.VideoWriter("Results/" + "result" + ".avi", self.fourcc, 50.0,
                                           self.env.observation_space.shape[1::-1])

    def evaluate(self):
        stacked_states = np.zeros((84, 84, 4), dtype=np.uint8)
        mean_ep_reward = []
        obs, int_rewards = [], []
        for ep in range(self.max_episode):
            self.env.seed(ep)
            s = self.env.reset()
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            clipped_ep_reward = 0
            done = False
            while not done:
                action, *_ = self.agent.get_actions_and_values(stacked_states)
                s_, r, done, info = self.env.step(action)
                #to get the room number pirnt the below item
                #print(self.env.unwrapped.ale.getRAM()[3])

                episode_reward += r
                clipped_ep_reward += np.sign(r)

                stacked_states = stack_states(stacked_states, s_, False)

                int_reward = self.agent.calculate_int_rewards(stacked_states[-1, ...].reshape(1, 84, 84), batch=False)
                int_rewards.append(int_reward)
                obs.append(s_)

                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                self.env.render()
                time.sleep(0.01)
            print(f"episode reward:{episode_reward}| "
                  f"clipped episode reward:{clipped_ep_reward}| "
                  f"Visited rooms:{info['episode']['visited_room']}")
            mean_ep_reward.append(episode_reward)
            self.env.close()
            self.VideoWriter.release()
            cv2.destroyAllWindows()
        print(f"Mean episode reward:{sum(mean_ep_reward) / len(mean_ep_reward):0.3f}")
