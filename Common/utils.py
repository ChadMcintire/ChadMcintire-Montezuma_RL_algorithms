import numpy as np
import gym

def update_mean_var_count_from_moments(mean, var, count, 
                                       batch_mean, batch_var, batch_count):
   delta = batch_mean - mean
   tot_count = count + batch_count

   new_mean = mean + delta * batch_count / tot_count
   m_a = var * count
   m_b = batch_var * batch_count
   M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
   new_var = M2 / tot_count

   return new_mean, new_var, new_count


##############
#As we get more batches we update the mean, variance, and count
#
##############
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #Parallel_algorithm # -> It's indeed batch normalization. :D
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
        self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def mean_of_list(func):
    def function_wrapper(*args, **kwargs):
        lists = func(*args, **kwargs)
        return [sum(list) / len(list) for list in lists[:-4]] + \
        [explained_variance(lists[-2], lists[-1])]

    return function_wrapper

def make_atari(env_id, max_episode_steps, sticky_action=True, max_and_skip=True):
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps * 4
    assert 'NoFrameskip' in env.spec.id
    if sticky_action:
        env = StickyActionEnv(env)
    if max_and_skip:
        env = RepeatActionEnv(env)
    env = MontezumaVisitedRoomEnv(env, 3)
    env = AddRandomStateToInfoEnv(env)

    return env

def StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()


        
