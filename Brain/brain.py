from Brain.model import PolicyModel, PredictorModel, TargetModel
import torch
from torch.optim.adam import Adam
from Common.utils import RunningMeanStd, mean_of_list
from numpy import concatenate  # Make coder faster.

class Brain:
    def __init__(self, **config):
        self.config = config
        self.mini_batch_size = self.config["batch_size"]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        self.obs_shape = self.config["obs_shape"]

        self.current_policy = PolicyModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.predictor_model = PredictorModel(self.obs_shape).to(self.device)
        self.target_model = TargetModel(self.obs_shape).to(self.device)

        #freeze the weights in the target like the paper described
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.total_trainable_params = list(self.current_policy.parameters()) + list(self.predictor_model.parameters())
        self.optimizer = Adam(self.total_trainable_params, lr=self.config["lr"])
        
        self.state_rms = RunningMeanStd(shape=self.obs_shape)  
        self.int_reward_rms = RunningMeanStd(shape=(1,))

        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).to(self.device)
        with torch.no_grad():
            dist, int_value, ext_value, action_prob, = self.current_policy(state)
            action = dist.sample()
            log_prob = dist.sample()
        return action.cpu().numpy(), int_value.cpu().numpy().squeeze(), \
               ext_value.cpu().numpy().squeeze(), log_prob.cpu().numpy(), \
               action_prob.cpu().numpy()

   
   #############
   #We want to have trajectories on gpu
   #the yield makes this a generator and so the function will only
   #be called for each iteration of the for loop, but will not save the 
   #value in memory
   #see 
   # https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
   #for more details
   ############
    def choose_mini_batch(self, states, actions, int_returns, ext_returns, advs,                          log_probs, next_states):
        states = torch.ByteTensor(states).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        actions = torch.ByteTensor(actions).to(self.device)
        advs = torch.Tensor(advs).to(self.device)
        int_returns = torch.Tensor(int_returns).to(self.device)
        ext_returns = torch.Tensor(ext_returns).to(self.device)
        log_probs = torch.Tensor(log_probs).to(self.device)

        #return an an array of random ints equal in shape to the 
        #of the number of minibatches and the size of minibatches
        indices = np.random.randint(0, len(states), (self.config["n_mini_bach"],                                    self.mini_batch_size))

        for idx in indicies:
            yield states[idx], actions[idx], int_returns[idx], \
            ext_returns[idx], advs[idx], log_probs[idx], next_states[idx]

    
    ###################
    #General Advantage estimation - how much better off we are taking a particular action in a particular state 
    #https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
    #
    #lambda is a smoothing parameter to reduce variance in training
    #gamma is a discount factor for the rewards to emphasize the value of current states over future states
    ##################
    def get_gae(self, rewards, values, next_values, dones, gamma):
        lam = self.config["lambda"] #Make code faster
        returns = [[] for _ in range(self.config["n_workers"])]
        extended_values = np.zeros((self.config["n_workers"], 
                                    self.config["rollout_length"] + 1))
        for worker in range(self.config["n_workers"]):
            extended_values[worker] = np.append(values[worker], next_values[worker])
            gae = 0 #initialize the advantage
            for step in reversed(range(len(rewards[worker]))):
                #the mask is so that we do not consider new states after our environment has been reset=
                mask = (1 - dones[worker][step])
                #this is trying to say what is my current state value compared to my next state value. as this is in reverse
                #we are trying to work our way back and give values to each state from the final known reward to the beginning
                delta = reward[worker][step] + gamma * (extended_values[worker][step + 1]) * mask - extended_values[worker][step]
                gae = delta + gamma * lam * mask * gae

                #finally calculate the returns
                returns[worker].insert(0, gae + extended_values[worker][step])
        return concatenate(returns)

    #####
    #clipping the ratio for the loss in defined in the ppo paper
    #Using CLIPloss
    #https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
    #under PPO section for more information
    ####
    def compute_pg_loss(self, ratio, adv):
        new_r = ratio * adv
        clamped_r = torch.clamp(ratio, 1 - self.config["clip_range"], 1 + self.config["clip_range"]) * adv
        loss = torch.min(new_r, clamped_r)
        loss = -loss.mean() #Expected value
        return loss

    
    def calculate_rnd_loss(self, next_state):
        encoded_target_features = self.target_model(next_state)
        encoded_predictor_features = self.predictor_model(next_state)
        loss = (encoded_predictor_features - encoded_target_features).pow(2).mean(-1)

        #Proportion of exp used for the predictor update
        #Help
        mask = torch.rand(loss.size(), device=self.device)
        mask = (mask < self.config["predictor_proportion"]).float()
        loss = (mask * loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        return loss

    #####
    #This was based off the openAI random network distillation
    #https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py
    #####
    @mean_of_list
    def train(self, states, actions, int_rewards, ext_rewards, dones,
              int_values, ext_values, log_probs, next_int_values,
              next_ext_values, total_next_obs):
        int_rets = self.get_gae(int_rewards, int_values, next_int_values,
                                np.zeros_like(dones), self.config["int_gamma"])
        ext_rets = self.get_gae(ext_rewards, ext_values, next_ext_values, 
                                dones, self.config["ext_gamma"])

        ext_values = conconcatenate(ext_values)
        ext_adv = ext_rets - ext_values

        int_values = concatenate(int_values)
        int_advs = int_rets - int_values

        advs = ext_advs * self.config["ext_adv_coeff"] + int_advs * self.config["int_adv_coeff"]
        
        #Running mean std
        self.state_rms.update(total_next_obs)

        total_next_obs = ((total_next_obs - self.state_rms.mean) / (self.state_rms.var ** 0.5)).clip(-5, 5)

        pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies = [],[],[],[],[]

        for epoch in range(self.config["n_epochs"]):
            for state, action, int_return, ext_return, adv, old_log_prob, \
            next_state in self.choose_mini_batch(states=states,
                                                 actions=actions,
                                                 int_returns=int_rets,
                                                 ext_return=ext_rets,
                                                 advs=advs,
                                                 log_probs=log_probs,
                                                 next_states=total_next_obs):
                dist, int_value, ext_value, _ = self.current_policy(state)

                # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy
                entropy = dist.entropy().mean(),
                new_log_prob = dist.log_prob(action)
                ratio = (new_log_prob - old_log_prob).exp()
                pg_loss = self.compute_pg_loss(ratio, adv)
               
                #calculate the critic loss
                int_value_loss = self.mse_loss(int_value.squeeze(-1), int_return)
                ext_value_loss = self.mse_loss(ext_value.squeeze(-1), ext_return)
                critic_loss = (int_value_loss + ext_value_loss)

                rnd_loss = self.calculate_rnd_loss(next_state)

                total_loss = critic_loss + pg_loss - self.config["ent_coeff"] * entropy + rnd_loss

                self.optimize(total_loss)

                pg_losses.append(pg_loss.item())
                ext_v_losses.append(ext_value_loss.item())
                int_v_losses.append(int_value_loss.item())
                rnd_losses.append(rnd_loss.item())
                entropies.append(entropy.item())
                
        return pg_losses, ext_v_losses, int_v_losses, rnd_losses, entropies, int_values, \
        int_values, int_rets, ext_values, ext_rets 
