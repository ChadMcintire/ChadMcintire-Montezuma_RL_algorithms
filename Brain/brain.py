from Brain.model import PolicyModel, PredictorModel, TargetModel
import torch
from torch.optim.adam import Adam
from Common.utils import RunningMeanStd

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
