### TODO
# 1 Rebuild uit het Broek 2019 and replace stochastic function with real battery and RUL
# 2 Make maintenance intervention movable
# 3 Integrate inventory, customer demand, etc.

### Initialization
## Init packages
library(reticulate)
source_python("get_RUL.py")

## Init Problem
prev_main_c  = 2 # preventive maintenance cost
corr_main_c  = 6 # corrective maintenance cost
time_steps   = 30 # time horizon
time_eqv     = 10 # real time equivalents of one time_step
# TODO: allow decimal intensities
actions      = 0:6 # possible intensities 
revenue      = 0.001 # Revenue per time and intensity level

## Init Reinforcement Learning
sizes        = c(2, 50, 1)  # 2 Input (RUL, intensity [action]), 50 hidden, 1 output
activs       = c("input", "swish", "linear")
n_epo        = 200
gamma0       = 0.9                                 # discount factor
epsilons     = pmin(0.05, 2/(log(2:(n_epo+1))^3))  # deviation 
alphas       = 0.0005/sqrt(log(2:(n_epo+2)))      # step size 
best_avg_rew = -Inf                                # check for best average per epoch
mean_rew_epo = rep(0, n_epo)
net_AVF      = net.create(sizes, activs)


for (epoch in 2:n_epo){
  print(paste("Epoch", epoch))
  # Draw alpha and epsilon for epoch
  alpha = alphas[epoch]
  eps = epsilons[epoch]
  # Initialize epoch history
  state_hist = rep(1, time_steps) # RUL history
  rew_hist = rep(0, time_steps) # Reward history
  action_hist  = rep(0, time_steps) # Action history
  
  # First job
  old_state = 1
  # Sample first action and store
  action_hist[1]  = old_action = sample(actions, 1) 
  old_AV = net.IO(net_AVF, data = c(old_state, old_action))
  
  for (i in 1:time_steps){
    # Calculate reward
    rew_hist[i] = reward = old_action*revenue*time_eqv
    # If machine failed, it stays failed
    if (old_state < 0){
      new_state = state_hist[i] = 0
      # Production is no longer possible
      action_hist[i] = new_action = 0
      # Add corrective maintenance cost
      rew_hist[i] = reward = reward-corr_main_c
    # Else, calculate RUL from inputs
    }else if(old_state == 0){
      new_state = state_hist[i] = 0
      # Production is no longer possible
      action_hist[i] = new_action = 0
    }else{
      # Build dataframe with real times and their production intensities
      action_time = data.frame(time = seq(time_eqv, time_steps*time_eqv+time_eqv, time_eqv)[1:i],
                               intensity = action_hist[1:i])
      new_state = state_hist[i] = round(get_rul(action_time), 2)
      # New state/action matrix (normalize actions)
      states_matrix = cbind(matrix(new_state, nrow = length(actions), ncol = length(new_state), byrow = TRUE),
                            actions/max(actions))
      current_AFV = net.IO(net = net_AVF, data = states_matrix)
      ## Select actions
      # Determine greedy and non-greedy action
      greedy_action         = which.max(current_AFV)
      nongreedy_actions     = (1:(length(actions)))[-(greedy_action)]
      # Determine probabilities
      probs                 = rep(eps/(length(actions)-1), length(actions))
      probs[greedy_action]  = 1-eps
      # Sample new action and store
      action_hist[i] = new_action = sample(actions, 1, prob = probs) 
    }
    
    # update ANN #
    new_AV = net.IO(net_AVF, data = c(new_state, new_action))
    grad            = net.gradient.RL(net_AVF, data = c(old_state, old_action/max(actions)))
    delta           = reward + gamma0 * new_AV - old_AV
    net_AVF         = net.add.gradient.RL(net_AVF, grad = grad, stepsize = alpha * delta)
    
    # old_stuff <- new_stuff
    old_state       = new_state
    old_action      = new_action
    old_AV          = new_AV
  }
  mean_rew_epo[epoch] = mean(rew_hist)
  if (mean(rew_hist) > best_avg_rew){
    best_net_AVF  = net_AVF
    best_actions  = action_hist
    best_rewards  = rew_hist
    best_RUL      = state_hist
    print(paste("A better mean reward of ", mean(rew_hist), " (previously ", best_avg_rew, ") has been found in epoch ", epoch, "."))
    best_avg_rew = mean(rew_hist)
  }
}
plot(mean_rew_epo)
lines(mean_rew_epo)
