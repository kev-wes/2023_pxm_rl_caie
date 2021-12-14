### Initialization
## Init packages
library(reticulate)
source_python("get_RUL.py")

## Init Problem
n_size          = 20 # Problem size
cost_f          = 100 # minimum predictive maintenance cost (degradation == delta)
cost_zero       = 1000 # maximum predictive maintenance cost spent (degradation == theta_deg)
cost_breakdown  = 2000 # cost for reactive / breakdown maintenance (degradation > delta)

## Init Jobs
p_vec           = floor(runif(n_size, 100, 200)) # Processing time
intensity_vec   = floor(runif(n_size, 1, 5))
job_df          = data.frame(job=1:n_size, time=p_vec, intensity = intensity_vec, m_flag=0)

## Init Reinforcement Learning
sizes         <- c(5, 50, 1)  # 5 Input (RUL, time*intensity (t-1), maintenance_flag (t-1),
                              # jobtime*intensity (t) [action], maintenance_flag (t)),
                              # 50 hidden, 1 output
activs        <- c("input", "swish", "linear")

N_epo                   = 3
gamma0                  = 0.9                                 # discount factor
epsilons                = pmin(0.05, 2/(log(2:(N_epo+1))^3))  # deviation 
alphas                  = 0.00005/sqrt(log(2:(N_epo+2)))      # step size 
best_mean_reward        = -Inf                                # check for best average per epoch
mean_rewards_over_epos  = rep(0, N_epo)
net_AVF                 = net.create(sizes, activs)
max_job_time_intensity  = max(job_df$time*job_df$intensity)

for (epoch in 1:N_epo){
  print(paste("Epoch", epoch))
  # Draw alpha and epsilon for epoch
  alpha = alphas[epoch]
  eps = epsilons[epoch]
  # Initialize epoch history
  rul_hist = 1 #rul_hist = rbind(rul_hist, 2.5)
  rewards = 0
  action_store  = data.frame(job = NULL, time = NULL, intensity = NULL, m_flag = NULL)
  # First job = maintenance
  old_action = data.frame(job = "M", time = 0, intensity = 0, m_flag = 1)
  action_store =  rbind(action_store, old_action)
  # Create action_stack that can be reduced
  action_stack = job_df
  # Generate first action value (RUL=1, last job = M, M_flag (t-1) = 1, current job = sampled, M_flag = 0) 
  old_states = c(1, 0, 0)
  old_AV = net.IO(net_AVF, data = c(old_states, action_store[1,2] * action_store[1,3], action_store[1,4]))
  
  # Initialize iterator
  i = 2
  while (dim(action_stack)[1] > 0){
    # Calculate state
    # TODO Only calculate RUL for batch since last maintenance action
    rul_hist = rbind(rul_hist, get_rul(action_store))
    # Calculate reward
    # On each maintenance action, give a negative reward (=real costs for batch)
    if(old_action[[1]] == "M"){
      current_reward = -(cost_zero + ( cost_f - cost_zero ) * (1-rul_hist[i-1]))
    # else give small reward proportional to used life of machine  
    }else if(rul_hist[i] >= 0){
      current_reward = (cost_zero - cost_f) * (1-rul_hist[i])
    # if machine is broken, give negative breakdown reward  
    }else{
      current_reward = -cost_breakdown
    }
    rewards = rbind(rewards, current_reward)
    # New state/action matrix (normalize actions)
    new_states = c(rul_hist[i], (action_store[i-1,2] * action_store[i-1,3])/max_job_time_intensity, action_store[i-1,4])
    # Always add possibility to maintain to action stack if last action was non-maintenance
    if (action_store[i-1,1] != "M"){
      temp_action_stack_w_m = rbind(action_stack, data.frame(job = "M", time = 0, intensity = 0, m_flag = 1))
    }else{
      temp_action_stack_w_m = action_stack
    }
    states_matrix = cbind(matrix(new_states, nrow = nrow(temp_action_stack_w_m), ncol = length(new_states), byrow = TRUE),
                          c((temp_action_stack_w_m[,2]*temp_action_stack_w_m[,3])/max_job_time_intensity), temp_action_stack_w_m[,4])
    current_AFV = net.IO(net = net_AVF, data = states_matrix)
    ## Select actions
    # If machine has RUL = 0, automatically assign maintenance (best case)
    if(rul_hist[i] < 0)
    {
      new_action_index      = nrow(temp_action_stack_w_m)
    }else{
      # Determine greedy and non-greedy action
      greedy_action         = which.max(current_AFV)
      nongreedy_actions     = (1:(nrow(temp_action_stack_w_m)))[-(greedy_action)]
      # Determine probabilities
      probs                 = rep(eps/(nrow(temp_action_stack_w_m)-1), nrow(temp_action_stack_w_m))
      probs[greedy_action]  = 1-eps
      # Sample new action and store
      new_action_index      = sample(nrow(temp_action_stack_w_m), 1, prob = probs) 
    }
    new_action            = temp_action_stack_w_m[new_action_index,]
    action_store          = rbind(action_store, new_action)
    # Remove last action from action_stack
    # If last action was maintenance, index will miss out of bounds w/o error,
    # and the action_stack will stay the same size
    action_stack = action_stack[-new_action_index,]
    ### TODO: If Action Store is empty thereafter, calculate overall reward?!
    
    ## Update NN
    new_AV          = current_AFV[new_action_index]
    grad            = net.gradient.RL(net_AVF, data = c(old_states, (old_action[[2]] * old_action[[3]])/max_job_time_intensity, old_action[[4]]))
    delta           = rewards[i] + gamma0 * new_AV - old_AV
    net_AVF         <- net.add.gradient.RL(net_AVF, grad = grad, stepsize = alpha * delta)
    
    # old_stuff <- new_stuff
    old_states      <- new_states
    old_action      <- new_action
    old_AV          <- new_AV
    
    i = i + 1
  }
}

# for epoch
#   init 
#     network mit 3 neurons: current RUL (state), job dauer (action), job intensität (action)
#     job storage mit N jobs, job dauern, job intensit?ten
#     maintenance storage mit N-1 maintenance interventions
#
#   repeat{
#     Simuliere anhand von letzter Action den neuen State (RUL)
#     berechne reward:
#       Wenn breakdown: -1000
#       wenn letzte action = M
#         überbleibende RUL*Kosten pro RUL
#       Sonst: 1-RUL [Lebensspanne der Maschine]/1000[Reduktionsfaktor]
#     
#     wenn job storage leer:
#       break
#     
#     for all jobs + maintenance, query value from NN
#     store the greedy action (highest value)
#     sample and store a new action with the chance of 1-eps = new action and eps = any other action
#     remove action from job or maintenance storage
#
#     Update NN
#       compute gradient with old state
#       Berechne temporal difference error
#       add gradient to net
#
#     Old stuff = new stuff
#   }