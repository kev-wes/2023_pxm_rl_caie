### Initialization
## Init packages
library(tidyverse)
library(ggplot2)
library(dplyr)
library(reticulate)
source_python("get_RUL.py")

## Init Problem
n_size          = 50 # Problem size
cost_f          = 100 # minimum predictive maintenance cost (degradation == delta)
cost_zero       = 1000 # maximum predictive maintenance cost spent (degradation == theta_deg)
cost_breakdown  = 1500 # cost for reactive / breakdown maintenance (degradation > delta)

## Init Jobs
p_vec           = floor(runif(n_size, 300, 600)) # Processing time
intensity_vec   = floor(runif(n_size, 1, 5))
job_df          = data.frame(job=1:n_size, time=p_vec, intensity = intensity_vec, m_flag=0)

## Init Reinforcement Learning
sizes         <- c(3, 50, 1)  # 3 Input (RUL, jobtime*intensity (t) [action], maintenance_flag (t)),
                              # 50 hidden, 1 output
activs        <- c("input", "swish", "linear")

N_epo                   = 10
gamma0                  = 0.9                                 # discount factor
epsilons                = pmin(0.05, 2/(log(2:(N_epo+1))^3))  # deviation 
alphas                  = 0.00005/sqrt(log(2:(N_epo+2)))      # step size 
best_mean_reward         = -Inf                                # check for best average per epoch
mean_rewards_over_epos   = rep(0, N_epo)
net_AVF                 = net.create(sizes, activs)
max_job_time_intensity  = max(job_df$time*job_df$intensity)

cost_fun = function(p_rul_hist, p_action_store){
  data.frame(p_rul_hist, p_action_store)
}

for (epoch in 1:N_epo){
  print(paste("Epoch", epoch))
  # Draw alpha and epsilon for epoch
  alpha = alphas[epoch]
  eps = epsilons[epoch]
  # Initialize epoch history
  rul_hist = 1
  rewards = 0
  action_store  = data.frame(job = NULL, time = NULL, intensity = NULL, m_flag = NULL)
  # Create action_stack that can be reduced
  action_stack = job_df
  
  # First job
  old_states = 1
  states_matrix = cbind(matrix(old_states, nrow = nrow(action_stack), ncol = length(old_states), byrow = TRUE),
                        c((action_stack[,2]*action_stack[,3])/max_job_time_intensity), action_stack[,4])
  current_AFV = net.IO(net = net_AVF, data = states_matrix)
  ## Select actions
  # Determine greedy and non-greedy action
  greedy_action         = which.max(current_AFV)
  nongreedy_actions     = (1:(nrow(action_stack)))[-(greedy_action)]
  # Determine probabilities
  probs                 = rep(eps/(nrow(action_stack)-1), nrow(action_stack))
  probs[greedy_action]  = 1-eps
  # Sample new action and store
  action_index      = sample(nrow(action_stack), 1, prob = probs) 
  old_action            = action_stack[action_index,]
  action_store          = rbind(action_store, old_action)
  # Remove last action from action_stack
  # If last action was maintenance, index will miss out of bounds w/o error,
  # and the action_stack will stay the same size
  action_stack = action_stack[-action_index,]
  old_AV = net.IO(net_AVF, data = c(old_states, action_store[1,2] * action_store[1,3], action_store[1,4]))
  
  # Initialize iterator
  i = 2
  while (dim(action_stack)[1] > 0){
    # Calculate state
    # Only calculate RUL for batch since last maintenance action or breakdown
    if (length(which(rul_hist < 0 | action_store$job == 'M')) != 0){
      rul_hist = rbind(rul_hist, get_rul(action_store[tail(which( rul_hist < 0 | action_store$job == 'M'), n = 1):nrow(action_store),]))
    # Before first maintenance or failure, use whole action store
      }else {rul_hist = rbind(rul_hist, get_rul(action_store))}
    # Calculate reward
    # On each maintenance action, give a negative reward (=real costs for batch)
    if(old_action[[1]] == "M"){
      current_reward = -(cost_zero + ( cost_f - cost_zero ) * (1-rul_hist[i-1]))
    # if machine is broken, give negative breakdown reward   
    }else if(rul_hist[i] < 0){
      current_reward = -cost_breakdown
    # else give no reward  
    }else{
      current_reward = 100
    }
    rewards = rbind(rewards, current_reward)
    # New state/action matrix (normalize actions)
    new_states = rul_hist[i]
    # Always add possibility to maintain to action stack if last action was non-maintenance
    if (action_store[i-1,1] != "M" & rul_hist[i] >=0){
      temp_action_stack_w_m = rbind(action_stack, data.frame(job = "M", time = 0, intensity = 0, m_flag = 1))
    }else{
      temp_action_stack_w_m = action_stack
    }
    states_matrix = cbind(matrix(new_states, nrow = nrow(temp_action_stack_w_m), ncol = length(new_states), byrow = TRUE),
                          c((temp_action_stack_w_m[,2]*temp_action_stack_w_m[,3])/max_job_time_intensity), temp_action_stack_w_m[,4])
    current_AFV = net.IO(net = net_AVF, data = states_matrix)
    ## Select actions
    if (nrow(temp_action_stack_w_m) == 1){
      action_index = 1
    }
    else{
      # Determine greedy and non-greedy action
      greedy_action         = which.max(current_AFV)
      nongreedy_actions     = (1:(nrow(temp_action_stack_w_m)))[-(greedy_action)]
      # Determine probabilities
      probs                 = rep(eps/(nrow(temp_action_stack_w_m)-1), nrow(temp_action_stack_w_m))
      probs[greedy_action]  = 1-eps
      # Sample new action and store
      action_index      = sample(nrow(temp_action_stack_w_m), 1, prob = probs) 
    }
    new_action            = temp_action_stack_w_m[action_index,]
    action_store          = rbind(action_store, new_action)
    # Remove last action from action_stack
    # If last action was maintenance, index will miss out of bounds w/o error,
    # and the action_stack will stay the same size
    action_stack = action_stack[-action_index,]
    ### TODO 4: If Action Store is empty thereafter, calculate overall reward?!
    ### How? Delayed reward? Bookmark Firefox
    
    ## Update NN
    new_AV          = current_AFV[action_index]
    grad            = net.gradient.RL(net_AVF, data = c(old_states, (old_action[[2]] * old_action[[3]])/max_job_time_intensity, old_action[[4]]))
    delta           = rewards[i] + gamma0 * new_AV - old_AV
    net_AVF         <- net.add.gradient.RL(net_AVF, grad = grad, stepsize = alpha * delta)
    
    # old_stuff <- new_stuff
    old_states      <- new_states
    old_action      <- new_action
    old_AV          <- new_AV
    
    i = i + 1
  }
  # TODO 4: Give reward at end of epoch
  #grad            = net.gradient.RL(net_AVF, data = c(old_states, (old_action[[2]] * old_action[[3]])/max_job_time_intensity, old_action[[4]]))
  # Calculate final reward
  #delta           = action_store
  #net_AVF         = net.add.gradient.RL(net_AVF, grad = grad, stepsize = alpha * delta)
  ### TODO 3: Visualization throughout one epoch (a la "5 Evaluation.R")
  #print(data.frame(RUL = rul_hist, reward = rewards, action_store))
  
  ### TODO 5: Evaluation and monitoring function per epoch
  ### Assess final score
  mean_rewards_over_epos[epoch] = mean(rewards)
  if (mean(rewards) > best_mean_reward){
    best_net_AVF  = net_AVF
    best_actions  = action_store
    best_rewards  = rewards
    best_RUL      = rul_hist
    print(paste("A better mean reward of ", mean(rewards), " (previously ", best_mean_reward, ") has been found in epoch ", epoch, "."))
    best_mean_reward = mean(rewards)
  }
}