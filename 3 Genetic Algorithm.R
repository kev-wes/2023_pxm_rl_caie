### Initialization
library(GA)

PopSize = 100 # Population size
alpha_ga = 0.7 # Initialization percentage
CrossProb = 0.7 # Crossover probability
MutProb = 0.01 # Mutation probability
beta_ga = 0.2 # Replacement percentage
CycleGen = 10 # Restart cycle mechanism
epsilon_min = 0.1 # Low dispersion coefficient of variation
epsilon_max = 0.6 # High dispersion coefficient of variation
Rst = 0.15 # Restart mechanism percentage
MaxGen = 200 # Stopping criteria

### FUNCTIONS ###

### Calculate degradation for a batch ###
degradation_fun = function(p_batch){
  # for each batch j calculate cumulative degradation
    v_cum_deg = 0
    # for each job k in batch j, retrieve degradation from delta_vec and cumulate
    for(k in 1:length(p_batch)){
      v_cum_deg = v_cum_deg + delta_vec[[p_batch[[k]]]]
    }
    return(v_cum_deg)
}

### Calculate fitness function 
## Calculate cost
fitness_fun = function(p_candidate){
    deg_ga = cost_ga = 0
    for (jj in 1:(length(p_candidate)-1)){ # for each batch (last batch does not incur costs)
      deg_ga = theta_deg + sum(job_df$degradation[p_candidate[[jj]]])  # set degradation to initial degradation +
      # sum of degradation of all jobs in batch jj
      cost_ga = cost_ga + cost_zero + ( cost_f - cost_zero ) * deg_ga
    }
    ## Calculate affinity/fitness = reciprocal of cost 
    affinity = 1/cost_ga
}

### Tournament selection
# Choose the fitter of two candidates for entire population
tournament_fun = function(p_population){
  # Shuffle order of candidates
  tournament_order = sample(length(p_population))
  # Create new list for tournament "winners"
  r_population_new = vector("list", length(p_population)/2)
  for(i in 1:(length(p_population)/2)){
    # Check which candidate score a higher fitness value
    if(fitness_fun(p_population[[tournament_order[[i]]]]) >= 
       fitness_fun(p_population[[tournament_order[[i*2]]]])){
      # First candidate is fitter, add to new population
      r_population_new[[i]] = p_population[[tournament_order[[i]]]]
      }
    else{
      # Second candidate is fitter, add to new population
      r_population_new[[i]] = p_population[[tournament_order[[i*2]]]]
      }
     
  }
  return(r_population_new)
}
fill_child_fun = function(p_parent, p_child){
  # Loop through batches j of parent
  for (j in 1:(length(p_parent))){
    # Loop through jobs k of parent batch j
    for (k in 1:(length(p_parent[[j]]))){
      v_job_contained = FALSE
      # Loop through batches l of child
      for (l in 1:(length(p_child))){
        # Check if job k of batch j is contained in batch l of child i
        # If yes, set flag to true
        if (p_parent[[j]][[k]] %in% p_child[[l]] == TRUE){ v_job_contained = TRUE }
      }
      # If job k is not contained in any batch of child i, insert it into child first-fit
      if (v_job_contained == FALSE){
        # Loop through batches l of child and see if there is still space for job k of parent batch j
        for (l in 1:(length(p_child))){
          v_job_inserted = FALSE
          if(degradation_fun(p_child[[l]]) + delta_vec[[p_parent[[j]][[k]]]] < delta){
            v_job_inserted = TRUE
            p_child[[l]] = append(p_child[[l]], p_parent[[j]][[k]])
            break
          }
        }
        # If all batches are full, open up new batch
        if (v_job_inserted == FALSE){p_child = append(p_child, list(p_parent[[j]][[k]]))}
      }
    }
  }
  return(p_child)
}


## Crossover of two parents
# p_population must have an even length()!
# Low prio TODO: When uneven, residual parent crosses over with random other parent
crossover_fun = function(p_population){
  # Create empty list of children with length of population as upper bound
  # (if every pair of parents gets two children)
  r_children = vector("list", length(p_population))
  crossover_order = sample(length(p_population))
  for(i in 1:(length(p_population)/2)){
    # Check if CrossProb "fires" and children are created
    if(CrossProb >= runif(1)){
      # Assign parents
      v_parent_1 = p_population[[crossover_order[[i]]]]
      v_parent_2 = p_population[[crossover_order[[i+(length(p_population)/2)]]]]
      # 1. In the first phase, blocks from both parents are sorted in
      # the order of non-increasing degradation
      # Join parent batches
      v_parents = c(v_parent_1, v_parent_2)
      # Create df with batch ids and cumulative degradation (initialized with 0s)
      v_parents_id_df = data.frame(batch=1:length(v_parents),
                                deg=numeric(length(v_parents)))
      # for each batch j calculate cumulative degradation
      for(j in 1:length(v_parents_id_df[[1]])){
        # for each job k in batch j, retrieve degradation from delta_vec and cumulate
        v_parents_id_df$deg[[j]] = degradation_fun(v_parents[[j]])
      }
      # order batch index df in descending order of cumulative degradation
      v_parents_id_df = v_parents_id_df[order(-v_parents_id_df$deg),] 
      
      # 2. Next, starting from two empty offspring, we copy the
      # fullest non-overlapping blocks from parents. In other
      # words, a block is copied in both offspring only if it contains
      # no duplicated job.
      # Always add first batch
      r_children[[i]][[1]] = v_parents[[v_parents_id_df$batch[[1]]]]
      r_children[[i+(length(p_population)/2)]][[1]] =  v_parents[[v_parents_id_df$batch[[1]]]]
      for(j in 2:length(v_parents_id_df[[1]])){
        # Check if any job of batch j is in any batch < j-1
        
        # set marker that any job of current v_parents[[v_parents_id_df$batch[[j]]]]
        # is already containend in children to FALSE
        v_job_contained = FALSE
        # Loop over all batches of children
        for (k in 1:length(r_children[[i]])){
          # Check if any job in parent is in any children
          # (only one children must be checked, as they are identical)
          if (length(intersect(v_parents[[v_parents_id_df$batch[[j]]]], r_children[[i]][[k]])) != 0){
            v_job_contained = TRUE
          }
        }
        # If no job of parent batch j was found in any children,
        # append batch j to both children
        if (v_job_contained == FALSE){
          r_children[[i]][[length(r_children[[i]])+1]] = v_parents[[v_parents_id_df$batch[[j]]]]
          r_children[[i+(length(p_population)/2)]][[length(r_children[[i+(length(p_population)/2)]])+1]] = v_parents[[v_parents_id_df$batch[[j]]]]
        }
      }
      ### Until now, both children r_children[[i]] and r_children[[i*2]] are completely identical
      ### They might be missing single jobs that could not be assigned, because they were in a
      ### a parent batch that also contained jobs that were already assigned to any child
      ### Now in step 3, we scan v_parent_1 from left to right and assign missing jobs to r_children[[i]]
      ### These jobs are inserted first fit, that means, that they can be assigned to batch that are not fully degraded
      ### If this is not possible, we, of course, open up a new batch
      ### For r_children[[i*2]], we do the same with v_parent_2
      ### Now, the children are different as the first inherits more from the first parent and vice versa
      # Loop through batches j of parent
      r_children[[i]] = fill_child_fun(v_parent_1, r_children[[i]])
      r_children[[i+(length(p_population)/2)]] = fill_child_fun(v_parent_2, r_children[[i+(length(p_population)/2)]])
    }
  }
  # remove empty children that were skipped by CrossProb
  r_children[sapply(r_children, is.null)] <- NULL 
  return(r_children)
}
  
mutation(p_population){
  for(i in 1:length(p_population)){
    # After the offspring are generated from the selection and crossover,
    # the offspring chromosomes may be mutated. Like crossover, there is
    # a mutation probability. If a randomly selected floating-point value
    # is less than the mutation probability (MutProb), mutation is performed
    # on the offspring; otherwise, no mutation occurs.
    
  }
  return(r_population)
}  




### START OF GA ###

## Initialization of Population
# create empty list where all candidates are stored
population <- list()
for(p in 1:PopSize){
  population[[p]] <-  vector("list", n_size)  # Generate n_size list as n_size is
                                                  # maximum amount of production batches
  batch_i <- 1
  # alpha_ga*PopSize candidates will be shuffled and then assigned first-fit
  if(p <= alpha_ga*PopSize){jobs_ordered = sample(n_size)}
  # (1-alpha_ga)*PopSize candidates will be sorted by degradation ascending and then assigned first-fit
  else{jobs_ordered = job_df[order(job_df$degradation, decreasing = TRUE),]$job}
  for(i in jobs_ordered){
    # assign jobs to batch until batch reaches full deg
    # if current batch + new job >= delta (= full deg)
    if(sum(job_df$degradation[c(population[[p]][[batch_i]], i)]) >= delta){ 
      # increase batch_i by one and open new batch
      batch_i <- batch_i + 1
    }
    population[[p]][[batch_i]] <- c(population[[p]][[batch_i]], i) # append job to batch_i
  }
  population[[p]][sapply(population[[p]], is.null)] <- NULL # remove empty batches
}

## Selection 
parents = tournament_fun(population)
## Crossover
children = crossover_fun(parents)
### TODO: mutation_function
mut_children = mutation_function(children)

### Transform candidate solution (list of vectors) to result format (data.frame)
result_ga <- data.frame(i=integer(), j=integer())
# TODO: Replace candidate index 1 by index of fittest candidate!
fittest_c = 5
for(j in 1:length(population[[fittest_c]])){
  for(i in 1:length(population[[fittest_c]][[j]])){
    result_ga = rbind(result_ga, data.frame(i=population[[fittest_c]][[j]][[i]], j=j))
  }
}
  
