### This implements

### Initialization
library(dplyr)

PopSize = 200 # Population size
alpha_ga = 0.8 # Initialization percentage
CrossProb = 0.7 # Crossover probability
MutProb = 0.015 # Mutation probability
beta_ga = 0.2 # Replacement percentage
CycleGen = 20 # Restart cycle mechanism
epsilon_min = 0.2 # Low dispersion coefficient of variation
epsilon_max = 0.7 # High dispersion coefficient of variation
Rst = 0.25 # Restart mechanism percentage
MaxGen = 300 # Stopping criteria

### FUNCTIONS ###

## Initialization of Population
generate_pop_fun = function(p_init=FALSE, p_popsize=PopSize){
  # create empty list where all candidates are stored
  r_population <- list()
  for(p in 1:p_popsize){
    r_population[[p]] <-  vector("list", n_size)  # Generate n_size list as n_size is
    # maximum amount of production batches
    v_batch_i <- 1
    # In the initialization assign jobs either first-fit (prob = alpha_ga) or
    # sort descending by degradation and then first-fit (prob = 1 - alpha_ga)
    # In non-initialization cases, assign always random first-fit
    # alpha_ga*p_popsize candidates will be shuffled and then assigned first-fit
    if(p <= alpha_ga*p_popsize | p_init == FALSE){v_jobs_ordered = sample(n_size)}
    # (1-alpha_ga)*p_popsize candidates will be sorted by degradation ascending and then assigned first-fit
    else{v_jobs_ordered = job_df[order(job_df$degradation, decreasing = TRUE),]$job}
    
    # Assignment
    for(i in v_jobs_ordered){
      # assign jobs to batch until batch reaches full deg 
      # if current batch + new job >= delta (= full deg)
      if(sum(job_df$degradation[c(r_population[[p]][[v_batch_i]], i)]) >= delta){ 
        # increase v_batch_i by one and open new batch
        v_batch_i <- v_batch_i + 1
      }
      r_population[[p]][[v_batch_i]] <- c(r_population[[p]][[v_batch_i]], i) # append job to v_batch_i
    }
    r_population[[p]][sapply(r_population[[p]], is.null)] <- NULL # remove empty batches
  }
  return(r_population)
}

## Calculate degradation for a batch ###
degradation_fun = function(p_batch){
  # for each batch j calculate cumulative degradation
    v_cum_deg = 0
    # for each job k in batch j, retrieve degradation from delta_vec and cumulate
    for(k in 1:length(p_batch)){
      v_cum_deg = v_cum_deg + delta_vec[[p_batch[[k]]]]
    }
    return(v_cum_deg)
}

## Calculate cost function 
ga_cost_fun = function(p_candidate, p_con_offset = 0, p_pow_offset = 1){
  v_deg_ga = r_cost_ga = 0
  for (jj in 1:(length(p_candidate)-1)){ # for each batch (last batch does not incur costs)
    v_deg_ga = theta_deg + sum(job_df$degradation[p_candidate[[jj]]])
    # set degradation to initial degradation +
    # sum of degradation of all jobs in batch jj
    r_cost_ga = r_cost_ga + cost_zero + ( cost_f - cost_zero ) * v_deg_ga
  }
  # Return cost. If sd parameters are set, substract constant offset (p_con_offset)
  # and raise to power of power offset (p_pow_offset)
  return((r_cost_ga-p_con_offset)**p_pow_offset)
}

## Calculate fitness function 
fitness_fun = function(p_candidate){
    ## Calculate affinity/fitness = reciprocal of cost 
    r_affinity = 1/ga_cost_fun(p_candidate)
}

## Tournament selection
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
# p_population must have an even length() as it gets halved!
# Optional Open Issue (ToDo, low prio): When uneven, residual parent crosses over with random other parent
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

## Mutate individuals randomly by Swapping
mutation_fun = function(p_population, p_mutprob){
  for(i in 1:length(p_population)){
    # After the offspring are generated from the selection and crossover,
    # the offspring chromosomes may be mutated. Like crossover, there is
    # a mutation probability. If a randomly selected floating-point value
    # is less than the mutation probability (p_mutprob), mutation is performed
    # on the offspring; otherwise, no mutation occurs.
    if(p_mutprob >= runif(1)){
      # For each mutation operation, the number of permutations is randomly
      # chosen between (5%*n + 1) and (15%*n + 1) where n is the number jobs
      for (j in 1:sample((ceiling(0.05*n_size+1):floor(0.15*n_size+1)), 1)){
        # Swapping, if possible, two randomly selected jobs from two different
        # blocks. We only allow mutations that guarantee the feasibility of
        # the obtained solutions. Thus, the maximal threshold Δ (delta) must be
        # respected for each block. We repeat until a valid swap is found.
        # Create a batch & job index data.frame and shuffle it
        v_batch_job_df = data.frame(batch=integer(), job=integer())
        for(k in 1:length(p_population[[i]])){
          v_batch_job_df = rbind(v_batch_job_df, data.frame(batch=rep(k, length(p_population[[i]][[k]])),
                                                            job=1:length(p_population[[i]][[k]])))
        }
        v_batch_job_df <- v_batch_job_df[sample(nrow(v_batch_job_df)), ]
        # Loop over random data.frame in case the first result has no valid swaps
        for(k in 1:nrow(v_batch_job_df)){
          v_stop = FALSE
          # Sample batch/job combination
          v_batch_job1 = v_batch_job_df[k,]
          # Filter all other batches
          v_batch2_job_df = v_batch_job_df %>% filter(batch != v_batch_job1$batch)
          # Sample batch/job combination from another batch
          for(l in 1:nrow(v_batch2_job_df)){
            v_batch_job2 = v_batch2_job_df[l,]
            # Try swap
            v_temp_population = p_population[[i]]
            v_job1 = v_temp_population[[v_batch_job1[[1]]]][[v_batch_job1[[2]]]]
            v_job2 = v_temp_population[[v_batch_job2[[1]]]][[v_batch_job2[[2]]]]
            v_temp_population[[v_batch_job1[[1]]]][[v_batch_job1[[2]]]] = v_job2
            v_temp_population[[v_batch_job2[[1]]]][[v_batch_job2[[2]]]] = v_job1
            # Check if swap lead to two feasible batches
            if(degradation_fun(v_temp_population[[v_batch_job1[[1]]]]) < delta &
               degradation_fun(v_temp_population[[v_batch_job2[[1]]]]) < delta){
              p_population[[i]] = v_temp_population
              # We do not need to test other feasible solutions
              v_stop = TRUE
              break
            }
          }
          if(v_stop){break} # Break the outer loop when the flag is fired
          # If all batches have been run through and break has not fired yet,
          # No swap is feasible for p_population[[i]]
          if(k == nrow(v_batch_job_df)){print(paste("No feasible swap found for chromosome", i))} # TODO add generation index to print
        }
      }
    }
  }
  return(p_population)
}  

## Calculate Coefficient of Variance for a population
CoV_fun = function(p_population){
  # Mean cost of the whole population
  v_mean = mean(sapply(p_population, ga_cost_fun))
  # Standard deviation of the whole population
  v_sd = sqrt(mean(sapply(p_population, ga_cost_fun, p_con_offset = v_mean, p_pow_offset = 2)))
  r_cov = v_sd/v_mean
  return(r_cov)
}

## Eliminate worst Rst*PopSize individuals and replace with randoms
exploration_fun = function(p_population){
  # Generate Rst*PopSize individuals random, first-fit
  v_new_rand = generate_pop_fun(p_init=FALSE, p_popsize = Rst*PopSize)
  v_pop_fitness = as.data.frame(sapply(p_population, fitness_fun))
  v_pop_order = order(-v_pop_fitness[,1])
  # Select and remove unfittest Rst*PopSize individuals
  v_fittest_Rst <- p_population[v_pop_order[1:(length(p_population)-round(Rst*PopSize))]]
  # Join fittest and new pop
  r_population = append(v_fittest_Rst, v_new_rand)
}

## Mutate fittest Rst*PopSize individuals and append to p_population
exploitation_fun = function(p_population){
  # get fittest Rst*PopSize individuals
  v_pop_fitness = as.data.frame(sapply(p_population, fitness_fun))
  v_pop_order = order(-v_pop_fitness[,1])
  # Select fittest Rst*PopSize individuals
  v_fittest_Rst <- p_population[v_pop_order[1:round(Rst*PopSize)]]
  r_population = append(p_population, mutation_fun(v_fittest_Rst, 1))
}

## Construct New Population with PopSize from parents and children
replacement_fun = function(p_population){
  v_pop_fitness = as.data.frame(sapply(p_population, fitness_fun))
  v_pop_order = order(-v_pop_fitness[,1])
  # Initialize new population list
  r_new_population <- list()
  v_cutoff = round(beta_ga*length(p_population))
  # Insert worst beta_ga share of individuals
  r_new_population[1:v_cutoff] =
    p_population[v_pop_order[(length(p_population) - v_cutoff+1):(length(p_population))]]
  # Fill up with best individuals
  r_new_population[(v_cutoff+1):PopSize] =
    p_population[v_pop_order[1:(PopSize-v_cutoff)]]
  return(r_new_population)
}



### START OF PROGRAM ###

# Initialize Population
population = generate_pop_fun(p_init=TRUE, p_popsize = PopSize)
fittest_candidate = NA
for (i in 1:MaxGen){
  print(paste("Generation: ", i))
  # Selection 
  parents = tournament_fun(population)
  # Crossover
  children = crossover_fun(parents)
  # Mutation
  mut_children = mutation_fun(children, MutProb)
  total_population = append(population, mut_children)
  if (i %% CycleGen == 0){
    # If coefficient of variation is higher than epsilon_max, delete worst Rst*PopSize
    # individuals and generate random new ones (receptor editing) 
    if (CoV_fun(total_population)<epsilon_min){
      print(paste0("CoV ", CoV_fun(total_population), " is smaller than epsilon_max (", epsilon_min, "). Explore"))
      new_population = exploration_fun(total_population)
      #new_population = replacement_fun(new_population) 
      # If coefficient of variation is higher than epsilon_max, generate Rst*PopSize
      # individuals by mutating best solutions and injecting them into population  
    } else if (CoV_fun(total_population)>epsilon_max){
      print(paste0("CoV ", CoV_fun(total_population), " is higher than epsilon_max (", epsilon_max, "). Exploit."))
      new_population = exploitation_fun(total_population)
      #new_population = replacement_fun(new_population)
    } else {
      new_population = total_population
    # If CoV is moderate, just replace
    # new_population = replacement_fun(total_population) 
    }
  }else{
    new_population = replacement_fun(total_population)
  }
  # Store fittest candidate in variable
  pop_fitness = order(-as.data.frame(sapply(new_population, fitness_fun))[,1])
  if(is.na(fittest_candidate) || fitness_fun(new_population[[pop_fitness[1]]])>fitness_fun(fittest_candidate)){
    fittest_candidate = new_population[[pop_fitness[1]]]
    print(paste0("A fittest candidate was found with a fitness of ", fitness_fun(fittest_candidate)))
  }else{
    print(paste("No fitter candidate was found"))
  }
}

# Optional Open Issue (ToDo): Calibration (5.2)
# If content with performance, do not implement as it is much effort
# Optional Open Issue (ToDo): Comparison with Standard GA (5.3.1) und lower bound (5.3.2)

## Transform candidate solution (list of vectors) to result format (data.frame)
result_ga <- data.frame(i=integer(), j=integer())
for(j in 1:length(fittest_candidate)){
  for(i in 1:length(fittest_candidate[[j]])){
    result_ga = rbind(result_ga, data.frame(i=fittest_candidate[[j]][[i]], j=j))
  }
}