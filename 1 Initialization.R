### Initialization
n_size <- 200 # Problem size
theta_deg <- 0 # Initial degradation
delta <- 1 # maximal degradation
cost_f = 100 # maximum predictive maintenance cost
cost_zero = 1000 # minimum predictive maintenance cost spent when the machine reaches a full degradation ??
p_vec <- runif(n_size, 1, 50) # Processing time
rul_vec <- runif(n_size, 100, 150) # Degradation time
delta_vec <- p_vec / rul_vec
job_df = data.frame(job=1:n_size, p_time=p_vec, degradation = delta_vec)
