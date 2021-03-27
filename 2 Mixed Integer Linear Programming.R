library(extraDistr)
library(dplyr)
library(ROI)
library(ROI.plugin.glpk)
library(ompr)
library(ompr.roi)

### SOF MILP
## Construct and solve model
model <- MILPModel() %>% 
  add_variable(x[i, j], type = "binary", i = 1:n_size, j = 1:n_size) %>% 
  add_variable(y[j], type = "binary", j = 1:n_size) %>% 
  set_objective(sum_expr(delta * y[j], j = 1:n_size) - sum_expr(sum_expr(colwise(job_df$degradation[i]) * x[i, j], j = 1:n_size), i = 1:n_size), sense = "min") %>% 
  add_constraint(sum_expr(x[i,j], j = 1:n_size) == 1, i = 1:n_size) %>% 
  add_constraint(sum_expr(colwise(job_df$degradation[i]) * x[i, j], i = 1:n_size) <= (delta - theta_deg)*y[j], j = 1:n_size) %>% 
  solve_model(with_ROI("glpk", verbose = TRUE))

## Calculate cost for each production batch j
result_mip = get_solution(model, x[i, j]) # Get solution for whole problem
# Cut down result to only show assigned jobs and batches
result_mip = result_mip %>% 
  dplyr::filter(value == 1)
result_mip = subset(result_mip, select = -c(variable, value))



