### Initialization
library(tidyverse)
library(ggplot2)
library(dplyr)


### Cost Evaluation
# "result" must have following structure:
# i   j
# 2   2
# 16  2   <- Job i=16 is assigned to production batch j=2
# .   .
# .   .
# .   .
# 10  10
# 23  10 
#
# Result MUST be ordered by i > j.
# The jobs in the last batch do not require maintenance.
# They must have the highest j!
cost_fun = function(p_result){
  total_cost = 0
  for (jj in unique(p_result$j)){ 
    # get partial solution for batch jj
    jj_result = p_result %>% 
      dplyr::filter(j == jj)
    deg_jj = theta_deg+sum(job_df$degradation[jj_result$i])
    cost_jj = cost_zero + (cost_f - cost_zero)*(theta_deg+sum(job_df$degradation[jj_result$i]))
    # The last batch does not incur any cost as no maintenance action is placed
    # calculate cost of jj
    if (jj == max(unique(p_result$j))){
      cost_jj = 0
    }
    # print degradation and cost for batch jj
    print(paste0("BATCH ", jj, ": cost = ", cost_jj, ", degradation = ", deg_jj))
    # Calculate total costs
    total_cost = total_cost + cost_jj
  }
  # print total cost
  print(total_cost)
}

### Plotting
# p_max_jobs = maximum number where job index is displayed in plot
plot_fun = function(p_result, p_max_jobs=20){

  ## Prepare plot dataframe
  df <- data.frame(job=NA, proc=0, deg=0, text=0) # initialize empty dataframe
  for(jj in unique(p_result$j)){
    plot_result = p_result %>% 
      dplyr::filter(j == jj)
    df = rbind(df, data.frame(job=c(NA, plot_result$i),
                              proc=c(df$proc[length(df[,1])], df$proc[length(df[,1])] + cumsum(job_df$p_time[plot_result$i])),
                              deg=c(0, cumsum(job_df$degradation[plot_result$i])), 
                              text=c(df$text[length(df[,1])], df$proc[length(df[,1])] + cumsum(job_df$p_time[plot_result$i]))-c(0, job_df$p_time[plot_result$i]/2)))
  }
  ## Plot results
  ggpl = ggplot() +
    geom_segment(aes(x = df$proc, y = 0, xend = df$proc, yend = 1), linetype="dotted", size=1)+
    geom_line(data=df, aes(x=proc, y=0, color = factor(job), group=1), size=2) +
    geom_line(data=df, aes(x=proc, y=deg, group=1)) +
    geom_point(data=df, aes(x=proc, y=deg, group=1)) +
    scale_x_continuous(name ='Time') +
    scale_y_continuous(name ='Degradation')+ 
    theme(legend.position="none")
  # Limit number of displayed job indices to max_jobs
  if(length(unique(df$job)[!is.na(df$job)])<=p_max_jobs){
    ggpl = ggpl + geom_text(data=df[-1,], aes(x=text, y=0, label = ifelse(is.na(job), "", sprintf("%d",job)), vjust=1.5))
  }
  ggpl
}

  
### Evaluate cost and plot
cost_fun(result_mip)
cost_fun(result_ga)
plot_fun(result_mip, 20)
plot_fun(result_ga, 20)