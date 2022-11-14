
library("readxl")
library("tidyverse")
library("ggplot2")

# Read data
data <- read_excel("PxM_PPC_RL/diagnostics/test_results.xlsx")
# Drop index column
data <- data[-1]

# Denote machine runs to-failure with incrementing id
k <- 0
for (i in seq_len(nrow(data))) {
    if (data[i, "t_1"] == 0 & data[i, "v_1"] == 0) {
        k <- k + 1
        }
        data[i, "run"] <- k
}

#### Figure 1: Predicted vs. True Health Figure ####

temp_fig1_data <- filter(data, run == "6")
temp_fig1_data$index <- as.numeric(x = row.names(temp_fig1_data))
fig1_data <- data.frame( Time = rep(temp_fig1_data$index, 2),
                        Var = c(rep("Predicted Health", nrow(temp_fig1_data)),
                                rep("True Health", nrow(temp_fig1_data))),
                        Health = c(temp_fig1_data$Pred, temp_fig1_data$health))

ggplot(fig1_data, aes(x = Time, y = Health, color = Var)) + 
  geom_line() +
  labs(colour = "Legend")

#### Figure 2: Temperature-Voltage-Health-Correlation Figure ####
library(plotly)
temp_fig2_data <- data %>% filter(health > 0)
plot <- plot_ly(x = temp_fig2_data$t,
        y = temp_fig2_data$v,
        z = temp_fig2_data$health,
        type = "scatter3d",
        mode = "markers",
        color = temp_fig2_data$health) %>%
        layout(scene = list(xaxis = list(title = 'Temperature (°C)'),
                            yaxis = list(title = 'Voltage'),
                            zaxis = list(title = 'Health')))
install.packages('reticulate')
reticulate::install_miniconda()
reticulate::conda_install('r-reticulate', 'python-kaleido')
reticulate::conda_install('r-reticulate', 'plotly', channel = 'plotly')
reticulate::use_miniconda('r-reticulate')                          
save_image(plot, file = "PxM_PPC_RL/visuals/scatter3d.svg")

#### Figure 3: Average Reward per Episode Figure ####
fig3_data <- read.csv("PxM_PPC_RL/visuals/PPO_rollout_ep_rew_mean.csv")
ggplot(fig3_data, aes(x = Step, y = Value)) + 
  geom_line()


#### Figure 5 ####
library(tidyverse)
library(Rtsne)
# Load data
fig4_data <- read_excel("PxM_PPC_RL/visuals/PPO_state_action.xlsx")
names(fig4_data)[1] <- "ID"

# Save action store for color
fig4_data_action <- fig4_data %>%
  select(ID, action)
# Let us select relevant columns, standardise the data using scale() function
# before applying Rstne() function to perform tSNE.
set.seed(42)
fig4_tsne_fit <- fig4_data %>%
  select(ID, health, order, inventory, sp_inventory) %>%
  column_to_rownames("ID") %>%
  scale() %>%
  Rtsne()

# The tSNE result object contains two tSNE components that we are interested in.
# We can extract the component and save it in a dataframe. 
fig4_tsne_df <- fig4_tsne_fit$Y %>% 
  as.data.frame() %>%
  rename(tSNE1="V1",
        tSNE2="V2") %>%
        mutate(ID=row_number())

# Using the unique row ID, we can combine the tSNE components with the meta data information.
fig4_tsne_df <- fig4_tsne_df %>%
  inner_join(fig4_data_action, by="ID")

# Let us make a tSNE plot, which is a scatter plot with two tSNE components on x and y-axis.
# Here we have colored the data points by action.
fig4_tsne_df %>%
  ggplot(aes(x = tSNE1, 
             y = tSNE2,
             color = factor(action)))+
  geom_point() +
  scale_color_manual(values = c("0" = "springgreen",
                                "1" = "springgreen1",
                                "2" = "springgreen2",
                                "3" = "springgreen3",
                                "4" = "springgreen4",
                                "5" = "cyan",
                                "6" = "cyan1",
                                "7" = "cyan2",
                                "8" = "cyan3",
                                "9" = "cyan4",
                                "10" = "red"))

# Interactive version
plot4 = plot_ly(x = fig4_tsne_df$tSNE1,
        y = fig4_tsne_df$tSNE2,
        type = "scatter",
        color = factor(fig4_tsne_df$action),
        colors = c("springgreen", "springgreen1", "springgreen2",
                "springgreen3", "springgreen4", "cyan3", "cyan4", "red"))
# Export to html
htmlwidgets::saveWidget(as_widget(plot4), "PxM_PPC_RL/visuals/tsne.html")
# Export to excel
library(writexl)
write_xlsx(fig4_data, "PxM_PPC_RL/visuals/raw_data.xlsx")
write_xlsx(fig4_tsne_df, "PxM_PPC_RL/visuals/ID_lookup_data.xlsx")

#### Figure 5a State-Value####
library(tidyverse)
library(Rtsne)
library(readxl)
# Load data
fig5a_data <- read_excel("C:/Users/w_kevi02/sciebo/PhD/01 Prescriptive Maintenance/30 RQ3 - Digital Twin/PxM & PPC (R)/RL-PxM/PxM_PPC_RL/visuals/state_value.xlsx")
names(fig5a_data)[1] <- "ID"

# Save action store for color
fig5a_data_action <- fig5a_data %>%
  select(ID, value)
# Let us select relevant columns, standardise the data using scale() function
# before applying Rstne() function to perform tSNE.
set.seed(42)
fig5a_tsne_fit <- fig5a_data %>%
  select(ID, health, order, inventory, sp_inventory) %>%
  column_to_rownames("ID") %>%
  scale() %>%
  Rtsne()

# The tSNE result object contains two tSNE components that we are interested in.
# We can extract the component and save it in a dataframe. 
fig5a_tsne_df <- fig5a_tsne_fit$Y %>% 
  as.data.frame() %>%
  rename(tSNE1="V1",
        tSNE2="V2") %>%
        mutate(ID=row_number())

# Using the unique row ID, we can combine the tSNE components with the meta data information.
fig5a_tsne_df <- fig5a_tsne_df %>%
  inner_join(fig5a_data_action, by="ID")

# Let us make a tSNE plot, which is a scatter plot with two tSNE components on x and y-axis.
# Here we have colored the data points by action.
fig5a_tsne_df %>%
  ggplot(aes(x = tSNE1, 
             y = tSNE2,
             color = value))+
  geom_point() + scale_color_continuous(type = "viridis")

# Interactive version
plot5a = plot_ly(x = fig5a_tsne_df$tSNE1,
                y = fig5a_tsne_df$tSNE2,
                type = "scatter",
                color = fig5a_tsne_df$value,
                colors = "viridis")
# Export to html
htmlwidgets::saveWidget(as_widget(plot5a),
                        "C:/Users/w_kevi02/sciebo/PhD/01 Prescriptive Maintenance/30 RQ3 - Digital Twin/PxM & PPC (R)/RL-PxM/PxM_PPC_RL/visuals/tsne_state_value.html")
# Export to excel
library(writexl)
write_xlsx(fig5a_data, "C:/Users/w_kevi02/sciebo/PhD/01 Prescriptive Maintenance/30 RQ3 - Digital Twin/PxM & PPC (R)/RL-PxM/PxM_PPC_RL/visuals/tsne_state_value_raw_data.xlsx")
write_xlsx(fig5a_tsne_df, "C:/Users/w_kevi02/sciebo/PhD/01 Prescriptive Maintenance/30 RQ3 - Digital Twin/PxM & PPC (R)/RL-PxM/PxM_PPC_RL/visuals/tsne_state_value_ID_lookup_data.xlsx")
