library(readxl)
library(tidyverse)
library(ggplot2)
library(plotly)
library(cowplot)

# Set directory
#path = 'C:/Users/w_kevi02/sciebo/PhD/01 Prescriptive Maintenance/30 RQ3 - Digital Twin/PxM & PPC (R)/RL-PxM/PxM_PPC_RL/visuals/Final Analysis 1/'
path = './'

# set font
#windowsFonts(A=windowsFont("Times New Roman"))
windowsFonts(A=windowsFont("Arial"))

# set colors
fun_color_range <- colorRampPalette(c("#D9F5B1", "#359054"))
my_colors1 <- fun_color_range(5)
fun_color_range <- colorRampPalette(c("#d1e5f0", "#2166ac"))
my_colors2 <- fun_color_range(5)
customized_clr <- c(my_colors1,my_colors2,"#b2182b")

# Helper function
round_any = function(x, accuracy, f=round){f(x/ accuracy) * accuracy}

# Read data
data <- read_excel(paste(path, 'eps.xlsx', sep=''))
# Drop index column
data <- data[-1]

# Denote machine runs to-failure with incrementing id
timestep =0
k=0
for (i in 1:nrow(data)) {
  if (data[i, "breakdown"] == "TRUE" | data[i, "action"] == 10) {
        k=k+1
        timestep = 0}
  else{timestep = timestep +1}
       data[i, "round"]=k
       data[i, "timestep"] = timestep}
t_step = 1
for (i in 1:nrow(data)) {
  data[i, "t_step"] = t_step
  t_step = t_step + 1
  }
#### Figure 1: Predicted vs. True Health Figure ####
r = 5
# data %>% 
#   #filter(round == r) %>% 
#   ggplot() +
#   geom_line(aes(x = t_step, y = true_health,color="True Health"),size =.2)+
#   geom_line(aes(x = t_step, y = health_rul,color="Predicted Health"),size =.2)+
#   scale_color_manual(name= NULL,values = c("True Health" = "#67a9cf", "Predicted Health" = "#000000"))+
#   theme_bw()+
#   theme(text = element_text(size=10,family="A"),
#         panel.border = element_blank(),
#         legend.text = element_text(size=10),
#         axis.text = element_text(size=10), 
#         panel.grid.minor = element_blank(),
#         axis.ticks = element_blank(),
#         legend.justification = "top",
#         legend.position = c(0.82, 1.08),
#         legend.background = element_blank(),
#         legend.box.background = element_blank())+
#   labs(x = "Time", y = "Health")+
#   #scale_x_continuous(breaks=seq(0, 30, 5))+
#   scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(0, 1.2))

data %>% 
  filter(round == r) %>% 
  ggplot() +
  geom_line(aes(x = timestep, y = health_rul,color="True Health"),size =.5)+
  geom_line(aes(x = timestep, y = true_health,color="Predicted Health"),size =.5)+
  scale_color_manual(name= NULL,values = c("True Health" = "#67a9cf", "Predicted Health" = "#000000"))+
  theme_bw()+
  theme(text = element_text(size=10,family="A"),
        panel.border = element_blank(),
        legend.text = element_text(size=10),
        axis.text = element_text(size=10), 
        panel.grid.minor = element_blank(),
        axis.ticks = element_blank(),
        legend.justification = "top",
        legend.position = c(0.8, 1.05))+
  labs(x = "Time", y = "Health")+
  scale_x_continuous(breaks=seq(0, 30, 5))+
  scale_y_continuous(breaks=seq(0, 1, 0.2), limits = c(0, 1))

  ggsave(file=paste(path, 'fig.eps', sep=''), width = 4, height = 3, units = "in")
#### Figure 2: Temperature-Voltage-Health-Correlation Figure ####
#### OUTDATED: For R1, I based my analysis here on one eps_df instead 
#### of a dataframe generated during diagnostics
data %>% filter(health > 0) %>% 
  ggplot(aes(x=t, y=v))+
  geom_point(aes(color = health),size=2)+
  scale_color_gradient(low="#80B1D3", high= "#FDB462")+
  theme_bw()+
  theme(text = element_text(size=10),
        panel.border = element_blank(),
        legend.text = element_text(size=10),
        axis.text = element_text(size=10), 
        axis.ticks = element_blank(),
        legend.justification = "top")+
  labs(x = "Temperature (°C) ", y = "Voltage",colour="Health")
ggsave(file=paste(path, 'fig.eps', sep=''), width = 6, height = 3, units = "in")

#### Figure 3: Average Reward per Episode Figure ####
fig3_data <- read.csv(paste(path, "10x Retraining/", "reward1.csv", sep=""))
fig3_data$Run = as.character(1)
for (i in 2:10){
  temp_fig3_data <- read.csv(paste(path, "10x Retraining/", "reward", i, ".csv", sep=""))
  temp_fig3_data$Run = as.character(i)
  fig3_data = rbind(fig3_data, temp_fig3_data)
}

fig3_data %>% ggplot(aes(x = Step, y = Value, colour=reorder(Run, sort(as.numeric(Run))))) + 
  geom_line(size=.5)+
  theme_bw()+
  theme(text = element_text(size=10,family="A"),
        panel.border = element_blank(),
        legend.text = element_text(size=10),
        axis.text = element_text(size=10),
        axis.title.y = element_text(margin = margin(r=10)))+
  labs(y = "Mean Reward per Episode", colour="Run")+
  scale_y_continuous(limits = c(-5000, 2000))+
  scale_x_continuous(labels = function(x) format(x, digits=2, scientific = TRUE))
ggsave(file=paste(path, 'fig.eps', sep=''), width = 6, height = 3.50, units = "in")  

#### Figure 4: Health-Reward-Action Figure ####

## Add column "round" and timestep (from 0)
data <- read_excel(paste(path, "eps.xlsx", sep =''))
k = 0 
timestep = 1
for (i in 1:nrow(data)) {
  data[i, "round"]= k
  data[i, "timestep"] = timestep
  timestep = timestep +1
  if (data[i, "true_health"] == 1) {
     k=k+1
     timestep = 1}}

## revise column name
colnames(data)[1]="time"

roundn = 6
## set colors
data1 = data %>% filter(round == roundn)
figureClr = customized_clr[sort(as.numeric(unique(data1$action)+1))]
order_color_range_fun <- colorRampPalette(c("#fee090", "#fc8d59"))
order_color_range <- order_color_range_fun(8)
inv_color_range_fun <- colorRampPalette(c("#CCFFFF", "#008080"))
inv_color_range <- inv_color_range_fun(10)

data$action<-factor(data$action,
                    levels = 0:10,
                    labels = c("0 - Produce 0", "1 - Produce 1", "2 - Produce 2","3 - Produce 3","4 - Produce 4","5 - Produce 0 +  Spare Part","6 - Produce 1 + Spare Part", "7 - Produce 2 + Spare Part", "8 - Produce 3 + Spare Part", "9 - Produce 4 + Spare Part", "10 - Maintain"))


## plot4b (bottom plot)
data$dummy = 1
data_b = data %>% filter(round == roundn)
data_b = data_b[-1,]


plot4b = data_b %>% 
  ggplot(aes(fill= factor(old_order), y=dummy, x=timestep, label = factor(substr(old_order, 1, 2)))) + 
  geom_bar(position="stack", stat="identity",width=1) +
  scale_fill_manual(values=order_color_range,name="Last Order") +
  theme_void() +
  theme(legend.margin = margin(r=1,t=-90), legend.position = c(0.2,14.5), plot.margin = margin(r=18, l=10)) +
  geom_text(size = 3, position = position_stack(vjust = 0.5))

plot4c = data_b %>% 
  ggplot(aes(fill= factor(action), y=dummy, x=timestep, label = factor(substr(action, 1, 2)))) + 
  geom_bar(position="stack", stat="identity",width=1) +
  scale_fill_manual(values=figureClr,name="Action") +
  theme_void() +
  theme(legend.margin = margin(r=1,t=-90), legend.position = c(0.55,15.53), plot.margin = margin(r=18, l=10)) +
  geom_text(size = 3, position = position_stack(vjust = 0.5))


plot4d = data_b %>% 
  ggplot(aes(fill= factor(inventory), y=dummy, x=timestep, label = factor(substr(inventory, 1, 2)))) + 
  geom_bar(position="stack", stat="identity",width=1) +
  scale_fill_manual(values=inv_color_range,name="Inventory") +
  theme_void() +
  theme(legend.margin = margin(r=1,t=-90), legend.position = c(0.85,16.55), plot.margin = margin(r=18, l=10)) +
  geom_text(size = 3, position = position_stack(vjust = 0.5)) 

## plot upper plot (plot4a)
data1 = data %>% filter(round == roundn) 
parameter =  max(data1$reward) / max(data1$health_rul)
plot4a = data1 %>% 
  ggplot(mapping=aes(x = timestep, y = health_rul))+
  theme_bw()+
  theme(panel.grid=element_blank())+
  geom_line(color = "#000000",linetype="solid", size= 0.5)+
  labs(x = "Time", y = "Prognosed Health")+
  scale_x_continuous(expand = c(0,0))+
  theme(text = element_text(size=10,family="A")) +
  theme(axis.title.x = element_text(hjust = 1),
        axis.title.y = element_text(color= "#000000")) +
  theme(axis.text = element_text(size=10)) +
  theme(plot.margin = margin(t=150, r=5, b=5, l=5)) +
  theme(axis.title.y.left = element_text(margin = margin(l=0)))+
  geom_line(mapping = aes(y = reward/parameter, x = timestep) , color = "#af8dc3", linetype="solid", size= .5)+
  theme(axis.title.y.right = element_text(angle = 90, margin = margin(l=5),color = "#af8dc3"))+
  theme(axis.title.y.left = element_text(margin = margin(r=5)))+
  scale_y_continuous(sec.axis = sec_axis( ~.*parameter, name = "Reward",
                                          breaks = seq(round_any(min(data1$reward), 10, f = floor),
                                                       round_any(max(data1$reward), 10, f = ceiling),
                                                       by = 10)),
                     breaks=c(0, 1) )+
  geom_hline(yintercept=0, linetype='dashed')
 
## Combine plot4a and plot4b
plot_grid(plot4a, plot4b, plot4c, plot4d, ncol = 1, rel_heights = c(6, .3, .3, .3))
ggsave(file=paste(path, 'fig.eps', sep=''), width = 6, height = 6, units = "in") 

#### Figure 5a State-Value####
library(tidyverse)
library(Rtsne)
library(readxl)
# Load data
fig5a_data <- read_excel("./state_value.xlsx")
names(fig5a_data)[1] <- "ID"

# Save action store for color
fig5a_data_action <- fig5a_data %>%
  select(ID, value)
# Let us select relevant columns, standardise the data using scale() function
# before applying Rstne() function to perform tSNE.
set.seed(42)
fig5a_tsne_fit <- fig5a_data %>%
  select(ID, health, inventory, sp_inventory) %>%
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
htmlwidgets::saveWidget(as_widget(plot5a), "./tsne_state_value.html")
# Export to excel
library(writexl)
write_xlsx(fig5a_data, "./tsne_state_value_raw_data.xlsx")
write_xlsx(fig5a_tsne_df, "./tsne_state_value_ID_lookup_data.xlsx")

#### Figure 5: tSNE Figure State-Action ####

fig5_tsne_df <- read_excel(paste(path, "tsne_state_action_raw_data.xlsx", sep=""))
figureClr = customized_clr[sort(as.numeric(unique(fig5_tsne_df$action)+1))]
fig5_tsne_df$action<-factor(fig5_tsne_df$action,
                       levels = 0:10,
                       labels = c("0 - Produce 0", "1 - Produce 1", "2 - Produce 2","3 - Produce 3","4 - Produce 4","5 - Produce 0 +  Spare Part","6 - Produce 1 + Spare Part", "7 - Produce 2 + Spare Part", "8 - Produce 3 + Spare Part", "9 - Produce 4 + Spare Part", "10 - Maintain"))

ggplot(fig5_tsne_df, aes(x = tSNE1, y = tSNE2,color = action))+
  geom_hline(yintercept= 0)+
  geom_vline(xintercept = 0)+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = figureClr)+
  theme(text = element_text(size=10,family="A"),panel.border = element_blank(),
        legend.title=element_blank(),legend.text = element_text(size=10),
        axis.text = element_text(size=10),axis.ticks = element_blank(),legend.justification = "top", legend.position = "bottom")
ggsave(file=paste(path, 'fig.eps', sep=''), width = 7, height = 6, units = "in") 

#### Figure 5 Alternative: tSNE Figure State-Value ####

library(grid)

fig5a_tsne_df <- read_excel(paste(path, "tsne_state_value_ID_lookup_data.xlsx", sep=""))
figureClr = customized_clr[sort(as.numeric(unique(fig5a_tsne_df$value)+1))]

ggplot(fig5a_tsne_df, aes(x = tSNE1, y = tSNE2,color = value))+
  geom_point()+
  theme_bw()+
  scale_color_continuous(name="v",
                         type = "viridis")+
  theme(text = element_text(size=10,family="A"),
        panel.border = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        panel.background = element_blank(),
        legend.text = element_blank(),
        legend.position = c(0.9,0.32),
        legend.key.width = unit(0.3, "cm"),
        legend.background = element_blank(),
        )+
  annotation_custom(
    grob = linesGrob(arrow=arrow(type="open", ends="last", length=unit(2,"mm")), gp=gpar(col="black", lwd=1)), 
    xmin = 65, xmax = 65, ymin = -26, ymax = -2
  )+
  # Top left
  annotate("segment", x = -30, xend = 1.71357948541521, y = 45, yend = -5.8075205060007, colour = "black", linetype=2)+
  annotate("label", x = -30, y = 45, label = expression("h' = 1, I = 0, " ~ Theta  ~ "= 0"))+
  annotate("segment", x = -48, xend = -20.8302633376322, y = 32.5, yend = -5.056067492, colour = "black", linetype=2)+
  annotate("label", x = -48, y = 32.5, label = expression("h' = 1, I = 5, " ~ Theta  ~ "= 0"))+
  annotate("segment", x = -65, xend = -45.0920951236658, y = 20, yend = -8.4838682289214, colour = "black", linetype=2)+
  annotate("label", x = -65, y = 20, label = expression("h' = 1, I = 1, " ~ Theta  ~ "= 0"))+
  # Bottom
  annotate("segment", x = 30, xend = 16.2839738067095, y = -40, yend = -25.0835349138347, colour = "black", linetype=2)+
  annotate("label", x = 30, y = -40, label = expression("h' = 0.01, I = 0, " ~ Theta  ~ "= 0"))+
  annotate("segment", x = -10, xend = -16.8628323726061, y = -50, yend = -30.3147461331003, colour = "black", linetype=2)+
  annotate("label", x = -10, y = -50, label = expression("h' = 0, I = 5, " ~ Theta  ~ "= 0"))+
  annotate("segment", x = -55, xend = -43.1888606311029, y = -40, yend = -21.2210909715793, colour = "black", linetype=2)+
  annotate("label", x = -55, y = -40, label = expression("h' = 0.5, I = 1, " ~ Theta  ~ "= 0"))+
  # Top right
  annotate("segment", x = 0, xend = 22.3005381391557, y = 65, yend = 41.3345026641135, colour = "black", linetype=2)+
  annotate("label", x = 0, y = 65, label = expression("h' = 0, I = 9, " ~ Theta  ~ "= 1"))+
  annotate("segment", x = 35, xend = 33.4032569274724, y = 55, yend = 33.4275579223582, colour = "black", linetype=2)+
  annotate("label", x = 35, y = 55, label = expression("h' = 0.5, I = 9, " ~ Theta  ~ "= 1"))+
  annotate("segment", x = 57, xend = 40.9415366670871, y = 35, yend = 23.1545446441797, colour = "black", linetype=2)+
  annotate("label", x = 57, y = 35, label = expression("h' = 1, I = 9, " ~ Theta  ~ "= 1"))+
  coord_cartesian(xlim = c(-80, 70), ylim = c(-45, 65), clip="off")

ggsave(file=paste(path, 'fig.eps', sep=''), width = 7, height = 6, units = "in") 

#### Figure 2b-1 Figure ####
data <- data.frame(read_excel(paste(path, "test_results_noise/test_results_pn1_mn0.xlsx", sep="")))
data = data[c("health", "Pred")]
data = aggregate(data$Pred, by=list(data$health), data=data, FUN= mean)
data = data %>% mutate(th = as.numeric(Group.1),ph = as.numeric(x))

data %>% 
  ggplot(mapping= aes(x = th, y = ph))+
  geom_abline(slope=1,intercept = 0,color="#67a9cf")+
  geom_line(size=.5)+
  theme_bw()+
  theme(text = element_text(size=10,family="A"),
        panel.border = element_blank(),
        legend.text = element_text(size=10),
        axis.text = element_text(size=10), 
        panel.grid.minor = element_blank(),
        axis.ticks = element_blank(),
        plot.margin=unit(c(1,1,1,1),'lines'))+
  scale_x_continuous(expand = c(0,0))+
  labs(x = "True Health", y = "Predicted Health")
ggsave(file=paste(path, 'fig.eps', sep=''), width = 6, height = 4, units = "in")
#### Figure 2b-2 Figure ####
##REMEMBER TO DELETE ALL THE PIVOT TABLES IN THE EXCEL FILES IN THE FOLDER "TASK2B"

files = list.files(paste(path, "test_results_noise", sep=""), pattern = ".xlsx", full.names = TRUE)
files = files[-1]
X = lapply(files, function(fn) {
  X = read_excel(fn)
  X$noise = gsub("test_results_pn", "",gsub("_mn0.xlsx", "", basename(fn)))
  return(X)
})

X = do.call(rbind, X)    
X=X[,c("health","Pred","noise")]
X$noise = as.numeric(X$noise)

cols <- c("health","noise")
T= X %>% group_by(across(all_of(cols))) %>% summarize_at("Pred",list(meanPred=mean)) %>% arrange(noise)

ggplot(data = T, mapping = aes(x = health, y = meanPred)) + 
  geom_line()+ 
  facet_wrap(~noise, ncol = 3,labeller = labeller(noise = c(
    "0" = "Noise 0",
    "1" = "Noise 0.1",
    "2" = "Noise 0.2",
    "3" = "Noise 0.3",
    "4" = "Noise 0.4",
    "5" = "Noise 0.5",
    "6" = "Noise 0.6",
    "7" = "Noise 0.7",
    "8" = "Noise 0.8",
    "9" = "Noise 0.9",
    "10" = "Noise 1")))+
  geom_abline(slope=1,intercept = 0,color="#67a9cf")+
  theme_bw()+
  theme(text = element_text(family="A"),
        #panel.border = element_blank(),
        panel.grid = element_blank(),
        axis.ticks = element_blank(),
        #axis.text = element_blank(),
        strip.background = element_rect(colour="white", fill="white"))+
  labs(x = "True Health", y = "Predicted Health")
ggsave(file=paste(path, 'fig.eps', sep=''), width = 7, height = 10.5, units = "in")

#### Figure 2C Figure ####

files = list.files(paste(path, "reward_noise", sep=""), pattern = ".csv", full.names = TRUE)
X = lapply(files, function(fn) {
  X = readr::read_delim(fn)
  X$noise = gsub("run-PPO_pn", "",gsub("_mn0_1-tag-rollout_ep_rew_mean.csv", "", basename(fn)))
  return(X)
})
X = do.call(rbind, X)    
X=X[,c("Step","Value","noise")]
X$noise = as.numeric(X$noise)/10

true_csv = read.csv(file = paste(path, "run-PPO_true-tag-rollout_ep_rew_mean.csv", sep=""), sep =";")
true_X = cbind(true_csv[,c("Step","Value")], noise="True")
X = rbind(true_X, X)
X$noise = factor(X$noise, levels = c("True", seq(0, 1, 0.1)))
#levels(X$noise) = c(seq(0, 1, 0.1), "True")


X %>%
  ggplot(aes(x = Step, y = Value,colour= noise)) + 
  geom_line(size=0.6)+
  theme_bw()+
  theme(text = element_text(size=10,family="A"),
        panel.border = element_blank(),
        legend.text = element_text(size=10),
        axis.text = element_text(size=10), 
        panel.grid.minor = element_blank(),
        axis.ticks = element_blank(),
        legend.justification = "top",
        legend.position = "bottom")+
        scale_y_continuous(breaks=seq(-2400, 400, 400))+
        scale_x_continuous(labels = function(x) format(x, digits=2, scientific = TRUE),
                     breaks=seq(0, 2000000, 400000))+
        labs(colour = "Noise")
ggsave(file=paste(path, 'fig.eps', sep=''), width = 6, height = 6, units = "in")

