
library(readxl)
library(tidyverse)
library(ggplot2)
library(plotly)
library(cowplot)

# Set directory
path = 'C:/Users/w_kevi02/sciebo/PhD/01 Prescriptive Maintenance/30 RQ3 - Digital Twin/PxM & PPC (R)/RL-PxM/PxM_PPC_RL/visuals/IFAC/'

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



#### Figure 4: Health-Reward-Action Figure ####

## Add column "round" and timestep (from 0)
data <- read_excel(paste(path, "episode1.xlsx", sep =''))


## revise column name
colnames(data)[1]="t"
data$Q = data$action*100
colnames(data)[3]="h"
data$I = data$inventory*100
data$d = data$next_order*100
data = data %>% mutate(Q = if_else(Q == "500",0,Q))


## plot upper plot (plot4a)
ggplot(data) +
  theme_bw()+
  geom_bar(aes(x=t, y=Q),stat="identity", fill="lightgrey") +
  geom_line( aes(x=t, y=I), color="#2166AC", stat="identity", size = 0.5) + 
  geom_point( aes(x=t, y=I), color="#2166AC", stat="identity", size = 1) + 
  geom_line( aes(x=t, y=d), color="#b2182b", stat="identity", size = 0.5) + 
  geom_point( aes(x=t, y=d), color="#b2182b", stat="identity", size = 1) +
  geom_line( aes(x=t, y=h*400), color="#359054", stat="identity", size = 0.5) +
  geom_point( aes(x=t, y=h*400), color="#359054", stat="identity", size = 1)+
  xlab("Time t") +
  scale_y_continuous(name="Production Q, Inventory I, Demand d", sec.axis = sec_axis( ~./400, name="Prognosed Health h'"))
ggsave(file=paste(path, 'fig.svg', sep=''), width = 9, height = 3.2, units = "in") 

