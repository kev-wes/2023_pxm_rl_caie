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
#       Sonst: +1
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
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
