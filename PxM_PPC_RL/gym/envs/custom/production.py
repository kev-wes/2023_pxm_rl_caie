from pickle import FALSE, TRUE
import gym
import numpy as np
from math import inf 
from prog_models.models import BatteryCircuit
from gym import spaces
from gym.utils import seeding
import warnings

def reset_states(machine):
    # Returns initial states of machine, e.g., {'tb': 18.95, 'qb': 7856.3254, 'qcp': 0, 'qcs': 0} for Battery
    return(machine.default_parameters['x0'])

# Use prog_models package for simulation of health
def produce_model(machine, states, action): # TODO: Allow seed
        
        # Define load of battery
        def future_loading(t, x=None):
            return {'i': action}

        # Set current state of machine
        machine.parameters['x0'] = states

        # Simulate 100 steps
        options = {
            'save_freq': 100,  # Frequency at which results are saved
            'dt': 2  # Timestep
        }
        (_, _, states, outputs, event_states) = machine.simulate_to(100, future_loading, **options)
        health = event_states[-1]['EOD']
        return(round(health, 2), states[-1], outputs[-1]['t'], outputs[-1]['v'])
def get_order_list(mean, sd, length):
    # Define demand dependent on prod_levels
    order_list = np.random.normal(mean, sd, length)
    # Null negative demands
    order_list[order_list < 0] = 0
    # Round demands
    order_list = np.around(order_list , decimals=2)
    return(order_list)

class ProductionEnv(gym.Env):
    """ Simple Production Process
    Observation:
            Type: Box(4) or Box(3) if reactive_mode or scheduled_time
            Num     Observation                     Min                     Max
            (0      Health                          0                       1) only if !reactive_mode and !scheduled_time
            1       Order Quantity t+1              self.min_order = 0      inf
            2       Finished Inventory              0                       self.max_inventory = 9
            3       Spare Parts Inventory           0                       self.spare_parts_max_inventory

    Actions:
        

    Reward:
        Lorem

    Starting State:
        Lorem

    Episode Termination:
        The episode terminates after 100 time steps

    Parameters:
        diag_model: sklearn regression model that is used to derive the health of the machine. If None, true health values or prog_model are used.
        prog_model: sklearn regression model that is used to derive the RUL of the machine. If None, true health values or diag_model are used.
        reactive_model: If True, action 10 (maintenance) is forbidden and the machine always runs to failure. Observations also do not contain condition or
            health values.
        scheduled_time: Time after which system is maintenance preventively. Resets after each breakdown or maintenance. If 0, scheduled mode is deactivated.
        spare_part: Boolean value signifying whether spare parts and cost should be regarded
        process_noise: Noise representing uncertainty in the model transition. Applied during state transition. Standard deviation.
        measurement_noise: Noise representing uncertainty in the measurement process; e.g., sensor sensitivity, sensor misalignments, environmental effects.
            Applied during estimation of outputs from states.
        forecast: number of next demand observations that are returned to the agent (default = 1 = only next order)
        prod_level: Number of production levels/quantities available
    """

    def __init__(self, natural=False, diag_model = None, prog_model = None, reactive_mode = False, scheduled_time = 0,
                 spare_part = True, process_noise = False, measurement_noise = False, forecast = 1, prod_levels = 5):
        # filter prog_models warnings
        warnings.filterwarnings("ignore")
        # If reactive and scheduled maintenance parameters are filled, raise error
        if reactive_mode and scheduled_time:
            raise ValueError('You cannot set maintenance strategy to reactive and scheduled.')
        # If reactive and scheduled maintenance parameters are filled, raise error
        if (diag_model or prog_model) and (reactive_mode or scheduled_time):
            raise ValueError('You cannot set maintenance strategy to reactive or scheduled while diag_model is used.')
        # If reactive and scheduled maintenance parameters are filled, raise error
        if (diag_model and prog_model):
            raise ValueError('You must only use either a diag_model or prog_model.')
        # Set diagnostics model
        self.diag_model = diag_model
        # Set prognostics model
        self.prog_model = prog_model
        # Set spare_part Boolean
        self.spare_part = spare_part
        # Set forecast horizon
        self.forecast = forecast
        

        # Set parameter
        self.max_ep_time = 100 # Time at which episode terminates
        # Define action space
        self.prod_levels = prod_levels
        
        # Define demand dependent on prod_levels
        self.mean_order = (self.prod_levels-1)*0.5 #0.5
        self.sd_order = (self.prod_levels-1)*0.5 # 0.5
        self.order_list = get_order_list(self.mean_order, self.sd_order, self.max_ep_time+self.forecast)

        self.maintenance_c = 100 # Predictive Maintenance cost at health = 1
        self.repair_c = 500 # Reactive Repair cost
        self.backorder_c = 10 # Backorder cost
        self.order_r = 10 # Order fulfillment reward
        if spare_part:
            self.spare_part_order_c = 10 # Spare part order cost
            self.spare_part_holding_c = 10 # Spare part holding cost
            self.sp_emergency_order_c = 300 # Spare parts emergency order costs if inventory is zero
        else:
            self.spare_part_order_c = 0 # Spare part order cost
            self.spare_part_holding_c = 0 # Spare part holding cost
            self.sp_emergency_order_c = 0 # Spare parts emergency order costs if inventory is zero 
        self.spare_parts_max_inventory = 1 # Maximal spare parts inventory
        self.holding_c = 5 # Holding cost
        self.max_inventory = float('inf') # Maximal inventory
        self.min_order = 0 # Minimal order quantity
        self.init_t = 19.410832360806385 # Initial temp value of battery
        self.init_v = 3.8838358089883327 # Initial volt value of battery
        self.breakdown = False # Reset breakdown indicator
        self.scheduled_time = scheduled_time # Maintenance interval
        self.scheduled_maintenance_counter = 0 # Scheduled maintenance time counter
        self.reactive_mode = reactive_mode
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.actions = self.prod_levels
        low = np.array([], dtype=np.float32,)
        high = np.array([], dtype=np.float32,)
        #Allow maintenance and return health/condition
        if not (self.reactive_mode or self.scheduled_time):
            self.actions = self.actions + 1
            low = np.append(low, [0])
            high = np.append(high, [1])
        # Append state array of self.forecast length and values self.min_order as lower bound
        low = np.append(low, np.full(self.forecast, self.min_order))   
        # Append state array of self.forecast length and inf as upper bound
        high = np.append(high, np.full(self.forecast, float('inf')))   
        # Append lower and upper bounds for inventory
        low = np.append(low, [0])
        high = np.append(high, [self.max_inventory])
        if self.spare_part:  
            self.actions = self.actions + self.prod_levels
            low = np.append(low, [0,])
            high = np.append(high, [self.spare_parts_max_inventory,])


        self.action_space = spaces.Discrete(self.actions)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        reward = 0
        self.breakdown = False
        # Increase scheduled maintenance counter by one time step
        self.scheduled_maintenance_counter = self.scheduled_maintenance_counter + 1
        if self.scheduled_time:
            # If counter reached the scheduled maintenance interval -> maintain 
            if self.scheduled_maintenance_counter > self.scheduled_time:
                action = self.actions-1

        # Spare parts inventory holding costs
        reward = reward - self.sp_inventory * self.spare_part_holding_c
        # Calculate health
        if action == self.actions-1: # no production & maintain
            intensity = 0
            # Reset states of battery
            self.states = reset_states(self.battery)
            self.t = self.init_t
            self.v = self.init_v
            # Maintenance is less punished the closer to failure it is performed
            if self.sp_inventory != 0: reward = reward - self.maintenance_c * self.true_health 
            # Spare part not available, so it must be ordered
            else: reward = reward - self.maintenance_c * self.true_health - self.sp_emergency_order_c 
            self.sp_inventory = 0
            self.health = 1
            self.true_health = 1
            # Reset scheduled maintenance counter
            self.scheduled_maintenance_counter = 0
        else:
            # If action > prod_levels-1 -> Spare parts order
            if action > (self.prod_levels-1):
                intensity = action - self.prod_levels
                # Calculate new spare parts inventory and order cost
                self.sp_inventory = 1
                reward = reward - self.spare_part_order_c
            else: intensity = action
         
            # Shift history by one time period
            self.v_3 = self.v_2
            self.t_3 = self.t_2
            self.v_2 = self.v_1
            self.t_2 = self.t_1
            self.v_1 = self.v
            self.t_1 = self.t
            # More complex, black box degradation
            self.true_health, self.states, self.t, self.v = produce_model(self.battery, self.states, intensity)
            if self.true_health < 0: # Breakdown & Repair 
                self.breakdown = True
                intensity = 0 # Assume failure led to zero production (e.g., scrap)
                # Reset states of battery
                self.states = reset_states(self.battery)
                # Reset condition values
                self.t = self.init_t
                self.v = self.init_v
                self.t_1 = self.t_2 = self.t_3 = self.v_1 = self.v_2 = self.v_3 = 0
                self.health = 1
                self.true_health = 1
                if self.sp_inventory > 0: reward = reward - self.repair_c 
                # Spare part not available, so it must be ordered 
                else: reward = reward - self.repair_c - self.sp_emergency_order_c 
                self.sp_inventory = 0
                # Reset scheduled maintenance counter
                self.scheduled_maintenance_counter = 0
        # Calculate inventory cost
        reward = reward - (self.inventory * self.holding_c)

        # Calculate new Inventory and (back)order rewards
        # Set new inventory to be old inventory
        # Cannot exceed maximum inventory of self.max_inventory
        inventory = min(self.inventory + intensity - self.order, self.max_inventory)
        if inventory < 0:
            reward = reward + ((self.inventory + intensity) * self.order_r) + (inventory * self.backorder_c)
            self.inventory = 0
        else:
            reward = reward + self.order * self.order_r
            self.inventory = inventory
        
        
        
        
        # If a diagnostics model is supplied, replace true health with approximated health
        if self.diag_model is not None:
            self.health = max(self.diag_model.predict([[self.t, self.v, self.t_1, self.v_1, self.t_2, self.v_2, self.t_3, self.v_3]])[0], 0)
        elif self.prog_model is not None:
            self.rul = max(self.prog_model.predict([[self.t, self.v, self.t_1, self.v_1, self.t_2, self.v_2, self.t_3, self.v_3]])[0], 0)
        else:
            self.health = self.true_health
        
        # Delete old order
        self.order_list = np.delete(self.order_list, 0)
        # Save old order
        self.old_order = self.order
        
        # Stop when time self.max_ep_time has been reached
        self.time = self.time + 1
        if self.time < self.max_ep_time:
            # New top of array = new order (only until last episode, as stack might be empty)
            self.order = self.order_list[0]
        else:
            done = True
        
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # In case of reactive or scheduled maintenance, knowledge about condition is not available
        ret = []

        if self.prog_model:
            ret.append(self.rul)
        elif self.reactive_mode or self.scheduled_time:
            ret = ret
        else:
            ret.append(self.health)
        ret.extend(self.order_list[0:self.forecast].tolist())
        ret.append(self.inventory)
        if self.spare_part:
            ret.append(self.sp_inventory)
        return(ret)
        

            
    def reset(self):
        self.battery = BatteryCircuit(process_noise = self.process_noise, measurement_noise = self.measurement_noise)
        # Initial states
        self.states = reset_states(self.battery)
        self.t = self.init_t
        self.v = self.init_v
        self.t_1 = self.t_2 = self.t_3 = self.v_1 = self.v_2 = self.v_3 = 0
        self.rul = 99
        self.health = 1
        self.true_health = 1
        self.test = []
        self.time = 0
        self.order_list = get_order_list(self.mean_order, self.sd_order, self.max_ep_time+self.forecast)
        self.order = self.order_list[0]
        self.old_order = 0
        self.inventory = 0
        self.sp_inventory = 0
        self.scheduled_maintenance_counter = 0
        return self._get_obs()
