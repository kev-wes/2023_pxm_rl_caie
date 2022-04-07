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

# Get random integer order quantity between min and max  
def get_order(p_min = 1, p_max = 4, p_seed = None):
    #if p_seed is not None:
     #   np.random.default_rng(p_seed)
     #   np.random.seed(5)
    #return(round(np.random.uniform(min, max))) 
    return(min(np.random.poisson(p_max, 1)[0], p_max))

class ProductionEnv(gym.Env):
    """ Simple Production Process
    Observation:
        ACTIVE Version 1: Direct health
            Type: Box(3)
            Num     Observation                     Min                     Max
            0       Health                          0                       1
            1       Order Quantity t+1              self.min_order = 0      self.max_order = 4
            2       Finished Inventory              0                       self.max_inventory = 9
            3       Spare Parts Inventory           0                       self.spare_parts_max_inventory
            TODO Add Order Qty Forecasts
            4       Order Quantity t+2              self.min_order = 0      self.max_order = 4
            5       Order Quantity t+3              self.min_order = 0      self.max_order = 4
            6       Order Quantity t+4              self.min_order = 0      self.max_order = 4
            7       Order Quantity t+5              self.min_order = 0      self.max_order = 4

    Actions:
        Type: Discrete(6)
        Num   Action 
        0-4   Produce w/ intensity 0-4 and no spare parts order
        5-9   Produce w/ intensity 0-4 and order spare part
        10    Repair

    Reward:
        Lorem

    Starting State:
        Lorem

    Episode Termination:
        The episode terminates after 100 time steps

    TODO: 
        1. Build Version 3: Predicted health based on Condition Values
        2. Normalize values (e.g., condition data)
        3. Add Noise

    """

    def __init__(self, natural=False, diag_model=None):
        warnings.filterwarnings("ignore")
        # Set prognostics model
        self.diag_model = diag_model
        # Set parameter
        self.max_ep_time = 100 # Time at which episode terminates
        self.maintenance_c = 100 # Predictive Maintenance cost at health = 1
        self.repair_c = 500 # Reactive Repair cost
        self.backorder_c = 4 # Backorder cost
        self.order_r = 2 # Order fulfillment reward
        self.spare_part_order_c = 10 # Spare part order cost
        self.spare_part_holding_c = 1 # Spare part holding cos
        self.spare_parts_max_inventory = 1 # Maximal spare parts inventory
        self.sp_emergency_order_c = 300 # Spare parts emergency order costs if inventory is zero
        self.holding_c = 1 # Holding cost
        self.max_inventory = 9 # Maximal inventory
        self.min_order = 0 # Minimal order quantity
        self.max_order = 4 # Maximal order quantity
        self.init_t = 19.410832360806385 # Initial temp value of battery
        self.init_v = 3.8838358089883327 # Initial volt value of battery

        self.action_space = spaces.Discrete(11)
        
        # Set floor and ceiling of state space
        low     = np.array([0, self.min_order, 0, 0,], dtype=np.float32,)
        high    = np.array([1, self.max_order, self.max_inventory, self.spare_parts_max_inventory,], dtype=np.float32,)

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

        # Spare parts inventory holding costs
        reward = reward - self.sp_inventory * self.spare_part_holding_c
        # Calculate health
        if action == 10: # no production & maintain
            intensity = 0
            # Reset states of battery
            self.states = reset_states(self.battery)
            self.t = self.init_t
            self.v = self.init_v
            if self.sp_inventory != 0: reward = reward - self.maintenance_c * self.health # Maintenance is less punished the closer to failure it is performed
            else: reward = reward - self.maintenance_c * self.health - self.sp_emergency_order_c # Spare part not available, so it must be ordered
            self.sp_inventory = 0
            self.health = 1
        else:
            if action > 4:
                intensity = action - 5
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
            self.health, self.states, self.t, self.v = produce_model(self.battery, self.states, intensity)
            if self.health < 0: # Breakdown & Repair 
                intensity = 0 # Assume failure led to zero production (e.g., scrap)
                # Reset states of battery
                self.states = reset_states(self.battery)
                # Reset condition values
                self.t = self.init_t
                self.v = self.init_v
                self.t_1 = self.t_2 = self.t_3 = self.v_1 = self.v_2 = self.v_3 = 0
                self.health = 1
                if self.sp_inventory > 0: reward = reward - self.repair_c 
                else: reward = reward - self.repair_c - self.sp_emergency_order_c # Spare part not available, so it must be ordered 
                self.sp_inventory = 0
        # Calculate inventory cost
        reward = reward - self.inventory * self.holding_c

        # Calculate new Inventory and (back)order rewards
        # Set new inventory to be old inventory
        # Cannot exceed maximum inventory of self.max_inventory
        inventory = min(self.inventory + intensity - self.order, self.max_inventory)
        if inventory < 0:
            reward = reward + inventory * self.backorder_c
            self.inventory = 0
        else:
            reward = reward + self.order * self.order_r
            self.inventory = inventory
        
        # Draw next order
        self.order = get_order(self.min_order, self.max_order, self.seed()[0])
        
        # If a prognostics model is supplied, replace true health with approximated health
        if self.diag_model is not None:
            self.health = max(self.diag_model.predict([[self.t, self.v, self.t_1, self.v_1, self.t_2, self.v_2, self.t_3, self.v_3]])[0], 0)         

        # Stop when time self.max_ep_time has been reached
        self.time = self.time + 1
        
        
        if self.time == self.max_ep_time:
            done = True
        
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.health, self.order, self.inventory, self.sp_inventory)

    def reset(self):
        self.battery = BatteryCircuit()
        # Initial states
        self.states = reset_states(self.battery)
        self.t = self.init_t
        self.v = self.init_v
        self.t_1 = self.t_2 = self.t_3 = self.v_1 = self.v_2 = self.v_3 = 0
        self.health = 1
        self.test = []
        self.time = 0
        self.order = get_order(self.min_order, self.max_order, self.seed()[0])
        self.inventory = 0
        self.sp_inventory = 0
        return self._get_obs()
