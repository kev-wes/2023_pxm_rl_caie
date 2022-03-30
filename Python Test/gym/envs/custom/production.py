import gym
import numpy as np
import random
import math
from prog_models.models import BatteryCircuit
from gym import spaces
from gym.utils import seeding

# Use prog_models package for simulation of RUL
def produce_model(machine, a_store): # TODO: Allow seed
        
        def rounddown100(x):
            return int(math.floor(x / 100.0)) * 100

        # Define load of battery
        def future_loading(t, x=None):
            # if time step exceeds a_store take last intensity as open ended intensity
            if(int(rounddown100(t)/100) > len(a_store)-1):
                i = a_store[-1]
            else:
                i = a_store[int(rounddown100(t)/100)]
            return {'i': i}

        # Simulate to (len(a_store)-)*100
        options = {
            'save_freq': (len(a_store))*100,  # Frequency at which results are saved
            'dt': 2  # Timestep
        }
        (times, inputs, states, outputs, event_states) = machine.simulate_to((len(a_store)-1)*100, future_loading, **options)
        rul = event_states[-1]['EOD']
        
        return(round(rul, 2))
# Use simple function for simulation of RUL (RUL = RUL - intensity/100)
def produce_simple(rul, action):
        return(round(rul-action/100, 2))

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
        Type: Box(3)
        Num     Observation                     Min                     Max
        0       Remaining Useful Life (RUL)     0                       1
        1       Order Quantity                  1                       Inf
        2       Pole Angle                      -0.209 rad (-12 deg)    0.209 rad (12 deg)

    Actions:
        Type: Discrete(6)
        Num   Action
        0     Repair
        1-4   Produce w/ intensity 1-4
        5     Repair

    Reward:
        Lorem

    Starting State:
        Lorem

    Episode Termination:
        Lorem
    """

    def __init__(self, natural=False):
        # Set parameter
        self.max_ep_time = 100 # Time at which episode terminates
        self.maintenance_c = 100 # Predictive Maintenance cost at RUL = 1
        self.repair_c = 500 # Reactive Repair cost
        self.backorder_c = 4 # Backorder cost
        self.order_r = 2 # Order fulfillment reward
        self.holding_c = 1 # Holding cost
        self.max_inventory = 9 # Maximal inventory
        self.min_order = 0 # Minimal order quantity
        self.max_order = 4 # Maximal order quantity test

        self.action_space = spaces.Discrete(6)
        
        low     = np.array([0, self.min_order, 0,], dtype=np.float32,)
        high    = np.array([1, self.max_order, self.max_inventory,], dtype=np.float32,)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.seed()

        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        reward = 0
        # a_store stores the intensities per each 100 time steps
        self.a_store = self.a_store + [action]

        # Calculate RUL
        if action == 5: # no production & maintain
            action = 0
            # Maintenance is less punished the closer to failure it is performed
            reward = reward - self.maintenance_c * self.rul
            self.rul = 1
        else:
            # More complex, black box degradation
            #self.rul = produce_model(self.battery, self.a_store)
            # Simple, transparent degradation
            self.rul = produce_simple(self.rul, action)
            if self.rul < 0: # Breakdown & Repair 
                action = 0 # Assume failure led to zero production (e.g., scrap)
                self.rul = 1
                reward = reward - self.repair_c
        # Calculate inventory cost
        reward = reward - self.inventory * self.holding_c

        # Calculate new Inventory and (back)order rewards
        # Set new inventory to be old inventory
        # Cannot exceed maximum inventory of self.max_inventory
        inventory = min(self.inventory + action - self.order, self.max_inventory)
        if inventory < 0:
            reward = reward + inventory * self.backorder_c
            self.inventory = 0
        else:
            reward = reward + self.order * self.order_r
            self.inventory = inventory

        # Draw next order
        self.order = get_order(self.min_order, self.max_order, self.seed()[0])               

        # Stop when time self.max_ep_time has been reached
        self.time = self.time + 1
        if self.time == self.max_ep_time:
            done = True
        

        
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (self.rul, self.order, self.inventory)

    def reset(self):
        self.battery = BatteryCircuit()
        self.rul = 1
        self.time = 0
        self.a_store = []
        self.order = get_order(self.min_order, self.max_order, self.seed()[0])
        self.inventory = 0
        return self._get_obs()
