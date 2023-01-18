from pickle import FALSE, TRUE
from pyexpat import model
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


class SteelEnv(gym.Env):
    """ Simple Production Process
    Observation:
            Num     Observation                     Min                     Max
            0       Health Roll 1                   0                       1
            1       Health Roll 2                   0                       1
            2       Health Roll 3                   0                       1
            3       Health Roll 4                   0                       1
            4       Quality                         0                       1

    Actions:
        Num     Action
        0       Maintain
        1       Position 1
        2       Position 2
        3       Position 3
        4       Position 4
        5       Position 5
        

    Reward:
        Lorem

    Starting State:
        Lorem

    Episode Termination:
        The episode terminates after 100 time steps

    Parameters:
        
    """

    def __init__(self, natural=False, diag_model = None):
        # filter prog_models warnings
        warnings.filterwarnings("ignore")
        # Set parameter
        self.diag_model = diag_model

        # Set variables
        self.max_ep_time = 100 # Time at which episode terminates
        self.c_repair = -10
        self.c_breakdown = -20
        self.r_prod = 1
        self.c_scrap = -1
        # Set constants
        self.init_t = 19.410832360806385 # Initial temp value of battery
        self.init_v = 3.8838358089883327 # Initial volt value of battery

        # Health, Product quality
        low = np.array([0, 0,], dtype=np.float32,)
        high = np.array([1, 1,], dtype=np.float32,)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        self.breakdown = False
        self.quality = 0


        # Calculate health
        if action == 0: # no production & maintain            
            # Reset states of battery
            self.states = reset_states(self.battery)
            self.t = self.init_t
            self.v = self.init_v
            self.health = 1
            self.true_health = 1
            reward = self.c_repair
            # Reset scheduled maintenance counter
            self.scheduled_maintenance_counter = 0
        else:        
            # Shift history by one time period
            self.v_3 = self.v_2
            self.t_3 = self.t_2
            self.v_2 = self.v_1
            self.t_2 = self.t_1
            self.v_1 = self.v
            self.t_1 = self.t
            # More complex, black box degradation
            self.true_health, self.states, self.t, self.v = produce_model(self.battery, self.states, action)
            if self.true_health < 0: # Breakdown & Repair 
                self.breakdown = True
                # Reset states of battery
                self.states = reset_states(self.battery)
                # Reset condition values
                self.t = self.init_t
                self.v = self.init_v
                self.t_1 = self.t_2 = self.t_3 = self.v_1 = self.v_2 = self.v_3 = 0
                self.health = 1
                self.true_health = 1
                
                # Spare part not available, so it must be ordered 
                reward = self.c_breakdown
                # Reset scheduled maintenance counter
                self.scheduled_maintenance_counter = 0
            else:
                # If production, calculate bar quality (non-faulty = 1, faulty = 0)
                self.quality = np.random.binomial(1, min(1, self.health+(action/5)))
                # If self.quality = 1 (non-faulty), receive production revenue
                if self.quality: reward = self.r_prod
                # If self.quality = 0 (faulty), incur scrap cost
                else: reward = self.c_scrap
        
        # If a diagnostics model is supplied, replace true health with approximated health
        if self.diag_model is not None:
            self.health = max(self.diag_model.predict([[self.t, self.v, self.t_1, self.v_1, self.t_2, self.v_2, self.t_3, self.v_3]])[0], 0)
        else:
            self.health = self.true_health      
        
        self.scheduled_maintenance_counter = self.scheduled_maintenance_counter + 1
        # Stop when time self.max_ep_time has been reached
        self.time = self.time + 1
        if self.time == self.max_ep_time:
           done = True
        
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # In case of reactive or scheduled maintenance, knowledge about condition is not available
        ret = []
        ret.append(self.health)
        ret.append(self.quality)
        return(ret)
        

            
    def reset(self):
        self.battery = BatteryCircuit()
        # Initial states
        self.quality = 1
        self.states = reset_states(self.battery)
        self.t = self.init_t
        self.v = self.init_v
        self.t_1 = self.t_2 = self.t_3 = self.v_1 = self.v_2 = self.v_3 = 0
        self.health = 1
        self.true_health = 1
        self.time = 0
        self.breakdown = False # Reset breakdown indicator
        self.scheduled_maintenance_counter = 0
        return self._get_obs()
