# Init packages
from prog_models.models import BatteryCircuit
import numpy as np
import pandas as pd

# Init machine model
batt = BatteryCircuit() # Create physical model

def get_rul(p_batch):
    # Loading function
    # t = time
    # loading_df = two column df with first column containing the cumulated production time and
    # second column containing the loading / intensity
    def future_loading(t, x=None):
        r_loading = p_batch[p_batch.time <= t].iloc[-1:].intensity.values[0].copy()
        return {'i': r_loading}
    options = { #configuration for this sim
        'save_freq': 100,  # Frequency at which results are saved (s)
        'horizon': p_batch.time.iloc[-1]  # Maximum time to simulate (s) - This is a cutoff. The simulation will end at this time, or when a threshold has been met, whichever is first
    }
    (times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading, **options)
    return (event_states[-1]["EOD"])

