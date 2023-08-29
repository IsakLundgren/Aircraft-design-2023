import numpy as np
import matplotlib.pyplot as plt

### Predefined parameters

passengerWeight = 105 # kg
passengerCount = 90

### Calculations

## Crew and payload weight calculation

# In

# Out
W_crew = 1
W_payload = 1

## Empty weight fraction

# In

# Out
W_e_by_W_0 = 1

## Fuel weight fraction

# In

# Out
W_f_by_W_0 = 1

## Take off weight

# In
W_crew = 1
W_payload = 1
W_e_by_W_0 = 1
W_f_by_W_0 = 1

# Out
W_0 = (W_crew + W_payload) / (1 - W_e_by_W_0 - W_f_by_W_0)
