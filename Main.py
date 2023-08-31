import numpy as np
import matplotlib.pyplot as plt

# Important notes:
# Camel case for all parameters, underscore means fraction
# No capital for next word if prev word is 1 letter long

### Predefined parameters

passengerWeight = 105 # kg
passengerCount = 90

### Calculations

## Crew and payload weight calculation Isak

# In
crewCount = 5

# Out
WCrew = 1
WPayload = 1

## Lift to drag ratio Isak

# In
Swet_Sref = 5.95
A = 1
KLD = 15.5 # For civil jets Isak

# Out
L_Dmax = 1

## Specific fuel consumption Isak

# Out
SFC = 1

## Empty weight fraction Isak

# In

# Out
We_W0 = 1

## Initial fuel fraction Isak

# Out
Winit_W0 = 1

## Climb fuel fraction Mustafa

# Out
Wclimb_Winit = 1

## Cruise fuel fraction Mustafa

# Out
Wcruise_Wclimb = 1

## Descent fuel fraction Mustafa

# Out
Wdescent_Wcruise = 1

## Loiter fuel fraction Mustafa

# Out
Wloiter_Wdescent = 1

## Final fuel fraction Mustafa

# Out
Wfinal_Wloiter = 1

## Contingency fuel fraction Jay

# Out
Wcont_Wfinal = 1

## Trapped fuel fraction Jay

# Out
Wtrapped_Wcont = 1

## Diversion fuel fraction Jay

# Out
Wdiv_Wtrapped = 1

## Fuel weight fraction Jay

# In

# Out
W_f_by_W_0 = 1

## Take off weight Jay

# In
W_crew = 1
W_payload = 1
W_e_by_W_0 = 1
W_f_by_W_0 = 1

# Out
W_0 = (W_crew + W_payload) / (1 - W_e_by_W_0 - W_f_by_W_0)
