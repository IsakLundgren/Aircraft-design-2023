import numpy as np
import matplotlib.pyplot as plt

# Important notes:
# Camel case for all parameters, underscore means fraction
# No capital for next word if prev word is 1 letter long

### Predefined parameters

passengerWeight = 105 # kg
passengerCount = 90
gravity = 9.8  # m/s
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

## Take off fuel fraction Isak

# Out
Winit_W0 = 1

## Climb fuel fraction Mustafa

# Out

Wclimb_Winit = 0.985  # From historical data
## Cruise fuel fraction Mustafa. The cruise fraction includes the descent part as mentioned in Raymer for initial sizing

# Out
range = 1100 * 1852  # metres   # convert 1100 nautical miles to metres
cruise_mach = 0.5
sound_speed = 309.696  # m/s  at FL250
cruise_speed = cruise_mach * sound_speed
SFC_cruise_power = 0.063/1000000  # kg/Ws  # Typical value of 0.07 for turbo prop from historical data with 10% improvement
efficiency_turboprop = 0.8  # From Raymer
SFC_cruise = (SFC_cruise_power * cruise_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
L_Dcruise = L_Dmax

Wcruise_climb = np.exp((-range*SFC_cruise*gravity)/(cruise_speed * L_Dcruise))


## Loiter fuel fraction Mustafa
Wdescent_cruise = 1


# Out
endurance = 20 * 60  # s #Converted from minutes
loiter_speed = 200 * 0.514  # m/s # Converted from knots
SFC_loiter_power = 0.101/1000000  # kg/Ws  # Typical value for turbo prop from historical data Raymer. 0.08 given in slides
efficiency_turboprop = 0.8  # From Raymer
SFC_loiter = (SFC_loiter_power * loiter_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
L_Dloiter = 0.866 * L_Dmax
Wloiter_Wdescent = np.exp((-endurance*SFC_loiter*gravity)/L_Dcruise)

## Landing fuel fraction Mustafa

# Out
Wfinal_Wloiter = 0.995 # from historical data

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
