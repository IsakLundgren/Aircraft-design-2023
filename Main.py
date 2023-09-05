import numpy as np
import matplotlib.pyplot as plt

# Important notes:
# Camel case for all parameters, underscore means fraction
# No capital for next word if prev word is 1 letter long

### Predefined parameters

passengerWeight = 105 # kg
passengerCount = 90
gravity = 9.81

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

""" ## Contingency fuel fraction Jay        DON'T
                                            NEED
# Out                                       THESE
Wcont_Wfinal = 1                            SINCE       
                                            THESE
## Trapped fuel fraction Jay                ARE CONSIDERED IN
                                            1.06 as (1%+5%) IN
# Out                                       W_f_by_W_0 EQUATION
Wtrapped_Wcont = 1 """                      

## Diversion fuel fraction - Climb Jay

# Out
Wdiv_climb = 0.985

## Diversion fuel fraction - Cruise Jay
# In
R = 1   
SFC_cruise = 1
cruise_speed = 1
L_Dcruise = 1

# Out
Wdiv_cruise = np.exp(((-R) * SFC_cruise) * gravity / (cruise_speed * L_Dcruise))

## Diversion fuel fraction - Descent Jay

# Out
Wdiv_descent = 1

## Diversion fuel fraction TOTAL - Descent Jay
# In
Wdiv_climb = 1
Wdiv_cruise = 1
Wdiv_descent = 1

# Out
Wdiv_final = Wdiv_climb * Wdiv_cruise * Wdiv_descent

## Fuel weight fraction Jay

# In
Wfinal_W0 = Wclimb_Winit * Wcruise_Wclimb * Wdescent_Wcruise * Wloiter_Wdescent * \
            Wfinal_Wloiter * Wdiv_final

# Out
Wf_W0 = 1.06 * (1-Wfinal_W0)

## Take off weight Jay

# In
Wcrew = 1
Wpayload = 1
We_W0 = 1
Wf_W0 = 1

# Out
W_0 = (Wcrew + Wpayload) / (1 - We_W0 - Wf_W0)
