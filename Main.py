import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Important notes:
# Camel case for all parameters, underscore means fraction
# No capital for next word if prev word is 1 letter long

### Predefined parameters

passengerWeight = 105  # kg
passengerCount = 90
releaseYear = 2035  # BC

### Calculations

## Crew and payload weight calculation Isak

# In
crewCount = 5

# Out
Wcrew = crewCount * passengerWeight
Wpayload = passengerCount * passengerWeight

## Lift to drag ratio Isak

# In
Swet_Sref = 5.95
A = 9.16  # Aspect ratio http://www.b737.org.uk/techspecsdetailed.htm
KLD = 12  # For "Turboprop"

# Out
L_Dmax = KLD * np.sqrt(A / Swet_Sref)

## Specific fuel consumption Isak

# Create plot
fig, ax = plt.subplots()

# Read excel SFC data
dfSFC = pd.read_excel('excel/SFC_prop.xlsx')
yearData = dfSFC.loc[:, 'year']
SFCData = dfSFC.loc[:, 'SFC kg/Ws']

ax.scatter(yearData, SFCData, c='b', label='Reference data')


# Create curvefit
def mockFunction(year, numerator, term):
    return numerator / year + term


# noinspection PyTupleAssignmentBalance
param, param_cov = curve_fit(mockFunction, yearData, SFCData)
interpYear = np.linspace(1940, 2050, 1000)
interpSFC = param[0] / interpYear + param[1]
ax.plot(interpYear, interpSFC, '--r', label='Interpolation line')


# Find
hydrogenSFCAdjustment = -0.05  # TODO adjust
SFC = (param[0] / releaseYear + param[1]) * (1 + hydrogenSFCAdjustment)
ax.scatter([releaseYear], [SFC],
           s=plt.rcParams['lines.markersize'] ** 2 * 2,
           c='k', marker='x', label='Model SFC (adjusted)')

# Plot settings
ax.set_title('SFC - year relation')
ax.set_ylabel('SFC [kg W-1 s-1]')
ax.set_xlabel('Year')
ax.grid()
ax.legend()

# Save figure
figureDPI = 200
fig.set_size_inches(8, 6)
fig.savefig('img/SFCYearRelation.png', dpi=figureDPI)

## Empty weight fraction Isak

# In
Wf_W0 = 0.3
a = 0.92  # TODO This needs adjustment
c = -0.05  # TODO This needs adjustment
We_W0initialGuess = 0.6 * np.ones(1)  # From lecture 1


def We_frac_equation(We_W0_inner):
    return a * ((Wcrew + Wpayload) / (1 - Wf_W0 - We_W0_inner)) ** c - We_W0_inner


# Out
We_W0 = float(fsolve(We_frac_equation, We_W0initialGuess))
# print(We_W0)

## Initial fuel fraction Isak

# Out
Winit_W0 = 0.97  # In lecture 1

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

# Out
W_0 = (W_crew + W_payload) / (1 - W_e_by_W_0 - W_f_by_W_0)

# Show the plots
plt.show(block=True)
