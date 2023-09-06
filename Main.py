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
gravity = 9.82

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
print(f'Swet/Sref: {Swet_Sref:.3g}.')
A = 9.16  # Aspect ratio http://www.b737.org.uk/techspecsdetailed.htm
print(f'Aspect ratio: {A:.3g}.')
KLD = 12  # For "Turboprop"

# Out
L_Dmax = KLD * np.sqrt(A / Swet_Sref)
print(f'L/D: {L_Dmax:.3g}.')

## Specific fuel consumption Isak

# Create plot
fig, ax = plt.subplots()

# Read excel SFC data
dfSFC = pd.read_excel('excel/SFC_prop.xlsx')
yearData = dfSFC.loc[:, 'year']
SFCData = dfSFC.loc[:, 'SFC kg/Ws']

ax.scatter(yearData, SFCData * 10 ** 6, c='b', label='Reference data')


# Create curvefit
def mockFunction(year, numerator, term):
    return numerator / year + term


# noinspection PyTupleAssignmentBalance
param, param_cov = curve_fit(mockFunction, yearData, SFCData)
interpYear = np.linspace(1940, 2050, 1000)
interpSFC = param[0] / interpYear + param[1]
ax.plot(interpYear, interpSFC * 10 ** 6, '--r', label='Interpolation line')

# Find sfcH_sfcjetA
# Jet A info from https://en.wikipedia.org/wiki/Jet_fuel
SEJetA = 43.15 * 10 ** 6  # J kg-1
EDJetA = 34.7 * 10 ** 6 * 10 ** 3  # J m-3
rhoJetA = 0.804 * 10 ** 3 # kg m-3
SEH = 142 * 10 ** 6  # J kg-1 https://www.alakai.com/hydrogen-details
EDH = 8 * 10 ** 6 * 10 ** 3  # J m-3 https://www.energy.gov/eere/fuelcells/hydrogen-storage
rhoH = 71  # kg m-3

SFCH_SFCJetA = (SEH ** 3 * EDH * rhoH) / (SEJetA ** 3 * EDJetA * rhoJetA)
ax.text(1941, 0.52e-7 * 10 ** 6, f'SFC_H by SFC_Jet A = {SFCH_SFCJetA:.3g}')

# Finalize SFC calculation
SFC = (param[0] / releaseYear + param[1]) * SFCH_SFCJetA
print(f'SFC: {SFC * 10**6:.3g} mg/(Ws).')
ax.scatter([releaseYear], [SFC * 10 ** 6],
           s=plt.rcParams['lines.markersize'] ** 2 * 2,
           c='k', marker='x', label='Model SFC (adjusted)')

# Plot settings
ax.set_title('SFC - year relation')
ax.set_ylabel('SFC [mg W-1 s-1]')
ax.set_xlabel('Year')
ax.grid()
ax.legend()

# Save figure
figureDPI = 200
fig.set_size_inches(8, 6)
fig.savefig('img/SFCYearRelation.png', dpi=figureDPI)

## Empty weight fraction Isak

# In
Wf_W0 = 0.3  # TODO Move this such that Wf_W0 is taken into account
futureTechNudgeFactor = 0.96  # Design guess
hydrogenTanksNudgeFactor = 1.12  # Design guess
a = 0.92 * futureTechNudgeFactor * hydrogenTanksNudgeFactor
c = -0.05
We_W0initialGuess = 0.6 * np.ones(1)  # From lecture 1


def We_frac_equation(We_W0_inner):
    return a * ((Wcrew + Wpayload) / (1 - Wf_W0 - We_W0_inner)) ** c - We_W0_inner


# Out
We_W0 = float(fsolve(We_frac_equation, We_W0initialGuess))
print(f'OEW/MTOW: {We_W0:.3g}')

## Initial fuel fraction Isak

# Out
Winit_W0 = 0.97  # In lecture 1

## Climb fuel fraction Mustafa

# Out
# Wclimb_Winit = 1

## Cruise fuel fraction Mustafa

# Out
# Wcruise_Wclimb = 1

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

Wcruise_Wclimb = np.exp((-range*SFC_cruise*gravity)/(cruise_speed * L_Dcruise))


## Loiter fuel fraction Mustafa
Wdescent_Wcruise = 1


# Out
endurance = 20 * 60  # s #Converted from minutes
loiter_speed = 200 * 0.514  # m/s # Converted from knots
SFC_loiter_power = 0.101/1000000  # kg/Ws  # Typical value for turbo prop from historical data Raymer. 0.08 given in slides
efficiency_turboprop = 0.8  # From Raymer
SFC_loiter = (SFC_loiter_power * loiter_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
L_Dloiter = 0.866 * L_Dmax
Wloiter_Wdescent = np.exp((-endurance*SFC_loiter*gravity)/L_Dcruise)

## Final fuel fraction Mustafa

# Out
Wfinal_Wloiter = 0.995 # from historical data

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
R = 1  # TODO Figure out what this is
SFC_cruise = 1  # TODO Redirect SFC from Isaks part
cruise_speed = 1  # TODO Set actual cruise speed
L_Dcruise = 1  # TODO Redirect SFC from Isaks part

# Out
Wdiv_cruise = np.exp(((-R) * SFC_cruise) * gravity / (cruise_speed * L_Dcruise))

## Diversion fuel fraction - Descent Jay

# Out
Wdiv_descent = 1  # TODO Find expression for this

## Diversion fuel fraction TOTAL - Descent Jay
# In
# Wdiv_climb = 1
# Wdiv_cruise = 1
# Wdiv_descent = 1

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
Wcrew = 1  # TODO redirect
Wpayload = 1  # TODO redirect
We_W0 = 1  # TODO redirect
Wf_W0 = 1  # TODO redirect

# Out
W_0 = (Wcrew + Wpayload) / (1 - We_W0 - Wf_W0)

# Show the plots
plt.show(block=True)
