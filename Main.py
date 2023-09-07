import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

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
A = 12 # 9.16  # Aspect ratio http://www.b737.org.uk/techspecsdetailed.htm
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
rhoJetA = 0.804 * 10 ** 3  # kg m-3
SEH = 142 * 10 ** 6  # J kg-1 https://www.alakai.com/hydrogen-details
EDH = 8 * 10 ** 6 * 10 ** 3  # J m-3 https://www.energy.gov/eere/fuelcells/hydrogen-storage
rhoH = 71  # kg m-3

SFCH_SFCJetA = (SEH ** 3 * EDH * rhoH) / (SEJetA ** 3 * EDJetA * rhoJetA)
ax.text(1941, 0.52e-7 * 10 ** 6, f'SFC_H by SFC_Jet A = {SFCH_SFCJetA:.3g}')

# Finalize SFC calculation
SFC_power = (param[0] / releaseYear + param[1]) * SFCH_SFCJetA
print(f'SFC_power: {SFC_power * 10**6:.3g} mg/(Ws).')
ax.scatter([releaseYear], [SFC_power * 10 ** 6],
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
rangeAircraft = 1100 * 1852  # metres   # convert 1100 nautical miles to metres
cruise_mach = 0.5
sound_speed = 309.696  # m/s  at FL250
cruise_speed = cruise_mach * sound_speed
efficiency_turboprop = 0.8  # From Raymer
SFC_cruise = (SFC_power * cruise_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
print(f'SFC: {SFC_cruise * 10**6:.3g} mg/Ns')
L_Dcruise = L_Dmax

Wcruise_Wclimb = np.exp((-rangeAircraft*SFC_cruise*gravity)/(cruise_speed * L_Dcruise))


## Loiter fuel fraction Mustafa
# Included in cruise for initial sizing, set to 1
Wdescent_Wcruise = 1


# Out
endurance = 20 * 60  # s #Converted from minutes
loiter_speed = 200 * 0.514  # m/s # Converted from knots
SFC_loiter_power = 0.101/1000000  # kg/Ws Typical value for turboprop from historical data Raymer. 0.08 given in slides
efficiency_turboprop = 0.8  # From Raymer
SFC_loiter = (SFC_loiter_power * loiter_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
L_Dloiter = 0.866 * L_Dmax
Wloiter_Wdescent = np.exp((-endurance*SFC_loiter*gravity)/L_Dcruise)

## Final fuel fraction Mustafa

# Out
Wfinal_Wloiter = 0.995  # from historical data

## Diversion fuel fraction - Climb Jay

# Out
WdivClimb_Wfinal = 0.985

## Diversion fuel fraction - Cruise Jay
# Assumption: diversion takes place in FL250, M = 0.5, same as cruise conditions
# In
L_Dcruise = L_Dmax 
rangeDiversion = 100 * 1852  #m

# Out
WdivCruise_WdivClimb = np.exp(((-rangeDiversion) * SFC_cruise) * gravity / (cruise_speed * L_Dcruise))

## Diversion fuel fraction - Descent Jay

# Out
# Included in cruise for initial sizing, set to 1
WdivDescent_WdivCruise = 1

## Fuel weight fraction Jay

# In
Wfinal_W0 = Wclimb_Winit * Wcruise_Wclimb * Wdescent_Wcruise * Wloiter_Wdescent * \
            Wfinal_Wloiter * WdivClimb_Wfinal * WdivCruise_WdivClimb * WdivDescent_WdivCruise

# Out
Wf_W0 = 1.06 * (1-Wfinal_W0)

## Empty weight fraction Isak

# In
futureTechNudgeFactor = 0.96  # Design guess
hydrogenTanksNudgeFactor = 1.12  # Design guess
a = 0.92 * futureTechNudgeFactor * hydrogenTanksNudgeFactor
c = -0.05
We_W0initialGuess = 0.6  # From lecture 1


def We_frac_equation(We_W0_inner):
    return a * ((Wcrew + Wpayload) / (1 - Wf_W0 - We_W0_inner)) ** c - We_W0_inner


# Out
We_W0 = float(fsolve(We_frac_equation, We_W0initialGuess))
print(f'OEW/MTOW: {We_W0:.3g}')

## Take off weight Jay

# Out
W0 = (Wcrew + Wpayload) / (1 - We_W0 - Wf_W0)
print(f'MTOW: {W0 * 10 ** -3:.3g} tonnes.')

# Show the plots
plt.show(block=True)
