import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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
crewCount = 5
Wcrew = crewCount * passengerWeight
Wpayload = passengerCount * passengerWeight

## Lift to drag ratio Isak
Swet_Sref = 6.1  # Provided as a suggestion for ATR-72, "Aircraft Design Studies Based on the ATR 72", https://www.fzt.haw-hamburg.de/pers/Scholz/arbeiten/TextNita.pdf
print(f'Swet/Sref: {Swet_Sref:.3g}.')
A = 12  # Aspect ratio https://www.rocketroute.com/aircraft/atr-72-212, https://en.wikipedia.org/wiki/ATR_72
print(f'Aspect ratio: {A:.3g}.')
KLD = 12  # For "Turboprop", Raymer 2018

L_Dmax = KLD * np.sqrt(A / Swet_Sref)
print(f'L/D max: {L_Dmax:.3g}.')
print(f'L/D cruise: {L_Dmax:.3g}.')

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
# https://iashulf.memberclicks.net/technical-q-a-fuels
LHVJetA = 0.0828 * 43.37 + 0.1328 * 43.38 + 0.7525 * 43.39 + 0.0318 * 43.246  # MJ kg-1
SEH = 142 * 10 ** 6  # J kg-1 https://www.alakai.com/hydrogen-details
EDH = 8 * 10 ** 6 * 10 ** 3  # J m-3 https://www.energy.gov/eere/fuelcells/hydrogen-storage
rhoH = 0.071 * 10 ** 3 # kg m-3
LHVH = 119.96  # MJ kg-1 https://h2tools.org/hyarc/calculator-tools/lower-and-higher-heating-values-fuels

SFCH_SFCJetA = LHVJetA / LHVH

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

Winit_W0 = 0.97  # In lecture 1
Wclimb_Winit = 0.985  # From historical data

## Cruise fuel fraction Mustafa. The cruise fraction includes the descent part as mentioned in Raymer for initial sizing
rangeAircraft = 1100 * 1852  # metres   # convert 1100 nautical miles to metres
cruise_mach = 0.5
sound_speed = 309.696  # m/s  at FL250
cruise_speed = cruise_mach * sound_speed
efficiency_turboprop = 0.8  # From Raymer
SFC_cruise = (SFC_power * cruise_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
print(f'SFC: {SFC_cruise * 10**6:.3g} mg/Ns.')
L_Dcruise = L_Dmax

Wcruise_Wclimb = np.exp((-rangeAircraft*SFC_cruise*gravity)/(cruise_speed * L_Dcruise))


## Loiter fuel fraction Mustafa
# Included in cruise for initial sizing, set to 1
Wdescent_Wcruise = 1

endurance = 20 * 60  # s #Converted from minutes
loiter_speed = 200 * 0.514  # m/s # Converted from knots
SFC_loiter_power = 0.101/1000000  # kg/Ws Typical value for turboprop from historical data Raymer. 0.08 given in slides
efficiency_turboprop = 0.8  # From Raymer
SFC_loiter = (SFC_loiter_power * loiter_speed) / efficiency_turboprop  # kg/Ns # Converted to turbojet equivalent
L_Dloiter = 0.866 * L_Dmax
print(f'L/D loiter: {L_Dloiter:.3g}.')
Wloiter_Wdescent = np.exp((-endurance*SFC_loiter*gravity)/L_Dcruise)

## Final fuel fraction Mustafa
Wfinal_Wloiter = 0.995  # from historical data

## Diversion fuel fraction - Climb Jay
WdivClimb_Wfinal = 0.985

## Diversion fuel fraction - Cruise Jay
# Assumption: diversion takes place in FL250, M = 0.5, same as cruise conditions
L_Dcruise = L_Dmax 
rangeDiversion = 100 * 1852  #m

WdivCruise_WdivClimb = np.exp(((-rangeDiversion) * SFC_cruise) * gravity / (cruise_speed * L_Dcruise))

## Diversion fuel fraction - Descent Jay

# Included in cruise for initial sizing, set to 1
WdivDescent_WdivCruise = 1

## Fuel weight fraction Jay
Wfinal_W0 = Wclimb_Winit * Wcruise_Wclimb * Wdescent_Wcruise * Wloiter_Wdescent * \
            Wfinal_Wloiter * WdivClimb_Wfinal * WdivCruise_WdivClimb * WdivDescent_WdivCruise

Wf_W0 = 1.06 * (1-Wfinal_W0)
print(f'FUEL/MTOW: {Wf_W0:.3g}.')

## Empty weight fraction Isak
futureTechNudgeFactor = 0.96  # Design guess
hydrogenTanksNudgeFactor = 1.12  # Design guess
a = 0.92 * futureTechNudgeFactor * hydrogenTanksNudgeFactor
c = -0.05
We_W0initialGuess = 0.6  # From lecture 1


def We_frac_equation(We_W0_inner):
    return a * ((Wcrew + Wpayload) / (1 - Wf_W0 - We_W0_inner)) ** c - We_W0_inner


We_W0 = float(fsolve(We_frac_equation, We_W0initialGuess))
print(f'OEW/MTOW: {We_W0:.3g}.')

## Take off weight Jay
W0 = (Wcrew + Wpayload) / (1 - We_W0 - Wf_W0)
print(f'MTOW: {W0 * 10 ** -3:.3g} tonnes.')

## Additional deliverables

# Fuel weight
Wf = W0 * Wf_W0
print(f'FUEL WEIGHT: {Wf * 10 ** -3:.3g} tonnes.')

# Tank volume
tankVolume = Wf / rhoH
print(f'Tank volume: {tankVolume:.3g} m3.')

# Show the plots
plt.show(block=True)
