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

### Conversion parameters
m_feet = 1 / 3.28084
m_s_knot = 1 / 1.944

### Predefined parameters

passengerWeight = 105  # kg
passengerCount = 90
releaseYear = 2035  # BC
gravity = 9.82
Vapproach = 119 * m_s_knot  # m s-1
rhoSL = 1.225  # kg m-3

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


# Hydrogen adjustments for fuel fractions where SFC is not used
def convFuelFrac(W1_W0, LHVA_LHVB):
    return 1 - LHVA_LHVB * (1 - W1_W0)


Winit_W0 = convFuelFrac(0.97, SFCH_SFCJetA)  # In lecture 1
Wclimb_Winit = convFuelFrac(0.985, SFCH_SFCJetA)  # From historical data

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
Wfinal_Wloiter = convFuelFrac(0.995, SFCH_SFCJetA) # from historical data

## Diversion fuel fraction - Climb Jay
WdivClimb_Wfinal = convFuelFrac(0.985, SFCH_SFCJetA)

## Diversion fuel fraction - Cruise Jay
# Assumption: diversion takes place in FL250, M = 0.5, same as cruise conditions
L_Dcruise = L_Dmax 
rangeDiversion = 100 * 1852  #m

WdivCruise_WdivClimb = np.exp(((-rangeDiversion) * SFC_cruise) * gravity / (cruise_speed * L_Dcruise))

## Diversion fuel fraction - Descent Jay

# Included in cruise for initial sizing, set to 1
WdivDescent_WdivCruise = 1

## Fuel weight fraction Jay
Wfinal_W0 = Winit_W0 * Wclimb_Winit * Wcruise_Wclimb * Wdescent_Wcruise * Wloiter_Wdescent * \
            Wfinal_Wloiter * WdivClimb_Wfinal * WdivCruise_WdivClimb * WdivDescent_WdivCruise

Wf_W0 = 1.06 * (1-Wfinal_W0)
print(f'\nSizing fractions:')
print(f'Winit_W0: {Winit_W0:.3g}')
print(f'Wclimb_Winit: {Wclimb_Winit:.3g}')
print(f'Wcruise_Wclimb: {Wcruise_Wclimb:.3g}')
print(f'Wdescent_Wcruise: {Wdescent_Wcruise:.3g}')
print(f'Wloiter_Wdescent: {Wloiter_Wdescent:.3g}')
print(f'Wfinal_Wloiter: {Wfinal_Wloiter:.3g}')
print(f'WdivClimb_Wfinal: {WdivClimb_Wfinal:.3g}')
print(f'WdivCruise_WdivClimb: {WdivCruise_WdivClimb:.3g}')
print(f'WdivDescent_WdivCruise: {WdivDescent_WdivCruise:.3g}')
print(f'\nFUEL/MTOW: {Wf_W0:.3g}.')

## Empty weight fraction Isak
futureTechNudgeFactor = 0.96  # Design guess
a = 0.92 * futureTechNudgeFactor
c = -0.05
Gi = 0.35  # Gravimetric index,
# added from https://chalmers.instructure.com/courses/25325/pages/project-kick-off?module_item_id=387491

We_W0initialGuess = 0.6  # From lecture 1


def We_frac_equation(We_W0_inner):
    Weorg_W0 = a * ((Wcrew + Wpayload) / (1 - Wf_W0 - We_W0_inner)) ** c
    Wt_W0 = Wf_W0 * ((1-Gi)/Gi)
    return Weorg_W0 + Wt_W0 - We_W0_inner


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

# Assume cylindrical tanks, give tank radius and heights
tankAspectRatio = 20  # h / r = TAR Design guess
tankRadius = np.power((tankVolume / 2) / (np.pi * tankAspectRatio), 1/3)
tankHeight = tankRadius * tankAspectRatio
print(f'Tank aspect ratio: {tankAspectRatio:.3g}.')
print(f'Tank radius: {tankRadius:.3g} m.')
print(f'Tank height: {tankHeight:.3g} m.')

## Calculate thrust to weight ratio

# Cruise
T_Wcruise = 1 / L_Dcruise

# Climb
takeoffToCruiseTime = 15.5 * 60  # s
takeoffAlt = 1500 * m_feet  # m
cruiseAlt = 250 * 100 * m_feet  # m
VclimbVertical = (cruiseAlt - takeoffAlt) / takeoffToCruiseTime  # m s-1
ATRClimbVertical = 1335 * m_feet / 60  # m s-1 http://www.atr-aircraft.com/wp-content/uploads/2020/07/Factsheets_-_ATR_72-600.pdf
ATRClimbSpeed = 170 * m_s_knot  # m s-1
climbSpeedRatio = ATRClimbVertical / ATRClimbSpeed
Vclimb = VclimbVertical / climbSpeedRatio
print(f'\nClimb speed: {Vclimb:.3g} m/s.')
L_Dclimb = L_Dcruise  # Could not find data, assume equal
T_Wclimb = 1 / L_Dclimb + climbSpeedRatio

# First do statistical approach
etaPropeller = 0.8  # Aircraft design book p. 518
a = 0.016  # Lect 5 twin turboprop
C = 0.5  # Lect 5 twin turboprop
P_W0statistical = a * cruise_speed ** C * 1000  # W kg-1
T_W0statistical = etaPropeller / cruise_speed * P_W0statistical / gravity  # -

# Now do thrust matching approach, Aircraft design book p. 122
P_Wcruise = gravity * cruise_speed / etaPropeller * T_Wcruise  # W kg-1
Wcruise_W0 = Wcruise_Wclimb * Wclimb_Winit * Winit_W0
eshpRatio = (60 + 80) * 1/100
P_W0thrustmatch = P_Wcruise * 1 / eshpRatio
T_W0thrustmatch = etaPropeller / cruise_speed * P_W0thrustmatch / gravity

# Take the maximum
T_W0takeoff = np.max([T_W0statistical, T_W0thrustmatch])

# Print thrust to weight ratios
print(f'\nTake-off thrust to weight ratio: {T_W0takeoff * 100:.3g}%.')
print(f'Climb thrust to weight ratio: {T_Wclimb * 100:.3g}%.')
print(f'Cruise thrust to weight ratio: {T_Wcruise * 100:.3g}%.')

# Find the propeller size by statistical approximations
Pcruise = P_Wcruise * Wcruise_W0 * W0
Ptakeoff = Pcruise * 1 / eshpRatio
Kp = 0.52  # Lecture 5 three blades
DpropellerStatistical = Kp * (Ptakeoff / 1000) ** (1/4)

# Print required powers and thrusts
Wclimb = Wclimb_Winit * Winit_W0 * W0
Wcruise = Wcruise_W0 * W0
print(f'\nCruise power: {Pcruise / 1000000:.3g} MW.')
print(f'Take-off thrust: {T_W0takeoff * W0 / 1000:.3g} kN.')
print(f'Climb thrust: {T_Wclimb * Wclimb / 1000:.3g} kN.')
print(f'Cruise thrust: {T_Wcruise * Wcruise / 1000:.3g} kN.')

# Find the propeller size by compressibility effects
rpsPropeller = 1200 / 60  # Taken from ATR 72 https://www.naval-technology.com/projects/atr-72-asw-anti-submarine-warfare-aircraft/
Mtip = 0.97
DpropellerCompressibility = np.sqrt((Mtip * sound_speed) ** 2 - cruise_speed ** 2) / (np.pi * rpsPropeller)

# Take the maximum
Dpropeller = max(DpropellerStatistical, DpropellerCompressibility)  # Probably do not want the diameter to reduce, pick the max
print(f'\nPropeller diameter: {Dpropeller:.3g} m.')

## Wing area

# Calculate wing loading for stall
Vstall = Vapproach / 1.3  # Raymer p.132 for commercial aircraft
Vtakeoff = Vstall * 1.10  # Lect 5
CLmax = 3  # Assume double slotted flap, 0 sweep angle from Raymer p.127
W_Sstall = 1 / (2 * gravity) * rhoSL * Vstall ** 2 * CLmax
Wloiter_W0 = Wloiter_Wdescent * Wdescent_Wcruise * Wcruise_Wclimb * Wclimb_Winit * Winit_W0  # Assume stall occurs in the loiter part of the mission
W_Stakeoff = W_Sstall / Wloiter_W0
S = W0 / W_Stakeoff
print(f'Wing reference area: {S:.3g} m^2.')

## Calculate span, taper-ratio, wing stuff
taper_ratio = 0.4       #Raymer - For most unswept wings
Dihedral_angle = 2      #General value to assume and begin with design
t_c = 0.15              #Thickness to chord ratio, From historical plots
c_HT = 0.9              #Constant, Raymer 160
c_VT = 0.08              #Constant, Raymer 160
dist_to_VT = 1
dist_to_HT = 1

# Aileron sizing - 50-90% of wingspan
#                - 20% of wing chord
# Rudders & Elevators - 0-90% of wingspan
#                     - 36% elevtor chord length, 46% rudder chord length

# Solving for Wing span
span = np.sqrt(A * S)
print('Span = ', span)
print('Half wing span =', span/2)

def eqn2(p):
    root_chord, tip_chord = p
    return (0.4*root_chord - tip_chord, \
            (2*S/span) - root_chord - tip_chord)
root_chord, tip_chord =  fsolve(eqn2, (1, 1))
print('Root chord = ', root_chord, '\nTip chord = ', tip_chord)

mean_chord = 0.666 * root_chord * ((1 + taper_ratio + taper_ratio**2)/(1 + taper_ratio))
print('Mean aerodynamic Chord =', mean_chord)

loc_mean_chord = (span/6) * ((1+2*taper_ratio)/(1+taper_ratio))
print('Location of Mean Aerodynamic Chord =', loc_mean_chord)

# Vertical tail
VT_area = c_VT * span * S / dist_to_VT

# Horizontal tail
HT_area = c_HT * mean_chord * S / dist_to_HT


### Show the plots
plt.show(block=True)
