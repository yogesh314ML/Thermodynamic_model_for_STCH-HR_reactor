import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
#==================Input Parameters==================#
# Operation Pressure
P_red = 9.86*10**-3 # Reactor total pressure (atm)
eff_pump = 0.07*np.log10(P_red) + 0.4 # efficiency of pump with pressure
T_pump = 373 #K

# Operating Temperature and Flow rates
ramp_rate = 80 #K/min (80 K/min with 4.1 KW input)
T_ox = 950+273.15 #K
T_red = 1500+273.15 #K
# moles of argon per min per mol of ceria
n_Ar = 0.002777
# Delta ox and delta red
d_ox = 0.0065924  # Obtained from oxidation code
d_red = 0.0479
# Gas Constant
R = 8.314  # J/k/mol

#====================Function definition=====================#
def dh(x):
    return 395 - 31.4*np.log10(x)  # bulfins correlation

#====================Energy Consumption Calculation================#
delta_change = d_red-d_ox
print(f"Delta_change: {round(delta_change, 7)}")

Cp_Ar = 0.02076 #KJ/molK

def h_Ce(T):
    return (16.761*(10**-3)*T + 1.108*(10**-6)*(T**2) + 2.392*(10**2)*(T**-1) - 5.898)*4.184

n_Ce = 1
#Total_cycle_time = max_iterations
Total_cycle_time = (T_red-T_ox)/ramp_rate

#Argon sensible heating with recovery
Energy_Ar = n_Ar*Total_cycle_time*Cp_Ar*(T_red-298)*0.05

#Ceria Sensible heating
Ce_sensible_heat = n_Ce*(h_Ce(T_red)-h_Ce(T_ox))

#Reduction Enthalpy
deltadummy = np.linspace(d_ox+0.00001, d_red,100)
H_dummy = dh(deltadummy)
Reduction_enthalpy = np.trapz(H_dummy,deltadummy)

#Pump energy
E_pump = (R*T_pump/eff_pump)*np.log(1/P_red)*(n_Ar*Total_cycle_time+delta_change/2)/1000  #Do I need to consider no. of moles of argon too

#Argon seperation energy
E_ar_sep = 20*n_Ar*Total_cycle_time # 20Kj/mol-Ar

#=====================Result===================#
print('\nEnery_Ar_th:', round(Energy_Ar,3), "KJ")
print('Enery_Ar_sep:', round(E_ar_sep,3), "KJ")
print('Vacumm Pump energy:', round(E_pump,3), "KJ")
print('Ce_sensible_heat:', round(Ce_sensible_heat,3), "KJ")
print('Reduction_enthalpy:', round(Reduction_enthalpy,3), "KJ")


#==================Exporting the data===============#
df_export = pd.DataFrame({
    'Tred': T_red,
    'Tox': T_ox,
    'delta_ox': d_ox,
    'delta_red': d_red,
    'deltachange': delta_change,
    'E_Ar_th': Energy_Ar,
    'E_Ar_sep': E_ar_sep,
    'E_Pump': E_pump,
    'E_Ce_sensible_heat': Ce_sensible_heat,
    'Reduction_enthalpy': Reduction_enthalpy
},
index = [0]
)

# Export to Excel
df_export.to_excel(fr'\oxidation_Vr_16_{T_ox-273.15}_marxer_reduction_conditions.xlsx', index=False)
print("Data exported to 'iteration_results.xlsx'")