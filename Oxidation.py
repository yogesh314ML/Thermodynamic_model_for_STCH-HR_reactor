#Code for H20 splitting with ceria oxidation reaction

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#============Input Parameters============#
N_cycles = 6
T_ox = 950 + 273.15 #K
T_red = 1500 + 273.15 #K
R = 8.314 #J/Kmol
n_H20 = 0.05 #mol/min  Staring flow
T_H20_initial = 150+273.15 #K
n_Ce = 1 #mol
T_amb = 25.01+273.15 #K
T_H2_initial = T_H20_initial
Y_H2_initial = 0
V_DIHE_V_r = 4

#============Thermochemical Properties ============#

#Formation of steam equilibrium constant Range 800 K to 1800 K
def logKvsT(T):
    return 12992.526780195172/T - 2.9382903998639245

#Partial molar enthapy h(KJ/mol) vs delta
def dh(x):
    return 395 - 31.4*np.log10(x)  # bulfins correlation

#Partial molar entropy s(J/K/mol) vs delta
def ds(x):
    return 160 + 2.9*8.314*(np.log(0.3448275862 - x) - np.log(x))  # bulfins correlation

#defining gibbs for gases
def dgo1(di,T,n_H20, df,Y_H2):
    return R*T*(np.log((n_H20-(df-di))/((Y_H2+df+0.00000000001-di)*10**(logKvsT(T)))))

#defining gibbs for solid
def dgo2(di,T):
    return -dh(di)*1000 + T*ds(di)

#defining equating function
def f(di,T,nH20,df,Y_H2):
    return (dgo1(di,T,nH20,df,Y_H2) - dgo2(di,T))**2

#defining enthapy of MO and gases
#Temp range 298.15 K to 2000 K
def h_Ce(T):
    return (16.761*(10**-3)*T + 1.108*(10**-6)*(T**2) + 2.392*(10**2)*(T**-1) - 5.898)*4.184

#Temp range 298.15 K to 3000 K
def h_H20(T):
    if(298.15<T<373.15):
        return (18.041*(10**-3)*T - 5.379)*4.184
    elif(373.15<=T<3000):
        return (7.272*(10**-3)*T + 1.246*(10**-6)*(T**2) + 0.094*(10**2)*(T**-1) + 8.201)*4.184

#Temp range 298.15 K to 3000 K
def h_H2(T):
    return (6.456*(10**-3)*T + 0.419*(10**-6)*(T**2) - 0.165*(10**2)*(T**-1) - 1.907)*4.184

#Temp range 298.15 K to 2000 K
def h_O2(T):
    return (7.230*(10**-3)*T + 0.503*(10**-6)*(T**2) + 0.452*(10**2)*(T**-1) - 2.352)*4.184

#Temp range 298.15 K to 2000 K
def h_diss_H20(T):
    if (298.15<T<373.15):
        return -(-70.611 + 7.970*(10**-3)*T - 0.671*(10**-6)*(T**2) - 6.100*(T**-1))*4.184
    elif(373.15<=T<2000):
        return -(-57.031 - 2.799*(10**-3)*T + 0.576*(10**-6)*(T**2) + 3.300*(T**-1))*4.184

#============Start of the cycle============#
for i in range(N_cycles):
    T = [] #K
    df = 0.0479025   #reduction delta initialization
    T_H20 = T_H20_initial #K
    T_H2 = T_H2_initial #K

    if (Y_H2_initial != 0):
        n_H20 = n_H20 + (n_H20 * (h_H20(T_ox)-h_H20(T_H20_initial)) + Y_H2_initial * (h_H2(T_ox)-h_H2(T_H2_initial)))/(h_H20(T_H20_initial)-h_H20(T_amb))

    # Iterative loop with tolerance
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0
    deltachange_array = []  # Initialize array for differences
    nH20_array = [n_H20] #flow over time
    Y_H2 = Y_H2_initial
    T_equil = T_red

    while iteration < max_iterations:
        #Calculating temperature profile
        T_prev = T_equil
        T_diff = 1
        while T_diff > 1e-04:
            # Calculate new delta (chemical equilibrium)
            dlta = minimize(f,0.03,bounds=[(max(0.00001,df - nH20_array[iteration] + 1e-08),df)],args=(T_equil,nH20_array[iteration],df,Y_H2))
            new_delta = dlta.x[0]

            # Calculate and append difference
            difference = df - new_delta


            deltaitof = np.linspace(new_delta, df, 100)
            H_deltaitof = dh(deltaitof)

            T_temperory = T_equil
            #==================Thermal equilibrium==================#
            h_equil1 = nH20_array[iteration] * h_H20(T_H20) + Y_H2 * h_H2(T_H2) + n_Ce * h_Ce(T_prev)
            def h_equil2(T):
                return ((nH20_array[iteration]-difference)*h_H20(T) + (Y_H2+difference)*h_H2(T) + n_Ce * h_Ce(T) + h_diss_H20(T)*difference - np.trapz(H_deltaitof,deltaitof) - h_equil1)**2

            T_equil_temperory = minimize(h_equil2, T_prev, bounds=[(373.16, T_red)])
            T_equil = T_equil_temperory.x[0]

            T_diff = np.abs(T_equil - T_temperory)

        T.append(T_equil)
        deltachange_array.append(difference)
        Y_H2 = Y_H2_initial + sum(deltachange_array)
        n_H20_temp = nH20_array[iteration] - difference

        #===================Calculating maximum amount of possible steam generation=================#
        n_new = (n_H20_temp * (h_H20(T_equil)-h_H20(T_H20_initial)) + Y_H2 * (h_H2(T_equil)-h_H2(T_H2_initial)))/(h_H20(T_H20_initial)-h_H20(T_amb))

        if (n_H20_temp + n_new)*(h_H20(T_ox)-h_H20(T_H20_initial)) + Y_H2*(h_H2(T_ox)-h_H2(T_H2_initial)) <= n_Ce*(h_Ce(T_equil)-h_Ce(T_ox)):
            nH20_array.append(n_H20_temp + n_new)
        else:
            #============Determining the fraction of liquid water to be injected as compared to maximum==============#
            alpha = 3*(T_equil-T_ox)/T_ox  # relaxation factor

            n_new = alpha*n_new

            h_H20_equil1 = n_H20_temp*h_H20(T_equil) + Y_H2 * h_H2(T_equil) + n_new*h_H20(T_amb)

            def h_H20_equil2(T):
                return (n_H20_temp*h_H20(T) + n_new*h_H20(T) + Y_H2*h_H2(T) - h_H20_equil1)**2

            T_H20_temperory = minimize(h_H20_equil2,T_equil,bounds=[(373.16, T_equil)])
            T_H20 = T_H2 = T_H20_temperory.x[0]
            nH20_array.append(n_H20_temp + n_new)

        df = new_delta

        # Check convergence
        if abs(T_equil-T_ox) < 0.1 :
            iteration += 1
            print(f"Converged after {iteration} iterations")
            print(f"Final Delta: {round(new_delta, 5)}")
            break

        # Update
        iteration += 1

        print(f"Iteration {iteration}: Difference = {round(difference, 7)}")
    if (N_cycles > 1 or Y_H2_initial !=0):
        n_H20 = nH20_array[-1] * (1/(1+V_DIHE_V_r)) * (V_DIHE_V_r/(1+V_DIHE_V_r))
        Y_H2_initial = Y_H2 * (1 / (1 + V_DIHE_V_r)) * (V_DIHE_V_r / (1 + V_DIHE_V_r))

if iteration >= max_iterations:
    print(f"Maximum iterations ({max_iterations}) reached without convergence")

delta_ox = df

#================ Printing and Plotting Final results ======================#
print(f"\nTotal iterations: {iteration}")
print(f"Differences: {[round(d, 7) for d in deltachange_array]}")
print(f"Delta ox: {round(delta_ox, 7)}")
print(f"Total ox: {round(sum(deltachange_array), 7)}")
print("nH20", nH20_array)

#Time array
Time = np.arange(1, iteration + 1)
Iteration_ = np.arange(1,iteration + 1)


#Including zero at first position for ploting
deltachange_array.insert(0, 0)
Time = np.insert(Time, 0, 0)
Iteration_ = np.insert(Iteration_, 0, 1)
T = np.insert(T, 0, T_red)

#Plotting the thermodynamic limits on rates
fig, ax1 = plt.subplots()
ax1.plot(Time, deltachange_array, 'b-', label='oxidation extent')
ax1.set_xlabel('Time [mins]')
ax1.set_ylabel('$\Delta\delta$', color='b')

ax2 = ax1.twinx()
ax2.plot(Time, T, 'r-', label='Temperature profile')
ax2.set_ylabel('Temperature [K]', color='r')

plt.show()

delta_ox_ = sum(deltachange_array) + df - np.cumsum(deltachange_array)
conv = [a*100/b for a,b in zip(deltachange_array,nH20_array)]

cumulative_conv = sum(deltachange_array)*100/(nH20_array[-1] + sum(deltachange_array))
Rate = [v*1000000/172.115/60 for v in deltachange_array]


#Calculating oxidation energy
n_H20_starting = n_H20
if (N_cycles > 1 or Y_H2_initial !=0):
    n_H20 = n_H20 + (n_H20 * (h_H20(T_ox) - h_H20(T_H20_initial)) + Y_H2_initial * (h_H2(T_ox) - h_H2(T_H2_initial))) / (h_H20(T_H20_initial) - h_H20(T_amb))

E_ox = nH20_array[-1]*h_H20(T_equil) + Y_H2*h_H2(T_equil) - ((nH20_array[-1]+Y_H2)-n_H20-Y_H2_initial)*h_H20(T_amb) - n_H20*h_H20(T_H20_initial) - Y_H2_initial*h_H2(T_H2_initial) - n_Ce*(h_Ce(T_red) - h_Ce(T_equil))

print("Oxidation energy:", E_ox)
print("cumulative conv:",cumulative_conv)
print("Tox:", T_equil)



#=========================== Saving data ==================#
df_export = pd.DataFrame({
    'Time': Time,
    'deltachange': deltachange_array,
    'delta_ox': delta_ox_,
    'conv': conv,
    'Rate mumole/g/s': Rate,
    'Temp':T,
    'n_H20':nH20_array,
    'conv_cumulative': cumulative_conv,
    'Starting n_H20': n_H20_starting,
    'Starting Y_H2': Y_H2_initial,
    'Total moles of product gases': (nH20_array[-1]+Y_H2),
    'E_ox': E_ox,
})

# Export to Excel
df_export.to_excel(fr'oxidation_Vr_{V_DIHE_V_r}_{T_ox-273.15}.xlsx', index=False)
print("Data exported to 'iteration_results.xlsx'")