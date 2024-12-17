#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:17:19 2024

@author: noahvaillant
"""



# Set Up and Imports
# =============================================================================

# Importing Important Modules
import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

# Setting Important Physical Constants. (Pi, Mass of Electron, Charge of Electron, Permeability Constant)

pi = np.pi
me = 9.10938356*(10**-31)
qe = 1.60217663 * 10**-19
gamma = qe/(2*me)
u0 = 4*pi*(10**-7)
R = 0.0775
uR = 0.0005

# We took the constants in the final f(I) equation all together for use later in the code


CF = ((4/5)**(3/2))*u0*320*(1/R)*(1/(2*pi))*(gamma)
CF1 = (1/2*pi)*gamma

# Importing the data we took for the frequency vs current for the large, medium and small coils.
# As explained in our report, all the uncertainty is on the amplitude
# Also as explained, we have two different sets of data  as our experiment set up
# Changed week-week

def load_data(a) -> np.array:
    'Inputs a raw file of frequency, amplitude and thier uncertianties and outputs'
    'an array of arrays containing unit-corrected frequency and amplitude with uncertainties'
    raw = np.loadtxt(a, comments='#', delimiter='\t', skiprows=1)
    freq = ((10**6)*raw[:,0],(10**6)*raw[:,1])
    amps = (1/np.sqrt(2))*raw[:,2],(1/np.sqrt(2))*raw[:,3]
    return freq, amps


large_freq, large_amps = load_data("ESR2BIG.txt")


med_freq, med_amps = load_data("ESR1MED.txt")

med_freq_2, med_amps_2 = load_data("ESR2MED.txt")


small_freq_1, small_amps_1 = load_data("ESR1SMALL.txt")

small_freq_2, small_amps_2 = load_data("ESR2SMALL.txt")

# Curve Fitting 
# =============================================================================

# Defining functions for the coil's field and the frequency amplitude relation
# We flipped the frequency eqution becuase the uncertianties we used were all realated to the current
# This way it works better in scipy

def B(I,r):
    return ((4/5)**(3/2))*u0*320*(1/r)*I

def I(f,g):
    return (1/CF)*f*g

# Putting the uncertainty of each measurement with an uncertainty into one to be used in curve fit
# Neglected frequency for reasons explained in the report

def unc_mod(amps: np.array) -> np.array:
    return np.sqrt((B(amps[0],1)/R**2)*(uR)**2+((B(1,1))**2*(amps[1])**2))




#g value and uncertianty from raw data
def fitting(raw) -> (float, float):
    freq, amps = load_data(raw)
    u = unc_mod(amps)
    print(u)
    g, pcov = sp.curve_fit(I,  freq[0], amps[0], sigma = u)
    return 1/g, np.diag(pcov)/g**2

large_g, unc_l = fitting("ESR2BIG.txt")

def l_I(f):
   return (1/CF)*1/large_g*f

med_1_g, unc_m_1 = fitting("ESR1MED.txt")
def m_I_1(f):
    return (1/CF)*1/med_1_g*f
med_2_g, unc_m_2 = fitting("ESR2MED.txt")
def m_I_2(f):
    return (1/CF)*1/med_2_g*f
    
small_1_g, unc_s_1 = fitting("ESR1SMALL.txt")
def s_I_1(f):
    return (1/CF)*1/small_1_g*f
small_2_g, unc_s_2 = fitting("ESR2SMALL.txt")
def s_I_2(f):
    return (1/CF)*1/small_2_g*f
    
# Curve Fitting the Total Data for Day 2

day_2_amps = np.append(large_amps[0],np.append(med_amps_2[0],small_amps_2[0]))
day_2_freq = np.append(large_freq[0], np.append(med_freq_2[0], small_freq_2[0]))
day_2_unc = np.append(unc_mod(large_amps), np.append(unc_mod(med_amps_2),unc_mod(small_amps_2) ))

day_2_g, pcov_day_2 = sp.curve_fit(I,  day_2_freq, day_2_amps, sigma = day_2_unc)
g_unc_day_2 = np.sqrt(np.diag(pcov_day_2))/(day_2_g)**2

def I_day_2(f):
    return (1/CF)*f*day_2_g

print('g list, large, small(1,2), med(1,2), day 2', '\n',
      'large', (large_g[0], unc_l[0]),'\n','Medium', (med_1_g[0], unc_m_1[0]),(med_2_g[0], unc_m_2[0]), '\n', 'small', (small_1_g[0],unc_s_1[0]) ,(small_2_g[0], unc_s_1[0]), '\n', 'day 2', (1/day_2_g[0] ,g_unc_day_2[0]))

print('gamma list, large, small(1,2), med(1,2), day 2', '\n',
      'large', (gamma*1/large_g[0], gamma*unc_l[0]),'\n','Medium', (gamma*1/med_1_g[0], gamma*unc_m_1[0]), (gamma*1/med_2_g[0], gamma*unc_m_2[0]), '\n', 'small', (gamma*1/small_1_g[0], gamma*unc_s_1[0]) , (gamma*1/small_2_g[0], gamma*unc_s_1[0]), '\n', 'day 2', (gamma*1/day_2_g[0] ,gamma*g_unc_day_2[0]))


# Plotting
# =============================================================================



def plot(a) -> plt.plot:
    'Plotting each data set against its fit, including errorbars and '
    amps, freq = load_data(a)
    g, unc = fitting(a)
    plt.figure(figsize=(10,6))
    plt.scatter(amps[0],freq[0], label= 'Data')
    plt.errorbar(amps[0], freq[0], xerr=amps[1], yerr=freq[1], fmt='r|', label='Uncertainty')
    plt.plot((CF)*(g)*freq[0], freq[0], label='Curve Fit')
    
    plt.title(a)
    plt.xlabel('Amplitude (A)')
    plt.ylabel('Frequency (Hz)')
    plt.grid()
    plt.legend()
    plt.show()

plot("ESR2BIG.txt")

plot("ESR1MED.txt")
plot("ESR2MED.txt")

plot("ESR1SMALL.txt")
plot("ESR2SMALL.txt")


# Day 2 frequency all together
plt.figure(figsize=(10,6))
plt.scatter(large_amps[0],large_freq[0], label='Large Coil Data')
plt.errorbar(large_amps[0], large_freq[0], xerr=large_amps[1], yerr=large_freq[1], fmt='r|')

plt.scatter (med_amps_2[0],med_freq_2[0], label='Medium Coil Data')
plt.errorbar(med_amps_2[0],med_freq_2[0], xerr=med_amps_2[1], yerr=med_freq_2[1], fmt='r|')

plt.scatter (small_amps_2[0],small_freq_2[0], label='Small Coil Data')
plt.errorbar(small_amps_2[0],small_freq_2[0], xerr=small_amps_2[1], yerr=small_freq_2[1], fmt='none', ecolor='red',label='Uncertainty' )

plt.plot( I_day_2(day_2_freq), day_2_freq, label='Curve Fit to Large Coil Data')

plt.title("All Day 2 Data Combined")
plt.xlabel('A')
plt.ylabel('Hz')
plt.legend()
plt.grid()
plt.show()

# Residuals
# =============================================================================


def plot_residual(raw):
    freq, amps = load_data(raw)
    g, unc = fitting(raw)
    residuals = amps[0] - (1/CF)*(1/g)*freq[0]
    plt.scatter(freq[0], residuals, label='Difference of Predicted and Measured Amplitude')
    plt.errorbar(freq[0], residuals, yerr=amps[1], fmt='none', ecolor='red', label='Error on Amplitude')
    plt.title('Large Coil Resiuals')
    plt.ylabel('A')
    plt.xlabel('Hz')
    plt.legend()
    plt.grid()
    plt.show()

plot_residual('ESR2BIG.txt')

plot_residual('ESR1MED.txt')
plot_residual('ESR2MED.txt')
    
plot_residual('ESR1SMALL.txt')
plot_residual('ESR2SMALL.txt')



# Calculating Chi-Sqaured Values
# =============================================================================

def chi_squared(yprediction, ydata, err):
    return sum((yprediction-ydata)**2/err**2)


def chi_red(yprediction, ydata, err, npram):
    return chi_squared(yprediction, ydata, err)/(len(ydata)-npram)


chi_l = chi_red(l_I(large_freq[0]), large_amps[0] , large_amps[1], 1)
chi_m_1 = chi_red(m_I_1(med_freq[0]), med_amps[0] , med_amps[1], 1)
chi_m_2 = chi_red(m_I_2(med_freq_2[0]), med_amps_2[0] , med_amps_2[1], 1)
chi_s_1 = chi_red(s_I_1(small_freq_1[0]), small_amps_1[0] , small_amps_1[1], 1)
chi_s_2 = chi_red(s_I_2(small_freq_2[0]), small_amps_2[0] , small_amps_2[1], 1)


print('chi list', chi_l,chi_m_1,chi_m_2,chi_s_1,chi_s_2)
