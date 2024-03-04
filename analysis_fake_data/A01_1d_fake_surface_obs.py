# %%
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')
sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

import matplotlib.pyplot as plt
#%matplotlib inline

#import m_general as M
#import m_tools as MT
import numpy as np

import m_general_ph3 as M
import os
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import h5py


import imp
import m_spectrum_ph3 as spec
import JONSWAP_gamma

sys.path.append(base_path +'analysis_fake_data/')
import wave_realizations as WR
font_for_pres()

# %%

f = np.arange(0.001, 0.2,0.001)
f_std =0.01
spec_power = JONSWAP_gamma.JONSWAP_default(f, 2e6, 10)
spec_power_gauss =  np.max(spec_power) * np.exp( - (  (f -f[spec_power.argmax()])/ f_std )**2 )
plt.plot(f, spec_power_gauss, label= 'gauss E= ' + str(np.trapz(spec_power_gauss, f)))
plt.plot(f, spec_power, label= 'JONSWAP E= ' + str(np.trapz(spec_power, f)))
plt.legend()


# %%


t = np.arange(0, 2000, 0.01)
# calculate the amplitudes(freq) of the gaussian spectrum
amps = WR.amplitudes_from_spectrum(spec_power_gauss, f)

# generate a realization of the spectrum
instance = WR.time_realization_from_spectrum(t, f, spec_power_gauss)


M.figure_axis_xy(5.5, 3.5, view_scale=0.9)

plt.subplot(211)
plt.title
plt.plot(f, amps, linewidth=0.5)
plt.title('Spectrum', loc='left')
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude [m]')


plt.subplot(212)
plt.plot(t, instance, label='realization')
plt.title('Realization', loc='left')
plt.legend()
plt.ylabel('height [m]')
plt.xlabel('time [s]')

WR.test_variance_conservations(f, spec_power_gauss, instance)


# %% test wave number version
imp.reload(WR)

x= np.arange(0, 1e4, 0.5)
k =(2 * np.pi * f)**2 / 9.81

amps = WR.amplitude_from_wavenumber_spectrum(spec_power_gauss, k)
instance = WR.space_realization_from_spectrum(x, k, spec_power_gauss)

M.figure_axis_xy(5.5, 3.5, view_scale=0.9)

plt.subplot(211)
plt.title
plt.plot(f, amps, linewidth=0.5)
plt.title('Spectrum', loc='left')
plt.xlabel('wavenumber [k= 2pi/lambda]')
plt.ylabel('amplitude [m]')


plt.subplot(212)
plt.plot(x, instance, label='realization')
plt.title('Realization', loc='left')
plt.legend()
plt.ylabel('height [m]')
plt.xlabel('space [m]')

WR.test_variance_conservations(k,  spec_power_gauss, instance, wave_number=True)


# %%
C= -3
k = 2* np.pi / 200
x = np.arange(0, 2000, 2)

phi = np.pi *3/2
instance = C* np.cos(k * x + phi )
plt.plot(x,  instance, '.')

A =     C * np.cos(phi)
B =   - C * np.sin(phi)

instance2 = A * np.cos(k * x) +  B * np.sin(k * x)
plt.plot(x,  instance2)

R = np.sqrt(A**2 + B**2)
phi2 = np.arctan2(-B, A)

instance3 = R* np.cos(k * x + phi2 )
plt.plot(x,  instance3, '+')


# %%

C= -3
k = 2* np.pi / 200
x = np.arange(0, 4000, 2)

phi = np.pi *3/2
instance = C* np.cos(k * x + phi )
plt.plot(x,  instance, '-')

instance2 = C* np.cos(k*1.05 * x + phi + np.pi*1.5 )
plt.plot(x,  instance2, '-')


plt.plot(x,  instance + instance2, '-k')
