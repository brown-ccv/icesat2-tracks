# %%
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())
exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())


base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')
sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

import matplotlib.pyplot as plt
#matplotlib inline

#import m_general as M
#import m_tools as MT
import numpy as np

import m_general_ph3 as M
import os
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import h5py

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io

import imp
import m_spectrum_ph3 as spec
import JONSWAP_gamma



plot_path = mconfig['paths']['plot']+ 'proposal/'
# %%

f = np.arange(0.001, 0.2,0.001)
spec_power = JONSWAP_gamma.JONSWAP_default(f, 2e6, 15)
plt.plot(f, spec_power)

# %%
amps = (spec_power * np.gradient(f)) **0.5

2/f[amps.argmax()]

t = np.arange(0, 500, 0.1)
tt, ww = np.meshgrid(2* np.pi * f, t)
phi = np.random.random(len(amps))*2*np.pi

instance = amps* np.cos( ww * tt + phi )

# %% fake ice

from scipy import stats

noise = stats.norm(0).rvs(size=t.size)

data = np.zeros(len(t))
alpha1= 0.98
xx = 1.0
for i in np.arange(1,len(t)):
    xx = data[i] = alpha1  * xx +  noise[i]

data = data/ data.std()
ice_mask = abs(data) > 0.8
ice = np.where(ice_mask, 1, np.nan)

SIC = ice_mask.sum()/ice.size
print(SIC)

plt.plot(ice[0:200])
# %

ice_height  = 0.04
font_for_print()
ylims = -0.05, ice_height* 2.5
F = M.figure_axis_xy(6, 3, view_scale=0.9, container = True)
plt.suptitle("Illustration of signal decomposition", y = 1)

gs = GridSpec(3,1,  wspace=0,  hspace=0.5)#figure=fig,
ax1 = F.fig.add_subplot(gs[0, :])
ax1.tick_params(labelbottom=False)
plt.title('Total observed mean surface height ' + str(int(np.round(SIC, 1)*100)) + '% SIC', loc = 'left')

#ax1 =plt.subplot(3, 1, 1)

#plt.plot(t, instance.mean(1) * 5 , linewidth = 0.5, color='gray')

total_y = instance.mean(1) * 5 + ice * ice_height
plt.plot(t, total_y, '.', c=col.cascade2,markersize = 0.5 )
plt.fill_between(t, total_y, color=col.cascade2, alpha = 0.6)


plt.xlim(0, 300)
plt.ylim(ylims[0], ylims[1])

ax2 = F.fig.add_subplot(gs[1, :])
ax2.tick_params(labelbottom=False)#, labelleft=False)
plt.title('Sea ice freeboard without waves', loc = 'left')


plt.plot(t, ice * ice_height, '.', c=col.cascade2,markersize = 0.5 )
plt.fill_between(t, ice * ice_height, color=col.cascade2, alpha = 0.6)
plt.xlim(0, 300)
plt.ylim(ylims[0], ylims[1])


ax3 = F.fig.add_subplot(gs[2, :])
plt.title('Waves only', loc = 'left')

plt.plot(t, instance.mean(1) * 5 , '-', c=col.black,linewidth = 0.8 )

plt.xlim(0, 300)
plt.ylim(ylims[0]*1.4, ylims[1])
plt.xlabel('meters')

ax1.axhline(0, linewidth =0.7, color = col.gridcolor)
ax2.axhline(0, linewidth =0.7, color = col.gridcolor)
ax3.axhline(0, linewidth =0.7, color = col.gridcolor)

F.save_pup(path = plot_path, name = "fake_height_model")


# %% ICesat 2 track density?
N_tracks = 1387
# 91 day repeat cycle, means 91/2 revisit, because of acending and descending.
Dtime = 91 * 24 # in hours

91/2
#*60 *60 # hours
#tacks on a 91-day repetitio cycle.

# circumface at 65 N

re= np.float(mconfig['constants']['radius'])/1e3
circumface = np.cos(np.pi * 65/ 180) * 2* np.pi * re # km

dx = circumface/N_tracks # km
dt = Dtime/N_tracks

dx, dt

# study area width 40 deg lon:

area_width = circumface * 40/360

sub_area_width = circumface * 5/360

# time to pass study area once:
dt * area_width / dx / 24

sub_area_width/dx

# %%
