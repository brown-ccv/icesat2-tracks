import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
#sys.path.append(base_path +'modules/')
#sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

import matplotlib.pyplot as plt
%matplotlib inline

#import m_general as M
#import m_tools as MT
import numpy as np

import m_general_ph3 as M

import netCDF4
import datetime
import os
from netCDF4 import Dataset
import xarray as xr
import pandas as pd

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import imp



#import s3fs
# %%

# Python reader based on Pandas. Other reader examples available in readers.py
plot_path = mconfig['paths']['plot'] + '/tests/'


track_name= 'ATL03_20190515060436_07170312_002_02'
load_path   = base_path + 'data/data1/'

# test which beams exist:
all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
low_beams = ['gt1l',  'gt2l',  'gt3l']
high_beams = ['gt1r',  'gt2r',  'gt3r']

# %%
Gall= xr.open_dataset(load_path+'/'+track_name +'_filtered_photon_heights.nc')

# Gmask = Gall.dist *0
# for k in Gall.var():
#
#     #print( (np.isnan( Gall[k] )).sum().load().data )
#     Gmask+= Gall[k]
#
# print( (np.isnan( Gmask )).sum().load().data )
#
# #io.save_pandas_table(B3, track_name + '_filted' , load_path)
#
# # %% fake time axis
# dx = np.diff(Gmask.dist.data)[0]
# from scipy.ndimage.measurements import label
#
# nan_mask = np.isnan(Gmask)
# lfields = label(~nan_mask )
#
# for i in np.arange(1,lfields[1]+1):
#     if sum(dx*(lfields[0] ==i)) > 6e2:
#         plt.plot( nan_mask.dist[ lfields[0] ==i ] )

# %%
L=600 # meters
imp.reload(spec)

G2= dict()
for k in high_beams:
    Gall[k+'_nans'] = np.isnan(Gall[k])
    Gall[k][Gall[k+'_nans']] = 0
    Gsel = Gall[k]
    dd = (Gsel -Gsel.mean('dist'))
    dd = np.gradient(dd)
    #np.isnan(dd).sum()
    S = spec.wavenumber_spetrogram(Gsel.dist, dd, L)
    G = S.cal_spectrogram()
    S.mean_spectral_error() # add x-mean spectal error estimate to xarray

    S.parceval(add_attrs= True)


    stancil_iter = spec.create_chunk_boundaries(L, Gall[k+'_nans'].size)
    def get_stancil_nans(stancil):
        idata = Gall[k+'_nans'][stancil[0]:stancil[-1]]
        return stancil[1], idata.sum().data/idata.size

    nan_list = np.array(list(dict(map(  get_stancil_nans,  stancil_iter )).values()))

    G2[k] = G
    G2[k].coords['nan_density'] = (('x'), nan_list)


#plt.plot(G)
# %%
#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
Gplot = G.rolling(k=10, x=2, min_periods= 1, center=True).mean()

import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

font_for_print()
F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =0.8)

plt.suptitle('FFT with gaps =0', y = 0.95)
gs = GridSpec(3,3,  wspace=0.1,  hspace=0.7)#figure=fig,
clev=np.arange(0, 6, 0.1)

for pos, k, pflag in zip([gs[0, 0],gs[0, 1],gs[0, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G2[k]#.rolling(k=10, x=2, min_periods= 1, center=True).mean()

    #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])

    plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap='PuBu', vmin = clev[0], vmax = clev[-1])

    plt.gca().set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')
    plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(20,800)
    plt.title(k, loc='left')

    plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
    ax0.axhline(30, color='black', linewidth=0.5)

    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()


G3_sum = G2[k]*0
G3_sum['nan_density'] =0
for k in high_beams:
    G3_sum += G2[k]

G3_sum= G3_sum/3
G3_sum['nan_density'] =G3_sum['nan_density']/3


pos, k, pflag = gs[1, 0], 'mean', True
ax0 = F.fig.add_subplot(pos)
Gplot = G3_sum.rolling(k=10, x=2, min_periods= 1, center=True).mean()

#plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])

plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap='PuBu', vmin = clev[0], vmax = clev[-1])

plt.gca().set_yscale('log')
# plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')
plt.xlabel('Distance from the Ice Edge (km)')
plt.ylim(20,800)
plt.title(k, loc='left')

plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
ax0.axhline(30, color='black', linewidth=0.5)

if pflag:
    plt.ylabel('Wave length\n(meters)')
    plt.legend()

F.save_light(path=plot_path, name='exmpl_spec_FFT_strong')
