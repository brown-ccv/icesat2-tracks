import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'modules/')
sys.path.append(base_path +'modules/ICEsat2_SI_tools/')

import matplotlib.pyplot as plt
%matplotlib inline

#import m_general as M
#import m_tools as MT
import numpy as np

import m_general_ph3 as M
import spicke_remover

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
import copy

import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec


#import s3fs
# %%

# Python reader based on Pandas. Other reader examples available in readers.py
plot_path = mconfig['paths']['plot'] + '/tests/'


track_name= 'ATL03_20190515060436_07170312_002_02'
load_path   = base_path + 'data/data1/'

#track_name= 'ATL03_20191215230028_12220512_004_01'
track_name= 'ATL03_20210414174223_03191110_004_01'
load_path   = base_path + 'data/data4/'

# test which beams exist:
all_beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
low_beams = ['gt1l',  'gt2l',  'gt3l']
high_beams = ['gt1r',  'gt2r',  'gt3r']

# %%

#Gall= xr.open_dataset(load_path+'/'+track_name +'_filtered_photon_heights.nc')
imp.reload(io)
Gfilt = io.load_pandas_table_dict(track_name + '_B01_corrected', load_path) # rhis is the rar photon data
Gd = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #




# %% test LS with an even grid where missing values are set to 0
imp.reload(spec)

Gi =Gd[high_beams[0]]

#Gi = Gi[~np.isnan(Gi['heights_c_weighted_mean'])].sort_values('dist')

# derive spectal limits
# Longest deserved period:
T_max = 40 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gi['dist'])
dx = np.diff(x).mean()
min_datapoint =  1/k_0/dx

Lpoints = int(np.round(min_datapoint) * 10)
Lmeters =Lpoints  * dx

#plt.plot(np.diff(np.array(Gi['dist'])))
print('L number of gridpoint:', Lpoints)
print('L length in meters:', Lmeters)
print('approx number windows', 2* Gi['dist'].iloc[-1] /Lmeters-1   )


S_pwelch_k, S_pwelch_dk = spec.calc_freq_LS( x, Lpoints, dx= dx, method='fft', minimum_frequency=None, maximum_frequency=None, samples_per_peak=0.005)
S_pwelch_k.shape
hkey= 'heights_c_weighted_mean'

imp.reload(spicke_remover)
font_for_pres()

G_LS= dict()
G_fft= dict()
for k in high_beams:
    #k = high_beams[1]
    x= Gd[k]['dist']
    xlims = x.iloc[0], x.iloc[-1]
    dd  = np.copy(Gd[k][hkey])



    M.figure_axis_xy(6, 3)
    plt.subplot(1, 2, 1)
    plt.plot(x, dd, 'gray', label='displacement (m) ')

    # compute slope spectra !!
    dd = np.gradient(dd)
    dd, _ = spicke_remover.spicke_remover(dd, spreed=5, verbose=False)
    dd_nans = (np.isnan(dd) ) + (Gd[k]['N_photos'] <= 5)

    dd_no_nans= dd[~dd_nans]
    x_no_nans = x[~dd_nans]


    # outlyers = abs(dd) >  dd.std() *5
    # dd= dd[~outlyers]
    # x= x[~outlyers]

    plt.plot(x_no_nans, dd_no_nans, 'black', label='slope (m/m)')
    plt.legend()

    print('LS')
    S = spec.wavenumber_spectrogram_LS( np.array(x_no_nans), np.array(dd_no_nans), Lmeters, waven_method = S_pwelch_k[1:]  ,  ov=None, window=None, kjumps=2)
    G_ls_i = S.cal_spectrogram(xlims= xlims)
    S.mean_spectral_error() # add x-mean spectal error estimate to xarray
    S.parceval(add_attrs= True)

    #G['x'] = G.x * dx
    def get_stancil_nans(stancil):
        x_mask= (stancil[0] < x) & (x <= stancil[-1])
        idata = Gd[k]['N_photos'][x_mask]
        return stancil[1], idata.sum()

    photon_list = np.array(list(dict(map(  get_stancil_nans,  copy.copy(S.stancil_iter) )).values()))

    G_LS[k] = G_ls_i
    G_LS[k].coords['N_photons'] = (('x'), photon_list)
    plt.subplot(1, 2, 2)
    plt.plot(G_ls_i.k, G_ls_i.mean('x'), 'k', label='mean LS')

    # standard FFT
    print('FFT')
    dd[dd_nans]    = 0
    S = spec.wavenumber_spectrogram(x, dd, Lpoints)
    G = S.cal_spectrogram()
    S.mean_spectral_error() # add x-mean spectal error estimate to xarray
    S.parceval(add_attrs= True)


    stancil_iter = spec.create_chunk_boundaries(int(Lpoints), dd_nans.size)
    def get_stancil_nans(stancil):
        idata = dd_nans[stancil[0]:stancil[-1]]
        return stancil[1], idata.size - idata.sum()

    N_list = np.array(list(dict(map(  get_stancil_nans,  stancil_iter )).values()))

    G_fft[k] = G
    G_fft[k].coords['N_per_stancil'] = (('x'), N_list)
    G_fft[k].coords['x'] = G_fft[k].coords['x'] * dx

    plt.plot(G.k, G.mean('x'), 'darkblue', label='mean FFT')

    plt.legend()
    plt.show()


    #print(np.isinf(G).sum().data)


# %%
font_for_pres()

plt.plot(G.k, G.mean('x'))

def dict_weighted_mean(Gdict, weight_key):
    """
    returns the weighted meean of a dict of xarray, data_arrays
    weight_key must be in the xr.DataArrays
    """
    #Gdict = G_LS
    # weight_key='N_per_stancil'

    akey = list( Gdict.keys() )[0]
    GSUM = Gdict[akey].copy()
    GSUM.data     = np.zeros(GSUM.shape)
    N_per_stancil = GSUM.N_per_stancil * 0
    N_photons     = np.zeros(GSUM.N_per_stancil.size)

    counter= 0
    for k,I in Gdict.items():
        GSUM                += I * I[weight_key] #.sel(x=GSUM.x)
        N_per_stancil       += I[weight_key]
        if 'N_photons' in GSUM.coords:
            N_photons    += I['N_photons']
        counter+=1

    GSUM             = GSUM  / N_per_stancil /counter

    if 'N_photons' in GSUM.coords:
        GSUM.coords['N_photons'] = (('x'), N_photons )

    return GSUM

G_LS_wmean = dict_weighted_mean(G_LS, 'N_per_stancil')
G_fft_wmean = dict_weighted_mean(G_fft, 'N_per_stancil')


# %%
def plot_wavenumber_spectrogram(ax, Gi, clev, title= None, plot_photon_density=True ):

    x_lambda= 1/Gi.k
    plt.pcolormesh(Gi.x/1e3, x_lambda , Gi, cmap=plt.cm.ocean_r , vmin = clev[0], vmax = clev[-1])

    ax.set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')

    if plot_photon_density:
        plt.plot(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10 , c='black', linewidth= 0.8, label='NAN-density' )
        ax.axhline(30, color='black', linewidth=0.5)

    #plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(x_lambda[-1], x_lambda[0])
    plt.title(title, loc='left')

#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
Gmean = G_LS_wmean.rolling(k=10, x=2, min_periods= 1, center=True).mean()
Gmean = Gmean.where(~np.isnan(Gmean), 0)

k_max_range = Gmean.k[Gmean.mean('x').argmax().data].data* 0.75, Gmean.k[Gmean.mean('x').argmax().data].data* 1, Gmean.k[Gmean.mean('x').argmax().data].data* 1.25



font_for_print()
F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =1)

plt.suptitle('LS and FFT Slope Spectrograms\n' + track_name, y = 0.98)
gs = GridSpec(3,3,  wspace=0.2,  hspace=.4)#figure=fig,
#clev=np.arange(0, 6, 0.1)*3
clev = M.clevels( Gmean.data, 31)* 1
xlims= Gmean.x[0]/1e3, Gmean.x[-1]/1e3

for pos, k, pflag in zip([gs[0, 0],gs[0, 1],gs[0, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G_LS[k]#.rolling(k=10, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    # #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])
    #
    plot_wavenumber_spectrogram(ax0, Gplot,  clev, title =k,  plot_photon_density=True )
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()

pos, k, pflag = gs[1, 0], 'weighted mean', True
ax0 = F.fig.add_subplot(pos)
Gplot = G_LS_wmean.rolling(k=5, x=2, min_periods= 1, center=True).median()


plot_wavenumber_spectrogram(ax0, Gplot,  clev, title =k, plot_photon_density= True)
plt.xlim(xlims)

# plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
# ax0.axhline(30, color='black', linewidth=0.5)

ax0.axhline(1/k_max_range[0], color='red', linestyle= '--', linewidth= 0.5)
ax0.axhline(1/k_max_range[1], color='red', linestyle= '-', linewidth= 0.5)
ax0.axhline(1/k_max_range[2], color='red', linestyle= '--', linewidth= 0.5)


if pflag:
    plt.ylabel('Wave length\n(meters)')
    plt.legend()


pos = gs[1, 1]
ax0 = F.fig.add_subplot(pos)
plt.title('Photons density ($m^{-1}$)', loc='left')

for k,I in G_LS.items():
    plt.plot(Gplot.x/1e3, I.N_photons/Lmeters, label=k, linewidth=0.8)
plt.plot(Gplot.x/1e3, G_LS_wmean.N_photons/3/Lmeters , c='black', label='ave Photons' , linewidth=0.8)
plt.xlim(xlims)
plt.xlabel('Distance from the Ice Edge (km)')

pos = gs[1, 2]
ax0 = F.fig.add_subplot(pos)

ax0.set_yscale('log')

plt.title('Peak Spectal Power', loc='left')

for k,I in G_LS.items():
    plt.scatter(I.x.data/1e3, I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k').data *1e3 ,  s=0.5, marker='.', color='red', alpha= 0.3)

for k,I in G_fft.items():
    I= I[:,I.N_per_stancil >=  I.N_per_stancil.max().data*0.9]
    plt.scatter(I.x/1e3, I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3 ,  s=0.5, marker='.', c='blue', alpha= 0.3)


Gplot= G_fft_wmean[:,G_fft_wmean.N_per_stancil >=  G_fft_wmean.N_per_stancil.max().data*0.9]
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3, '.', markersize=1.5 , c='blue', label= 'FFT')

Gplot= G_LS_wmean
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3 , '.' , markersize=1.5, c='red', label= 'LS')

plt.ylabel('1e-3 $(m)^2~m$')
plt.legend()
#plt.ylim(Gplot.min()*1.4, Gplot.max()*1.4 )
plt.xlim(xlims)

F.save_light(path=plot_path, name=track_name +'_exmpl_spec_LS_strong_L'+str(Lmeters))
