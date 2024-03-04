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

#Gall= xr.open_dataset(load_path+'/'+track_name +'_filtered_photon_heights.nc')
imp.reload(io)
Gfilt = io.load_pandas_table_dict(track_name + '_B01_corrected', load_path)
Gd = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)

# %%
Gi =Gd[high_beams[0]]
Gi = Gi[~np.isnan(Gi['heights_c_weighted_mean'])].sort_values('dist')
#np.isnan(Gi['dist']).sum()
Gi = Gi[(Gi['dist'] > 140000) & (Gi['dist'] < 160001)]

# %%

plt.plot( Gi['dist'], Gi['heights_c_weighted_mean'], '-')
#plt.xlim(140000, 145000)
plt.ylim(-1, 1)

#np.diff(np.array(Gi['dist']))
# %% test and compare to FFT
from scipy.signal import detrend
y =detrend(np.array(Gi['heights_c_weighted_mean']) )
f_fft2, df2 = spec.calc_freq_fft( np.array(Gi['dist']), y.size)
spec_fft2 = spec.calc_spectrum_fft(y, df2, np.array(Gi['dist']).size)
f_fft2_regular, spec_fft2_regular = f_fft2, spec_fft2


# %% simple spectra
from astropy.timeseries import LombScargle
ls = LombScargle( np.array(Gi['dist']) , np.array(Gi['heights_c_weighted_mean']) )

ls_auto_power = ls.power(f_fft2 , normalization='psd')
#ls_auto_f , ls_auto_power = ls.autopower(normalization='psd', samples_per_peak=0.1)

plt.plot(f_fft2 , spec.LS_power_to_PSD(ls_auto_power, y.size, df2) )

plt.plot(f_fft2, spec_fft2 )
plt.xlim(0, 0.01)

# %% tests
# maximum to be resolved wavenumber
T = 40 #sec
k_0 = (2 * np.pi/ T)**2 / 9.81

1/k_0
x= np.array(Gi.index)
dt = np.diff(x).mean()
#dt = np.diff(np.array(Gi['dist'])).mean()
min_datapoint =  1/k_0/dt
L = int(np.round(min_datapoint) * 10)
S_pwelch_fft = spec.wavenumber_spectrogram(x , np.array(Gi['heights_c_weighted_mean']), L)
G_pwelch_fft = S_pwelch_fft.cal_spectrogram()


S_pwelch_ls = spec.wavenumber_spectrogram(x, np.array(Gi['heights_c_weighted_mean']), L)
G_pwelch_fft = S_pwelch_fft.cal_spectrogram()

# %%
imp.reload(spec)
y =detrend(np.array(Gi['heights_c_weighted_mean']) )


S_pwelch_k, S_pwelch_dk = spec.calc_freq_LS( x, L, method='fftX2', minimum_frequency=None, maximum_frequency=None, samples_per_peak=0.5)

S_pwelch_ls = spec.wavenumber_spectrogram_LS_even(x, y, L, waven_method = S_pwelch_k , dy=None ,  ov=None, window=None, kjumps=2)
G_pwelch_ls = S_pwelch_ls.cal_spectrogram()
S_pwelch_ls.parceval()
plt.plot(G_pwelch_ls.k , G_pwelch_ls.mean('x') )
plt.plot(G_pwelch_fft.k , G_pwelch_fft.mean('x') , 'k')


plt.xlim(0, 0.02)


# %% test LS with an even grid where missing values are set to 0
imp.reload(spec)

Gi =Gd[high_beams[0]]
#Gi = Gi[~np.isnan(Gi['heights_c_weighted_mean'])].sort_values('dist')
# derive spectal limits
T_max = 50 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gi.index)
dx = np.diff(x).mean()
min_datapoint =  1/k_0/dx
L = int(np.round(min_datapoint) * 20)

plt.plot(np.diff(np.array(Gi['dist'])))

print(L)
print(L*dx)
print(Gi['dist'].iloc[-1] /L  )

S_pwelch_k, S_pwelch_dk = spec.calc_freq_LS( x, L, dx= dx, method='fftX2', minimum_frequency=None, maximum_frequency=None, samples_per_peak=0.01)
hkey= 'heights_c_weighted_mean'
G2= dict()
for k in high_beams:

    x= Gd[k].index
    dd  =Gd[k][hkey]
    dd_nans = np.isnan(dd)
    dd= dd[~dd_nans]
    x= x[~dd_nans]
    dd = (dd -dd.mean())
    #dd = np.gradient(dd)
    #np.isnan(dd).sum()
    S = spec.wavenumber_spectrogram_LS_even( np.array(x), np.array(dd), L, waven_method = S_pwelch_k , dy=None ,  ov=None, window=None, kjumps=2)
    G = S.cal_spectrogram()
    S.mean_spectral_error() # add x-mean spectal error estimate to xarray
    S.parceval(add_attrs= True)

    G['x'] = G.x * dx

    stancil_iter = spec.create_chunk_boundaries(L, np.array(dd).size)
    def get_stancil_nans(stancil):
        idata = Gd[k]['N_photos'][stancil[0]:stancil[-1]]
        return stancil[1], idata.sum()

    photon_list = np.array(list(dict(map(  get_stancil_nans,  stancil_iter )).values()))

    G2[k] = G
    G2[k].coords['N_photos'] = (('x'), photon_list)



# %%
font_for_pres()

plt.plot(G.k, G.mean('x'))
# %%
#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
Gplot = G.rolling(k=10, x=2, min_periods= 1, center=True).mean()

import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

font_for_print()
F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =0.8)

plt.suptitle('LS on regular grid', y = 0.95)
gs = GridSpec(3,3,  wspace=0.1,  hspace=0.7)#figure=fig,
#clev=np.arange(0, 6, 0.1)*3
clev = M.clevels(Gplot.data, 31)* 0.8


for pos, k, pflag in zip([gs[0, 0],gs[0, 1],gs[0, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G2[k]#.rolling(k=10, x=2, min_periods= 1, center=True).mean()

    #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])

    plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap=plt.cm.ocean_r, vmin = clev[0], vmax = clev[-1])

    plt.gca().set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')
    plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(20,L*dx/2)
    plt.title(k, loc='left')

    # plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
    # ax0.axhline(30, color='black', linewidth=0.5)

    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()


G3_sum = G2[k]*0
G3_sum['N_photos'] = G3_sum['N_photos']*0
G3_sum= G3_sum.sel(x=slice(G3_sum.x[0],G3_sum.x[-3]) )

for k in high_beams:
    G3_sum += G2[k].sel(x=G3_sum.x)
    G3_sum['N_photos'] += G2[k].sel(x=G3_sum.x)['N_photos']

G3_sum= G3_sum/3
G3_sum['N_photos'] =G3_sum['N_photos']/3


pos, k, pflag = gs[1, 0], 'mean', True
ax0 = F.fig.add_subplot(pos)
Gplot = G3_sum.rolling(k=5, x=2, min_periods= 1, center=True).mean()

#plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])

plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap=plt.cm.ocean_r, vmin = clev[0], vmax = clev[-1])

plt.gca().set_yscale('log')
# plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')
plt.xlabel('Distance from the Ice Edge (km)')
plt.ylim(20,L*dx/2)
plt.title(k, loc='left')

# plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
# ax0.axhline(30, color='black', linewidth=0.5)

if pflag:
    plt.ylabel('Wave length\n(meters)')
    plt.legend()


pos = gs[1, 1]
ax0 = F.fig.add_subplot(pos)
plt.title('ave photon per chunk')
plt.plot(Gplot.x/1e3, Gplot.N_photos , c='black')


pos = gs[1, 2]
ax0 = F.fig.add_subplot(pos)

plt.title('spectal power decay')
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(1/500, 1/100)).integrate('k') , c='black')

F.save_light(path=plot_path, name='exmpl_spec_LS_strong_L'+str(L*dx))


# %% rar photon data

Gi = Gfilt[high_beams[0]]
Gi = Gi[(Gi['dist'] > 140000) & (Gi['dist'] < 160001)]

plt.plot( Gi['dist'], Gi['heights_c_weighted_mean'], '.', alpha= 0.2)
#plt.xlim(140000, 145000)
plt.ylim(-1, 1)

#np.diff(np.array(Gi['dist']))
# %%
from scipy.signal import detrend
y =detrend(np.array(Gi['heights']) )
f_fft2, df2 =   spec.calc_freq_fft( np.array(Gi['dist']), y.size)
f_fft2      =   f_fft2[1:]
spec_fft2   =   spec.calc_spectrum_fft(y, df2, np.array(Gi['dist']).size)



# %% simple spectra
from astropy.timeseries import LombScargle
ls = LombScargle( np.array(Gi['dist']) , np.array(Gi['heights']) )

ls_power = ls.power(f_fft2_regular[1::1] , normalization='psd')
plt.plot(f_fft2_regular[1::1] , (spec.LS_power_to_PSD(ls_power, y.size, df2)) , 'r', zorder=12)

# ls_auto_f , ls_auto_power = ls.autopower(normalization='psd', samples_per_peak=0.1)
# plt.plot(ls_auto_f , spec.LS_power_to_PSD(ls_auto_power, y.size, df2) , 'r', zorder=12)

#plt.plot(f_fft2, spec_fft2 )
plt.plot(f_fft2_regular, spec_fft2_regular, 'k-' )

plt.xlim(0, 0.01)
plt.ylim(0, 80)













# %% ueven version

font_for_pres()

Gi = Gfilt[high_beams[0]]
dist_diffs= Gi.sort_values('dist')['dist'].diff()
dist_diffs.median()


imp.reload(spec)

# derive spectal limits
T_max = 30 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81

x = np.array(Gi.sort_values('dist')['dist'])
xdiff = np.diff(x)
xdiff = xdiff[xdiff< .1]

#plt.hist( xdiff, bins= 40 )
dx =np.mean(xdiff)*1e2
#dx = 0.01#dist_diffs.median() #np.diff(np.array(Gi['dist'])).mean()
1/dx
min_datapoint =  1/k_0/dx
longest_wavelength=  1/k_0 # meters

Lmeters = int(np.round(longest_wavelength) * 20)
print(Lmeters)
print(Gi['dist'].max()/ Lmeters )

#
shortest_wavenumber = 1/2 # 2 meter wave
desired_dx = 1/shortest_wavenumber

Lpoints = Lmeters / dx
print(Lpoints)
Gi.shape[0]/ Lpoints

S_pwelch_k, S_pwelch_dk = spec.calc_freq_LS( None, Lmeters, method='fixed_ratio', dx= dx)

S_pwelch_k, S_pwelch_dk = spec.calc_freq_LS( np.array(Gi['dist']), Lpoints, method='LS_auto', minimum_frequency=f_fft2_regular[1], maximum_frequency=f_fft2_regular[-1], samples_per_peak=0.1)

S_pwelch_k[-1]
S_pwelch_k[0]
S_pwelch_k.size

#f_regular, df2 = spec.calc_freq_fft_given_dx(desired_dx,Lmeters)
f_regular = np.linspace(k_0/100, 0.1, 300)

#f_regular = 1/np.linspace(1/shortest_wavenumber, 10/k_0, 1000)[::-1]

dk = np.diff(f_regular).mean()
f_regular[1:].size
f_regular[-1]

G2= dict()
#k = high_beams[0]
for k in high_beams:
    Gi = Gfilt[k]
    # for testing amplitude
    #Gi = Gi[(Gi['dist'] > 140000) & (Gi['dist'] < 160001)]
    #Gi['nans'] = np.isnan(Gi['heights'])
    #Gall[k][Gall[k+'_nans']] = 0
    #Gsel = Gi[~Gall[k+'_nans']]
    dd = np.array(  Gi['heights'] - Gi['heights'].mean() )
    #dd = np.gradient(dd)
    #np.isnan(dd).sum()
    S = spec.wavenumber_spectrogram_LS(np.array(Gi.dist), dd, Lmeters, waven_method = f_regular , dy=None ,  ov=None, window=None, kjumps=1)
    G = S.cal_spectrogram()
    S.mean_spectral_error() # add x-mean spectal error estimate to xarray
    S.parceval(add_attrs= True)

    G2[k] = G


font_for_pres()

plt.plot(G.k, G, 'b', alpha = 0.2)
plt.plot(G.k, G.mean('x'), 'r-' )

# for testing amplitude
# !!!! vanriance is not conserved yet !!!
plt.plot(f_fft2_regular,  spec_fft2_regular, 'k-' )
plt.xlim(0, 0.01)
plt.ylim(0, 40)

# %%

font_for_pres()

plt.plot(G.k, G.mean('x'))
# %%
#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
Gplot = G.rolling(k=10, x=2, min_periods= 1, center=True).mean()
# %%
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec

font_for_print()
F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =0.8)

plt.suptitle('LS on uneven grid (dx= '+ str(int(np.diff(Gplot.x).mean())) +' m)', y = 0.95)
gs = GridSpec(3,3,  wspace=0.1,  hspace=0.7)#figure=fig,
#clev=np.arange(0, 6, 0.1)*3
clev = M.clevels(Gplot.data, 31)* 0.9


for pos, k, pflag in zip([gs[0, 0],gs[0, 1],gs[0, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G2[k]#.rolling(k=10, x=2, min_periods= 1, center=True).mean()

    #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])
    ax0.set_yscale('log')
    #plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap=plt.cm.ocean_r)
    plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap=plt.cm.ocean_r, vmin = clev[0], vmax = clev[-1])

    #plt.gca().set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')

    plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(20,Lmeters/2)
    plt.title(k, loc='left')


    #ax0.axhline(30, color='black', linewidth=0.5)

    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()


k= high_beams[0]
G3_sum = G2[k]*0
N_grid=G2[k]*0
x_common =G2[k].x
for k in high_beams:
    gii = G2[high_beams[0]]
    N_grid +=~np.isnan(gii)
    G3_sum += gii.where(~np.isnan(gii), 0).interp(x = x_common)

G3_sum= G3_sum/N_grid


pos, k, pflag = gs[1, 0], 'mean', True
ax0 = F.fig.add_subplot(pos)
Gplot = G3_sum.rolling(k=5, x=5, min_periods= 1, center=True).mean()

#plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])

plt.pcolormesh(Gplot.x/1e3, 1/Gplot.k , Gplot, cmap=plt.cm.ocean_r, vmin = clev[0], vmax = clev[-1])

plt.gca().set_yscale('log')
# plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')
plt.xlabel('Distance from the Ice Edge (km)')
plt.ylim(20,Lmeters/2)
plt.title(k, loc='left')

#plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
ax0.axhline(30, color='black', linewidth=0.5)


if pflag:
    plt.ylabel('Wave length\n(meters)')
    plt.legend()


pos = gs[1, 1]
ax0 = F.fig.add_subplot(pos)
plt.title('mean # of datapoints / chunk (1e3)')
plt.plot(Gplot.x/1e3, Gplot.N_per_sample/1e3 , c='black')

pos = gs[1, 2]
ax0 = F.fig.add_subplot(pos)

plt.title('spectal power decay')
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(1/500, 1/100)).integrate('k') , c='black')

F.save_light(path=plot_path, name='exmpl_spec_LS_uneven_strong')
