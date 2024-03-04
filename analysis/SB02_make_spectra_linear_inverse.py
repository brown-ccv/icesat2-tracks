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

import lmfit
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
Gi = Gi[(Gi['dist'] >= 140000) & (Gi['dist'] <= 180001)]
Gi = Gi[(Gi['dist'] >= 140000) & (Gi['dist'] <= 170001)]

# %%




#plt.xlim(140000, 145000)
#plt.ylim(-1, 1)

#np.diff(np.array(Gi['dist']))
# %% test and compare to FFT
from scipy.signal import detrend
y =detrend(np.array(Gi['heights_c_weighted_mean']) )

y_gap= np.copy(y)
y_gap[200:2800] = np.nan
nan_mask =np.isnan(y_gap)
data_filled = np.copy(y)
data_filled[nan_mask] = 0

nan_mask.sum()/y_gap.size

plt.plot( Gi['dist'], y, '-')
plt.plot( Gi['dist'], y_gap, '-')

# %%
x= Gi['dist'] - Gi['dist'].iloc[0]
dx = np.diff(x).mean()
plt.plot( x, y_gap)


k_full = np.copy(np.fft.rfftfreq(x.size, d=dx) * 2* np.pi)
#k = k_full[1:201]
klim= 301# 301
k = k_full[1:klim]
MM = k.size
NN = x.size

2*np.pi/k[-1]

x_gap = x[~nan_mask]

#plt.plot( x_gap , np.cos(np.outer(x_gap, k[0:10])), '.' )

G = np.vstack([ np.cos(np.outer(x_gap, k)).T ,  np.sin(np.outer(x_gap, k)).T ] ).T
G_full = np.vstack([ np.cos(np.outer(x, k)).T ,  np.sin(np.outer(x, k)).T ] ).T
G.shape

b = np.zeros([2 *MM])
b[4] = 1
b[300] = 1

b.shape
G.shape
y_gap.shape

plt.plot( (G *b).sum(1) )

# Model Penalty
# regullization . 1/weight
# choose simple decay along diagonal

import ICEsat2_SI_tools.generalized_FT as gFT
imp.reload(gFT)
k_fft = np.fft.rfftfreq(x.size, d=dx) * 2* np.pi
f_weight= np.sqrt(9.81 * k_fft) / (2 *np.pi)
data_weight = spec.Z_to_power(np.fft.rfft(data_filled), np.diff(f_weight).mean(), x.size)

S = gFT.get_prior_spec(f_weight, data_weight )
pars = S.set_parameters()

f= np.sqrt(9.81 * k) / (2 *np.pi)
weight = S.create_weight(freq = f, plot_flag= False)
weight = weight + weight.max()* 0.1 # add pemnalty floor
#weight[weight< 0] =0.001

plt.plot(k, weight)
#weights = np.concatenate([ weight ,  weight  ])
#weights = np.concatenate([ weight ,  weight  ])

#plt.plot( np.arange(MM, 0, -1)/MM )
#weights = np.concatenate([ np.arange(MM, 0, -1)/MM ,  np.arange(MM, 0, -1)/MM  ])
weights = np.concatenate([ np.arange(MM, 0, -1)/MM ,  np.arange(MM, 0, -1)/MM  ])# *0+1
weights= weights+ 0.1
plt.plot(weights)

#weights = np.concatenate([ np.exp(-4 *np.arange(0, MM)/MM),  np.exp(-4 *np.arange(0, MM)/MM) ])
P = np.diag(weights)
P.shape



# uncertainty of the data

uncertaintiy = np.array(Gi['heights_c_std'])[~nan_mask]
#uncertaintiy = np.array(Gi['heights_c_std'])
R = np.diag( uncertaintiy )
R.shape

from numpy import linalg
# R = dvar is either a scalar or a vector n_data long
# estimate is:
# phat = (G'*R^-1*G + P^-1)^-1 * G'*R^-1*y  < -------------------------------

inv = linalg.inv
inv(R).shape
G.T.shape

#inv(  (G.T @ inv(R) @ G  ) + inv(P) ).shape
b_hat = inv(  (G.T @ inv(R) @ G  ) + inv(P) ) @ G.T @ inv(R) @ y[~nan_mask]

plt.plot(b_hat)
# %%

M.figure_axis_xy(10,5,  view_scale =0.6)

plt.subplot(2,1, 1)

#plt.plot( x_gap, y_gap[~nan_mask], '-k', linewidth= 2)
plt.plot( x, y, '-k', linewidth= 2)

#plt.plot( x , (b_hat * G).sum(1), 'r-', markersize= 1 )
plt.plot( x[~nan_mask] , (b_hat * G).sum(1), 'r-', markersize= 1 )
plt.plot( x[nan_mask] , (b_hat * G_full).sum(1)[nan_mask], '-', color='orange', linewidth = 0.5 )
#plt.xlim(1000,8000)
plt.ylim(-1, 1)

plt.subplot(2,1, 2)

plt.plot( x[~nan_mask], (y[~nan_mask] - (b_hat * G).sum(1)) , '.k', markersize= 0.8 )
#plt.plot( x, y_gap - (b_hat * G).sum(1) , '.k', markersize= 0.8 )
# rr= y - (b_hat * G).sum(1)
# y.var() - (b_hat * G).sum(1).var() - rr.var()



rr= y[~nan_mask] - (b_hat * G).sum(1)
y[~nan_mask].var() - (b_hat * G).sum(1).var() - rr.var()

# rr= y - (b_hat * G).sum(1)
# y.var() - (b_hat * G).sum(1).var() - rr.var()


rr.var()/uncertaintiy.var()

plt.xlim(1000,8000)
plt.ylim(-1, 1)



# %%




# %%
k.shape

sum1 = (b_hat * G).sum(0)
R = sum1[0:MM]
I = sum1[MM:]
C  =(G**2)[:, 0:MM].sum(0)
S  =(G**2)[:, MM:].sum(0)
ampl_spec  =  np.sqrt( 16*  MM * (R**2/ C + I**2/ S) )
plt.plot( k, ampl_spec)

power_spec  =  ( MM * (R**2/ C + I**2/ S) ) * 2* np.pi / y_gap[~nan_mask].std()/4
power_spec.sum()




dk = np.diff(k).mean()


1/dk
dx


phi_ft = np.arctan2(I, R)
Z2_real = ampl_spec * np.cos(phi_ft)
Z2_imag = ampl_spec * np.sin(phi_ft)

plt.plot(Z.real)
plt.plot(Z2_real)
#plt.ylim(0,0.06)


# %%
Z_fft = np.fft.rfft(y)

Z = b_hat[0:MM] - b_hat[MM:] *1j # this is per size(k)


x.shape
 #%%

plt.plot(k, power_spec)
############################################
# %%
Z = b_hat[0:MM] - b_hat[MM:] *1j # this is per size(k)

(b_hat * G).sum(1).var()
(b_hat * G).sum(1).var() / y[~nan_mask].var()

(b_hat * G_full).sum(1).var()

#y.var() - rr.var()
k.shape
k_full.size
x.size
x[~nan_mask].size
x_gap.size

k.shape

# for non gappy data
#Z_power = spec.Z_to_power(Z * k_full.size, dk, x.size)
k.size/k_full.size
k_full.size/k.size

x_gap.size/x.size

datat_point_frac= x_gap.size/x.size

1/datat_point_frac
#1/np.sqrt(datat_point_frac)

#Z_power = spec.Z_to_power(Z * k_full.size* datat_point_frac , dk, x_gap.size)
#Z_power = spec.Z_to_power(Z * k_full.size , dk, x.size) #/datat_point_frac
#Z_power = spec.Z_to_power(Z * k_full.size , dk, x.size) #/datat_point_frac
Z_power = spec.Z_to_power(Z * (x.size/2+1) , dk, x.size) #/datat_point_frac
#Z_power = spec.Z_to_power(Z * k.size , dk, x_gap.size) #/datat_point_frac

Z_power.shape
Z_power.sum() * dk
Z_power.sum() * dk / (b_hat * G_full).sum(1).var()


# Z_power2 = spec.Z_to_power(Z * (x.size/2 +1)  , dk, x_gap.size)
# Z_power2.sum() * dk * datat_point_frac
Z_power2 = (Z*Z.conj()).real * x.size / 2 / x_gap.size
Z_power2.sum()
Z_power2.sum() / (y[~nan_mask].var() - rr.var())


Z_power2 = (Z*Z.conj()).real * x.size / 2 / x_gap.size /dk
Z_power2.sum() *dk
Z_power2.sum() *dk / (y[~nan_mask].var() - rr.var())
#np.sqrt(2)
#Z_power.sum() * dk / datat_point_frac

Z_power2 =  ((Z * (x.size/2+1))*(Z * (x.size/2+1)).conj()).real * 2 / ( x.size * x_gap.size  * dk)
Z_power2.sum() *dk
Z_power2.sum() *dk / (y[~nan_mask].var() - rr.var())


1/dk/x.size
#plt.plot(k, 4*abs(Z* k_full.size)/dk/x_gap.size**2, 'b')
plt.plot(k, Z_power, 'k*')

#
# plt.plot(Z_power, abs(Z))
# #plt.plot( abs(Z_fft)[1:klim], abs(Z), '.')
# Z = b_hat[0:MM] - b_hat[MM:] *1j # this is per size(k)
# plt.plot(Z_power,  abs(Z.imag) + abs(Z.real) )#, abs(Z))
#
# plt.plot(abs(b_hat[0:MM]) + abs(b_hat[MM:]), abs(Z), '.')
#
# tpi= 2*np.pi
#

Z_b_hat_full = np.fft.rfft( (b_hat * G_full).sum(1) )
Z_b_hat_full_power = spec.Z_to_power(Z_b_hat_full , dk, x.size)
Z_b_hat_full_power
plt.plot(k, Z_b_hat_full_power[1:klim], '-g')

Z_b_hat_full_power[1:klim].sum() *dk


Z_fft = np.fft.rfft(data_filled)
# #Z_fft = np.fft.rfft(y)
Z_fft_power = spec.Z_to_power(Z_fft, dk, x.size)
plt.plot(k, Z_fft_power[1:klim], 'r')
#Z_fft_power[1:klim].sum() *dk/tpi
Z_fft_power[1:klim].sum() *dk
#Z_fft_power.sum() *dk


################################################
# %% define functions based onthe above:
def complex_represenation(b_hat, M, N_x_full):
    """
    returns the fourrier coefficiens in b_hat as comples number Z
    b_hat is the model coefficient matrix
    M   number if wavenumbers, ie.e size of the model matrix /2
    N_x_full it the size of the data if there wouldn't be gaps. = Lmeters / dx
    i.e twice the size of the wavenumber space of a standard FFT.

    returns:
    The fourier coeffcients as complex vector with the same amplitudes as from an fft, but for the model wavenumbers of b_hat.

    this returns a power spectral density with the same variance as the data without gaps.
    """
    Z = b_hat[0:M] - b_hat[M:] *1j
    Z = Z * (N_x_full/2+1) # this
    return Z


def Z_to_power_gFT(Z, dk, N_x,  N_x_full):
    """ compute the 1d Power spectral density of a field Z
    inputs:
    Z       complex fourier coefficients, output of .complex_represenation method
    dk      delta wavenumber asssuming Z is on regular grid
    N_x the actual size of the data
    N_x_full it the size of the data if there wouldn't be gaps. = Lmeters / dx

    returns
    spec_incomplete     spectral density respesenting the incomplete data ( [b_hat]^2 / dk)
    spec_complete       spectal density representing the (moddeled) complete data ( [b_hat]^2 / dk)
    """

    spec = 2.*(Z*Z.conj()).real
    neven = True if (N_x_full%2) else False
    # the zeroth frequency should be counted only once
    spec[0] = spec[0]/2.
    if neven:
        spec[-1] = spec[-1]/2.

    # spectral density respesenting the incomplete data ( [b_hat]^2 / dk)
    spec_incomplete = spec / dk / N_x / N_x_full
    # spectal density representing the (moddeled) complete data ( [b_hat]^2 / dk)
    spec_complete = spec / dk / N_x_full**2


    return spec_incomplete, spec_complete



def power_from_model(b_hat, dk,  M,  N_x,  N_x_full):
    """ compute the 1d Power spectral density from the model coefficients in b_hat

    b_hat       is the model coefficient matrix
    M     size of the model vector, size of k
    N_x the     actual size of the data
    N_x_full    is the size of the data if there wouldn't be gaps. = Lmeters / dx

    returns:

    spectral density respesenting the incomplete data ( [b_hat]^2 / dk)
    """

    Z = b_hat[0:M] - b_hat[M:] *1j
    spec = (Z*Z.conj()).real * N_x_full / 2 / N_x /dk

    neven = True if (N_x_full%2) else False
    spec[0] = spec[0]/2.
    if neven:
        spec[-1] = spec[-1]/2.

    # spectral density respesenting the incomplete data
    return spec


ZZ = complex_represenation(b_hat, MM, x.size)
spec_in, spec_c = Z_to_power_gFT(ZZ, dk, x_gap.size, x.size)
spec_c.sum() * dk
(b_hat * G_full).sum(1).var()

spec_in.sum() *dk
power_from_model(b_hat, dk, MM, x_gap.size, x.size).sum() *dk
(b_hat * G).sum(1).var()

# %%
plt.plot(k, abs(Z) )
plt.plot( k, ampl_spec , 'k')
#np.median( abs(Z_fft) / (abs(Z)*MM) )

# %% compate phase
plt.plot(Z_fft.real[1:klim]/ k_full.size, Z.real, '.')
plt.axis('equal')


(Z_fft.real[1:klim]/ k_full.size - (Z.real)).mean()
Z_fft.imag[1:klim].sum()/ k_full.size
Z.imag.sum()

Z_fft.real[1:klim].sum()/ k_full.size
Z.real.sum()

( (Z_fft.imag[1:klim]/ k_full.size) - (Z.imag) ).mean()
#plt.plot( (Z_fft[1:klim].imag) - (Z.imag*k_full.size))
plt.plot( (Z_fft[1:klim].imag) )
plt.plot( (Z.imag*k_full.size))

plt.plot( (Z_fft[1:klim].real) )
plt.plot( (Z.real*k_full.size))

plt.plot( (Z_fft[1:klim].imag) , (Z.imag*k_full.size), '.')

# plt.plot( (Z_fft.real) - (Z.real*MM))
#plt.plot(  ( (Z_fft.imag) - (Z.imag*MM) ) -  (  (Z_fft.real) - (Z.real*MM) ) )
