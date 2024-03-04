# %%
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
#%matplotlib inline

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

import generalized_FT as gFT
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
Gi = Gi[(Gi['dist'] >= 140000) & (Gi['dist'] <= 170001)][0:2500]

# %% test and compare to FFT
from scipy.signal import detrend
y =detrend(np.array(Gi['heights_c_weighted_mean']) )
y.shape
y_gap= np.copy(y)
y_gap[200:1000] = np.nan
nan_mask =np.isnan(y_gap)
data_filled = np.copy(y)
data_filled[nan_mask] = 0

nan_mask.sum()/y_gap.size

plt.plot( Gi['dist'], y, '-')
plt.plot( Gi['dist'], y_gap, '-')

# %%
x= Gi['dist'] - Gi['dist'].iloc[0]
dx = np.median(np.diff(x))
plt.plot( x, y_gap)

y_gap_fill=np.copy(y_gap)
y_gap_fill[nan_mask]= 0


# %%
import ICEsat2_SI_tools.generalized_FT as gFT
imp.reload(gFT)

#plt.plot(data)

# %%
# choose wavenumber range
T_max = 40 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gi['dist'])
dx = np.diff(x).mean()
min_datapoint =  2*np.pi/k_0/dx

Lpoints = int(np.round(min_datapoint) * 10 )
Lmeters =Lpoints  * dx

#plt.plot(np.diff(np.array(Gi['dist'])))
print('L number of gridpoint:', Lpoints)


print('L length in km:', Lmeters/1e3)
print('approx number windows', 2* Gi['dist'].iloc[-1] /Lmeters-1   )


T_min=5
lambda_min = 9.81 * T_min**2/ (2 *np.pi)
flim = 1/T_min

oversample = 2
dlambda = (Lpoints *dx ) * oversample
dk = 2 * np.pi/ dlambda
kk =np.arange(0, 1/lambda_min,  1/dlambda) * 2*np.pi
#kk.shape
#kk = kk[k_0<=kk]
k = kk[1:]
kk.shape
dk = np.diff(kk).mean()

print('2 M = ',  kk.size *2 )

k = kk
# %%

#k_full = np.copy(np.fft.rfftfreq(x.size, d=dx) * 2* np.pi)
#1/np.diff(k_full/2/np.pi).mean()
#k = k_full[1:201]

x_gap = x[~nan_mask]


MM = k.size
MM

NN = x_gap.size

data_var = y_gap[~nan_mask].var()
print('2 MM= ', 2*MM, 'NN=', NN)

#plt.plot( x_gap , np.cos(np.outer(x_gap, k[0:10])), '.' )
G = np.vstack([ np.cos(np.outer(x_gap, k)).T ,  np.sin(np.outer(x_gap, k)).T ] ).T
G_full = np.vstack([ np.cos(np.outer(x, k)).T ,  np.sin(np.outer(x, k)).T ] ).T

G.shape
k.size
# Model Penalty
# regullization . 1/weight


imp.reload(gFT)
k_fft = np.fft.rfftfreq(x.size, d=dx) * 2* np.pi
f_weight= np.sqrt(9.81 * k_fft) / (2 *np.pi)
data_weight = spec.Z_to_power(np.fft.rfft(y_gap_fill), np.diff(f_weight).mean(), x.size)

imp.reload(gFT)

S = gFT.get_prior_spec(f_weight, data_weight )
pars = S.set_parameters()
S.params['gamma'].set(value =1 ,vary= False)
f= np.sqrt(9.81 * k) / (2 *np.pi)

weight = S.create_weight(freq = f, plot_flag= True)
weight = weight + weight.max()* 0.05 # add pemnalty floor
weight = weight/weight.max()  # add pemnalty floor
weight = weight *data_var

plt.title('FFT(data) and fitted Pior model')
plt.show()
#weight[weight< 0] =0.001
plt.plot([kk[0], kk[0]], [0, 1/weight.min()], 'k')
plt.plot([kk[-1], kk[-1]], [0, 1/weight.min()], 'k')

plt.plot(kk, 1/weight)
#plt.plot(kk, weight)
plt.title('penalty')

plt.plot(f, (1/f)**(1/4))

#next(iter(S.params.items()))[1].value

#plt.plot( np.arange(MM, 0, -1)/MM )
#weights = np.concatenate([ np.arange(MM, 0, -1)/MM ,  np.arange(MM, 0, -1)/MM  ])
weights = np.concatenate([ weight ,  weight  ])
weights.shape

# weights = np.concatenate([ np.arange(MM, 0, -1)/MM ,  np.arange(MM, 0, -1)/MM  ]) *0+1
# weights.shape
#weights = np.concatenate([ np.exp(-4 *np.arange(0, MM)/MM),  np.exp(-4 *np.arange(0, MM)/MM) ])
P = np.diag(weights)
P_1d  = weights
#plt.plot(1/weights)
P.shape

# uncertainty of the data

data_uncertainty = np.array(Gi['heights_c_std'])[~nan_mask]
noise_amp = 0.1
R = np.diag( noise_amp *  data_uncertainty/data_uncertainty.std() )
R_1d = np.diag(R)
#np.array(Gi['heights_c_std'])[~nan_mask].std()

from numpy import linalg
inv = linalg.inv

plt.plot(  - 1/np.diag(P) )


# Hessian_int
# %timeit Hess =(G.T @ inv(R) @ G  ) + inv(P)
# %timeit b_hat = inv( Hess) @ G.T @ inv(R) @ y[~nan_mask]

Hess =(G.T @ inv(R) @ G  ) + inv(P)
b_hat = inv( Hess) @ G.T @ inv(R) @ y[~nan_mask]

# %timeit G_T_R_inv       = G.T * (1/R_noise)
# %timeit Hess    = (G_T_R_inv @ G  ) + np.diag(1/P_1d)
# %timeit Hess_inv = inv(Hess)
# %timeit b_hat = Hess_inv @ G_T_R_inv @ y[~nan_mask]


# %%

model_error_k         = np.diag(inv( Hess))
model_error_real  = ((G**2) @ inv( Hess)).sum(1)

residual = y_gap[~nan_mask] - (b_hat * G).sum(1)
normalized_residual = residual.var() /np.diag(R).var()

data_var = y_gap[~nan_mask].var()
model_var = (b_hat * G).sum(1).var()
residual.var()

print('normlized residual ', normalized_residual)
print('data variance=', data_var,'model variance', model_var, ' residual variance', residual.var())
print('sum', data_var-model_var- residual.var())

# %%

plt.plot(k, gFT.power_from_model(b_hat, dk, MM, NN, x.size) )
plt.title('Power(b_hat)')
plt.show()

plt.title('model error')
plt.plot(k, np.sqrt(model_error_k[0:MM]**2 + model_error_k[MM:]**2 ) )
plt.show()
#plt.xlim(0, 0.01)
# %%
M.figure_axis_xy(10,5,  view_scale =0.6)

plt.subplot(2,1, 1)

plt.plot( x, y_gap, '-k', linewidth= 2)
#plt.plot( x, y, '-k', linewidth= 2)
x_gap.size
k.size *2

plt.plot( x_gap , (b_hat * G).sum(1), 'r-', markersize= 3 , label='model')
#plt.plot( x[~nan_mask] , (b_hat * G_full).sum(1)[~nan_mask], '-', color='orange', linewidth = 0.5 , label='prediction')
plt.plot( x[nan_mask] , (b_hat * G_full).sum(1)[nan_mask], '-', color='orange', linewidth = 0.5 , label='prediction')


plt.plot(x_gap, np.sqrt(model_error_real) , 'b.', markersize= 1 , label= 'sqrt(model error)')

#plt.xlim(1000,28000)
plt.ylim(-1, 1)

plt.legend()
# %
plt.subplot(2,1, 2)

#plt.plot( x, (y - (b_hat * G).sum(1)) , '.k', markersize= 0.8 )
plt.plot( x_gap,residual, '.k', markersize= 0.8 , label= 'residual')

plt.plot( x_gap, np.diag(R) , '-b', markersize= 0.8, label= 'given data uncertainty' )
plt.legend()


#plt.xlim(1000,28000)
plt.ylim(-1, 1)


# %% Auto correlation function of the Prior



plt.plot( (G @ P @ G.T)[0,0:200] )
plt.title('Prior auto correlation')


# %%
plt.plot(  (G  * np.diag(P ) ).sum(1) )
plt.plot( np.abs( (np.diag(P ) * G).sum(1) )[0:100]  )

# %%
M.figure_axis_xy(7,3,  view_scale =0.6)

Z = b_hat[0:MM] - b_hat[MM:] *1j
#Z_err = model_error_k[0:MM] - model_error_k[MM:] *1j

plt.subplot(1, 2, 1)
plt.title('abs(Z) model')

plt.plot(k, weight/weight.max()*abs(Z).max())

plt.plot(k, abs(Z) )
plt.xlabel('k ( 2 pi/ lambda) ')
plt.subplot(1, 2, 2)
plt.title('model error')
plt.plot(k, abs(Z_err))
plt.xlabel('k ( 2 pi/ lambda) ')

# %%
