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
%matplotlib inline

#import m_general as M
#import m_tools as MT
import numpy as np
import m_general_ph3 as M

import netCDF4
import datetime
import os
import xarray as xr

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import imp
#import s3fs
# %%
Lx=10000
xx = np.arange(0, Lx, 0.1)

lam = 10
k = 1 / lam
_logger.debug(k)
k2 = 1  / 0.21
_logger.debug(k2)
#yy = np.sin( k * xx  * 2 * np.pi ) + np.sin(  k2 * xx  * 2 * np.pi ) + 0.1 *np.random.randn(len(xx))

decay = np.exp( -  0.2 *(np.arange(len(xx))/Lx))
plt.plot(decay)
yy = np.sin( k * xx  * 2 * np.pi ) * decay   + np.sin( 10 * k * xx  * 2 * np.pi ) * decay   +   0.3 *np.random.randn(len(xx))
plt.plot(yy)


# %% test simple pwelch wavenumber spectrum

L=400
imp.reload(specs)
S =specs.wavenumber_pwelch(yy, xx, L, ov=None, window=None, save_chunks=False, plot_chunks=False)

S.specs.shape
S.spec_est.shape
S.k[1]
S.k[S.spec_est.argmax()]

plt.plot(S.k, S.spec_est)

# %% test class
imp.reload(specs)
S = specs.wavenumber_spetrogram(xx, yy, L)
G = S.cal_spectrogram()
S.mean_spectral_error() # add x-mean spectal error estimate to xarray

S.parceval(add_attrs= True)
G.attrs

plt.plot(G.T)
G.plot()

# test of frequency is recovered
G.k[G.mean('x').argmax().data].data

# %% lomb-scargle
