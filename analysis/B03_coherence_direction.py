
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline

import ICEsat2_SI_tools.convert_GPS_time as cGPS
import h5py
import ICEsat2_SI_tools.io as io
import ICEsat2_SI_tools.spectral_estimates as spec

import imp
import copy
import spicke_remover
import datetime
import concurrent.futures as futures
#import s3fs
# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False




#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

save_path   = mconfig['paths']['work'] + '/B03_spectra_'+hemis+'/'
save_name   = 'B03_'+track_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/B_spectra/'
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data

load_path   = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #
load_path   = mconfig['paths']['work'] + '/B02_spectra_'+hemis+'/'
Gpars   = io.load_pandas_table_dict('B02_'+ track_name + '_params' , load_path)  #
Gspec   = xr.open_dataset(load_path + 'B02_'+ track_name + '_LS.nc' )  #
Gspec['Y_model_hat'] = Gspec.Y_model_hat_real + Gspec.Y_model_hat_imag *1j
Gspec = Gspec.drop('Y_model_hat_real').drop('Y_model_hat_imag')

dk = Gspec.k.diff('k').mean().data
Lpoints = Gspec.Lpoints

Gspec = Gspec.sel(k = slice(0.000125, 0.025)).isel(x =slice(4, 30))

Gspec.coords['f'] = (('k'), np.sqrt(Gspec.k.data * 9.81)/ 2/np.pi )
Gspec.coords['T'] = 1/Gspec.coords['f']
Gspec=Gspec.swap_dims({'k': 'f'})

#Gspec.spectral_power_optm.sel(beam='weighted_mean').plot()
# %%
#k_lim= 0.02
A, B = Gspec.sel(beam= 'gt2r').Y_model_hat , Gspec.sel(beam= 'gt2l').Y_model_hat

r_ave_kargs={'x':2, 'f':10, 'center':True, 'min_periods':2}
r_ave_kargs2={'f':10, 'center':True, 'min_periods':2}
#(abs(B) - abs(A)).plot()
S_aa = (A*A.conj()).real
S_bb = (B*B.conj()).real
#co_spec = (A.conj() *B) /S_aa/S_bb
# abs(co_spec).plot()
co_spec = (A.conj() *B)
np.log(abs(co_spec)).plot(levels=np.arange(-2, 3, 0.1))

co_spec = (A.conj() *B).rolling(**r_ave_kargs).mean()#
np.log(abs(co_spec)).plot(levels=np.arange(-2, 3, 0.1))

(abs(co_spec)).plot(levels=np.exp(np.arange(-3, 2, 0.1)))

abs(co_spec).mean('x').plot()

#(abs(co_spec)/(S_aa *S_bb).rolling(**r_ave_kargs).mean()).plot()
#
# (abs(A.conj() *B)/(S_aa *S_bb)).rolling(**r_ave_kargs).mean()[:,:].plot()

from xhistogram.xarray import histogram

phase = np.arctan2( -co_spec.imag, co_spec.real )
phase.plot()

# %%
pi_sel = phase.isel(x=0)

f = phase.f
f_pos = abs(co_spec).isel(x=0).rolling(f=10).mean().argmax().data



def adjust_phase_forward(pi, f_pos, f_size):

    for fi in np.arange(f_pos+1, f_size):
        #print(fi,pi[fi]  )
        dpi1 = pi[fi] - pi[fi-1]
        if abs(dpi1)/(2*np.pi) > (2 *np.pi):
            dpi1= dpi1 - np.sign(dpi1) * np.floor(abs(dpi1)/(2*np.pi)) * 2 *np.pi

        dpi2 = - np.sign(dpi1) * (2* np.pi - np.sign(dpi1) * dpi1)
        dds = np.array([dpi1, dpi2])
        pi[fi] = pi[fi-1] + dds[np.argmin(abs(dds))]
        #print(pi[fi]  )
    return pi

def adjust_phase_backward(pi, f_pos, f_size):

    for fi in np.arange(f_pos-1, -1, -1):
        #print(fi,pi[fi]  )
        dpi1 = pi[fi] - pi[fi+1]
        if abs(dpi1)/(2*np.pi) > (2 *np.pi):
            dpi1= dpi1 - np.sign(dpi1) * np.floor(abs(dpi1)/(2*np.pi)) * 2 *np.pi

        dpi2 = - np.sign(dpi1) * (2* np.pi - np.sign(dpi1) * dpi1)
        dds = np.array([dpi1, dpi2])
        pi[fi] = pi[fi+1] + dds[np.argmin(abs(dds))]
        #print(pi[fi]  )
    return pi

def adjust_phase_freq(pi, f_pos, f_size):

    pi = adjust_phase_forward(pi, f_pos, f_size)

    pi = adjust_phase_backward(pi, f_pos, f_size)
    return pi
# %%
pi = np.copy(pi_sel)
pi =adjust_phase_freq(pi, f_pos, f.size)
#pi =adjust_phase_freq(pi, f_pos, f.size)

plt.plot(pi_sel/np.pi, 'k--')
plt.plot(pi/np.pi)
plt.grid()

# %%
phase = np.arctan2( -co_spec.imag, co_spec.real )
phase_data = np.zeros(phase.T.shape)*np.nan

for xi in range(phase.x.size):
    #print(xi.data)
    pi = np.copy(phase.isel(x= xi))
    phase_data[xi,:] =adjust_phase_freq(pi, f_pos, f.size)

phase_adjust = phase.copy()
phase_adjust.data = phase_data.T / np.pi
phase_adjust.plot()


phase_adjust.where(phase_adjust <  np.pi, phase_adjust - 2* np.pi).plot()
# for fi in range(phase_adjust.f.size):
#     #print(xi.data)
#     pi = np.copy(phase_adjust.isel(f= fi))
#     phase_data[:, fi] =adjust_phase_forward(pi, 0, phase_adjust.x.size)
#
# phase_adjust.plot()
#
# for xi in range(phase.x.size):
#     #print(xi.data)
#     pi = np.copy(phase_adjust.isel(x= xi))
#     phase_data[xi,:] =adjust_phase_freq(pi, f_pos, f.size)
#
#
# phase_adjust.plot()
# %%
phase_adjust = phase_adjust.where(phase_adjust < 1 , phase_adjust -2)
phase_adjust = phase_adjust.where(phase_adjust > -1 , phase_adjust +2)
phase_adjust.plot()
# %%
phase_filtered= (phase_adjust.where(abs(co_spec)>0.2, np.nan))
phase_filtered.plot()


x_prime= (phase_filtered.x - phase_filtered.x[0] )/1e3

y_prime = (phase_filtered.k/ 2/ np.pi)
plt.pcolormesh( x_prime , y_prime, phase_filtered.data)
plt.ylabel('1/lambda')

# %%

font_for_pres()
lala= 2* np.pi/phase_filtered.k
plt.pcolormesh( x_prime ,  1/y_prime, phase_filtered * lala  /2, vmin=-1000, vmax=1000, cmap=plt.cm.coolwarm)
plt.ylabel('wave length (meters)')
plt.colorbar(label='meters of phase shift')
plt.yscale('log')
plt.ylim(0, 1e4)

# %%


plt.plot(1/y_prime,  (phase_filtered* lala  /2), '-')
# %%

histogram(phase_adjust, bins=[20], dim=['x'], weights=abs(co_spec)).rolling(**r_ave_kargs2).mean().T.plot()

# %%

phase_hist    = histogram(phase_adjust, bins=[20], dim=['x'], weights=abs(co_spec))

phase_hist.rolling(**r_ave_kargs2).mean().T.plot(cmap=plt.cm.ocean_r, levels=np.arange(-1, 18, 0.5))
plt.ylabel('radients (1/pi)')
plt.grid()
#histogram(phase_adjust, bins=[20], dim=['x']).rolling(**r_ave_kargs2).mean().T.plot()

# %%
phase_hist    = histogram(phase_adjust, bins=[100], dim=['f'], weights=abs(co_spec))
phase_hist.T.plot(cmap=plt.cm.ocean_r, levels=np.arange(-1, 20, 0.5) )
plt.ylabel('radients (1/pi)')
plt.grid()
#phase_hist.rolling(**r_ave_kargs2).mean().T.plot(cmap=plt.cm.ocean_r, levels=np.arange(-1, 20, 0.5))
# %%

phase_adjust = phase_adjust.where(phase_adjust < 1 , phase_adjust -2)
phase_adjust = phase_adjust.where(phase_adjust > -1 , phase_adjust +2)
phase_hist    = histogram(phase_adjust, bins=[60], dim=['f'], weights=abs(co_spec))
phase_hist.T.plot(cmap=plt.cm.ocean_r, levels=np.arange(-1, 20, 0.5) )
plt.ylabel('radients (1/pi)')
plt.grid()
# %%
phase_hist.T.rolling(Y_model_hat_bin = 5, x=10, center= True, min_periods= 2).mean().plot(cmap=plt.cm.ocean_r, levels=np.arange(-1, 10, 0.5) )
plt.ylabel('radients (1/pi)')
plt.grid()

# %%
histogram(phase, bins=[60], dim=['f']).T.plot()

plt.plot(phase, 'k', linewidth= 1,  alpha =0.5)
#plt.plot(phase.k, phase, 'k', linewidth=0.8, alpha= 0.6)
phase.sel(x=slice(500000, 600000)).mean('x').rolling(k=20, center=True, min_periods=2).mean().plot()#.hist(bins= 40)
phase.sel(x=slice(500000, 600000)).mean('x').rolling(k=20, center=True, min_periods=2).std().plot()#.hist(bins= 40)

# %%
phase.mean('x').plot()#.hist(bins= 40)

phase.where(phase > np.pi/2, phase + np.pi).plot()


C_ab= abs(co_spec)**2 #/ S_aa/ S_bb
np.log(C_ab).sel(k=slice(0, 0.03)).plot()

C_ab= abs(co_spec)**2 / S_aa/ S_bb
(C_ab).sel(k=slice(0, 0.03)).plot(levels = np.arange(0.5, 1+0.1, 0.05))
#(C_ab.where(~np.isnan(C_ab), 0)).sel(k=slice(0, 0.03)).plot.hist(bins= 30)

x_nan_mask = (~np.isnan(C_ab)).sum('k') == 0

C_ab[:, ~x_nan_mask].min()

phase = np.arctan2( -co_spec.imag, co_spec.real )

plt.plot( phase )
plt.plot(S_bb, 'k', linewidth= 0.5, alpha =0.5)

np.nanmin(C_ab.imag)
np.nanmax(C_ab.imag)
Gspec.beam

# %%
Ai = A.sel(x= 500000, method='nearest')
Bi = B.sel(x= 500000, method='nearest')

M.figure_axis_xy(10, 2.5, view_scale= 0.7)
plt.plot(np.fft.irfft(Ai) )
plt.plot(np.fft.irfft(Bi) )

# %%


co_spec=Ai.imag *Bi.real
C_ab= abs(co_spec)**2 / ( (Ai * Ai.conj()).real * (Bi * Bi.conj()).real  )
plt.plot(C_ab.k, C_ab)
#plt.semilogx(C_ab.k, abs(Ai.imag *Bi.real)**2)


phase = np.arctan2( -co_spec.imag, co_spec.real )
phase[phase <- np.pi/2] = phase[phase < - np.pi/2] + np.pi
phase.plot.hist()
plt.plot(phase.k, phase)
# %%


def get_coherency(d1, d2, Lint, dt=None):
    """
    returns the squared coherence Cxy and its phase
    """
    from scipy import signal

    if dt is None:
        dt =np.diff((dd.time - dd.time[0]).data.astype('m8[D]')).mean().astype('int')
    else:
        dt= dt

    coh = signal.coherence( d1, d2, fs=1/dt, window ='hann',  nperseg=Lint,  noverlap=None, detrend='linear')

    co_spec = signal.csd( d1, d2, fs=1/dt, window ='hann',  nperseg=Lint,noverlap=None, detrend='linear')
    phase = np.arctan2( -co_spec[1].imag, co_spec[1].real )

    #Spec = m_spec.Spectrum(dd, dt=dt, verbose=False)
    #co_spec = signal.csd( d1, d2, fs=1/dt, window ='hann',  nperseg=Lint)
    #np.arctan2(-)
    return {'f' : coh[0], 'coh2' : coh[1], 'coh' : np.sqrt(abs(coh[1])), 'cospec' :co_spec , 'phase':phase}
