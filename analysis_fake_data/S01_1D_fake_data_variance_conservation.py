# %%
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file test the wave realizations and the spectral estimates and the variance conservation between GFT, FFT, and real space.
"""

exec(open(os.environ['PYTHONSTARTUP']).read())
exec(open(STARTUP_2021_IceSAT2).read())

#%matplotlib inline
import ICEsat2_SI_tools.spectral_estimates as spec

import imp
import copy
import spicke_remover
#import datetime
import generalized_FT as gFT
from scipy.ndimage.measurements import label

base_path='/Users/Shared/Projects/2021_IceSAT2_tracks/'
sys.path.append(base_path +'analysis_fake_data/')
import wave_realizations as WR
import JONSWAP_gamma
#import s3fs
# %%
# load_path   = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
# load_file   = load_path + 'processed_' + ATlevel + '_' + track_name + '.h5'

# save_path   = mconfig['paths']['work'] + '/B02_spectra_'+hemis+'/'
# save_name   = 'B02_'+track_name

# plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/B_spectra/'
# MT.mkdirs_r(plot_path)
# MT.mkdirs_r(save_path)
# bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'


# %% create fake 1D data
# define spectral power and its reaalization
imp.reload(WR)
f = np.arange(0.001, 0.2,0.001) # frequency range
x_obs= np.arange(0, 3e5, 0.5) # space range
k =(2 * np.pi * f)**2 / 9.81 # wavenumber range

f_std =0.005
spec_power = JONSWAP_gamma.JONSWAP_default(f, 1e6, 10)
spec_power_gauss =  np.max(spec_power) * np.exp( - (  (f -f[spec_power.argmax()])/ f_std )**2 )

amps = WR.amplitude_from_wavenumber_spectrum(spec_power_gauss, k)
instance = WR.space_realization_from_spectrum(x_obs, k, spec_power_gauss)
WR.test_variance_conservations(f, spec_power_gauss, instance )

M.figure_axis_xy(5.5, 5.5, view_scale=0.9)

plt.subplot(311)
plt.plot(k, amps, linewidth=0.5)
plt.title('Amplitude Spectrum', loc='left')
plt.xlabel('wavenumber [k= 2pi/lambda]')
plt.ylabel('amplitude [m]')

plt.subplot(312)
plt.plot(x_obs/1e3, instance, label='realization')
plt.title('Realization', loc='left')
plt.legend()
plt.ylabel('height [m]')
plt.xlabel('space [km]')

plt.subplot(313)
plt.plot(x_obs[0:12000], instance[0:12000], label='realization')
plt.title('Realization', loc='left')
plt.legend()
plt.ylabel('height [m]')
plt.xlabel('space [m]')

WR.test_variance_conservations(k,  spec_power_gauss, instance, wave_number=True)


# %% Make standard DFT estimate and test variance conservation

import spectral_estimates as spec
instance2= instance[0:3000]
x2 = x_obs[0:3000]
Lpoints= len(instance2)
kk, dkk = spec.calc_freq_fft(x2, Lpoints)
kk, dkk = 2 * np.pi *kk, 2 * np.pi * dkk
spec_FFT = spec.calc_spectrum_fft(instance2, dkk, len(instance2))
S = spec.wavenumber_spectrogram(x2, instance2, 12000 , window='hann')

spec_FFT_welch =S.cal_spectrogram().mean('x')
S.parceval()
M.figure_axis_xy(5.5, 5.5, view_scale=0.9)
#plt.plot(kk, spec_FFT)
plt.plot( spec_FFT_welch.k, spec_FFT_welch )


print('variance of the initial spectrum', np.trapz(spec_power_gauss, k))
plt.plot(k, spec_power_gauss, linewidth=0.8, color='k', label='Input data')
plt.xlim(0, 0.05)


# %% GFT inverse of a single case
instance2= instance[0:3000]
x2 = x_obs[0:3000]
Lpoints= len(instance2)
Lmeters = x2[-1] - x2[0]
kk, dkk = spec.calc_freq_fft(x2, Lpoints)
kk, dkk = 2 * np.pi *kk, 2 * np.pi * dkk

dk_GFT = dkk* 1
kk_GFT= np.arange(kk[0], kk[-1], dk_GFT)

import generalized_FT as gFT
G = gFT.generalized_Fourier(x2, instance2, kk_GFT)

weight = ( np.interp(kk_GFT, k, spec_power_gauss) /spec_power_gauss.max())  * 0.8 * instance2.var()
err = instance2 * 0 + 0.1

G.define_problem(weight, err)
p_hat = G.solve()

# %% test realization
GFT_model = G.model()

M.figure_axis_xy(5.5, 5.5, view_scale=0.9)
plt.subplot(211)
plt.plot(x2, instance2, label='realization')
plt.plot(x2, GFT_model, label='GFT model')

# %% test of different spectral estimate calculations
font_for_pres()
MM = len(kk_GFT)
Nx = len(x2)
Mtwo = 2 *MM 

Z = (p_hat[0:MM] - p_hat[MM:] *1j) * (Nx/2+1)

DFT  = np.fft.rfft(instance2)

M.figure_axis_xy(5.5, 5.5, view_scale=0.6)
plt.subplot(211)

plt.plot(kk[0:25], DFT[0:25].real)
plt.plot(kk[0:25], DFT[0:25].imag)

plt.plot(kk_GFT[0:50], Z[0:50].real, '-*')
plt.plot(kk_GFT[0:50], Z[0:50].imag, '-*')

plt.subplot(212)

spec_DFT = 2.*(DFT*DFT.conj()).real / dkk /Nx**2
plt.plot(kk[0:50], spec_DFT[0:50], label='DFT')


spec_GFT = ( 2.*(Z*Z.conj()).real / dk_GFT /Nx**2 ) #/ ( Nx  / Mtwo )
plt.plot(kk_GFT[0:100], spec_GFT[0:100], '*', label='GFT')


dk_fake = 2 * np.pi/ Lmeters
spec_GFT2 = 2.*(Z*Z.conj()).real / dk_fake /Nx**2
plt.plot(kk_GFT[0:100], spec_GFT2[0:100], '-+', label='GFT')

Z = (p_hat[0:MM] - p_hat[MM:] *1j) * (MM/2+1)
#dk_fake = 2 * np.pi/ Lmeters
spec_GFT3 = 2.*(Z*Z.conj()).real / dk_GFT /MM**2 
plt.plot(kk_GFT[0:100], spec_GFT3[0:100], '--', label='GFT')

# test variance conservation
#print('variance of the initial spectrum', np.trapz(spec_power_gauss, k))
print('variance of the data', instance2.var())
print('variance of the DFT spectrum', np.trapz(spec_DFT, kk))
print('-----------')
print('variance of the GFT model', GFT_model.var())
print('variance of the GFT spectrum', np.trapz(spec_GFT, kk_GFT))
print('variance of the GFT2 spectrum', np.trapz(spec_GFT2, kk_GFT))
print('variance of the GFT3 spectrum', np.trapz(spec_GFT3, kk_GFT))
print('2M', str(2* MM) , 'N', str(Nx) )


# %% Define limits of the GFT wave numbers

# derive spectal limits
# Longest deserved period:
def define_minimal_datapoints_from_period(T0, x):
    k_0         = (2 * np.pi/ T0)**2 / 9.81
    dx          =  np.diff(x).mean()
    lambda_0  = 2 * np.pi/ k_0
    min_datapoint =  lambda_0/dx
    return int(np.ceil(min_datapoint)), k_0, dx

def define_minimal_datapoints_from_wavenumber(k0, x):
    lambda_min  = 2 * np.pi/ k0
    dx          =  np.diff(x).mean()
    min_datapoint =  lambda_min/dx
    return int(np.ceil(min_datapoint)), k0, dx

min_datapoint, k_0, dx = define_minimal_datapoints_from_period(20, x_obs)
Lpoints     = min_datapoint * 10 
Lmeters     = Lpoints  * dx

#plt.plot(np.diff(np.array(Gi['dist'])))
print('L number of gridpoint:', Lpoints)
print('L length in km:', Lmeters/1e3)
print('approx number windows', 2* x_obs[-1] /Lmeters-1   )

T_min       = 8
lambda_min  = 9.81 * T_min**2/ (2 *np.pi)
flim        = 1/T_min

#2* np.pi/dx

oversample  = 4
dlambda     = Lmeters * oversample
dk          = 2 * np.pi/ dlambda
kk          = np.arange(0, 1/lambda_min,  1/dlambda) * 2*np.pi
kk          = kk[k_0<=kk]
#dk = np.diff(kk).mean()

dk_adjustment = (np.pi *2 ) / (dk * Lmeters )

print('oversample ', oversample)
print('2 M = ',  kk.size *2 )
print('N = ', Lpoints)
#print('2M / N  = ', (kk.size *2 ) / Lpoints  )
print('N / 2M  = ', Lpoints / (kk.size *2 )  )
print('2M / N ', (kk.size *2 ) / Lmeters   )
print('Lmeter / 2M ', Lmeters / (kk.size *2 )  )
print('dk = ', dk)
print('dk_adjustment = ', dk_adjustment)   


imp.reload(spec)

x       = x_obs#[0:Lpoints*4]
dd      = instance#[0:Lpoints*4]

xlims   = x[0], x[-1]

(xlims[1] - xlims[0]) / Lmeters

dd_error = dd * 0 + 1 #np.copy(Gd[k]['heights_c_std'])
#dd_error[np.isnan(dd_error)] = 100

np.std(dd_error)
#sum(np.isnan(dd_error))
#plt.hist(1/dd_weight, bins=40)

# F = M.figure_axis_xy(6, 3)
# plt.subplot(2, 1, 1)
# plt.plot(x, dd, 'gray', label='displacement (m) ')

# compute slope spectra !!
dd      = dd #np.gradient(dd)
dd, _   = spicke_remover.spicke_remover(dd, spreed=10, verbose=False)
dd_nans = (np.isnan(dd) )#)# + (Gd[k]['N_photos'] <= 5)

# using gappy data
dd_no_nans = dd[~dd_nans] # windowing is applied here
x_no_nans  = x[~dd_nans]
dd_error_no_nans = dd_error[~dd_nans]
#plt.subplot(2, 1, 2)

#plt.plot(x_no_nans, dd_no_nans, 'black', label='slope (m/m)')
#plt.legend()
#plt.show()

#xlims = xlims[0], xlims[0] + (xlims[1] -xlims[0])/2


# %%

print('gFT')
font_for_print()
#S_pwelch_k2 = np.arange(S_pwelch_k[1], S_pwelch_k[-1], S_pwelch_dk*2 )
imp.reload(gFT)
S = gFT.wavenumber_spectrogram_gFT( np.array(x_no_nans), np.array(dd_no_nans), Lmeters, dx, kk, data_error = dd_error_no_nans ,  ov=None)

GG, GG_x, Params = S.cal_spectrogram(xlims= xlims, max_nfev = None, plot_flag = True)


M.figure_axis_xy(4.5, 4.5)

ave_data_var=GG_x.y_data.var('eta').mean().data 
font_for_pres()
# np.nanmean( (GG.gFT_PSD_model.sum('k') *dk) )
# (dk_adjustment*  GG.gFT_PSD_model.mean('x')).plot(marker = '+', label='GFT model')
# (dk_adjustment* GG.gFT_PSD_data.mean('x')).plot(label='GFT data')

(GG.gFT_PSD_model.mean('x')).plot(marker = '+', label='GFT model')
(GG.gFT_PSD_data.mean('x')).plot(label='GFT data')

# GG.gFT_PSD_data.median('x').rolling(k= 5).mean().plot(label='GFT data')
# GG.gFT_PSD_model.median('x').rolling(k= 5).mean().plot(label='GFT model')

plt.plot(k, spec_power_gauss, linewidth=0.8, color='k', label='Input data')
plt.title('Power Spectrum', loc='left')
plt.xlabel('wavenumber [k= 2pi/lambda]')
plt.ylabel('amplitude [m^2/k]')

plt.xlim(k[0], 0.05)
plt.legend()

print('data energy / mean model energy = ' , np.array(dd_no_nans).var() / ( GG.gFT_PSD_model.sum('k') *dk ).mean('x').data )

#plt.plot( spec_power_gauss / GG.gFT_PSD_model.median('x').interp(k= k)    )

GG.spec_adjust

# %%

GGi = GG.isel(x= 1)
GGxi = GG_x.isel(x= 1)
spec_man = GGi.gFT_cos_coeff**2 +  GGi.gFT_sin_coeff **2
np.trapz(spec_man, GGi.k)

GGxi.y_data.var('eta').data / (GGi.gFT_PSD_model.sum('k') *dk ).data

# %%
def linear_gap_fill(F, key_lead, key_int):

    """
    F pd.DataFrame
    key_lead   key in F that determined the independent coordindate
    key_int     key in F that determined the dependent data
    """
    y_g = np.array(F[key_int])

    nans, x2= np.isnan(y_g), lambda z: z.nonzero()[0]
    y_g[nans]= np.interp(x2(nans), x2(~nans), y_g[~nans])

    return y_g


font_for_pres()
# plot_data_model=True
# if plot_data_model:
for i in np.arange(1,2,1):
    c1= 'blue'
    c2= 'red'

    GGi = GG.isel(x= i)

    xi_1=GG_x.x[i]
    xi_2=GG_x.x[i+1]
    #if k%2 ==0:

    F = M.figure_axis_xy(16, 2)
    eta = GG_x.eta

    # gFT model
    y_model = GG_x.squeeze().y_model[:, i]
    plt.plot(eta +xi_1, y_model ,'-', c=c1, linewidth=0.8, alpha=1, zorder=12)
    y_model = GG_x.squeeze().y_model[:, i+1]
    plt.plot(eta +xi_2, y_model,'-', c=c2, linewidth=0.8, alpha=1, zorder=12)

    # iterpolated model in gaps
    FT = gFT.generalized_Fourier(eta +xi_1, None,GG.k )
    _ = FT.get_H()
    FT.p_hat=np.concatenate([ GGi.gFT_cos_coeff, GGi.gFT_sin_coeff ])
    # FT.ydata_std = GG_x.y_data.std()
    # FT.ydata_mean = GG_x.y_data.mean()
    plt.plot(eta +xi_1, FT.model() ,'-', c='orange', linewidth=0.8, alpha=1,zorder= 2)

    FT = gFT.generalized_Fourier(eta +xi_2, None,GG.k )
    _ = FT.get_H()
    FT.p_hat=np.concatenate([ GGi.gFT_cos_coeff, GGi.gFT_sin_coeff ])
    plt.plot(eta +xi_2, FT.model() ,'-', c='orange', linewidth=0.8, alpha=1,zorder= 2)


    # oringial data
    #plt.plot(x, dd, '-', c='k',linewidth=3,  alpha =0.6, zorder=11)
    #plt.plot(x, dd, '.', c='k',markersize=3,  alpha =0.5)

    # oringal data from model_output
    y_data = GG_x.y_data[:, i]
    plt.plot(eta +xi_1, y_data ,'-', c='k',linewidth=3,  alpha =0.3, zorder=11)
    y_data = GG_x.y_data[:, i+1]
    plt.plot(eta +xi_2, y_data,'-', c='k',linewidth=3,  alpha =0.3, zorder=11)


    #plt.plot(x[~dd_nans], dd_error[~dd_nans], '.', c='green',linewidth=1,  alpha =0.5)

    F.ax.axvline(xi_1 + eta[0].data , linewidth=4,  color=c1, alpha=0.5)
    F.ax.axvline(xi_1 + eta[-1].data, linewidth=4,  color=c1, alpha=0.5)
    F.ax.axvline(xi_2 + eta[0].data , linewidth=4,  color=c2, alpha=0.5)
    F.ax.axvline(xi_2 + eta[-1].data, linewidth=4,  color=c2, alpha=0.5)

    ylims= -np.nanstd(dd)*2, np.nanstd(dd)*2
    plt.text(xi_1 + eta[0].data, ylims[-1], '  N='+ str(GG.sel(x=xi_1, method='nearest').N_per_stancil.data) + ' N/2M= '+ str(GG.sel(x=xi_1, method='nearest').N_per_stancil.data/2/kk.size) )
    plt.text(xi_2 + eta[0].data, ylims[-1], '  N='+ str(GG.sel(x=xi_2, method='nearest').N_per_stancil.data) + ' N/2M= '+ str(GG.sel(x=xi_2, method='nearest').N_per_stancil.data/2/kk.size) )
    plt.xlim(xi_1 + eta[0].data*1.2, xi_2 + eta[-1].data*1.2 )
    #plt.xlim(xi_1, xi_2 )


    plt.ylim(ylims[0], ylims[-1])
    plt.show()

