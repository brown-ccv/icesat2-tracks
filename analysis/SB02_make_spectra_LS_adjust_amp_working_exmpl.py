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
Gi = Gi[(Gi['dist'] > 140000) & (Gi['dist'] < 170001)]

# %%




#plt.xlim(140000, 145000)
#plt.ylim(-1, 1)

#np.diff(np.array(Gi['dist']))
# %% test and compare to FFT
from scipy.signal import detrend
y =detrend(np.array(Gi['heights_c_weighted_mean']) )

data_gap= np.copy(y)
data_gap[500:2000] = np.nan
nan_mask =np.isnan(data_gap)
data_filled = np.copy(y)
data_filled[nan_mask] = 0

nan_mask.sum()/data_gap.size

plt.plot( Gi['dist'], y, '-')
plt.plot( Gi['dist'], data_gap, '-')

# %%
def create_weighted_window(data, window=None):
    """
    define window function and weight it to conserve variance
    if window is not None it show have a length of N
    """
    import scipy.signal.windows as WINDOWS

    L = data.size
    if window is None:
        #win=np.hanning(L)
        win =  WINDOWS.tukey(L, alpha=0.1, sym=True)
    else:
        win=window


    factor  = np.sqrt( np.var(data)  / np.var(( data* win) ) )
    #factor=np.sqrt( L/(win**2).sum())
    win     *= factor
    return win


def create_window_mask(nan_mask, ddata):

    from scipy.ndimage.measurements import label
    nlabel = label(~nan_mask +0)

    windows = np.zeros(nan_mask.size)
    for i in np.arange(1, nlabel[1]+1):

        sub_mask  = nlabel[0] == i
        win= create_weighted_window(ddata[sub_mask])
        #win = create_window(sum(sub_mask))
        windows[sub_mask] = win

    return windows#, window_inv

#windows = create_window_mask(nan_mask, data_gap )
win = create_weighted_window(data_filled)

np.nanvar(data_gap)
np.nanvar(( data_gap* win)[~nan_mask] )

plt.plot(data_gap* win)

# %%

f_fft2, df2 = spec.calc_freq_fft( np.array(Gi['dist']), y.size)

Y = spec.calc_spectrum_fft(y, df2, y.size)
Y_win= spec.calc_spectrum_fft(y * win, df2, y.size)
Y_gap_win = spec.calc_spectrum_fft(data_filled * win, df2, y.size)


# %% simple spectra
from astropy.timeseries import LombScargle
ls_gap_win = LombScargle( np.array(Gi['dist'])[~nan_mask], ( data_gap* win)[~nan_mask] , fit_mean=False)
ls_gap_win_power = ls_gap_win.power(f_fft2 , normalization='psd', assume_regular_frequency= False)

F= M.figure_axis_xy(10, 4, view_scale= 0.8)
plt.subplot(2, 1,1)
#plt.plot(f_fft2[1::] , spec.LS_power_to_PSD(ls_auto_power_windowed, y.size, df2), '^'  , label= 'LS single window, no gaps')
plt.plot(f_fft2[:-1], Y_win, '-' , c='k' , label= 'single windowed FFT, no gaps')
plt.plot(f_fft2[:-1], Y_gap_win, '-' , label= 'single windowed FFT, gaps')

plt.plot(f_fft2 , spec.LS_power_to_PSD(ls_gap_win_power, y.size, df2)  , label= 'LS_gap_win')

plt.legend()
plt.xlim(0, 0.01)

plt.subplot(2, 1,2)

plt.plot(f_fft2[:-1],np.angle(np.fft.rfft(y * win)), c='k' , label= 'windowed standard FFT')
plt.plot(f_fft2[:-1],np.angle(np.fft.rfft(y)), '--', c='k', label= 'standard FFT')
plt.legend()

plt.xlim(0, 0.01)




# %% reconstruct data
# test getting phase

t_base =np.array(Gi['dist'])
t_base = t_base# - t_base[0]

y_model          = ls_gap_win.offset() * np.ones(len(t_base))
thetas = np.zeros([2])

for fi in f_fft2[1:]:#f_fft2[1:]:
    theta = ls_gap_win.model_parameters(fi)
    thetas = np.vstack([thetas, theta])

    y_model +=  theta[1]*  np.cos(t_base * 2 * np.pi *fi) +  theta[0] * np.sin(t_base * 2 * np.pi *fi )

N=y.size

# %% rewrite model as FFT
Y_from_LS = np.fft.rfft(y_model)


# %% test reconstructed model
plt.plot( t_base , data_gap *win , linewidth=2, alpha=0.9 , c='black', label=' data_grap')
plt.plot(t_base, y_model, '-', label='LS model gap')

plt.plot(t_base[1:], np.fft.irfft(Y_from_LS), '-', label='inv FFT from fft(LS model)')

plt.xlim(t_base[0], t_base[700])
plt.legend()



# %% fft again
Y_from_data_win = np.fft.rfft(y * win)
df3 = np.gradient(f_fft2).mean()

# % parcevel test

print('data var:', y.var() )
print('Data gap var:', data_gap[~nan_mask].var() )

print('with window')
print('data * win var:', (y*win).var() )
spec_data_fft3 = 2.*(Y_from_data_win*Y_from_data_win.conj()).real / df3 /y.size**2
print('data * win fft sum:', df3 * spec_data_fft3.sum().data )

print('LS model var:', y_model.var() )
spec_LS_model_fft3 = 2.*(Y_from_LS*Y_from_LS.conj()).real / df3 /y.size**2
print('LS model fft sum:', df3 * spec_LS_model_fft3.sum().data )


# %%




# %%



# %%
def tanh_weight(x, x_cutoff , x_max_pos,  LF_amp, HF_amp, Gauss_amp, sigma_g):
    """
        zdgfsg
    """
    HF_amp1 = (LF_amp-HF_amp)
    decay   =  0.5 - np.tanh( (x-x_cutoff)/sigma_g  )/2
    y       =  decay * HF_amp1 + (1 - HF_amp1)
    y  = y- y[0] +LF_amp

    def gaus(x, x_0, amp, sigma_g ):
        return amp* np.exp(-0.5 * (  (x-x_0)/sigma_g)**2)

    y += gaus(x, x_max_pos, Gauss_amp, sigma_g )

    #y =  y * LF_amp
    return y

plt.plot( f_fft2, tanh_weight( f_fft2,  0.025 + 0.02 , 0.025 ,  1 ,  0.1, 0.5, 0.003)  )
# %%


# %%

def objective_func(params, data_x, Z_results, weight_func, freq, nan_mask = None, plot_flag=False):
    # model_x= model_real_space
    # data= y

    alpha =1e7
    def model_real_space(Z, weights):
        """
        Both inputs must have the same length
        """
        return np.fft.irfft(Z*weights)

    weights = weight_func(freq[1:], params)

    if nan_mask is not None:
        model = model_real_space(Z_results, weights)[~nan_mask[1:]]
        dd = data_x[~nan_mask][1:]
        #print(sum(np.isnan(dd)))
    else:
        model = model_real_space(Z_results, weights)[:]
        dd = data_x[1:]

    if plot_flag:

        F= M.figure_axis_xy(10, 4.1 * 2.5, view_scale= 0.5, container = True)

        gs = GridSpec(5,1,  wspace=0.1,  hspace=0.4)#figure=fig,
        pos0,pos1,pos2 = gs[0:3, 0],gs[3, 0],gs[4, 0]#,gs[3, 0]
        ax1 = F.fig.add_subplot(pos0)
        plt.title('Stacked Timeseries', loc='left')

        chunk_l= 400
        chunk_iter = spec.create_chunk_boundaries(chunk_l, data_x.size, ov=0, iter_flag = True)

        ofsett0= 6
        ofsett = np.copy(ofsett0)
        for chi in chunk_iter:

            v1= np.round(np.nanvar(dd), 4)
            plt.plot(ofsett+ data_x[chi[0]:chi[-1]] , linewidth=3, alpha=0.5 , c='black', label=' org. data (var:'+str(v1)+')')

            model_init = model_real_space(Z_results, weights*0 +1)[~nan_mask[1:]]
            v1= np.round(model_init.var(), 4)
            plt.plot(ofsett + model_real_space(Z_results, weights*0 +1)[chi[0]:chi[-1]] ,linewidth= 0.8, c='red', label='LS model init (var:'+str(v1)+')')

            v1= np.round(model.var(), 4)
            plt.plot(ofsett + model_real_space(Z_results, weights)[chi[0]:chi[-1]],linewidth= 0.8, c='blue', label='LS model weighted (var:'+str(v1)+')')

            if ofsett == ofsett0:
                plt.legend()
            ofsett -= 1

        plt.ylim(ofsett, ofsett0+1)
        plt.xlim(0, chunk_l*2)


        ax2 = F.fig.add_subplot(pos1)
        #ax2 = plt.subplot(3, 1, 2)
        plt.title('Amplitude Weight Function', loc='left')
        plt.plot(weights , c='black')
        ax2.set_xscale('log')

        ax3 = F.fig.add_subplot(pos2)
        plt.title('Initial and tuned |Z|', loc='left')

        #ax3 = plt.subplot(3, 1, 3)

        # v2_fft= np.fft.rfft(data_x)
        # v2 = np.round( (2.*(v2_fft*v2_fft.conj()).real  /data_x.size**2 ).sum(), 4)
        # plt.plot(abs(v2_fft) , linewidth=2, alpha=0.5 , c='black', label='org data (var: '+str(v2) +')')

        v2 = np.round( (4.*(Z_results*Z_results.conj()).real  /data_x.size**2 ).sum(), 4)
        plt.plot(abs(Z_results), linewidth= 0.8,  c='red', label='Z (var: '+str(v2) +')')
        plt.plot(M.runningmean(abs(Z_results) , 20, tailcopy=True), linewidth= 1.5, c='red', zorder=12)

        Z2= Z_results* weights
        v2 = np.round( (4.*(Z2*Z2.conj()).real  /data_x.size**2 ).sum(), 4)
        plt.plot(abs(Z2), linewidth= 0.8, c='blue', label='weighted Z(var: '+str(v2) +')')
        plt.plot(M.runningmean(abs(Z2) , 20, tailcopy=True), linewidth= 1.5, c='blue', zorder=12)
        plt.legend()

        plt.ylim( np.percentile(abs(Z_results), 0.5), abs(Z_results).max()*1.3 )
        plt.xlabel('wavenumber k')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

    fitting_cost =( abs(dd - model) / dd.std() )**2
    variance_cost =( abs(dd.var() -  model.var())  / dd.std() ) **2

    return fitting_cost.sum() , alpha* variance_cost


def tanh_weight_function(ff, params):
    return tanh_weight(ff, params['x_cutoff'].value,
                        params['x_max_pos'].value,
                        params['LF_amp'].value,
                        params['HF_amp'].value,
                        params['Gauss_amp'].value,
                        params['sigma_g'].value )

params = lmfit.Parameters()

p_smothed = M.runningmean(np.abs(Y_from_LS), 20, tailcopy=True)
f_max = f_fft2[p_smothed[~np.isnan(p_smothed)].argmax()]

lambda_max = 9.81 * 5 **2/ (2* np.pi)
params.add('x_cutoff', 1/lambda_max , min=f_max*0.75, max=f_max*5,  vary=False)
params.add('x_max_pos', f_max       , min=f_max*0.75, max=f_max*5,  vary=False)
params.add('LF_amp',        1       , min=0.5       , max=1.2,      vary= True)
params.add('HF_amp', 0.5            ,  min=0        , max=1.5,      vary= True)
params.add('sigma_g', 0.002         , min=0.001     , max=0.05,     vary= False)
params.add('Gauss_amp', 0.5         , min=0.01      , max=2,        vary= True)

# %
dx = t_base[2]- t_base[1]
stancil=t_base[0], t_base[-1]
x_model           = np.arange(stancil[0], stancil[-1] + dx, dx)

x_pos =  np.round( (t_base[~nan_mask])/ 10.0, 0).astype('int')
x_pos = x_pos- x_pos[0]
#x_pos[x_pos >= t_base.size] = t_base.size-1

y_gridded= np.copy(x_model) * 0
y_gridded[x_pos] = (data_gap*win)[~nan_mask]
y_gridded=y_gridded[:-1]

data_gap.shape
y_gridded.shape
#tanh_weight_function(f_fft3, params


objective_func(params, (data_gap*win), Y_from_LS, tanh_weight_function, f_fft2,  nan_mask = nan_mask , plot_flag= True)
# %%

params.pretty_print()
D_data = (data_gap * win)#[~nan_mask]
Z_data= Y_from_LS# np.fft.rfft(m1_gap_windowed)
D_data.shape
# Z_mask= Z_data_final <.5
# Z_2 = np.copy(Z_data_final)
# Z_2[Z_mask] =0
# Z_data = Z_2

1/f_fft2[450]

f_fft2[450]

1/30



fitter = lmfit.minimize(objective_func, params, args=(D_data, Z_data, tanh_weight_function, f_fft2),
        kws={'nan_mask':nan_mask} , method='dual_annealing')

fitter.params.pretty_print()

objective_func(fitter.params, D_data, Z_data, tanh_weight_function, f_fft2, nan_mask = nan_mask, plot_flag= True)

plt.plot(M.runningmean(np.abs(Y_from_data_win) , 20, tailcopy=True), linewidth = 1.2, c='black' ,label= 'LS full data', zorder=12)
plt.legend()

data_gap

# %%
f_fft2


Z_data_final = Z_data * tanh_weight_function(f_fft2[1:], fitter.params)

Z_mask= Z_data_final <3
plt.plot(Z_data_final)

Z_2 = np.copy(Z_data_final)
Z_2[Z_mask] =0


# %%
plt.plot( data_gap )
plt.plot( np.fft.irfft(Z_2), 'k' )
plt.xlim(0, 10`00)
# %%

# %%
imp.reload(spec)
P = spec.conserve_variance(Z_data, f_fft2, D_data, nan_mask = nan_mask )
P.set_parameters()
P.params.pretty_print()
P.test_ojective_func(P.tanh_weight_function, plot_flag=False)

fitter = P.optimize()


P.plot_result()
plt.plot(np.fft.ifft(P.best_guess_Z() ))
