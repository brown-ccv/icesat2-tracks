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

nan_mask.sum()/data_gap.size

plt.plot( Gi['dist'], y, '-')
plt.plot( Gi['dist'], data_gap, '-')

# %%
def create_weighted_window(data, window=None):
    """
    define window function and weight it to conserve variance
    if window is not None it show have a length of N
    """
    L = data.size
    if window is None:
        #win=np.hanning(L)
        import scipy.signal.windows as WINDOWS
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

windows = create_window_mask(nan_mask, data_gap )

np.nanvar(data_gap)
np.nanvar(( data_gap* windows)[~nan_mask] )

plt.plot(data_gap* windows)
# %%

f_fft2, df2 = spec.calc_freq_fft( np.array(Gi['dist']), y.size)

spec_fft2 = spec.calc_spectrum_fft(y, df2, y.size)
f_fft2_regular, spec_fft2_regular = f_fft2, spec_fft2

window_single = spec.create_window(y.size)
win = create_weighted_window(y)

dd = np.copy(data_gap)
dd[nan_mask]= 0
spec_fft_gap_window1 = spec.calc_spectrum_fft(dd * window_single, df2, y.size)
spec_fft_windowed = spec.calc_spectrum_fft(y * window_single, df2, y.size)

spec_fft_gap_windowed = spec.calc_spectrum_fft(y * windows, df2, y.size)


# %% simple spectra
from astropy.timeseries import LombScargle
ls = LombScargle( np.array(Gi['dist']) , y, fit_mean=False)
fft_sub = f_fft2[1:]
ls_auto_power = ls.power(fft_sub , normalization='psd', assume_regular_frequency= False)

ls_windowed = LombScargle( np.array(Gi['dist']) , y *win, fit_mean=False)
ls_auto_power_windowed = ls_windowed.power(fft_sub , normalization='psd', assume_regular_frequency= False)

ls_gap = LombScargle( np.array(Gi['dist'])[~nan_mask] , data_gap[~nan_mask] , fit_mean=False)
ls_gap_auto_power = ls_gap.power(fft_sub , normalization='psd', assume_regular_frequency= False)

ls_gap_windowed = LombScargle( np.array(Gi['dist'])[~nan_mask], ( data_gap* windows)[~nan_mask] , fit_mean=False)
ls_auto_power_gap_windowed = ls_windowed.power(fft_sub , normalization='psd', assume_regular_frequency= False)

ls_gap_window1 = LombScargle( np.array(Gi['dist'])[~nan_mask], (data_gap* window_single)[~nan_mask] , fit_mean=False)
ls_auto_power_gap_window1 = ls_windowed.power(fft_sub , normalization='psd', assume_regular_frequency= False)



#ls_auto_f , ls_auto_power = l s.autopower(normalization='psd', samples_per_peak=0.1)
F= M.figure_axis_xy(10, 4, view_scale= 0.8)
plt.subplot(2, 1,1)
#plt.plot(f_fft2[1::] , spec.LS_power_to_PSD(ls_auto_power_windowed, y.size, df2), '^'  , label= 'LS single window, no gaps')
plt.plot(f_fft2[:-1], spec_fft_windowed, '*-' , label= 'single windowed FFT, no gaps')
plt.plot(f_fft2[:-1], spec_fft_gap_windowed, '-' , label= 'single windowed FFT, gaps')

#plt.plot(f_fft2[:-1], spec_fft_gap_window1 , label= 'single windowed standard FFT, gaps')

#plt.plot(fft_sub , spec.LS_power_to_PSD(ls_auto_power, y.size, df2)  , label= 'LS no window', zorder=12)
plt.plot(f_fft2[1::] , spec.LS_power_to_PSD(ls_auto_power_gap_windowed, y.size, df2)  , label= 'LS_gap_windows')
plt.plot(f_fft2[1::] , spec.LS_power_to_PSD(ls_auto_power_gap_window1, y.size, df2)  , label= 'LS_gap_1 window')
#plt.plot(f_fft2[1::] , spec.LS_power_to_PSD(ls_gap_auto_power, y.size, df2)  , label= 'LS_gap no windowed')

#plt.plot(f_fft2[1::] , spec.LS_power_to_PSD(spec_fft_gap_windowed, y.size, df2)  , label= 'FFFT_gap_windowed')

#plt.plot(f_fft2[:-1], spec_fft2 , label= 'standard FFT')

plt.legend()
plt.xlim(0, 0.01)

plt.subplot(2, 1,2)


# yn= 1
# y_test= y[::yn]
# f_fft_sub, df2 = spec.calc_freq_fft( np.array(Gi['dist'])[::yn], y_test.size)
# plt.plot(f_fft_sub, np.angle(np.fft.rfft(y_test)) , label= 'standard FFT')

plt.plot(f_fft2[:-1],np.angle(np.fft.rfft(y * windows)) , label= 'windowed standard FFT')

plt.plot(f_fft2[:-1],np.angle(np.fft.rfft(y)) , label= 'standard FFT')

plt.xlim(0, 0.01)




# %%
# test getting phase

t_base =np.array(Gi['dist'])
t_base = t_base# - t_base[0]

m1          = ls.offset() * np.ones(len(t_base))
m1_windowed = ls_windowed.offset() * np.ones(len(t_base))
m1_gap      = ls_gap.offset() * np.ones(len(t_base))
m1_gap_window1 = ls_gap_window1.offset() * np.ones(len(t_base))
m1_gap_windowed = ls_gap_windowed.offset() * np.ones(len(t_base))

fi_divide= 1000
thetas = np.zeros([2])
thetas_v2 = np.zeros([2])

for fi in fft_sub:#f_fft2[1:]:
    #m1 += ls.model(t_base, fi)
    #print(ls.model_parameters(fi))
    theta = ls.model_parameters(fi)
    m1 +=   theta[0] * np.sin(t_base * 2 * np.pi *fi ) + theta[1]*  np.cos(t_base * 2 * np.pi *fi)

    theta = ls_windowed.model_parameters(fi)
    m1_windowed +=   theta[0] * np.sin(t_base * 2 * np.pi *fi ) + theta[1]*  np.cos(t_base * 2 * np.pi *fi)

    #m1_man += theta[0] +  theta[1] * np.sin(t_base * 2 * np.pi *fi ) + theta[2]*  np.cos(t_base * 2 * np.pi *fi)
    thetas = np.vstack([thetas, theta])

    theta = ls_gap.model_parameters(fi)
    m1_gap +=   theta[0] * np.sin(t_base * 2 * np.pi *fi ) + theta[1]*  np.cos(t_base * 2 * np.pi *fi)

    theta = ls_gap_window1.model_parameters(fi)
    m1_gap_window1 +=   theta[0] * np.sin(t_base * 2 * np.pi *fi ) + theta[1]*  np.cos(t_base * 2 * np.pi *fi)

    theta = ls_gap_windowed.model_parameters(fi)
    m1_gap_windowed +=   theta[0] * np.sin(t_base * 2 * np.pi *fi ) + theta[1]*  np.cos(t_base * 2 * np.pi *fi)


    # ii= np.array([theta[0] * np.sqrt(2)  *np.sin(np.arctan(theta[1]/ theta[0]) ) , theta[1]*  np.sqrt(2)  * np.cos(np.arctan(theta[1]/ theta[0])) ])
    # thetas_v2 = np.vstack([ thetas_v2, ii])

thetas = thetas[1:, :]
# thetas_v2 = thetas_v2[1:, :]

N=y.size

# test
# fft_contstruct =  N*  np.complex128(thetas[:,1] ) /2
# fft_contstruct.imag = N* (thetas[:,0]) /2

# %%

# m2 = ls.offset() * np.ones(len(t_base))
# for fi in f_fft2[fi_divide+1::1]:
#     m2 += ls.model(t_base, fi)

# %%
plt.plot( t_base , data_gap , linewidth=2, alpha=0.5 , c='black', label=' data_grap')
plt.plot(t_base, m1_gap, '-', label='LS model gap')
plt.xlim(t_base[100], t_base[1000])
plt.legend()
# %%

#plt.plot( t_base , y , linewidth=2, alpha=0.5 , c='black', label=' org. data')
plt.plot( t_base , data_gap *window_single , linewidth=2, alpha=0.5 , c='black', label=' data_grap * win')


#plt.plot(t_base, m1)
#plt.plot(t_base, m1, '*', label='LS model', alpha=0.2)

#plt.plot(t_base, m1_windowed, '-', label='LS model windowed', alpha =0.6)
plt.plot(t_base, m1_gap_window1, '-', label='LS model gap 1 window', alpha =0.6)

plt.xlim(t_base[100], t_base[1000])
plt.legend()


# %%
#plt.plot( t_base[1:-1] , (y -m1)[1:-1], linewidth=2, alpha=0.5 , c='black', label=' org. data')
#plt.plot( t_base[1:-1] , (y -m1_windowed)[1:-1], linewidth=2, alpha=0.5 , c='black', label=' org. data')
#plt.xlim(t_base[500], t_base[600])


# %% fft again
phih_from_LS = np.fft.rfft(m1 * win)
phih_from_LS_gap = np.fft.rfft(m1_gap * win)
phih_from_data = np.fft.rfft(y * win)
f_fft3 = np.fft.rfftfreq(m1.size, d=10.0)
df3 = np.gradient(f_fft3).mean()

# % parcevel test

print('with Hanning window')
print('data var:', y.var() )
spec_data_fft3 = 2.*(phih_from_data*phih_from_data.conj()).real / df3 /y.size**2
print('data fft sum:', df3 * spec_data_fft3.sum().data )

print('LS model var:', m1.var() )
spec_LS_model_fft3 = 2.*(phih_from_LS*phih_from_LS.conj()).real / df3 /m1.size**2
print('LS model fft sum:', df3 * spec_LS_model_fft3.sum().data )


print('Data gap var:', data_gap[~nan_mask].var() )
print('LS model without gap var:', m1_gap[~nan_mask].var() )

print('LS model with gap var:', m1_gap.var() )
spec_LS_model_gap_fft3 = 2.*(phih_from_LS_gap*phih_from_LS_gap.conj()).real / df3 /y.size**2
print('Ls model with gap fft sum:', df3 * spec_LS_model_gap_fft3.sum().data )


# %% fft again
phih_from_LS = np.fft.rfft(m1)
phih_from_LS_gap = np.fft.rfft(m1_gap)
phih_from_data = np.fft.rfft(y)
f_fft3 = np.fft.rfftfreq(m1.size, d=10.0)
df3 = np.gradient(f_fft3).mean()

# % parcevel test
print('withOUT Hanning window')
print('data var:', y.var() )
spec_data_fft3 = 2.*(phih_from_data*phih_from_data.conj()).real / df3 /y.size**2
print('data fft sum:', df3 * spec_data_fft3.sum().data )

print('LS model var:', m1.var() )
spec_LS_model_fft3 = 2.*(phih_from_LS*phih_from_LS.conj()).real / df3 /m1.size**2
print('LS model fft sum:', df3 * spec_LS_model_fft3.sum().data )


print('Data gap var:', data_gap[~nan_mask].var() )
print('LS model without gap var:', m1_gap[~nan_mask].var() )

print('LS model with gap var:', m1_gap.var() )
spec_LS_model_gap_fft3 = 2.*(phih_from_LS_gap*phih_from_LS_gap.conj()).real / df3 /y.size**2
print('Ls model with gap fft sum:', df3 * spec_LS_model_gap_fft3.sum().data )



# %%
np.angle(phih_from_LS_gap)


F= M.figure_axis_xy(12, 4.1 * 2.5, view_scale= 0.5, container=True)
gs = GridSpec(4,1,  wspace=0.1,  hspace=0.7)#figure=fig,

pos0,pos1,pos2,pos3  = gs[0, 0],gs[1, 0],gs[2, 0],gs[3, 0]
ax0 = F.fig.add_subplot(pos0)

#plt.subplot(4, 1, 1)
plt.plot(t_base, m1, '.', c='blue', label='LS model')
plt.plot(t_base, m1_gap, '-',c='red', label='LS model')
#plt.plot(t_base, m2 - m2.mean())

plt.plot( t_base , y , linewidth=2, alpha=0.5 , c='black', label=' org. data')
plt.xlim(t_base[10], t_base[2500])
plt.ylim(-1, 1)
plt.legend()


def gaus(x, x_0, amp, sigma_g ):
    return amp* np.exp(-0.5 * (  (x-x_0)/sigma_g)**2)


def gauss_weight(x, x_positions,  amplitudes, sigma_g=5):
    """
    postions and amplitudes must have the same length

    """

    y= np.ones(x.size)
    for p,a in zip(x_positions, amplitudes):
        y += gaus(x, p, a,  sigma_g)

    return y




weights = gauss_weight(f_fft3, [0.001, 0.005, 0.045], [0, -.5, -.9], sigma_g= 0.002)

plt.plot(f_fft3,  weights )

ax2 = F.fig.add_subplot(pos1)
#ax2 = plt.subplot(4, 1, 2)
plt.plot(f_fft3,  np.abs(phih_from_LS), c='blue', linewidth = 0.5 )
plt.plot(f_fft3, M.runningmean(np.abs(phih_from_LS) , 10, tailcopy=True), linewidth = 1.2, c='blue' ,label= 'LS full data')
plt.plot(f_fft3, np.abs(phih_from_LS_gap), c='red', linewidth = 0.5 , alpha= 0.6)
plt.plot(f_fft3, M.runningmean(np.abs(phih_from_LS_gap) , 10, tailcopy=True), linewidth = 1.2, c='red' ,label= 'LS gap data')


plt.plot(f_fft3, np.abs(phih_from_LS_gap) *weights , linewidth = 0.5, c='orange', alpha= 0.6)
plt.plot(f_fft3, M.runningmean(np.abs(phih_from_LS_gap) *weights, 10, tailcopy=True), linewidth = 1.2, c='orange' , label='LS gap weighted' )


plt.legend()

plt.xlim(f_fft3[1], f_fft3[-1])
ax2.set_xscale('log')
ax2.set_yscale('log')


ax3 = F.fig.add_subplot(pos2)
plt.plot(f_fft3, weights , linewidth = 0.5, c='k', label='LS gap weighted' )



plt.xlim(f_fft3[0], f_fft3[-1])

plt.xlim(f_fft3[1], f_fft3[-1])
ax3.set_xscale('log')

#ax3.set_yscale('log')

ax4 = F.fig.add_subplot(pos3)
plt.plot(t_base, m1, '-', c='blue', linewidth = 0.5,label='LS model')
#plt.plot(t_base, m1_gap, '-',linewidth = 0.5, c='red', label='LS model', zorder=10)

data_model = np.fft.irfft(phih_from_LS_gap*weights)
plt.plot(t_base[1:], data_model , alpha= 0.8, c='orange',  label='LS weighted model')
#plt.plot(t_base, m2 - m2.mean())

#plt.plot( t_base , y , linewidth=2, alpha=0.5 , c='black', label=' org. data')
plt.xlim(t_base[10], t_base[2500])

plt.ylim(-1, 1)
plt.legend()


print(' cost', (abs(data_gap[~nan_mask][1:] - data_model[~nan_mask[1:]])**2).sum() )

# %%



def objective_func(params, data_x, Z_results, weight_func, nan_mask = None, plot_flag=False):
    # model_x= model_real_space
    # data= y

    alpha =1e3
    def model_real_space(Z, weights):
        """
        Both inputs must have the same length
        """
        return np.fft.irfft(Z*weights)

    weights = weight_func(f_fft3, params)

    if nan_mask is not None:
        model = model_real_space(Z_results, weights)[~nan_mask[1:]]
        dd = data[~nan_mask][1:]
    else:
        model = model_real_space(Z_results, weights)[:]
        dd = data[1:]

    if plot_flag:

        F= M.figure_axis_xy(10, 4.1 * 2.5, view_scale= 0.5, container = True)

        gs = GridSpec(5,1,  wspace=0.1,  hspace=0.2)#figure=fig,
        pos0,pos1,pos2 = gs[0:3, 0],gs[3, 0],gs[4, 0]#,gs[3, 0]
        ax1 = F.fig.add_subplot(pos0)

        chunk_l= 400
        chunk_iter = spec.create_chunk_boundaries(chunk_l, data_x.size, ov=0, iter_flag = True)

        ofsett0= 6
        ofsett = np.copy(ofsett0)
        for chi in chunk_iter:

            v1= np.round(np.nanvar(dd), 4)
            plt.plot(ofsett+ data_x[chi[0]:chi[-1]] , linewidth=3, alpha=0.5 , c='black', label=' org. data (var:'+str(v1)+')')

            v1= np.round(model_x(Z_results, weights*0 +1)[~nan_mask[1:]].var(), 4)
            plt.plot(ofsett + model_x(Z_results, weights*0 +1)[chi[0]:chi[-1]] ,linewidth= 0.8, c='red', label='LS model init (var:'+str(v1)+')')

            v1= np.round(model.var(), 4)
            plt.plot(ofsett + model_x(Z_results, weights)[chi[0]:chi[-1]],linewidth= 0.8, c='blue', label='LS model weighted (var:'+str(v1)+')')

            if ofsett == ofsett0:
                plt.legend()
            ofsett -= 1

        plt.ylim(ofsett, ofsett0+1)
        plt.xlim(0, chunk_l*2)


        ax2 = F.fig.add_subplot(pos1)
        #ax2 = plt.subplot(3, 1, 2)
        plt.plot(weights )
        ax2.set_xscale('log')

        ax3 = F.fig.add_subplot(pos2)
        #ax3 = plt.subplot(3, 1, 3)

        # v2_fft= np.fft.rfft(data_x)
        # v2 = np.round( (2.*(v2_fft*v2_fft.conj()).real  /data_x.size**2 ).sum(), 4)
        # plt.plot(abs(v2_fft) , linewidth=2, alpha=0.5 , c='black', label='org data (var: '+str(v2) +')')

        v2 = np.round( (2.*(Z_results*Z_results.conj()).real  /data_x.size**2 ).sum(), 4)
        plt.plot(abs(Z_results), linewidth= 0.8,  c='red', label='Z (var: '+str(v2) +')')
        plt.plot(M.runningmean(abs(Z_results) , 20, tailcopy=True), linewidth= 1.5, c='red', zorder=12)

        Z2= Z_results* weights
        v2 = np.round( (2.*(Z2*Z2.conj()).real  /data_x.size**2 ).sum(), 4)
        plt.plot(abs(Z2), linewidth= 0.8, c='blue', label='weighted Z(var: '+str(v2) +')')
        plt.plot(M.runningmean(abs(Z2) , 20, tailcopy=True), linewidth= 1.5, c='blue', zorder=12)
        plt.legend()

        plt.ylim( np.percentile(abs(np.fft.rfft(data)), 1), abs(np.fft.rfft(data)).max()*1.3 )

        ax3.set_xscale('log')
        ax3.set_yscale('log')


    fitting_cost =( abs(dd - model) / dd.std() )**2
    variance_cost =( abs(dd.var() -  model.var())  / dd.std() ) **2

    return fitting_cost.sum() ,alpha* variance_cost

import lmfit

# find energy peak and seed center points for ajustment
p_smothed = M.runningmean(np.abs(phih_from_LS_gap), 20, tailcopy=True)
f_max = f_fft3[p_smothed[~np.isnan(p_smothed)].argmax()]

f_list   = [f_max/ 2,  f_max, f_max]# + 1 * (f_fft3[-1] - f_max)/2]#, f_max + 2 * (f_fft3[-1] - f_max)/3 ]
amp_list = [0 for i in f_list]

fi_low  = 0
params = lmfit.Parameters()
for i in np.arange(len(f_list)):
    f_name   = 'freq'+str(i)
    fi      = f_list[i]
    if i+1 < len(f_list):
        fi_high = f_list[i+1]
    else:
        fi_high = f_fft3[-1]
    params.add(f_name, fi, min=fi_low, max=fi_high)
    fi_low  = fi


for i in np.arange(len(amp_list)):
    a_name   = 'amp'+str(i)
    params.add(a_name, 0.5, min=-1, max=2)

params.add('sigma_g', 0.001, min=0.001, max=0.1, vary= False)

params['freq1'].vary = False
params['freq2'].vary = False
params.pretty_print()


def gauss_weight_function(ff, params):

    def reorder_values(params):
        freq_l = list()
        amp_l = list()
        for i in list(params):
            if 'freq' in i:
                freq_l.append(params[i].value)
            elif 'amp' in i:
                amp_l.append(params[i].value)

        return freq_l, amp_l

    if params is not None:
        freq_l, amp_l = reorder_values(params)
        weights = gauss_weight(f_fft3, freq_l, amp_l, sigma_g= params['sigma_g'])
    else:
        weights = gauss_weight(f_fft3, [0.001, 0.005, 0.045], [0, -.5, -.9], sigma_g= params['sigma_g'])

    return weights

plt.plot(f_fft3,  gauss_weight_function(f_fft3, params) )


objective_func(params, data_gap, phih_from_LS_gap, gauss_weight_function, nan_mask = nan_mask, plot_flag= True)

# %% simplest model
params = lmfit.Parameters()
params.add('amp', 0, min=-1, max=1)

def constant_weight(ff, amp):
    """
        zdgfsg
    """
    y =  (ff*0+1) + amp
    return y



def constant_weight_function(ff, params):
    return constant_weight(ff, params['amp'].value)

objective_func(params, data_gap, phih_from_LS_gap, constant_weight_function, nan_mask = nan_mask, plot_flag= True)

# %%
cost_list=list()
amp_list= np.arange(-0.8, 1, 0.05)
for aa in amp_list:
    params['amp'].value = aa
    cost= objective_func(params, data_gap, phih_from_LS_gap, constant_weight_function, nan_mask = nan_mask, plot_flag= False)
    cost_list.append(cost)

cost_list = np.array(cost_list)
plt.plot(amp_list, cost_list[:,0], label='fitting cost')
plt.plot(amp_list, cost_list[:,1], label='variance cost')
plt.plot(amp_list, cost_list[:,0]+ cost_list[:,1], label='total cost')
plt.legend()

# %%

#args=(y, model_real_space)
#kws={'nan_mask':nan_mask, 'sigma_g':0.002 }

fitter = lmfit.minimize(objective_func, params, args=(data_gap, phih_from_LS_gap, constant_weight_function),
        kws={'nan_mask':nan_mask} , method='least_squares')

fitter.params.pretty_print()
objective_func(fitter.params, data_gap, phih_from_LS_gap, constant_weight_function, nan_mask = nan_mask, plot_flag= True)

# %%
def tanh_weight(x, x_positions,  LF_amp, HF_amp, sigma_g):
    """
        zdgfsg
    """
    HF_amp1 = (LF_amp-HF_amp)
    decay   =  0.5 - np.tanh( (x-x_positions)/sigma_g  )/2
    y       =  decay * HF_amp1 + (1 - HF_amp1)
    y  = y- y[0] +LF_amp
    #y =  y * LF_amp
    return y

#plt.plot( f_fft3, tanh_weight( f_fft3,  0.025, 1 ,  1, 0.005)  )

def tanh_weight_function(ff, params):
    return tanh_weight(ff, params['x_pos'].value,params['LF_amp'].value,params['HF_amp'].value,params['sigma_g'].value, )

params = lmfit.Parameters()

p_smothed = M.runningmean(np.abs(phih_from_LS_gap), 20, tailcopy=True)
f_max = f_fft3[p_smothed[~np.isnan(p_smothed)].argmax()]
params.add('x_pos', f_max, min=f_max*0.75, max=f_max*1.25, vary=True)

params.add('LF_amp', 1, min=.2, max=1.5)
params.add('HF_amp', 1, min=.2, max=1.5)
params.add('sigma_g', 0.002, min=0.001, max=0.05, vary= False)


#tanh_weight_function(f_fft3, params)
objective_func(params, data_gap, phih_from_LS_gap, tanh_weight_function, nan_mask = nan_mask, plot_flag= True)


# %%

def objective_func(params, data_x, Z_results, weight_func, nan_mask = None, plot_flag=False):
    # model_x= model_real_space
    # data= y

    alpha =1e7
    def model_real_space(Z, weights):
        """
        Both inputs must have the same length
        """
        return np.fft.irfft(Z*weights)

    weights = weight_func(f_fft3, params)

    if nan_mask is not None:
        model = model_real_space(Z_results, weights)[~nan_mask[1:]]
        dd = data_x[~nan_mask][1:]
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

            v1= np.round(model_x(Z_results, weights*0 +1)[~nan_mask[1:]].var(), 4)
            plt.plot(ofsett + model_x(Z_results, weights*0 +1)[chi[0]:chi[-1]] ,linewidth= 0.8, c='red', label='LS model init (var:'+str(v1)+')')

            v1= np.round(model.var(), 4)
            plt.plot(ofsett + model_x(Z_results, weights)[chi[0]:chi[-1]],linewidth= 0.8, c='blue', label='LS model weighted (var:'+str(v1)+')')

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

        plt.ylim( np.percentile(abs(np.fft.rfft(data)), 0.5), abs(np.fft.rfft(data)).max()*1.3 )
        plt.xlabel('wavenumber k')
        ax3.set_xscale('log')
        ax3.set_yscale('log')

    fitting_cost =( abs(dd - model) / dd.std() )**2
    variance_cost =( abs(dd.var() -  model.var())  / dd.std() ) **2

    return fitting_cost.sum() , alpha* variance_cost





# D_data = data_gap *window_single
#
# Z_data = np.copy(data_gap *win)
# Z_data[nan_mask]=0
# Z_data= np.fft.rfft(Z_data)
#Z_data.shape

D_data = data_gap * window_single
Z_data= np.fft.rfft(m1_gap_window1)

D_data = data_gap# * window_single
Z_data= np.fft.rfft(m1_gap)

# D_data = data_gap# * window_single
# Z_data= np.fft.rfft(m1_gap_window1)

# D_data = data_gap * windows
# Z_data= np.fft.rfft(m1_gap_windowed)

# Z_data=  phih_from_LS_gap
# Z_data.shape




fitter = lmfit.minimize(objective_func, params, args=(D_data, Z_data, tanh_weight_function),
        kws={'nan_mask':nan_mask} , method='dual_annealing')

fitter.params.pretty_print()
objective_func(fitter.params, D_data, Z_data, tanh_weight_function, nan_mask = nan_mask, plot_flag= True)


# v2_fft= np.fft.rfft(data_x)
#v2 = np.round( (2.*(v2_fft*v2_fft.conj()).real  /data_x.size**2 ).sum(), 4)
# plt.plot(abs(v2_fft) , linewidth=2, alpha=0.5 , c='black', label='org data (var: '+str(v2) +')')

#plt.plot(np.abs(phih_from_LS), c='gray', linewidth = 2, alpha=0.5 )
plt.plot(M.runningmean(np.abs(phih_from_data) , 20, tailcopy=True), linewidth = 1.2, c='black' ,label= 'LS full data', zorder=12)
plt.legend()

data_gap

# %%
data_gap * windows

# %%
data_gap * windows


# %%

phih_from_LS



## %%
phih_from_data.shape
plt.plot(y , linewidth=3, alpha=0.5 , c='black')

plt.plot(np.fft.irfft(phih_from_data), 'b')

zz = np.copy(phih_from_data)
zz[300:]= 0
plt.plot(np.fft.irfft(zz))
plt.xlim(0,200)
