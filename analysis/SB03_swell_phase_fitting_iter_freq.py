
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
#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
track_name, batch_key, test_flag = '20190601093502_09790310_004_01', 'SH_batch02', False



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

col.colormaps2(21)
# %%


T_max = 40 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gd['gt1r']['dist'])
dx = np.diff(x).mean()
xlims   = x[0], x[-1]
min_datapoint =  1/k_0/dx

Lpoints = int(np.round(min_datapoint) * 10)
Lmeters =Lpoints  * dx

def get_stacil_data(stancil, G3, key = 'dist' ):

    mask = (G3[key] >= stancil[0]) & (G3[key] < stancil[2])
    return G3[np.array(mask)]

def make_slopes(G3, key = 'heights_c_weighted_mean', spreed =10, verbose= False, photon_min =5):

    dd      = np.copy(G3['heights_c_weighted_mean'])
    dd      = np.gradient(dd)
    dd, _   = spicke_remover.spicke_remover(dd, spreed=spreed, verbose=verbose)
    dd_nans = (np.isnan(dd) ) + (G3['N_photos'] <= photon_min)

    return dd, dd_nans

stancil_iter = spec.create_chunk_boundaries_unit_lengths(Lmeters, xlims, ov= None , iter_flag=False)

# %%
sys.path.append(mconfig['paths']['analysis'])
import SB04_2d_wavefield_emulator as WaveEmulator

# test wave model
k_abs = 2 * np.pi/100
imp.reload(WaveEmulator)
kk, ll, amps, mesh_shape = WaveEmulator.get_stancils_polar(k_abs,0, size=1, dk= k_abs/10, amp_std=k_abs/20  , plot_flag=False)


def wavemodel(XX, YY, ks, ls, amps, group_phase = 0, amp_height= 1):

    import numpy as np

    G = np.vstack([ np.cos(np.outer(XX, ks) + np.outer(YY, ls) ).T ,  np.sin(np.outer(XX, ks) + np.outer(YY, ls) ).T ] ).T

    #phase1 = np.random.rand(1, amp_list.size) *  np.pi*2
    #phase = np.arange(0, amp_list.size) *  np.pi/2

    b = np.hstack([ np.cos(group_phase)*amps, np.sin(group_phase) *amps]).squeeze() * amp_height
    z_model = (G @ b)

    return z_model


# %% select data
Gd.keys()

def get_beam(G3, key):
    G1 = G3[key]
    G1['slopes'], _ = make_slopes(G1)
    G1['beam']= key
    return G1

B1= get_beam(Gd, 'gt1r')
B2= get_beam(Gd, 'gt1l')

dist_y_mean = np.sqrt( (B1['x']-B2['x'])**2 + (B1['y']-B2['y'])**2 ).mean()
dist_y_mean

B1['dist_y'] = 0
B2['dist_y'] = dist_y_mean

GG0 = pd.concat( [B1, B2] )
# GG = get_beam(Gd, 'gt3r')
#GG0['slopes'].plot()
#plt.plot( GG0['dist'] , GG0['slopes'])

#plt.plot( GG0['x']/1e3 , GG0['y']/1e3)
plt.plot( GG0['x']/1e3 , GG0['y']/1e3)

# %%
stancil = stancil_iter[:,  stancil_iter[0,:] > 0.9 *1e6][:, 0]
#stancil = stancil_iter[:, 200]
#stancil_iter.shape
GG = get_stacil_data(stancil, GG0).sort_values(['beam','dist'])
GG = GG.loc[~np.isnan(GG['slopes'])]

GG['x_prime'] = GG['x'] - GG['x'].min()
GG['y_prime'] = GG['y'] - GG['y'].max()
plt.plot(GG['dist'], GG['slopes'] , '-', markersize = 0.8)
plt.show()
plt.plot(GG['x_prime']/1e3, GG['y_prime']/1e3)
plt.show()

# plt.axis('equal')

# %% get peak wavenumber, power and spreed

def get_fft(G4, beam, dx):
    G4 = G4[G4['beam'] == beam]
    ys = np.copy(G4['slopes'])
    ys[np.isnan(ys)] =0
    k_fft = np.fft.rfftfreq(ys.size, d=dx) * 2* np.pi
    return np.fft.rfft(ys), k_fft

B1_fft, k_fft = get_fft(GG, 'gt1l', dx)
B2_fft, k_fft = get_fft(GG, 'gt1r', dx)

Z_fft      = (B1_fft+ B2_fft)/2
Z_fft_rm   = M.runningmean(abs(Z_fft), 4, tailcopy=True)

k_fft      = k_fft[~np.isnan(Z_fft_rm)]
Z_fft_rm   = Z_fft_rm[~np.isnan(Z_fft_rm)]

Z_max_pos = Z_fft_rm.argmax()
k_fft_max  = k_fft[Z_max_pos]
Z_max      = Z_fft_rm[Z_max_pos]

# Z_sort = Z_fft_rm.argsort()[::-1]
# k_fft[Z_sort].shape
# K_fft_sel = k_fft[Z_sort][np.cumsum(Z_fft_rm[Z_sort])/np.sum(Z_fft_rm) <0.2]
# Z_fft_sel = Z_fft_rm[Z_sort][np.cumsum(Z_fft_rm[Z_sort])/np.sum(Z_fft_rm) <0.2]
#
# plt.plot(K_fft_sel, Z_fft_sel, '.')
plt.plot(k_fft, Z_fft_rm )

# %%
font_for_pres()
plt.plot(k_fft, abs(Z_fft)[:-1] )
plt.plot(k_fft, Z_fft_rm )
treshold = np.nanmean(Z_fft_rm)  + np.nanstd(Z_fft_rm) *2
k_fft_max_list  = k_fft[Z_fft_rm > treshold]
Z_max_list      = Z_fft_rm[Z_fft_rm > treshold]

plt.plot(k_fft_max, Z_fft_rm[Z_max_pos], '.', markersize= 20)
plt.plot(k_fft_max_list, Z_max_list, '.', markersize= 20)
2*np.pi/.15

#plt.plot(k_fft_max_list,Z_max_list, '.', markersize= 20)


def gaus(x, x_0, amp, sigma_g ):
    return amp* np.exp(-0.5 * (  (x-x_0)/sigma_g)**2)

k_sigma =0.02
plt.plot(k_fft, gaus(k_fft,  k_fft_max, Z_max , k_sigma ), 'r', linewidth = 2 )


k_list = np.arange(k_fft_max - 1* k_sigma, k_fft_max + 1* k_sigma+ np.diff(k_fft).mean(), np.diff(k_fft).mean())
plt.plot(k_list, gaus(k_list,  k_fft_max, Z_max , k_sigma ), 'b.', linewidth = 2 )

#plt.xlim(0, 0.09)

# %%

def get_wavenumbers_polar( amp, angle_rad):
    """
    inputs:

    amp     length of peak k vector in radial coordinates
    angle_rad   angle of peak k vector in radians between - pi/2 to  + pi/2

    returns:
    wavenumber k,l
    """
    import numpy as np
    k0 = amp * np.cos(angle_rad)
    l0 = amp * np.sin(angle_rad)

    return k0, l0

def wavemodel(XX, YY, ks, ls, amps, group_phase = 0):

    import numpy as np

    G = np.vstack([ np.cos(np.outer(XX, ks) + np.outer(YY, ls) ).T ,  np.sin(np.outer(XX, ks) + np.outer(YY, ls) ).T ] ).T

    #phase1 = np.random.rand(1, amp_list.size) *  np.pi*2
    #phase = np.arange(0, amp_list.size) *  np.pi/2

    b = np.hstack([ np.cos(group_phase) * amps , np.sin(group_phase) * amps ]).squeeze()
    z_model = (G @ b)

    return z_model

# %%

def get_z_model(x_positions, y_position, K_prime, K_amp,  alpha_rad, group_phase):

    K_abs = K_prime   / np.cos(alpha_rad)

    k = K_abs * np.cos(alpha_rad)
    l = K_abs * np.sin(alpha_rad)

    return wavemodel( x_positions,y_position, k, l, np.array(K_amp ), group_phase= group_phase)



def objective_func(pars, x, y, z, test_flag= False ):

    z_model = get_z_model(x, y, pars['K_prime'], pars['K_amp'], pars['alpha'],pars['group_phase'])
    cost =( abs(z - z_model) )**2  /z.std()**2
    if test_flag:
        return z_model
    else:
        return cost


def plot_instance(GG3, data_key, model_key, non_dim=False , view_scale = 0.3):

    import itertools
    F = M.figure_axis_xy(8, 5, view_scale = view_scale)

    beam_list = list(set(GG3['beam']))


    col_list = itertools.cycle([col.cascade2, col.rascade2, col.cascade1, col.rascade1, col.cascade3, col.rascade3])
    subz = len(beam_list)
    for beam, i in zip(beam_list, np.arange(1, subz+1, 1)):

        plt.subplot(subz, 1, i)
        plt.title(beam  )
        #plt.plot(GG['x_prime'], GG['z_model'], '.' )

        GGsel =GG3[GG3['beam'] == beam]
        plt.plot(GGsel['dist'], GGsel[model_key],  c=col.gray)

        if non_dim:
            # normalized version
            plt.plot(GGsel['dist'], GGsel['slopes']/GGsel['slopes'].std() , '-', c=next(col_list))
        else:
            # diumensional version
            plt.plot(GGsel['dist'], GGsel[data_key] , '-', c=next(col_list))

    return F



#GG = pd.concat( [B1, B2] )
#GG = pd.concat( [B1, B2] )

# stancil = stancil_iter[:, 200]
# GG = get_stacil_data(stancil, GG).sort_values(['beam','dist'])
# GG = GG.loc[~np.isnan(GG['slopes'])]
# np.isnan(GG['slopes']).sum()
#GG['slopes'] = zm
#slopes1 = GG[GG['beam'] == 'gt3r']['slopes']
#GG['slopes'] = np.concatenate([slopes1, slopes1])

# alpha_true = 75 * np.pi/180 # 0.95 * (np.pi/2)
# dy  = 60 # meters
# np.tan(alpha_true)
# GG['slopes'] = np.concatenate([slopes1, np.roll(slopes1, int(dy  * np.tan(alpha_true) /dx ))])

# %%
# normalized version
#amp_Z = abs(Z_max)/np.sqrt(2)/2

# dimensional version
N = GG.shape[0]
amp_Z = 2* abs(Z_max)**2 /N

import lmfit as LM
params0 = LM.Parameters()
params0.add('alpha', 0       ,  vary=True  , min=-0.95 * np.pi /2, max=0.95 * np.pi /2)
params0.add('group_phase', 0       ,  vary=True  , min=0, max= 2*np.pi)

# test: this should be 0
np.isnan(GG['slopes']).sum()
fitting_args = (GG['dist'],GG['dist_y'], GG['slopes'])
#fitting_kargs = {'size' :1}
L = pd.DataFrame(index=['alpha', 'group_phase', 'K_prime', 'K_amp'] )

for k_prime_max,Z_max in zip(k_list, gaus(k_list,  k_fft_max, Z_max , k_sigma )):
    #for k_prime_max,Z_max in zip(k_fft_max_list,Z_max_list):
    print(k_prime_max)
    amp_enhancement = 4
    amp_Z = amp_enhancement * abs(Z_max)**2 /N

    params = params0.copy()
    params.add('K_prime', k_prime_max ,  vary=False  , min=k_prime_max*0.5, max=k_prime_max*1.5)
    params.add('K_amp', amp_Z         ,  vary=False  , min=amp_Z*.5       , max=amp_Z*5)

    fitter = LM.minimize(objective_func, params, args=fitting_args, method='dual_annealing',max_nfev=None)
    #fitter = LM.minimize(objective_func, params, args=fitting_args, method='brute', Ns=120, )
    GG['z_model'] = objective_func(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'] , test_flag= True)
    plot_instance(GG, 'slopes', 'z_model' )
    print(fitter.params.pretty_print())
    plt.show()

    L[k_prime_max]  = fitter.params.valuesdict().values()

# %%



F = M.figure_axis_xy(5, 4.5, view_scale = 0.5)
# L.T['alpha'].hist(bins = 60)
# F.ax.axvline(alpha_true, linewidth = 0.8, color = 'black')
xlims= [L.T.index.min()* 0.9, L.T.index.max()* 1.1]

ax0 = plt.subplot(2, 1, 1)
plt.plot( L.T.index,  L.T['alpha'], '.r', markersize = 10)
#ax0.axhline(alpha_true, linewidth = 0.8, color = 'black')
plt.xlim(xlims[0], xlims[1])

plt.subplot(2, 1, 2)

plt.plot(k_fft, Z_fft_rm )
#plt.plot(k_fft_max, Z_fft_rm[Z_max_pos], '.', markersize= 20)
plt.plot(k_fft_max_list,Z_max_list, '.r', markersize= 10)

plt.xlim(xlims[0], xlims[1])

# %% construct spectrum

L
amp_Z = 2* abs(Z_max)**2 /N

GG['z_model'] = get_z_model(GG['dist'],GG['dist_y'], L.T['K_prime'], L.T['K_amp']/amp_enhancement, L.T['alpha'],L.T['group_phase'])
plot_instance(GG, 'slopes', 'z_model',  view_scale = 0.8)
# %%
params_big = LM.Parameters()
L2 = L.T.sort_values('K_prime')

N_wavenumbers = L2.shape[0]

L2['num'] = np.arange(0, L2.shape[0])

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    ki = p[1]['K_prime']
    params_big.add('K_prime_' +  numi  , ki     ,  vary=False , min=ki*0.5       , max=ki*1.5)

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    ampi = p[1]['K_amp']/amp_enhancement
    params_big.add('K_amp_' + numi   , ampi      ,  vary=False , min=ampi*.5      , max=ampi*5)

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    grpi = p[1]['group_phase']
    params_big.add('group_phase_' + numi  , grpi         ,  vary=False  , min=0           , max= 2*np.pi)

for p in L2.T.items():
    numi= str(int(p[1]['num']))
    grpi = p[1]['alpha']
    params_big.add('alpha_' + numi  , grpi                ,  vary=False  , min=-0.95 * np.pi /2, max=0.95 * np.pi /2)

# def get_max_angle(a):
#     hist = np.histogram(a, np.linspace(- np.pi/2, np.pi/2 , int(np.round(a.size/2)) ) )
#     return hist[1][hist[0].argmax()]
#
#
params_big.add('angle_shift', 0 ,  vary=False  )#, min=-0.95 * np.pi /2, max=0.95 * np.pi /2)

# %%
#params_big

def objective_func_big(params_big, x, y, z, test_flag= False ):

    pardict_val         = np.array(list(params_big.valuesdict().values()))
    K_prime_array       = pardict_val[0:N_wavenumbers]
    K_amp_array         = pardict_val[N_wavenumbers:N_wavenumbers*2]
    group_phase_array   = pardict_val[N_wavenumbers*2:N_wavenumbers*3]
    #alpha     = pardict_val[-1]
    alpha               = pardict_val[N_wavenumbers*3:N_wavenumbers*4] + pardict_val[-1]

    #print(pardict_val[-1])
    #print(alpha)
    z_model =   get_z_model(GG['dist'],GG['dist_y'], K_prime_array, K_amp_array, alpha, group_phase_array)
    cost    =   ( abs(z - z_model) )**2  /z.std()**2
    #alpha_cost  =  (alpha - alpha.mean())**2 / (2* np.pi/180)**2
    if test_flag:
        return z_model
    else:
        return cost

def turn_vary_key_on(pard, key):

    for pi in pard.keys():
        if key in pi:
            pard[pi].vary = True
        else:
            pard[pi].vary = False
    return pard


GG['z_model'] = objective_func_big(params_big, GG['dist'], GG['dist_y'], GG['slopes'], test_flag= True)
plot_instance(GG, 'slopes', 'z_model',  view_scale = 0.6)

# %%
#help(LM.minimize)
# params_big = turn_vary_key_on(params_big, 'group_phase')
# fitter = LM.minimize(objective_func_big, params_big, args=fitting_args, method='ampgo',max_nfev=10000)
#fitter

params2 = turn_vary_key_on(params_big, 'K_prime')
fitter = LM.minimize(objective_func_big, params2, args=fitting_args, method='least_squares',max_nfev=10000)

params2 = turn_vary_key_on(fitter.params, 'K_amp')
fitter = LM.minimize(objective_func_big, params2, args=fitting_args, method='least_squares',max_nfev=10000)

#print('angle_shift')
# params2 = turn_vary_key_on(fitter.params, 'angle_shift')
# fitter = LM.minimize(objective_func_big, params2, args=fitting_args, method='least_squares',max_nfev=10000)
#fitter = LM.minimize(objective_func_big, fitter.params, args=fitting_args, method='emcee')

# %%
GG['z_model'] = objective_func_big(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= True)
plot_instance(GG, 'slopes', 'z_model',  view_scale = 0.6)

# %%
z_mdoel_init = objective_func_big(params_big, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= True)
objective_func_big(params_big, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= False).sum()
z_mdoel_fit = objective_func_big(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= True)
objective_func_big(fitter.params, GG['dist'],GG['dist_y'], GG['slopes'], test_flag= False).sum()
plt.plot(z_mdoel_init)
plt.plot(z_mdoel_fit)


# %%
#fitter.params.valuesdict()

def params_to_Dataframe(pard):

    pardict_val         = np.array(list(pard.valuesdict().values()))

    pardict = dict()
    pardict['K_prime_array']       = pardict_val[0:N_wavenumbers]
    pardict['K_amp_array']         = pardict_val[N_wavenumbers:N_wavenumbers*2]
    pardict['group_phase_array']   = pardict_val[N_wavenumbers*2:N_wavenumbers*3]
    pardict['alpha']               = pardict_val[N_wavenumbers*3:N_wavenumbers*4]

    return pd.DataFrame.from_dict(pardict)

Fits = params_to_Dataframe(fitter.params)
plt.plot( k_fft_max_list, abs( (Fits['alpha']) ), '.' )

Fits['alpha']

plt.hist(fitter.flatchain, 30)
