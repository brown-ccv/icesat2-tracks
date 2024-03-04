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

#import s3fs
# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190205231558_06030212_004_01', 'SH_batch02', False



#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')
#track_name= '20190605061807_10380310_004_01'
ATlevel= 'ATL03'

load_path   = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
load_file   = load_path + 'processed_' + ATlevel + '_' + track_name + '.h5'

save_path   = mconfig['paths']['work'] + '/B02_spectra_'+hemis+'/'
save_name   = 'B02_'+track_name

plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/B_spectra/'
MT.mkdirs_r(plot_path)
MT.mkdirs_r(save_path)
bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data
Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #


# %% test amount of nans in the data
nan_fraction= list()
for I in Gd.values():
    nan_fraction.append( np.sum(np.isnan(I['heights_c_std'])) / I['heights_c_std'].shape[0] )


if np.array(nan_fraction).mean() > 0.95:
    print('nan fraction > 95%, pass this track, add to bad tracks')
    MT.json_save(track_name, bad_track_path, {'nan_fraction': np.array(nan_fraction).mean(), 'date': str(datetime.date.today()) })
    print('exit.')
    exit()

# %% test LS with an even grid where missing values are set to 0
imp.reload(spec)
print(Gd.keys())
Gi =Gd[ list(Gd.keys())[0] ] # to select a test  beam

# derive spectal limits
# Longest deserved period:
T_max = 40 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gi['dist'])
dx = np.diff(x).mean()
min_datapoint =  1/k_0/dx

Lpoints = int(np.round(min_datapoint) * 20)
Lmeters =Lpoints  * dx

#plt.plot(np.diff(np.array(Gi['dist'])))
print('L number of gridpoint:', Lpoints)
print('L length in meters:', Lmeters)
print('approx number windows', 2* Gi['dist'].iloc[-1] /Lmeters-1   )


S_pwelch_k, S_pwelch_dk = spec.calc_freq_LS( x, Lpoints, dx= dx, method='fft', minimum_frequency=None, maximum_frequency=None, samples_per_peak=0.005)
S_pwelch_k.shape
hkey= 'heights_c_weighted_mean'

G_LS= dict()
G_fft= dict()
imp.reload(spec)

# %%
k =all_beams[0]

for k in all_beams:

    x       = Gd[k]['dist']
    xlims   = x.iloc[0], x.iloc[-1]
    dd      = np.copy(Gd[k][hkey])
    print(xlims)
 # %%
    F = M.figure_axis_xy(6, 3)
    plt.subplot(2, 1, 1)
    plt.plot(x, dd, 'gray', label='displacement (m) ')

    # compute slope spectra !!
    dd      = np.gradient(dd)
    dd, _   = spicke_remover.spicke_remover(dd, spreed=10, verbose=False)
    dd_nans = (np.isnan(dd) ) + (Gd[k]['N_photos'] <= 5)

    dd_no_nans = dd[~dd_nans]
    x_no_nans  = x[~dd_nans]

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

    # assign beam coordinate
    G_ls_i.coords['beam']    = str(k)#(('beam'), str(k))
    G_ls_i                   = G_ls_i.expand_dims(dim = 'beam', axis = 2)
    # repack such that all coords are associated with beam
    G_ls_i.coords['N_per_stancil'] = (('x', 'beam' ), np.expand_dims(G_ls_i['N_per_stancil'], 1))
    G_ls_i.coords['mean_El'] = (('k', 'beam' ), np.expand_dims(G_ls_i['mean_El'], 1))
    G_ls_i.coords['mean_Eu'] = (('k', 'beam' ), np.expand_dims(G_ls_i['mean_Eu'], 1))

    # calculate number data points
    def get_stancil_nans(stancil):
        x_mask = (stancil[0] < x) & (x <= stancil[-1])
        idata  = Gd[k]['N_photos'][x_mask]
        return stancil[1], idata.sum()

    photon_list = np.array(list(dict(map(  get_stancil_nans,  copy.copy(S.stancil_iter) )).values()))
    G_ls_i.coords['N_photons'] = (('x', 'beam' ), np.expand_dims(photon_list, 1))

    G_LS[k] = G_ls_i

    plt.subplot(2, 1, 2)
    GG =G_ls_i.squeeze()
    plt.plot(GG.k, GG.mean('x'), 'k', label='mean LS')


    # standard FFT
    print('FFT')
    dd[dd_nans]    = 0
    S = spec.wavenumber_spectrogram(x, dd, Lpoints)
    G = S.cal_spectrogram()
    S.mean_spectral_error() # add x-mean spectal error estimate to xarray
    S.parceval(add_attrs= True)

    # assign beam coordinate
    G.coords['beam']      = str(k)#(('beam'), str(k))
    G                     = G.expand_dims(dim = 'beam', axis = 2)
    G.coords['mean_El']   = (('k', 'beam' ), np.expand_dims(G['mean_El'], 1))
    G.coords['mean_Eu']   = (('k', 'beam' ), np.expand_dims(G['mean_Eu'], 1))
    G.coords['x']         = G.coords['x'] * dx # adjust x-coodinate definition

    stancil_iter = spec.create_chunk_boundaries(int(Lpoints), dd_nans.size)
    def get_stancil_nans(stancil):
        idata = dd_nans[stancil[0]:stancil[-1]]
        return stancil[1], idata.size - idata.sum()

    N_list = np.array(list(dict(map(  get_stancil_nans,  stancil_iter )).values()))

    # repack such that all coords are associated with beam
    G.coords['N_per_stancil'] = (('x', 'beam' ), np.expand_dims(N_list, 1))


    G_fft[k] = G
    GG =G.squeeze()

    plt.plot(G.k, GG[:, GG['N_per_stancil'] > 10 ].mean('x'), 'darkblue', label='mean FFT')
    #plt.plot(G.k, GG.mean('x'), 'lightblue', label='mean FFT')
    plt.legend()
    plt.show()

    F.save_light(path=plot_path, name = 'B02_control_'+k+'_' + track_name)
    #print('saved as '+'B02_control_'+k+'_' + track_name)
    #print(np.isinf(G).sum().data)



# %%

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
        I =I.squeeze()
        GSUM                += I.where( ~np.isnan(I), 0) * I[weight_key] #.sel(x=GSUM.x)
        N_per_stancil       += I[weight_key]
        if 'N_photons' in GSUM.coords:
            N_photons    += I['N_photons']
        counter+=1

    GSUM             = GSUM  / N_per_stancil /counter

    if 'N_photons' in GSUM.coords:
        GSUM.coords['N_photons'] = (('x', 'beam'), np.expand_dims(N_photons, 1) )

    GSUM['beam'] = ['weighted_mean']
    GSUM.name='power_spec'

    return GSUM

G_LS_wmean = dict_weighted_mean(G_LS, 'N_per_stancil')
G_fft_wmean = dict_weighted_mean(G_fft, 'N_per_stancil')


# %% plot
def plot_wavenumber_spectrogram(ax, Gi, clev, title= None, plot_photon_density=True ):

    x_lambda= 1/Gi.k
    plt.pcolormesh(Gi.x/1e3, x_lambda , Gi, cmap=plt.cm.ocean_r , vmin = clev[0], vmax = clev[-1])

    ax.set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')

    if plot_photon_density:

        plt.plot(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10 , c='black', linewidth= 0.8, label='NAN-density' )
        plt.fill_between(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10,  0, color='gray', alpha = 0.3)
        ax.axhline(30, color='black', linewidth=0.3)

    #plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(x_lambda[-1], x_lambda[0])
    plt.title(title, loc='left')

#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
#Gmean = G_LS_wmean.rolling(x=2, min_periods= 1, center=True).mean()
Gmean = G_LS_wmean.rolling(k=5, center=True).mean()
#Gmean = Gmean.where(~np.isnan(Gmean), 0)

k_max_range = Gmean.k[Gmean.mean('x').argmax().data].data* 0.75, Gmean.k[Gmean.mean('x').argmax().data].data* 1, Gmean.k[Gmean.mean('x').argmax().data].data* 1.25

font_for_print()
F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =1)

plt.suptitle('LS and FFT Slope Spectrograms\n' + track_name, y = 0.98)
gs = GridSpec(3,3,  wspace=0.2,  hspace=.5)#figure=fig,
#clev=np.arange(0, 6, 0.1)*3

#%matplotlib inline

clev = M.clevels( [Gmean.quantile(0.01).data, Gmean.quantile(0.99).data * 1.2], 31)* 1
xlims= Gmean.x[0]/1e3, Gmean.x[-1]/1e3

for pos, k, pflag in zip([gs[0, 0],gs[0, 1],gs[0, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G_LS[k].squeeze()#.rolling(k=10, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    # #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])
    #
    plot_wavenumber_spectrogram(ax0, Gplot,  clev, title =k,  plot_photon_density=True )
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()


for pos, k, pflag in zip([gs[1, 0],gs[1, 1],gs[1, 2] ], low_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = G_LS[k].squeeze()#.rolling(k=10, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    # #plt.pcolormesh(G.x/1e3, 1/G.k , G, norm=colors.LogNorm(vmin=G.min()*1e6, vmax=G.max()), cmap='PuBu')#, vmin = clev[0], vmax = clev[-1])
    #
    plot_wavenumber_spectrogram(ax0, Gplot,  clev, title =k,  plot_photon_density=True )
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length\n(meters)')
        plt.legend()

pos, k, pflag = gs[2, 0], 'Power(weighted mean) \n10 $\log_{10}( (m/m)^2 m )$', True
ax0 = F.fig.add_subplot(pos)
Gplot = G_LS_wmean.squeeze().rolling(k=5, min_periods= 1, center=True).median().rolling(x=3, min_periods= 1, center=True).median()

dd = 10 * np.log10(Gplot)
dd= dd.where(~np.isinf(dd), np.nan )
clev_log = M.clevels( [dd.quantile(0.01).data, dd.quantile(0.98).data * 1.2], 31)* 1
plot_wavenumber_spectrogram(ax0, dd, clev_log  , title =k, plot_photon_density= True)
plt.xlim(xlims)

# plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
# ax0.axhline(30, color='black', linewidth=0.5)

ax0.axhline(1/k_max_range[0], color='red', linestyle= '--', linewidth= 0.5)
ax0.axhline(1/k_max_range[1], color='red', linestyle= '-', linewidth= 0.5)
ax0.axhline(1/k_max_range[2], color='red', linestyle= '--', linewidth= 0.5)


if pflag:
    plt.ylabel('Wave length\n(meters)')
    plt.legend()

pos = gs[2, 1]
ax0 = F.fig.add_subplot(pos)
plt.title('Photons density ($m^{-1}$)', loc='left')

for k,I in G_LS.items():
    plt.plot(Gplot.x/1e3, I.N_photons/Lmeters, label=k, linewidth=0.8)
plt.plot(Gplot.x/1e3, G_LS_wmean.N_photons/3/Lmeters , c='black', label='ave Photons' , linewidth=0.8)
plt.xlim(xlims)
plt.xlabel('Distance from the Ice Edge (km)')

pos = gs[2, 2]
ax0 = F.fig.add_subplot(pos)

ax0.set_yscale('log')

plt.title('Peak Spectal Power', loc='left')

for k,I in G_LS.items():
    plt.scatter(I.x.data/1e3, I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k').data *1e3 ,  s=0.5, marker='.', color='red', alpha= 0.3)

for k,I in G_fft.items():
    I= I.squeeze()
    I= I[:,I.N_per_stancil >=  I.N_per_stancil.max().data*0.9]
    plt.scatter(I.x/1e3, I.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3 ,  s=0.5, marker='.', c='blue', alpha= 0.3)


Gplot= G_fft_wmean.squeeze()
Gplot = Gplot[:,Gplot.N_per_stancil >=  Gplot.N_per_stancil.max().data*0.9]
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3, '.', markersize=1.5 , c='blue', label= 'FFT')

Gplot= G_LS_wmean
plt.plot(Gplot.x/1e3, Gplot.sel(k=slice(k_max_range[0], k_max_range[2])).integrate('k') *1e3 , '.' , markersize=1.5, c='red', label= 'LS')

plt.ylabel('1e-3 $(m)^2~m$')
plt.legend()
#plt.ylim(Gplot.min()*1.4, Gplot.max()*1.4 )
plt.xlim(xlims)

F.save_light(path=plot_path, name = 'B02_specs_' + track_name +'_L'+str(Lmeters))


# %% repack data
def repack_attributes(DD):
    attr_dim_list = list(DD.keys())
    for k in attr_dim_list:
        for ka in list(DD[k].attrs.keys()):
            I = DD[k]
            I.coords[ka] = ( 'beam', np.expand_dims(I.attrs[ka], 1) )
    return DD

G_LS[G_LS_wmean.beam.data[0]] =G_LS_wmean
G_fft[G_fft_wmean.beam.data[0]] =G_fft_wmean

G_LS = repack_attributes(G_LS)
G_fft = repack_attributes(G_fft)

# %% save results
G_LS_DS         = xr.merge(G_LS.values())
G_LS_DS['name'] = 'LS_power_spectra'
G_LS_DS.to_netcdf(save_path+save_name+'_LS.nc')

G_fft_DS        = xr.merge(G_fft.values())
G_fft_DS['name']= 'FFT_power_spectra'
G_fft_DS.to_netcdf(save_path+save_name+'_FFT.nc')
