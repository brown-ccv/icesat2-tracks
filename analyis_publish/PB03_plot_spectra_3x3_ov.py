
# %%
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

import time
import imp
import copy
import spicke_remover
import datetime
import generalized_FT as gFT
from scipy.ndimage.measurements import label

# %%
track_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#track_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#track_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190207002436_06190212_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190206022433_06050212_004_01', 'SH_batch02', False


#track_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
#track_name, batch_key, test_flag = '20190502021224_05160312_004_01', 'SH_batch02', False

#track_name, batch_key, test_flag = 'SH_20190224_08800210', 'SH_publish', False
#track_name, batch_key, test_flag = 'SH_20190219_08070210', 'SH_publish', False
track_name, batch_key, test_flag = 'SH_20190502_05160312', 'SH_publish', False
#track_name, batch_key, test_flag = 'SH_20190502_05180312', 'SH_publish', False


#print(track_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')

load_path   = mconfig['paths']['work'] +batch_key +'/B02_spectra/'
load_file   = load_path + 'B02_' + track_name #+ '.nc'
#plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/'
plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/publish/' + track_name + '/'
MT.mkdirs_r(plot_path)

Gk = xr.open_dataset(load_file+'_gFT_k.nc')
Gx = xr.open_dataset(load_file+'_gFT_x.nc')
Gfft = xr.open_dataset(load_file+'_FFT.nc')

# %%
all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']
#Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data
#Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #

col.colormaps2(31, gamma=1)
col_dict= col.rels


# %%
def dict_weighted_mean(Gdict, weight_key):
    """
    returns the weighted meean of a dict of xarray, data_arrays
    weight_key must be in the xr.DataArrays
    """
    #Gdict = G_rar_fft
    #weight_key='N_per_stancil'

    akey = list( Gdict.keys() )[0]
    GSUM = Gdict[akey].copy()
    GSUM.data     = np.zeros(GSUM.shape)
    N_per_stancil = GSUM.N_per_stancil * 0
    N_photons     = np.zeros(GSUM.N_per_stancil.size)

    counter= 0
    for k,I in Gdict.items():
        #print(k)
        I =I.squeeze()
        print(len(I.x) )
        if len(I.x) !=0:
            GSUM                += I.where( ~np.isnan(I), 0) * I[weight_key] #.sel(x=GSUM.x)
            N_per_stancil       += I[weight_key]
        if 'N_photons' in GSUM.coords:
            N_photons    += I['N_photons']
        counter+=1

    GSUM             = GSUM  / N_per_stancil

    if 'N_photons' in GSUM.coords:
        GSUM.coords['N_photons'] = (('x', 'beam'), np.expand_dims(N_photons, 1) )

    GSUM['beam'] = ['weighted_mean']
    GSUM.name='power_spec'

    return GSUM


#G_gFT_wmean = (Gk['gFT_PSD_model'].where( ~np.isnan(Gk['gFT_PSD_model']), 0) * Gk['N_per_stancil']).sum('beam')/ Gk['N_per_stancil'].sum('beam')

G_gFT_wmean = (Gk.where( ~np.isnan(Gk['gFT_PSD_model']), 0) * Gk['N_per_stancil']).sum('beam')/ Gk['N_per_stancil'].sum('beam')
G_gFT_wmean['N_photons'] = Gk['N_photons'].sum('beam')

G_fft_wmean = (Gfft.where( ~np.isnan(Gfft), 0) * Gfft['N_per_stancil']).sum('beam')/ Gfft['N_per_stancil'].sum('beam')
G_fft_wmean['N_per_stancil'] = Gfft['N_per_stancil'].sum('beam')


# %% plot
#col.colormaps2(31, gamma=1)
#col.plot()

def plot_wavenumber_spectrogram(ax, Gi, clev, plot_photon_density=True , cmap=None ):

    if Gi.k[0] ==0:
        Gi= Gi.sel(k=Gi.k[1:])
    x_lambda= 2 * np.pi/Gi.k
    if cmap is None:
        cmap = col.white_base_blgror #plt.cm.
    if clev is None:
        clev = None, None

    css= plt.pcolormesh(Gi.x/1e3, x_lambda , Gi, cmap=cmap , vmin = clev[0], vmax = clev[-1])
    ax.set_yscale('log')
    # plt.colorbar(orientation='vertical', pad=0.06, label='Spectral Power (m^2/m)')

    if plot_photon_density:

        plt.plot(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10 , c='black', linewidth= 0.8, label='NAN-density' )
        plt.fill_between(Gi.x/1e3, x_lambda[-1] + (Gi.N_per_stancil/Gi.N_per_stancil.max() ) * 10,  0, color='gray', alpha = 0.3)
        ax.axhline(30, color='black', linewidth=0.3)

    #plt.xlabel('Distance from the Ice Edge (km)')
    plt.ylim(x_lambda[-1], x_lambda[0])

    return css

#Gplot = G.rolling(x=5, min_periods= 1, center=True).mean()
#Gmean = G_gFT_wmean.rolling(x=2, min_periods= 1, center=True).mean()

#Gmean = G_gFT_wmean['gFT_PSD_data'].rolling(k=5, center=True).mean()


Gmean,_ = gFT.rebin(G_gFT_wmean['gFT_PSD_data'], 10)

#Gmean = Gmean.where(~np.isnan(Gmean), 0)

# define mean first for colorbar
#Gplot = G_gFT_wmean.squeeze()#.rolling(k=10, min_periods= 1, center=True).median().rolling(x=3, min_periods= 1, center=True).median()
dd = 10 * np.log10(Gmean)
dd= dd.where(~np.isinf(dd), np.nan )
clev_log = M.clevels( [dd.quantile(0.01).data*1, dd.quantile(0.98).data * 1], 31)* 1


try:
    k_max_range = Gmean.k[Gmean.isel(x= slice(0, 5)).mean('x').argmax().data].data* 0.75, Gmean.k[Gmean.isel(x= slice(0, 5)).mean('x').argmax().data].data* 1, Gmean.k[Gmean.isel(x= slice(0, 5)).mean('x').argmax().data].data* 1.25
except:
    k_max_range = Gmean.k[Gmean.isel(x= slice(0, 20)).mean('x').argmax().data].data* 0.75, Gmean.k[Gmean.isel(x= slice(0, 20)).mean('x').argmax().data].data* 1, Gmean.k[Gmean.isel(x= slice(0, 20)).mean('x').argmax().data].data* 1.25


# %
font_for_print()
fn = copy.copy(lstrings)

F = M.figure_axis_xy(6.5, 5.6, container= True, view_scale =1)
Lmeters = Gk.L.data[0]


plt.suptitle('Generalized Fourier Transform Slope Spectral Power\n' + io.ID_to_str(track_name), y = 0.98)
gs = GridSpec(7,3,  wspace=0.25,  hspace=1.2)#figure=fig,
#clev=np.arange(0, 6, 0.1)*3

#%matplotlib inline



#clev = M.clevels( [Gmean.quantile(0.6).data * 1e4, Gmean.quantile(0.99).data * 1e4], 31)/ 1e4

xlims= Gmean.x[0]/1e3, Gmean.x[-1]/1e3

k =high_beams[0]
for pos, k, pflag in zip([gs[0:2, 0],gs[0:2, 1],gs[0:2, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    ax0.tick_params(labelbottom=False)
    Gplot = Gk.sel(beam = k).gFT_PSD_model.squeeze()#rolling(k=5, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    dd2 = 10 * np.log10(Gplot)
    dd2= dd2.where(~np.isinf(dd2), np.nan )
    plot_wavenumber_spectrogram(ax0, dd2,  clev_log, plot_photon_density=False )
    plt.title(next(fn)+k, color= col_dict[k], loc= 'left')
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length (m)')
        plt.legend()

for pos, k, pflag in zip([gs[2:4, 0],gs[2:4, 1],gs[2:4, 2] ], low_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    ax0.tick_params(labelbottom=False)
    Gplot = Gk.sel(beam = k).gFT_PSD_model.squeeze()#.rolling(k=10, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()
    dd2 = 10 * np.log10(Gplot)
    dd2= dd2.where(~np.isinf(dd2), np.nan )
    plot_wavenumber_spectrogram(ax0, dd2,  clev_log, plot_photon_density=False )
    plt.title(next(fn)+k, color= col_dict[k], loc= 'left')
    plt.xlim(xlims)
    #
    if pflag:
        plt.ylabel('Wave length (m)')
        plt.legend()


pos = gs[4:6, 0]
ax0 = F.fig.add_subplot(pos)
plt.title(next(fn)+'Photons density ($m^{-1}$)', loc='left')

max_list = list()
for k in all_beams:
    I = Gk.sel(beam = k)['gFT_PSD_model']
    plt.plot(Gplot.x/1e3, I.N_photons/I.L.data, color= col_dict[k], linewidth=0.9)
    max_list.append((I.N_photons/I.L.data).max())

plt.plot(Gplot.x/1e3, G_gFT_wmean['N_photons']/I.L.data/6, c='black', label='mean photons density' , linewidth=0.8)
plt.xlim(xlims)
plt.ylim(2,max(max_list)*1.3)
plt.legend(ncol= 3,loc=1)
plt.xlabel('Distance from the ice edge (km)')


ax0 = F.fig.add_subplot(gs[4:6, 1])

css = plot_wavenumber_spectrogram(ax0, dd, clev_log  , plot_photon_density= False)
plt.title(next(fn)+'Beam weighted mean', loc= 'left') #\n10 $\log_{10}( (m/m)^2 m )$
plt.xlim(xlims)

# plt.plot(Gplot.x/1e3, 10* nan_list +20 , c='black', label='NAN-density' )
# ax0.axhline(30, color='black', linewidth=0.5)

# ax0.axhline(2* np.pi/k_max_range[0], color='black', linestyle= '--', linewidth= 0.5)
# ax0.axhline(2* np.pi/k_max_range[1], color='black', linestyle= '-', linewidth= 0.8)
# ax0.axhline(2* np.pi/k_max_range[2], color='black', linestyle= '--', linewidth= 0.5)
# ax0.axhspan(2* np.pi/k_max_range[0], 2* np.pi/k_max_range[2], color='gray', alpha = 0.4)


plt.xlabel('Distance from the ice edge (km)')
#plt.ylabel('Wave length (m)')
plt.legend(loc=1)



cbaxes = F.fig.add_subplot(gs[-1, 1])
cbaxes.axis('off')
cbpos  = cbaxes.get_position()
#cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0,cbpos.width/5,cbpos.height])
cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0+ 2*cbpos.height/4,cbpos.width,cbpos.height/3])
cb = plt.colorbar(css , cax =cbaxes2, orientation= 'horizontal')

cb.set_label('Power($(m/m)^2/k$)')
cb.outline.set_visible(False)
#cbaxes2.tick_params(axis='both', colors=col.gray)



Lpoints=  Gk.Lpoints.mean('beam').data
N_per_stancil = Gk.N_per_stancil.mean('beam').data#[0:-2]

G_error_model =dict()
G_error_data =dict()

for bb in Gk.beam.data:
    I = Gk.sel(beam= bb)
    b_bat_error =  np.concatenate([ I.model_error_k_cos.data , I.model_error_k_sin.data ])
    Z_error = gFT.complex_represenation(b_bat_error, Gk.k.size, Lpoints)
    PSD_error_data, PSD_error_model = gFT.Z_to_power_gFT(Z_error, np.diff(Gplot.k)[0],N_per_stancil  , Lpoints )

    #np.expand_dims(PSD_error_model, axis =)
    G_error_model[bb] =  xr.DataArray(data = PSD_error_model, coords = I.drop('N_per_stancil').coords, name='gFT_PSD_model_error' ).expand_dims('beam')
    G_error_data[bb] =  xr.DataArray(data = PSD_error_data, coords = I.drop('N_per_stancil').coords, name='gFT_PSD_data_error' ).expand_dims('beam')

gFT_PSD_model_error_mean = xr.merge(G_error_model.values(),compat='override').gFT_PSD_model_error
gFT_PSD_data_error_mean = xr.merge(G_error_data.values(),compat='override').gFT_PSD_data_error

PSD_model_error_mean = ( gFT_PSD_model_error_mean.where( ~np.isnan(gFT_PSD_model_error_mean), 0) * Gk['N_per_stancil']).sum('beam')/Gk['N_per_stancil'].sum('beam')
PSD_data_error_mean = ( gFT_PSD_data_error_mean.where( ~np.isnan(gFT_PSD_data_error_mean), 0) * Gk['N_per_stancil']).sum('beam')/Gk['N_per_stancil'].sum('beam')


dd2 = 10 * np.log10(PSD_data_error_mean)
#dd2 = PSD_data_error_mean
#dd= np.where(~np.isinf(dd),  dd, np.nan )
dd2= dd2.where(~np.isinf(dd2), np.nan )

pos = gs[4:6, 2]
ax0 = F.fig.add_subplot(pos)
ax0.set_yscale('log')
#clev_log

clev_log = M.clevels( [dd2.quantile(0.01).data*0.8, dd2.quantile(0.98).data*0.9 ], 31)
plt.cm.OrRd
css_err = plot_wavenumber_spectrogram(ax0, dd2, clev_log,  plot_photon_density= False, cmap= plt.cm.OrRd)
plt.title(next(fn)+'Mean error', loc= 'left') #\n10 $\log_{10}( (m/m)^2 m )$
plt.xlabel('Distance from the ice edge (km)')
plt.xlim(xlims)
#plt.colorbar()


cbaxes = F.fig.add_subplot(gs[-1, 2])
cbaxes.axis('off')
cbpos  = cbaxes.get_position()
#cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0,cbpos.width/5,cbpos.height])
cbaxes2 = F.fig.add_axes([cbpos.x0,cbpos.y0+ 2*cbpos.height/4,cbpos.width,cbpos.height/3])
cb = plt.colorbar(css_err , cax =cbaxes2, orientation= 'horizontal')

cb.set_label('Power($(m/m)^2/k$)')
cb.outline.set_visible(False)
#cbaxes2.tick_params(axis='both', colors=col.gray)


F.save_light(path=plot_path, name = 'PB03_specs_ov_'+str(track_name))
F.save_pup(path=plot_path, name = 'PB03_specs_ov_'+str(track_name))

# %%
