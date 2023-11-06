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
import ICEsat2_SI_tools.lanczos as lanczos
import time
import imp
import copy
import spicke_remover
import datetime
import generalized_FT as gFT
from scipy.ndimage.measurements import label

xr.set_options(display_style='text')
#import s3fs
# %%
ID_name, batch_key, test_flag = io.init_from_input(sys.argv) # loads standard experiment
#ID_name, batch_key, test_flag = '20190605061807_10380310_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190601094826_09790312_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190207111114_06260210_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190208152826_06440210_004_01', 'SH_batch01', False
#ID_name, batch_key, test_flag = '20190213133330_07190212_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190207002436_06190212_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190206022433_06050212_004_01', 'SH_batch02', False

#ID_name, batch_key, test_flag = '20190219073735_08070210_004_01', 'SH_batch02', False
#ID_name, batch_key, test_flag = '20190502021224_05160312_004_01', 'SH_batch02', False

#ID_name, batch_key, test_flag =  'SH_20190208_06440212', 'SH_publish', True
#ID_name, batch_key, test_flag =  'SH_20190219_08070210', 'SH_publish', True
#ID_name, batch_key, test_flag =  'SH_20190502_05160312', 'SH_publish', True

#ID_name, batch_key, test_flag =  'NH_20190311_11200203', 'NH_batch06', True
#ID_name, batch_key, test_flag =  'NH_20210312_11961005', 'NH_batch07', True

#ID_name, batch_key , test_flag = 'SH_20190502_05180312', 'SH_testSLsinglefile2' , True

#print(ID_name, batch_key, test_flag)
hemis, batch = batch_key.split('_')

all_beams   = mconfig['beams']['all_beams']
high_beams  = mconfig['beams']['high_beams']
low_beams   = mconfig['beams']['low_beams']

load_path_work    = mconfig['paths']['work'] +'/'+ batch_key +'/'
B3_hdf5    = h5py.File(load_path_work +'B01_regrid'+'/'+ID_name + '_B01_binned.h5', 'r')


load_path_angle   = mconfig['paths']['work'] +'/'+ batch_key +'/B04_angle/'

B3 = dict()
for b in all_beams:
    B3[b] = io.get_beam_hdf_store(B3_hdf5[b])

B3_hdf5.close()

# B2          = io.load_pandas_table_dict(ID_name + '_B01_regridded'  , load_path1) # rhis is the rar photon data
# B3          = io.load_pandas_table_dict(ID_name + '_B01_binned'     , load_path1)  #

load_file   = load_path_work +'/B02_spectra/' + 'B02_' + ID_name #+ '.nc'
Gk = xr.open_dataset(load_file+'_gFT_k.nc')
Gx = xr.open_dataset(load_file+'_gFT_x.nc')
Gfft = xr.open_dataset(load_file+'_FFT.nc')


#plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + ID_name + '/'
plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + ID_name + '/B06_correction/'
MT.mkdirs_r(plot_path)

save_path   = mconfig['paths']['work'] +batch_key+'/B06_corrected_separated/'
MT.mkdirs_r(save_path)


# %%

#Gfilt   = io.load_pandas_table_dict(ID_name + '_B01_regridded', load_path) # rhis is the rar photon data
#Gd      = io.load_pandas_table_dict(ID_name + '_B01_binned' , load_path)  #

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


#G_gFT_wmean = (Gk['gFT_PSD_data'].where( ~np.isnan(Gk['gFT_PSD_data']), 0) * Gk['N_per_stancil']).sum('beam')/ Gk['N_per_stancil'].sum('beam')

G_gFT_wmean = (Gk.where( ~np.isnan(Gk['gFT_PSD_data']), 0) * Gk['N_per_stancil']).sum('beam')/ Gk['N_per_stancil'].sum('beam')
G_gFT_wmean['N_photons'] = Gk['N_photons'].sum('beam')

G_fft_wmean = (Gfft.where( ~np.isnan(Gfft), 0) * Gfft['N_per_stancil']).sum('beam')/ Gfft['N_per_stancil'].sum('beam')
G_fft_wmean['N_per_stancil'] = Gfft['N_per_stancil'].sum('beam')


# %% plot

# derive spectral errors:
Lpoints=  Gk.Lpoints.mean('beam').data
N_per_stancil = Gk.N_per_stancil.mean('beam').data#[0:-2]

G_error_model =dict()
G_error_data =dict()

for bb in Gk.beam.data:
    I = Gk.sel(beam= bb)
    b_bat_error =  np.concatenate([ I.model_error_k_cos.data , I.model_error_k_sin.data ])
    Z_error = gFT.complex_represenation(b_bat_error, Gk.k.size, Lpoints)
    PSD_error_data, PSD_error_model = gFT.Z_to_power_gFT(Z_error, np.diff(Gk.k)[0],N_per_stancil  , Lpoints )

    #np.expand_dims(PSD_error_model, axis =)
    G_error_model[bb] =  xr.DataArray(data = PSD_error_model, coords = I.drop('N_per_stancil').coords, name='gFT_PSD_data_error' ).expand_dims('beam')
    G_error_data[bb] =  xr.DataArray(data = PSD_error_data, coords = I.drop('N_per_stancil').coords, name='gFT_PSD_data_error' ).expand_dims('beam')

gFT_PSD_data_error_mean = xr.concat(G_error_model.values(), dim='beam')
gFT_PSD_data_error_mean = xr.concat(G_error_data.values(), dim='beam')

gFT_PSD_data_error_mean = ( gFT_PSD_data_error_mean.where( ~np.isnan(gFT_PSD_data_error_mean), 0) * Gk['N_per_stancil']).sum('beam')/Gk['N_per_stancil'].sum('beam')
gFT_PSD_data_error_mean = ( gFT_PSD_data_error_mean.where( ~np.isnan(gFT_PSD_data_error_mean), 0) * Gk['N_per_stancil']).sum('beam')/Gk['N_per_stancil'].sum('beam')

G_gFT_wmean['gFT_PSD_data_err'] = gFT_PSD_data_error_mean
G_gFT_wmean['gFT_PSD_data_err'] = gFT_PSD_data_error_mean

Gk['gFT_PSD_data_err'] = xr.concat(G_error_model.values(), dim='beam')
Gk['gFT_PSD_data_err']  = xr.concat(G_error_data.values(), dim='beam')


# %%

G_gFT_smth = G_gFT_wmean['gFT_PSD_data'].rolling(k=30, center=True, min_periods=1).mean()
G_gFT_smth['N_photons'] = G_gFT_wmean.N_photons
G_gFT_smth["N_per_stancil_fraction"] = Gk['N_per_stancil'].T.mean('beam')/Gk.Lpoints.mean('beam')

k = G_gFT_smth.k

# %%
# GG_no_nan = G_gFT_smth.isel( x = ~np.isnan(G_gFT_smth.mean('k')) )
# k_lead_peak = GG_no_nan.k[GG_no_nan.isel(x=0).argmax().data].data
# if k_lead_peak== k[0].data or k_lead_peak == k[-1].data:
#     #raise ValueError('wavenumber Peak on Boundary!')
#     print('wavenumber Peak on Boundary!')
#     MT.json_save('B06_fail', plot_path+'../',  {'time':time.asctime( time.localtime(time.time()) ) , 'reason': 'wavenumber Peak on Boundary!'})
#     print('exit()')
#     #exit()
#
# # %%
# k_lims =0.01
# k_span = [k_lead_peak- k_lims , k_lead_peak, k_lead_peak+ k_lims]

F = M.figure_axis_xy()
#plt.loglog(k, k**(-2))
# plt.loglog(k, 1e-4 *k**(-2))
# plt.loglog(k, 1e-5 *k**(-3))

# F.ax.axvline(k_span[0])
# F.ax.axvline(k_span[1])
# F.ax.axvline(k_span[2])
#plt.plot(np.log(k), np.log( k**(-3) ) )
#plt.loglog(k, (k)**(-3) - 1e5)

plt.loglog(k, G_gFT_smth/k)
# dd= dd.where(~np.isinf(dd), np.nan )
#plt.grid()
plt.title('displacement power Spectra', loc='left')

# %%
def define_noise_wavenumber_tresh_simple(data_xr, k_peak, k_end_lim =None,  plot_flag = False):

    """
    returns noise wavenumber on the high end of a spectral peak. This method fits a straight line in loglog speace using robust regression.
    The noise level is defined as the wavenumber at which the residual error of a linear fit to the data is minimal.

    inputs:
    data_xr xarray.Dataarray with the power spectra with k as dimension
    k_peak  wavenumber above which the searh should start
    dk      the intervall over which the regrssion is repeated

    returns:
    k_end   the wavenumber at which the spectrum flattens
    m       slope of the fitted line
    b       intersect of the fitted line
    """
    #data_xr, k_peak =    G_gFT_smth.isel(x=0), k_lead_peak
    #k_end_lim = None#
    #k_end_lim= 0.06396283#0.0224938*1.05
    from scipy.ndimage.measurements import label

    if k_end_lim is None:
        k_end_lim =data_xr.k[-1]

    k_lead_peak_margin = k_peak *1.05
    try:
        data_log = np.log(data_xr).isel(k =(data_xr.k > k_lead_peak_margin)).rolling(k =10,  center=True, min_periods=1).mean()

    except:
        data_log = np.log(data_xr).isel(k =(data_xr.k > k_lead_peak_margin/2)).rolling(k =10,  center=True, min_periods=1).mean()

    k_log= np.log(data_log.k)
    try:
        d_grad = data_log.differentiate('k').rolling(k =40, center=True, min_periods=4).mean()
    except:
        d_grad = data_log.differentiate('k').rolling(k =20, center=True, min_periods=2).mean()
    ll = label( d_grad >=-5  )

    #test if plausible minium exist:
    # #print(ll[0][d_grad.k <= k_end_lim] )
    # if sum(  ll[0][d_grad.k <= k_end_lim] ==0) == 0:
    #     #print(sum(  ll[0][d_grad.k <= k_end_lim] ==0) == 0)
    #     print('no gradient in range, set to peak')
    #     return k_peak

    if ll[0][0] !=0:
        #print(sum(  ll[0][d_grad.k <= k_end_lim] ==0) == 0)
        print('no decay, set to peak')
        return k_peak

    if sum(ll[0]) == 0:
        k_end = d_grad.k[-1]
    else:
        k_end = d_grad.k[(ll[0] == 1) ][0].data

    if plot_flag:
        # plt.plot(np.log(d_grad.k), d_grad)
        # plt.show()
        plt.plot(np.log(data_xr.k), np.log(data_xr))
        plt.plot(k_log, data_log )
        plt.plot([np.log(k_end), np.log(k_end)], [-6, -5])
        #print(k_end)
    return k_end



# %% new version
def get_correct_breakpoint(pw_results):
    br_points   = list()
    for i in pw_results.keys():
        [br_points.append(i) if 'breakpoint' in i else None]
    br_points_df = pw_results[br_points]
    br_points_sorted = br_points_df.sort_values()

    alphas_sorted = [i.replace('breakpoint', 'alpha') for i in br_points_df.sort_values().index]
    alphas_sorted.append('alpha'+ str(len(alphas_sorted)+1) )


    betas_sorted = [i.replace('breakpoint', 'beta') for i in br_points_df.sort_values().index]

    #betas_sorted
    alphas_v2 = list()
    alpha_i = pw_results['alpha1']
    for i in [0] + list(pw_results[betas_sorted]):
        alpha_i += i
        alphas_v2.append(alpha_i)

    alphas_v2_sorted   = pd.Series(index = alphas_sorted, data =alphas_v2)
    br_points_sorted['breakpoint'+ str(br_points_sorted.size+1)] = 'end'

    print('all alphas')
    print(alphas_v2_sorted)
    slope_mask = alphas_v2_sorted < 0

    if sum(slope_mask) ==0:
        print('no negative slope found, set to lowest')
        breakpoint = 'start'
    else:

        # take steepest slope
        alpah_v2_sub = alphas_v2_sorted[slope_mask]
        print(alpah_v2_sub)
        print(alpah_v2_sub.argmin())
        break_point_name =  alpah_v2_sub.index[alpah_v2_sub.argmin()].replace('alpha', 'breakpoint')

        # take first slope
        #break_point_name = alphas_v2_sorted[slope_mask].index[0].replace('alpha', 'breakpoint')
        breakpoint = br_points_sorted[break_point_name]

    return breakpoint

def get_breakingpoints(xx, dd):

    import piecewise_regression
    x2, y2 = xx, dd
    convergence_flag =True
    n_breakpoints= 3
    while convergence_flag:
        pw_fit = piecewise_regression.Fit(x2, y2, n_breakpoints=n_breakpoints)
        print('n_breakpoints', n_breakpoints, pw_fit.get_results()['converged'])
        convergence_flag = not pw_fit.get_results()['converged']
        n_breakpoints += 1
        if n_breakpoints >=4:
            convergence_flag = False

    pw_results = pw_fit.get_results()
    #pw_fit.summary()

    if pw_results['converged']:
        # if pw_results['estimates']['alpha1']['estimate'] < 0:
        #     print('decay at the front')
        #     print('n_breakpoints',pw_fit.n_breakpoints )

        pw_results_df = pd.DataFrame(pw_results['estimates']).loc['estimate']

        breakpoint = get_correct_breakpoint(pw_results_df)

        return pw_fit, breakpoint

    else:
        return pw_fit, False

def define_noise_wavenumber_piecewise(data_xr, plot_flag = False):

    data_log = data_xr
    data_log = np.log(data_xr)

    k =data_log.k.data
    k_log= np.log(k)

    pw_fit, breakpoint_log   = get_breakingpoints(k_log, data_log.data)

    if breakpoint_log is 'start':
        print('no decay, set to lowerst wavenumber')
        breakpoint_log =  k_log[0]
    if (breakpoint_log is 'end') | (breakpoint_log is False) :
        print('higest wavenumner')
        breakpoint_log =  k_log[-1]

    breakpoint_pos                  = abs(k_log -breakpoint_log).argmin()
    breakpoint_k                    = k[breakpoint_pos]

    #plot_flag= False
    if plot_flag:
        # plt.plot(np.log(d_grad.k), d_grad)
        # plt.show()
        pw_fit.plot()
        #plt.plot(np.log(data_xr.k), np.log(data_xr))
        plt.plot(k_log, data_log )
        #plt.gca().set_xscale('log')
        #plt.plot([np.log(breakpoint_k), np.log(breakpoint_k)], [-6, -5])
        #print(k_end)

    return breakpoint_k, pw_fit

#G_gFT_smth.isel(x=7).plot()

k_lim_list = list()
k_end_previous = np.nan
x = G_gFT_smth.x.data[0]
k = G_gFT_smth.k.data

for x in G_gFT_smth.x.data:
    #x = G_gFT_smth.isel(x=9).x
    #x= 237500.0
    print(x)
    # use displacement power spectrum
    k_end, pw_fit = define_noise_wavenumber_piecewise(G_gFT_smth.sel(x=x)/k, plot_flag =False )
    #pw_fit.get_results()
    #pw_fit.n_breakpoints

    #pw_fit.summary()
    #k_end, slope = define_noise_wavenumber_piecewise(G_gFT_smth.sel(x=x), k_lead_peak, k_end_lim= k_end_0, plot_flag =True )
    #k_end = define_noise_wavenumber_tresh_simple(G_gFT_smth.sel(x=x), k_lead_peak, k_end_lim= k_end_0, plot_flag =True )


    k_save = k_end_previous if k_end == k[0] else k_end
    #k_save = k_end_previous if k_end >= k[-1]*0.95 else k_end

    #k_save = k_end_previous if k_end == k[-1] else k_end
    k_end_previous = k_save #if k_end_0 is None else k_end_0
    k_lim_list.append(k_save)

    #k_save = np.nan if slope >= 0 else k_end
    # plt.gca().axvline(np.log(k_save), linewidth= 2, color='red')
    # plt.show()
    print('--------------------------')
# %%
# write k limits to datasets
# lanczos.lanczos_filter_1d(G_gFT_smth.x, k_lim_list, 2)
# lanczos.lanczos_filter_1d_wrapping

font_for_pres()
G_gFT_smth.coords['k_lim'] = ('x', k_lim_list )
G_gFT_smth.k_lim.plot()
#G_gFT_smth.k_lim.rolling(x=4,  center=True, min_periods=1).median().plot()
k_lim_smth = G_gFT_smth.k_lim.rolling(x=3,  center=True, min_periods=1).mean()
k_lim_smth.plot(c='r')

plt.title('k_c filter', loc='left')
F.save_light(path=plot_path, name = str(ID_name)+ '_B06_atten_ov')

G_gFT_smth['k_lim']  = k_lim_smth #G_gFT_smth.k_lim.rolling(x=3,  center=True, min_periods=1).mean().plot(c='r').data
G_gFT_wmean.coords['k_lim'] = k_lim_smth #('x', k_lim_smth )


# %%
font_for_print()

fn = copy.copy(lstrings)
F = M.figure_axis_xy(fig_sizes['two_column'][0], fig_sizes['two_column'][0]* 0.9, container= True, view_scale =1)


plt.suptitle('Cut-off Frequency for Displacement Spectral\n' + io.ID_to_str(ID_name), y = 0.97)
gs = GridSpec(8,3,  wspace=0.1,  hspace=1.5)#figure=fig,#

#
# #clev = M.clevels( [Gmean.quantile(0.6).data * 1e4, Gmean.quantile(0.99).data * 1e4], 31)/ 1e4
#
k_lims = G_gFT_wmean.k_lim
xlims= G_gFT_wmean.k[0], G_gFT_wmean.k[-1]
#
k =high_beams[0]
for pos, k, pflag in zip([gs[0:2, 0],gs[0:2, 1],gs[0:2, 2] ], high_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = Gk.sel(beam = k).isel(x = slice(0, -1)).gFT_PSD_data.squeeze().rolling(k=20, x=2, min_periods= 1, center=True).mean()
    #Gplot.plot()

    Gplot= Gplot.where(Gplot["N_per_stancil"] / Gplot["Lpoints"] >= 0.1)#.plot()
    #Gplot.plot()


    alpha_range= iter(np.linspace(1,0, Gplot.x.data.size))
    for x in Gplot.x.data:
        ialpha =next(alpha_range)
        plt.loglog(Gplot.k, Gplot.sel(x=x)/Gplot.k, linewidth = 0.5, color= col.rels[k], alpha= ialpha)
        ax0.axvline(k_lims.sel(x=x), linewidth= 0.4, color= 'black', zorder= 0, alpha=ialpha)

    plt.title(next(fn) + k, color= col_dict[k], loc= 'left')
    plt.xlim(xlims)
    #
    if pflag:
        ax0.tick_params(labelbottom=False, bottom=True)
        plt.ylabel("Power (m$^2$/k')")
        plt.legend()
    else:
        ax0.tick_params(labelbottom=False, bottom=True, labelleft=False)

for pos, k, pflag in zip([gs[2:4, 0],gs[2:4, 1],gs[2:4, 2] ], low_beams, [True, False, False] ):
    ax0 = F.fig.add_subplot(pos)
    Gplot = Gk.sel(beam = k).isel(x = slice(0, -1)).gFT_PSD_data.squeeze().rolling(k=20, x=2, min_periods= 1, center=True).mean()
    #Gplot.mean('x').plot()

    Gplot= Gplot.where(Gplot["N_per_stancil"] / Gplot["Lpoints"] >= 0.1)#.plot()

    alpha_range= iter(np.linspace(1,0, Gplot.x.data.size))
    for x in Gplot.x.data:
        ialpha =next(alpha_range)
        plt.loglog(Gplot.k, Gplot.sel(x=x)/Gplot.k, linewidth = 0.5, color= col.rels[k], alpha= ialpha)
        ax0.axvline(k_lims.sel(x=x), linewidth= 0.4, color= 'black', zorder= 0, alpha=ialpha)

    plt.title(next(fn) + k, color= col_dict[k], loc= 'left')
    plt.xlim(xlims)
    plt.xlabel("observed wavenumber k' ")

    #
    if pflag:
        ax0.tick_params( bottom=True)
        plt.ylabel("Power (m$^2$/k')")
        plt.legend()
    else:
        ax0.tick_params(bottom=True, labelleft=False)

F.save_light(path=plot_path, name =str(ID_name) + '_B06_atten_ov_simple')
F.save_pup(path=plot_path, name = str(ID_name) + '_B06_atten_ov_simple')

# %
pos = gs[5:, 0:2]
ax0 = F.fig.add_subplot(pos)

lat_str = str(np.round( Gx.isel(x = 0).lat.mean().data, 2)  ) +' to ' + str(np.round( Gx.isel(x = -1).lat.mean().data, 2)  )
plt.title(next(fn) + 'Mean Displacement Spectra\n(lat='+ lat_str +')', loc='left')

dd = (10 * np.log( (G_gFT_smth/G_gFT_smth.k) .isel(x = slice(0, -1))))#.plot()
dd = dd.where(~np.isinf(dd), np.nan)

## filter out segments with less then 10% of data points
dd= dd.where(G_gFT_smth["N_per_stancil_fraction"] >= 0.1)#.plot()

dd_lims = np.round(dd.quantile(0.01).data*0.95, 0) , np.round(dd.quantile(0.95).data*1.05, 0)
plt.pcolor(dd.x/1e3, dd.k, dd, vmin=dd_lims[0], vmax= dd_lims[-1], cmap = col.white_base_blgror)
cb = plt.colorbar(orientation= 'vertical')

cb.set_label('Power (m$^2$/k)')
plt.plot( G_gFT_smth.isel(x = slice(0, -1)).x/1e3 ,  G_gFT_smth.isel(x = slice(0, -1)).k_lim , color= col.black, linewidth = 1)
plt.ylabel('wavenumber k')
plt.xlabel('X (km)')

pos = gs[6:, -1]
ax9 = F.fig.add_subplot(pos)

plt.title('Data Coverage (%)', loc ='left')
plt.plot(G_gFT_smth.x/1e3 , G_gFT_smth["N_per_stancil_fraction"]*100 , linewidth = 0.8, color = 'black')
ax9.spines['left'].set_visible(False)
ax9.spines['right'].set_visible(True)
ax9.tick_params(labelright=True, right=True, labelleft=False, left=False)
ax9.axhline(10, linewidth = 0.8, linestyle= '--', color ='black')
#plt.ylabel('(%)')
plt.xlabel('X (km)')


F.save_light(path=plot_path, name =str(ID_name) + '_B06_atten_ov')
F.save_pup(path=plot_path, name = str(ID_name) + '_B06_atten_ov')


# %% reconstruct slope displacement data
def fit_offset(x, data,  model, nan_mask, deg):

    #x, data,  model, nan_mask, deg = dist_stencil, height_data, height_model, dist_nanmask, 1
    p_offset = np.polyfit(x[~nan_mask], data[~nan_mask] - model[~nan_mask], deg)
    p_offset[-1] = 0
    poly_offset = np.polyval(p_offset,x )
    return poly_offset

def tanh_fitler(x, x_cutoff , sigma_g= 0.01):
    """
        zdgfsg
    """

    decay   =  0.5 - np.tanh( (x-x_cutoff)/sigma_g  )/2
    return decay


#plt.plot(x, tanh_fitler(Gk_1.k, k_thresh, sigma_g= 0.003) )


def reconstruct_displacement(Gx_1, Gk_1, T3, k_thresh):

    """
    reconstructs photon displacement heights for each stancil given the model parameters in Gk_1
    A low-pass frequeny filter can be applied using k-thresh

    inputs:
    Gk_1    model data per stencil from _gFT_k file with sin and cos coefficients
    Gx_1    real data per stencil from _gFT_x file with mean photon heights and coordindate systems
    T3
    k_thresh (None) threshold for low-pass filter

    returns:
    height_model  reconstucted displements heights of the stancil
    poly_offset   fitted staight line to the residual between observations and model to account for low-pass variability
    nan_mask      mask where is observed data in
    """

    dist_stencil = Gx_1.eta + Gx_1.x
    dist_stencil_lims = dist_stencil[0].data, dist_stencil[-1].data

    gFT_cos_coeff_sel = np.copy(Gk_1.gFT_cos_coeff)
    gFT_sin_coeff_sel = np.copy(Gk_1.gFT_sin_coeff)

    gFT_cos_coeff_sel = gFT_cos_coeff_sel*tanh_fitler(Gk_1.k, k_thresh, sigma_g= 0.003)
    gFT_sin_coeff_sel = gFT_sin_coeff_sel*tanh_fitler(Gk_1.k, k_thresh, sigma_g= 0.003)

    # gFT_cos_coeff_sel[Gk_1.k > k_thresh] = 0
    # gFT_sin_coeff_sel[Gk_1.k > k_thresh] = 0


    FT_int = gFT.generalized_Fourier(Gx_1.eta + Gx_1.x, None,Gk_1.k )
    _ = FT_int.get_H()
    FT_int.p_hat = np.concatenate([ -gFT_sin_coeff_sel /Gk_1.k, gFT_cos_coeff_sel/Gk_1.k ])

    dx = Gx.eta.diff('eta').mean().data
    height_model = FT_int.model() /dx# + T3_sel['heights_c_weighted_mean'].iloc[0]

    dist_nanmask = np.isnan(Gx_1.y_data)
    height_data  = np.interp(dist_stencil, T3_sel['dist'],  T3_sel['heights_c_weighted_mean']) #[~np.isnan(Gx_1.y_data)]
    #poly_offset = fit_offset(dist_stencil, height_data, height_model, dist_nanmask, 1)

    return height_model, np.nan, dist_nanmask

# cutting Table data


# %%
G_height_model=dict()
k       = 'gt2l'
for bb in Gx.beam.data:
    G_height_model_temp= dict()
    for i in np.arange(Gx.x.size):
        #k_thresh= 4

        Gx_1    = Gx.isel(x= i).sel(beam = bb)
        Gk_1    = Gk.isel(x= i).sel(beam = bb)
        k_thresh= G_gFT_smth.k_lim.isel(x=0).data


        dist_stencil        = Gx_1.eta + Gx_1.x
        dist_stencil_lims   = dist_stencil[0].data, dist_stencil[-1].data
        dist_stencil_lims_plot = dist_stencil_lims#Gx_1.eta[0]*0.25 + Gx_1.x, Gx_1.eta[-1]*0.25 + Gx_1.x
        dist_stencil_lims_plot = Gx_1.eta[0]*1 + Gx_1.x, Gx_1.eta[-1]*1 + Gx_1.x

        T3_sel              = B3[k].loc[( (B3[k]['dist']   >= dist_stencil_lims[0])    & (B3[k]['dist']    <= dist_stencil_lims[1])   )]
        #T2_sel              = B2[k].loc[(  B2[k]['x_true'] >= T3_sel['x_true'].min() ) & ( B2[k]['x_true'] <= T3_sel['x_true'].max()  )]

        if T3_sel.shape[0] != 0:
            # if T3_sel['x_true'].iloc[-1] < T3_sel['x_true'].iloc[0]:
            #     dist_T2_temp =np.interp(T2_sel['x_true'][::-1], T3_sel['x_true'][::-1],  T3_sel['dist'][::-1] )
            #     T2_sel['dist']      = dist_T2_temp[::-1]
            # else:
            #     dist_T2_temp =np.interp(T2_sel['x_true'], T3_sel['x_true'],  T3_sel['dist'] )
            #     T2_sel['dist']      = dist_T2_temp

            height_model, poly_offset, dist_nanmask = reconstruct_displacement(Gx_1, Gk_1, T3_sel, k_thresh = k_thresh)
            poly_offset = poly_offset*0
            G_height_model_temp[str(i) + bb]     = xr.DataArray(height_model, coords=Gx_1.coords, dims= Gx_1.dims, name = 'height_model' )
        else:
            G_height_model_temp[str(i) + bb]     = xr.DataArray(Gx_1.y_model.data, coords=Gx_1.coords, dims= Gx_1.dims, name = 'height_model' )

        #G_height_nans[i]      = xr.DataArray(dist_nanmask, coords=Gx_1.coords, dims= Gx_1.dims, name = 'nanmask' )

        # jsut for plotting:
        # # corrected rar Photon heights
        # T2_sel['heights_c_residual']            = photon_height_residual = T2_sel['heights_c'] - np.interp(T2_sel['dist'], dist_stencil, height_model +  poly_offset)
        #
        # # interpolate rebinned photon heights
        # heights_c_weighted_mean_stancil         = np.interp(dist_stencil, T3_sel['dist'], T3_sel['heights_c_weighted_mean'] )
        #
        # # corrected rebinned photon heights
        # photon_height_residual_mean                = heights_c_weighted_mean_stancil   - (height_model + poly_offset)
        # photon_height_residual_mean[dist_nanmask]  = np.nan
        # T3_sel['heights_c_weighted_mean_residual'] = T3_sel['heights_c_weighted_mean'] - np.interp(T3_sel['dist'], dist_stencil, height_model +  poly_offset )

        #plot
        # font_for_pres()
        # M.figure_axis_xy(5.5, 6, view_scale = 0.8)
        #
        # plt.subplot(3,1 ,1)
        # plt.scatter(T2_sel['dist'], T2_sel['heights_c'], s= 1,  marker='o', color='black',   alpha =0.2, edgecolors= 'none' )
        # #plt.scatter(T3_sel['dist'], T3_sel['heights_c_weighted_mean'], s= 1,  marker='o', color='black',   alpha =0.2, edgecolors= 'none' )
        # plt.plot(T3_sel['dist'], T3_sel['heights_c_weighted_mean'] , color =col.rascade1, linewidth = 0.5, label = 'residual $h_c$')
        # plt.xlim(dist_stencil_lims_plot)
        # plt.ylim(0, 1.5)
        #
        # ax1 = plt.subplot(3,1 ,2)
        # plt.plot(dist_stencil, height_model + poly_offset ,'-', c='red', linewidth=0.8, alpha=1,zorder= 12, label = 'GFT height model + correction')
        # plt.plot(dist_stencil, height_model ,'-', c='orange', linewidth=0.8, alpha=0.5,zorder= 2, label = 'GFT height model')
        # plt.legend(loc = 1)
        # plt.xlim(dist_stencil_lims_plot)
        # ax1.axhline(0, linewidth=0.5, color= 'black')
        #
        # plt.subplot(3,1 ,3)
        # plt.scatter(T2_sel['dist'], T2_sel['heights_c_residual'], s= 1,  marker='o', color='black',   alpha =0.5, edgecolors= 'none', zorder=6 )
        # #plt.scatter(T2_sel['dist'], T2_sel['heights_c_residual'], s= 1,  marker='o', color='black',   alpha =1, edgecolors= 'none' )
        #
        # plt.plot(T3_sel['dist'], T3_sel['heights_c_weighted_mean_residual'],'-', c=col.rascade2, linewidth=0.5, alpha=1, zorder= 10, label = 'GFT height model + correction')
        # #plt.plot(dist_stencil, photon_height_residual_mean,'-', c='red', linewidth=0.3, alpha=1, zorder= 2, label = 'GFT height model + correction')
        # plt.fill_between(dist_stencil , photon_height_residual_mean, color= col.cascade2, edgecolor = None, alpha = 1, zorder= 0)
        #
        # plt.xlim(dist_stencil_lims_plot)
        # plt.ylim(0, 1.5)

    G_height_model[bb] = xr.concat(G_height_model_temp.values(), dim= 'x').T

Gx['height_model'] = xr.concat(G_height_model.values(), dim= 'beam').transpose('eta', 'beam', 'x')

# %%
Gx_v2, B2_v2, B3_v2 = dict(), dict(), dict()
for bb in Gx.beam.data:
    print(bb)
    Gx_k                 = Gx.sel(beam = bb)
    #Gx_k['height_model'] = xr.concat(G_height_model.values(), dim= 'x').T#.plot()
    Gh          = Gx['height_model'].sel(beam = bb).T
    Gh_err      = Gx_k['model_error_x'].T
    Gnans       = np.isnan(Gx_k.y_model)

    concented_heights   = Gh.data.reshape(Gh.data.size)
    concented_err       = Gh_err.data.reshape(Gh.data.size)
    concented_nans      = Gnans.data.reshape(Gnans.data.size)
    concented_x         = (Gh.x+Gh.eta).data.reshape(Gh.data.size)

    dx                      = Gh.eta.diff('eta')[0].data
    continous_x_grid        = np.arange(concented_x.min(), concented_x.max(), dx)
    continous_height_model  = np.interp(continous_x_grid, concented_x, concented_heights )
    concented_err           = np.interp(continous_x_grid, concented_x, concented_err )
    continous_nans          = np.interp(continous_x_grid, concented_x, concented_nans ) ==1

    T3              = B3[bb]#.loc[( (B3[k]['dist']   >= dist_stencil_lims[0])    & (B3[k]['dist']    <= dist_stencil_lims[1])   )]
    #T2              = B2[bb]#.loc[(  B2[k]['x_true'] >= T3_sel['x_true'].min() ) & ( B2[k]['x_true'] <= T3_sel['x_true'].max()  )]

    #T2 = T2.sort_values('x_true')
    T3 = T3.sort_values('x')
    #T2['dist']    = np.interp(T2['x_true'], T3['x_true'],  T3['dist'] )
    #T2 = T2.sort_values('dist')
    T3 = T3.sort_values('dist')

    #T2              = T2.sort_index()
    #T2['dist']      = np.interp(T2['x_true'], T3['x_true'],  T3['dist'] )

    T3['heights_c_model']     = np.interp(T3['dist'], continous_x_grid, continous_height_model)
    T3['heights_c_model_err'] = np.interp(T3['dist'], continous_x_grid, concented_err)
    T3['heights_c_residual']  = T3['heights_c_weighted_mean'] - T3['heights_c_model']

    #T2['heights_c_model']     = np.interp(T2['dist'], continous_x_grid, continous_height_model)
    #T2['heights_c_residual']  = T2['heights_c'] - T2['heights_c_model']


    #B2_v2[bb] = T2
    B3_v2[bb] = T3
    Gx_v2[bb] = Gx_k

    # font_for_print()
    # F = M.figure_axis_xy(6, 2, view_scale= 0.7)
    #
    # plt.plot(T2['dist'] , T2['heights_c']+2,'ok', markersize=0.8, alpha=0.5, label='org photon height_c')
    # plt.plot(T3['dist'] , T3['heights_c_weighted_mean']+2,'.r', markersize=1, alpha=0.5, label='org photon wmean')
    #
    # plt.plot(T2['dist'] , T2['heights_c_model'], '.', markersize=1, alpha=0.8, label='height model', color=col.orange, zorder= 12)
    # F.ax.axhline(2, linewidth = .7, color= 'black')
    # F.ax.axhline(0, linewidth = .7, color= 'black')
    # F.ax.axhline(-2, linewidth = .7, color= 'black')
    #
    # plt.plot(T2['dist'] , T2['heights_c_residual']-2,'ob', markersize=0.5, alpha=0.5, label='residual photons')
    # plt.plot(T3['dist'], T3['heights_c_residual']-2 , 'r', linewidth= 0.8, zorder=12, label='photon height_c resodual')
    #
    # xlims = np.nanmean(T2['dist']), np.nanmean(T2['dist'])+7e3
    # plt.xlim(xlims)
    # dlim = np.nanmax(T3['heights_c_residual'][(T3['dist']> xlims[0]) & (T3['dist'] < xlims[1])])
    # #plt.ylim(-dlim*1.5, dlim*1.5)
    # try:
    #     plt.ylim((-2-1.5*dlim), 2+1.5*dlim)
    # except:
    #     plt.ylim(-5, 5)
    # plt.legend( ncol= 4)
    #F.save_light(path = plot_path , name = 'B06_'+bb+'__check')


# %% correct wave incident direction

#load_path = mconfig['paths']['work'] + '/B04_angle/'

try:
    G_angle = xr.open_dataset(load_path_angle+ '/B05_'+ID_name + '_angle_pdf.nc' )

    font_for_pres()

    Ga_abs = (G_angle.weighted_angle_PDF_smth.isel(angle = G_angle.angle > 0).data + G_angle.weighted_angle_PDF_smth.isel(angle = G_angle.angle < 0).data[:,::-1])/2
    Ga_abs = xr.DataArray(data=Ga_abs.T, dims = G_angle.dims, coords=G_angle.isel(angle = G_angle.angle > 0).coords)

    Ga_abs_front = Ga_abs.isel(x= slice(0, 3))
    Ga_best = ((  Ga_abs_front * Ga_abs_front.N_data ).sum('x')/Ga_abs_front.N_data.sum('x'))

    theta = Ga_best.angle[Ga_best.argmax()].data
    theta_flag = True

    font_for_print()
    F = M.figure_axis_xy(3, 5, view_scale= 0.7)

    plt.subplot(2, 1, 1)
    plt.pcolor(Ga_abs)
    plt.xlabel('abs angle')
    plt.ylabel('x')

    ax = plt.subplot(2, 1, 2)
    Ga_best.plot()
    plt.title('angle front ' + str(theta*180/np.pi), loc='left')
    ax.axvline(theta, color= 'red')
    F.save_light(path = plot_path , name = 'B06_angle_def')
except:

    print('no angle data found, skip angle corretion')
    theta= 0
    theta_flag = False

# %%
lam_p   = 2 *np.pi/Gk.k
lam     = lam_p * np.cos(theta)

if theta_flag:
    k_corrected  = 2 * np.pi/lam
    x_corrected  = Gk.x * np.cos(theta)
else:
    k_corrected  = 2 * np.pi/lam *np.nan
    x_corrected  = Gk.x * np.cos(theta) *np.nan

# %% spectral save
G5 = G_gFT_wmean.expand_dims(dim = 'beam', axis = 1)
G5.coords['beam'] = ['weighted_mean']#(('beam'), 'weighted_mean')
G5 = G5.assign_coords(N_photons= G5.N_photons)
G5['N_photons'] = G5['N_photons'].expand_dims('beam')
G5['N_per_stancil_fraction'] = G5['N_per_stancil_fraction'].expand_dims('beam')

Gk_v2 = xr.merge([Gk, G5])

Gk_v2 = Gk_v2.assign_coords(x_corrected=("x", x_corrected.data)).assign_coords(k_corrected=("k", k_corrected.data))

Gk_v2.attrs['best_guess_incident_angle'] = theta

# save collected spectral data
Gk_v2.to_netcdf(save_path+'/B06_'+ID_name + '_gFT_k_corrected.nc' )
Gx
# %% save real space data
Gx.to_netcdf(save_path+'/B06_'+ID_name + '_gFT_x_corrected.nc' )
try:
    io.save_pandas_table(B2_v2, 'B06_' +ID_name + '_B06_corrected_resid' , save_path) # all photos but heights adjusted and with distance coordinate
except:
    os.remove(save_path+'B06_' +ID_name + '_B06_corrected_resid.h5')
    io.save_pandas_table(B2_v2, 'B06_' +ID_name + '_B06_corrected_resid' , save_path) # all photos but heights adjusted and with distance coordinate

try:
    io.save_pandas_table(B3_v2, 'B06_' +ID_name + '_binned_resid' , save_path) # regridding heights
except:
    os.remove(save_path+'B06_' +ID_name + '_binned_resid.h5')
    io.save_pandas_table(B3_v2, 'B06_' +ID_name + '_binned_resid' , save_path) # regridding heights

MT.json_save('B06_success', plot_path + '../', {'time':time.asctime( time.localtime(time.time()) )})
print('done. saved target at ' + plot_path + '../B06_success' )

# %%
