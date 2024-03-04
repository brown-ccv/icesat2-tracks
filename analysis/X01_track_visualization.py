
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
Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #\
# %%

Gi      = Gd['gt1r'][['dist', 'heights_c_weighted_mean', 'heights_c_std', 'N_photos', 'y','x']]
x       = Gi['dist']
xlims   = x.iloc[0], x.iloc[-1]




#Gi['slopes']  = dd


# %%

T_max = 40 #sec
k_0 = (2 * np.pi/ T_max)**2 / 9.81
x= np.array(Gi['dist'])
dx = np.diff(x).mean()
min_datapoint =  1/k_0/dx

Lpoints = int(np.round(min_datapoint) * 20)
Lmeters =Lpoints  * dx

stancil_iter = spec.create_chunk_boundaries_unit_lengths(Lmeters, xlims, ov= None , iter_flag=False)
# %%





# %%
def plot_beam(x,y, z, alpha_mean= 0.5, levels_std = 1, point_size= None ):

    if alpha_mean is False:
        alphas = 1
    else:
        alphas= (abs(z/z.std()/2) )**(1/2)
        alphas= alphas*alpha_mean
        alphas[alphas>=1] = 1
        alphas[alphas<0] = 0
        alphas[np.isnan(alphas)] = 0

    if type(levels_std) is not np.array:
        levels = np.linspace(-z.std()*levels_std, z.std()*levels_std, 30)
    else:
        levels = levels_std

    if point_size is None:
        psize = abs(z*400)#**(1.5)
    else:
        point_size = point_size

    plt.scatter(x,y, s =psize ,vmin=levels[0], vmax=levels[-1], c= z, marker='o', cmap= plt.cm.coolwarm, alpha= alphas, edgecolors= 'none' )



def get_stacil_data(stancil, G3, key = 'dist' ):

    mask = (G3[key] >= stancil[0]) & (G3[key] < stancil[2])
    return G3[np.array(mask)]

def make_slopes(G3, key = 'heights_c_weighted_mean', spreed =10, verbose= False, photon_min =5):

    dd      = np.copy(G3['heights_c_weighted_mean'])
    dd      = np.gradient(dd)
    dd, _   = spicke_remover.spicke_remover(dd, spreed=spreed, verbose=verbose)
    dd_nans = (np.isnan(dd) ) + (G3['N_photos'] <= photon_min)

    return dd, dd_nans

y_positions = {
'gt1l': 1,
'gt1r': 1.1,
'gt2l': 2,
'gt2r': 2.1,
'gt3l': 3,
'gt3r': 3.1
}
Gd.keys()

F = M.figure_axis_xy(8, 2, view_scale= 0.5)


stancil_iter = spec.create_chunk_boundaries_unit_lengths(Lmeters, xlims, ov= None , iter_flag=False)
#stancil = stancil_iter[:, 30]
stancil_iter.shape

for stancil in stancil_iter[:,6:10].T:
    print(stancil)

    for k,I in Gd.items():

        GG = get_stacil_data(stancil, I)
        slopes, nans = make_slopes(GG)
        GG['slopes']  = slopes
        #x,y, z = GG['lons'],GG['lats'] , GG['slopes']
        x,y, z = GG['dist'], GG['dist']*0 + y_positions[k] , GG['slopes']

        plot_beam(x,y, z)
        plt.plot(x.iloc[0], y.iloc[0], 'o', color='orange', markersize=5, alpha = 1)
        plt.plot(x.iloc[-1], y.iloc[-1], '|', color='green', markersize=20, alpha = 1)
        #plt.axis('equal')
