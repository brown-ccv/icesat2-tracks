
import os, sys
#execfile(os.environ['PYTHONSTARTUP'])

"""
This file open a ICEsat2 track applied filters and corections and returns smoothed photon heights on a regular grid in an .nc file.
This is python 3
"""
if __name__ == '__main__':
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

    #plot_path   = mconfig['paths']['plot'] + '/'+hemis+'/'+batch_key+'/' + track_name + '/B_spectra/'
    plot_path   = mconfig['paths']['plot'] + '/phase_fitting_fake/2D_fake/'
    MT.mkdirs_r(plot_path)
    MT.mkdirs_r(save_path)
    bad_track_path =mconfig['paths']['work'] +'bad_tracks/'+ batch_key+'/'
# %%
# if __name__ == '__main__':
#     all_beams   = mconfig['beams']['all_beams']
#     high_beams  = mconfig['beams']['high_beams']
#     low_beams   = mconfig['beams']['low_beams']
#     #Gfilt   = io.load_pandas_table_dict(track_name + '_B01_regridded', load_path) # rhis is the rar photon data
#
#     load_path   = mconfig['paths']['work'] +'/B01_regrid_'+hemis+'/'
#     Gd      = io.load_pandas_table_dict(track_name + '_B01_binned' , load_path)  #
#     load_path   = mconfig['paths']['work'] + '/B02_spectra_'+hemis+'/'
#     Gpars   = io.load_pandas_table_dict('B02_'+ track_name + '_params' , load_path)  #
#     Gspec   = xr.open_dataset(load_path + 'B02_'+ track_name + '_LS.nc' )  #
#     Gspec['Y_model_hat'] = Gspec.Y_model_hat_real + Gspec.Y_model_hat_imag *1j
#     Gspec = Gspec.drop('Y_model_hat_real').drop('Y_model_hat_imag')
#
#     dk = Gspec.k.diff('k').mean().data
#     Lpoints = Gspec.Lpoints
#
#     Gspec = Gspec.sel(k = slice(0.000125, 0.025)).isel(x =slice(4, 30))
#
#     Gspec.coords['f'] = (('k'), np.sqrt(Gspec.k.data * 9.81)/ 2/np.pi )
#     Gspec.coords['T'] = 1/Gspec.coords['f']
#     Gspec=Gspec.swap_dims({'k': 'f'})
#
#     #Gspec.spectral_power_optm.sel(beam='weighted_mean').plot()
#     # %%
#     #k_lim= 0.02
#     A, B = Gspec.sel(beam= 'gt2r').Y_model_hat , Gspec.sel(beam= 'gt2l').Y_model_hat
#
#     r_ave_kargs={'x':2, 'f':10, 'center':True, 'min_periods':2}
#     r_ave_kargs2={'f':10, 'center':True, 'min_periods':2}
#     #(abs(B) - abs(A)).plot()
#     S_aa = (A*A.conj()).real
#     S_bb = (B*B.conj()).real
#     # co_spec = (A.conj() *B) /S_aa/S_bb
#     # abs(co_spec).plot()
#     co_spec = (A.conj() *B).rolling(**r_ave_kargs).mean()
#     np.log(abs(co_spec)).plot(levels=np.arange(-2, 3, 0.1))
#
#     (abs(co_spec)).plot(levels=np.exp(np.arange(-3, 2, 0.1)))
#
#     abs(co_spec).mean('x').plot()

# %%


x = np.linspace(-np.pi, np.pi, 200)

plt.plot(x, np.cos(x /2)**1*2 )
plt.plot(x, np.cos(x /2)**1.5*2 )
plt.plot(x, np.cos(x /2)**2*2 )
plt.grid()

# %%
fp= 1/12.0
f = np.arange(1/50, 1/5, 0.001)
beta = 2.4*(f / (0.95 *fp  ) )
bb, xx= np.meshgrid(beta, x)

# %%
plt.contourf(x, f, (1 /np.cosh( (bb *xx) )**2).T )

plt.grid()


#(abs(co_spec)/(S_aa *S_bb).rolling(**r_ave_kargs).mean()).plot()
#
# (abs(A.conj() *B)/(S_aa *S_bb)).rolling(**r_ave_kargs).mean()[:,:].plot()

# # %%
# if __name__ == '__main__':
#     L1 = 50
#     k1 = 2* np.pi /L1
#     l1 = 2* np.pi /L1
#
#     L2 = 65
#     k2 = 2* np.pi /L2
#
#     x=np.arange(-250, 250, 0.5)
#     y=np.arange(-200, 200, 0.5)
#     Nx, Ny= x.size, y.size
#     XX, YY = np.meshgrid(x, y)
#     XX, YY =  XX.reshape(XX.size), YY.reshape(YY.size)
#
#     alpha = 35
#     kk, ll = np.cos(alpha * np.pi/180) * np.array([0.9*k1, k1, 1.1* k1]), np.sin(alpha * np.pi/180) * np.array([0.9* k1, 1*k1, 1.1* k1])
#     M_k, M_l = kk.size, ll.size
#     #y =np.sin(k1* x) + np.sin(k2* x)
#     kk_mesh, ll_mesh = np.meshgrid(kk, ll)
#     kk_mesh, ll_mesh = kk_mesh.reshape(kk_mesh.size), ll_mesh.reshape(ll_mesh.size)
#     G = np.cos(np.outer(XX, kk_mesh) + np.outer(YY, ll_mesh)).T# + np.sin(np.outer(XX, kk_mesh) + np.outer(YY, ll_mesh)).T
#     #G = np.vstack([ np.cos(np.outer(x, k) + np.outer(y, l)).T ,  np.sin(np.outer(x, k)  + np.outer(y, l) ).T ] ).T
#     G.shape
#
#     plt.contourf(x, y, G.sum(0).reshape(Ny, Nx) )
#     plt.axis('equal')

 # %% radial coordincates

def gaus_2d(x, y, pos_tuple, sigma_g ):
    #grid = ( (XX - pos_tuple[0])  * (YY - pos_tuple[1]) )
    gx = np.exp(-0.5 * (x - pos_tuple[0])**2 /sigma_g**2  )
    gy = np.exp(-0.5 * (y - pos_tuple[1])**2 /sigma_g**2  )
    return  np.outer(gx ,  gy).T

if __name__ == '__main__':

    k_range = np.linspace(0, 0.1,   30)
    l_range = np.linspace(-0.1, .1 ,  60)
    kk, ll = np.meshgrid(k_range, l_range)
    gaus_lk = gaus_2d( k_range,  l_range, [0.02, 0.0] , 0.01)
    #
    # M.figure_axis_xy(4, 4, view_scale= 0.5)
    # plt.contourf(k_range, l_range, gaus_lk  )
    # plt.axis('equal')

# %%

k_0 = 0.03
l_0 = 0
dk = 0.01
stancil_size =0
def get_stancils_kl(k_0, l_0, size =1 , dk= 0.01, mesh = True):
    import numpy as np
    """
    size    is the stancil half width. if 0 the stancil is 1, if one the stancil is 3 and so on.

    """
    if size ==0:
        stancil_k = np.array(k_0)
        stancil_l = np.array(l_0)
    else:
        stancil_k = (np.arange(-size, size  +1 , 1)  *dk + k_0 )
        stancil_l = (np.arange(-size, size  +1 , 1)  *dk + l_0 )

    if mesh:
        stancil_k_mesh, stancil_l_mesh = np.meshgrid(stancil_k, stancil_l)
    else:
        stancil_k_mesh, stancil_l_mesh = stancil_k, stancil_l

    return stancil_k_mesh, stancil_l_mesh


def get_rand_stancils(a_mean, size =1 , dk= 0.01):
    import numpy as np
    """
    size    is here the total stancil size.
    dk      is the 2d std of the gaussian

    """
    if size == 1:
        stancil_k = np.array(a_mean)
    else:

        stancil_k       = np.random.normal(a_mean, dk,size-1)
        stancil_k       = np.insert(stancil_k ,0, a_mean )
    return stancil_k


def gaus_2d_mesh(XX,YY, pos_tuple, sigma_g ):
    #grid = ( (XX - pos_tuple[0])  * (YY - pos_tuple[1]) )
    import numpy as np
    gx = np.exp(-0.5 * (XX - pos_tuple[0])**2 /sigma_g**2  )
    gy = np.exp(-0.5 * (YY - pos_tuple[1])**2 /sigma_g**2  )
    return  (gx *  gy).T

if __name__ == '__main__':

    k_mesh, l_mesh = get_stancils_kl(k_0, l_0, size= stancil_size, dk= dk)
    amp_mesh  =gaus_2d_mesh( k_mesh, l_mesh , [k_0, l_0] , dk)


    stancil_k_mesh, stancil_l_mesh = k_mesh.reshape(k_mesh.size), l_mesh.reshape(l_mesh.size)
    stancil_amp_mesh = amp_mesh.reshape(amp_mesh.size)

    # plt.contourf(k_mesh, l_mesh, amp_mesh)
    # plt.axis('equal')


# %% radial coodinates
def get_stancils_polar( amp, angle_rad,  size=1, dk = 0.01, mesh = True, plot_flag = True, amp_std= None, random=True):
    """
    inputs:

    amp     length of peak k vector in radial coordinates
    angle_rad   angle of peak k vector in radians between - pi/2 to  + pi/2
    size        determines number of wave numbers used M = size*2+1. if 0, it returns single wave number, if 1 it returns 3 wavenumbers
    dk          spread between the wavenumber
    mesh        (True) the returns are MxM stancils (M = size*2+1), if False only the terms along the cross are used.
    plot_flag   plots the wavegroup in k-l space

    returns:
    list of k wave numbers, list of l wave numbers, list of relative amplitudes, shape of the stancil
    """
    import numpy as np
    k0 = amp * np.cos(angle_rad)
    l0 = amp * np.sin(angle_rad)

    if amp_std is None:
        amp_std = dk
    else:
        amp_std = amp_std

    if random:

        k_mesh = get_rand_stancils(k0, size= size, dk= dk)
        l_mesh = get_rand_stancils(l0, size= size, dk= dk)

        amp_mesh        =  get_rand_stancils(0, size= size, dk= 0.2)
        amp_mesh=  np.ones(amp_mesh.size) - abs(amp_mesh)
        stancil_k_mesh, stancil_l_mesh, stancil_amp_mesh = k_mesh, l_mesh, amp_mesh
    else:

        k_mesh, l_mesh = get_stancils_kl(k0, l0, size= size, dk= dk, mesh= mesh)
        amp_mesh       = gaus_2d_mesh( k_mesh, l_mesh , [k0, l0] , amp_std)

        stancil_k_mesh, stancil_l_mesh  = k_mesh.reshape(k_mesh.size), l_mesh.reshape(l_mesh.size)
        stancil_amp_mesh                = amp_mesh.reshape(amp_mesh.size)

    amp_mesh = amp_mesh/amp_mesh.sum()


    #print(k_mesh, l_mesh, amp_mesh)
    if plot_flag:
        import matplotlib.pyplot as plt
        if size == 1:
            plt.plot(k_mesh, l_mesh,  '.', markersize= amp_mesh*3, color= 'black')
        else:
            if random:
                plt.scatter(k_mesh, l_mesh, amp_mesh*3,  color= 'black')
            else:
                plt.contour(k_mesh, l_mesh, amp_mesh, colors= 'black', linewidths= 1)


    return stancil_k_mesh, stancil_l_mesh, stancil_amp_mesh, k_mesh.shape


# %%
font_for_pres()
#k_mesh.shape
#for angle in np.arange(-80, 80+20, 40):
#for phase in np.arange(0, 2*np.pi, np.pi/3):
#for k_abs in np.arange(0.01, 0.09, 0.01):

angle =30
amp =1
phase = 0
#k_abs = 0.1
k_abs = (2* np.pi/10)**2 / 9.81
k_abs_noise = (2* np.pi/4)**2 / 9.81



x=np.arange(-200, 200, 0.5) * 2
y=np.arange(-200, 200, 0.5) * 2
Nx, Ny= x.size, y.size
XX, YY = np.meshgrid(x, y)
XX, YY =  XX.reshape(XX.size), YY.reshape(YY.size)

#for dk in np.arange(0.005+0.005, 0.02, 0.002):
#for size in [4, 5, 6]:
# %%

    dk = 0.016
    size = 4

    #for angle in np.arange(-80, 80+20, 40):
    #for k_abs in np.arange(0.01, 0.09, 0.01):
    #for phase in np.arange(0, 2*np.pi, np.pi/3):


    F = M.figure_axis_xy( fig_sizes['one_column_high'][0], fig_sizes['one_column_high'][1]* 1.5, view_scale = 0.8, container =True)
    #plt.suptitle('k_abs=' + str(k_abs) +' \nangle=' + str(angle) + ' \nsize=' + str(size) +' dk=' + str(dk) )
    gs = GridSpec(8,1,  wspace=0.1,  hspace=30.8)
    ax1 = F.fig.add_subplot(gs[0:4, 0])
    plt.title('Narrow-banded waves in \nspectral space', loc= 'left')

    #ax = plt.subplot(1, 2, 1)
    #k_list, l_list, amp_weights, stancil_shape = get_stancils_polar(k_abs, angle * np.pi/180, size=size, dk = dk, mesh = True , plot_flag= True, random = True)
    k_list, l_list, amp_weights, stancil_shape = [0.03485149, 0.03299441, 0.03838695, 0.02980313], [0.02012152, 0.0399972 , 0.02650097, 0.01930944], [1.        , 0.77879636, 0.91494225, 0.95463948],20
    plt.scatter(k_list, l_list, s = 15*np.array(amp_weights)**4,  color= col.rascade1)
    plt.plot(k_list[0], l_list[0], '.', markersize= 9*amp_weights[0]**4,  color=  col.rascade2)
    #np.array(amp_weights)**2
    circle1 = plt.Circle((k_list[0], l_list[0]), dk, color=col.black,linewidth= 0.5,  fill=False)
    ax1.add_patch(circle1)

    kk= np.arange(0, 0.1, 0.001)
    plt.contourf(kk, kk, gaus_2d(kk, kk, (k_list[0], l_list[0]), dk), np.linspace(0,1.7, 21) ,cmap= plt.cm.Greys , zorder=0)

    k_noise, l_noise, amp_noise, stancil_shape = get_stancils_polar(0.25, 0 * np.pi/180, size=20, dk = 0.6, mesh = True , plot_flag= False, random = True)
    amp_noise = (amp_noise *0+1) * 0

    plt.xlim(0, 0.1)
    #plt.axis('equal')
    plt.ylim(0, 0.1)
    plt.xlabel('along track\nwavenumber k')
    plt.ylabel('across track\nwavenumber l')
    #plt.axis('equal')

    # % Derive real space model
    #plt.subplot(1, 2, 2)
    ax2 = F.fig.add_subplot(gs[4:, 0])
    plt.title('Real space', loc= 'left')
    k_all = np.concatenate([k_list, k_noise])
    l_all = np.concatenate([l_list, l_noise])
    amp_all = np.concatenate([amp_weights, amp_noise])
    amp_all.shape


    G = np.vstack([ np.cos(np.outer(XX, k_all) + np.outer(YY, l_all)).T ,  np.sin(np.outer(XX, k_all) + np.outer(YY, l_all)).T ] ).T
    #G = np.vstack([ np.cos(np.outer(XX, k_noise) + np.outer(YY, l_noise)).T             ,  np.sin(np.outer(XX, k_noise)  + np.outer(YY, l_noise) ).T ] ).T

    #phase1 = np.random.rand(1, amp_list.size) *  np.pi*2
    phase = np.arange(0, amp_all.size) *  np.pi/1

    #amp_all.shape
    b = np.hstack([ np.cos(phase)*amp_all, np.sin(phase) *amp_all]).squeeze() * amp
    z_model = (G @ b).reshape(Ny, Nx)

    ax2.axhline(-45, color= col.rels['gt2l'])
    ax2.axhline(45, color= col.rels['gt2r'])
    plt.pcolor(x, y, z_model, cmap =plt.cm.coolwarm )

    plt.axis('equal')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    F.save_light(path = plot_path, name = 'fake_2d_publish_dk' +str(dk) +'_s' + str(int(size)))
    #plt.show()
