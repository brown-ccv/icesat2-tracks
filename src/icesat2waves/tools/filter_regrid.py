
import numpy as np
from numba import jit


def correct_heights(T03, T03c, coord = 'delta_time'):
    """
    returns the corrected photon heigts in T03 given SSSH approxiamtion 'dem_h' in T03c
    """

    T03['heights_c']= T03['heights'] -  np.interp( T03[coord],T03c[coord], T03c['dem_h'] )
    return T03


# def track_type_beam(hdf5_file):
#     """
#     Returns True if track is acending
#     hdf5_file is a hdf5 beam file
#
#     sc_orient - spacecraft orientation
#     This parameter tracks the spacecraft orientation between ‘forward’ and ‘backward’ orientations, to allow mapping between ATLAS hardware and the beam orientation on the ground. Forward == 1; backward == 0; transition ==2.
#
#
#     """
#     return hdf5_file['orbit_info/sc_orient'][:][0] ==0

def track_pole_ward_file(hdf5_file, product='ALT03'):
    """
    Returns true if track goes poleward
    hdf5_file is a an HFD5 object in read mode
    """

    if product == 'ALT03':
        T_lat = hdf5_file['gt1r/geolocation/reference_photon_lat'][:]
        T_time = hdf5_file['gt1r/geolocation/delta_time'][:]
    elif product == 'ALT10':
        T_lat = hdf5_file['gt1r/freeboard_beam_segment/latitude'][:]
        T_time = hdf5_file['gt1r/freeboard_beam_segment/delta_time'][:]
    #return ( T_lat[T_time.argmax()] - T_lat[T_time.argmin()] ) < 0
    print('1st lat =' + str(abs(T_lat[T_time.argmin()])) , ';last lat =' + str(abs(T_lat[T_time.argmax()])) )
    return abs(T_lat[T_time.argmax()]) > abs(T_lat[T_time.argmin()])



def track_type(T):
    """
    Returns if track acending or desending
    T is a pandas table
    """
    #T = B[k]
    #T = B[beams_list[0]]
    return (T['lats'].iloc[T['delta_time'].argmax()] - T['lats'].iloc[T['delta_time'].argmin()] ) < 0

def lat_min_max_extended(B, beams_list, accent=None):
    """
    defines common boundaries for beams_list in B
    iunputs:
    beams_list list of concidered beams
    B is dict of Pandas tables with beams
    accent if track is accending or decending. if None, this will try to use the track time to get this

    returns:
    min_lat, max_lat, accent   min and max latitudes of the beams, (True/False) True if the track is accending
    """
    #B, beams_list = B , high_beams
    accent = regrid.track_type( B[beams_list[0]] ) if accent is None else accent

    if B[beams_list[0]]['lats'].iloc[0] < 0:
        hemis = 'SH'
    else:
        hemis = 'NH'

    track_pos_start, track_pos_end= list(), list()
    for k in beams_list:
        if (hemis == 'SH'):
            track_pos_start.append( B[k].loc[B[k]['lats'].argmax()][ ['lats', 'lons']] )
            track_pos_end.append( B[k].loc[B[k]['lats'].argmin()][ ['lats', 'lons']] )
        else:
            track_pos_start.append( B[k].loc[B[k]['lats'].argmin()][ ['lats', 'lons']] )
            track_pos_end.append( B[k].loc[B[k]['lats'].argmax()][ ['lats', 'lons']] )


    track_lat_start, track_lat_end = list(), list()
    track_lon_start, track_lon_end = list(), list()

    for ll in track_pos_start:
        track_lat_start.append(ll['lats'])
        track_lon_start.append(ll['lons'])


    for ll in track_pos_end:
        track_lat_end.append(ll['lats'])
        track_lon_end.append(ll['lons'])

        # track_lat_start.append( B[k]['lats'].min() )
        # track_lat_end.append( B[k]['lats'].max() )
        #
        # track_lon_left.append(B[k]['lons'].min())
        # track_lon_right.append(B[k]['lons'].max())

    if accent:
        track_lon_start
    #track_lat_start.min(), track_lon_right.max()

    if (hemis == 'SH') & accent:
        return [max(track_lat_start) , min(track_lat_end)], [max(track_lon_start), min(track_lon_end)], accent # accenting SH mean start is in the top right
    elif (hemis == 'SH') & ~accent:
        return [max(track_lat_start) , min(track_lat_end)], [min(track_lon_start), max(track_lon_end)], accent # decent SH mean start is in the top left
    elif (hemis == 'NH') & accent:
        return [min(track_lat_start) , max(track_lat_end)], [min(track_lon_start), max(track_lon_end)], accent # accent NH mean start is in the lower left
    elif (hemis == 'NH') & ~accent:
        return [min(track_lat_start) , max(track_lat_end)], [max(track_lon_start), min(track_lon_end)], accent # decent NH mean start is in the lower right
    else:
        raise ValueError('some defintions went wrong')



def lat_min_max(B, beams_list, accent=None):
    """
    defines common boundaries for beams_list in B
    iunputs:
    beams_list list of concidered beams
    B is dict of Pandas tables with beams
    accent if track is accending or decending. if None, this will try to use the track time to get this

    returns:
    min_lat, max_lat, accent   min and max latitudes of the beams, (True/False) True if the track is accending
    """
    #B, beams_list = B , high_beams
    accent = track_type( B[beams_list[0]] ) if accent is None else accent

    if B[beams_list[0]]['lats'].iloc[0] < 0:
        hemis = 'SH'
    else:
        hemis = 'NH'

    track_lat_mins, track_lat_maxs= list(), list()
    for k in beams_list:
        track_lat_mins.append( B[k]['lats'].min() )
        track_lat_maxs.append( B[k]['lats'].max() )

    if hemis == 'SH':
        return max(track_lat_maxs) , min(track_lat_mins), accent
    else:
        return min(track_lat_mins), max(track_lat_maxs), accent

def derive_axis(TT, lat_lims = None):
    """
    returns TT distance along track 'dist' in meters
    input:
    TT pandas table with ICEsat2 track data
    lat_lims (None) tuple with the global latitude limits used to define local coodinate system
    returns:
    TT with x,y,dist and order by dist
    """
    #TT, lat_lims = B[key], lat_lims_high
    # derive distances in meters
    r_e= 6.3710E+6
    dy= r_e*2*np.pi/360.0
    #deglon_in_m= np.cos(T2['lats']*np.pi/180.0)*dy

    # either use position of the 1st photon or use defined start latitude
    if lat_lims is None:
        TT['y']=(TT['lats'].max() - TT['lats']) *dy
    else:
        TT['y']=(lat_lims[0] - TT['lats']) *dy

    #TT['y']     =   (TT['lats']) *dy


    if (lat_lims[2] == True):
        # accending track
        lon_min = TT['lons'].max()
    else:
        # decending track
        lon_min = TT['lons'].min()

    #print(lon_min)
    TT['x']     = (TT['lons'] - lon_min) * np.cos( TT['lats']*np.pi/180.0 ) * dy
    #TT['x']     = (TT['lons'] ) * np.cos( TT['lats']*np.pi/180.0 ) * dy
    TT['dist']  =   np.sqrt(TT['x']**2 + TT['y']**2)

    # set 1st dist to 0, not used if global limits are used
    if lat_lims is None:
        TT['dist']= TT['dist']- TT['dist'].min()
    else:
        TT['dist']= TT['dist']#- lat_lims[0]

    TT=TT.sort_values(by='dist')
    return TT

def reduce_to_height_distance(TT, key, dx=1, lat_lims = None):
    """
    interpolates key (photos heights) to regular grid using 'dist' in pandas table TT.
    dx          is the interpolation interval
    lat_lims    (None) tuple with the global latitude limits used to define local coodinate system
                if None 'dist' min and max are used

    returns:
    x1, y1     position, height
    """
    from scipy.interpolate import interp1d
    if type(dx) is np.ndarray:
        x1 = dx
    else:
        x1 = np.arange(0,TT['dist'].max(), dx)
    y1 = np.interp(x1, TT['dist'], TT[key] )

    return x1, y1

# this is not need anymore
def poly_correct(x, y, poly_order=7, plot_flag=False):

    """
    subtracts a fitted polynom to y
    inputs:
    x,y     position, height
    poly_order  order of polynom
    plot_flag   if true plots the fit
    returns
    y'      y - polynom fit
    """
    z = np.polyfit(x , y , poly_order)
    p = np.poly1d(z)
    if plot_flag:
        plt.plot(x,y, '.',  markersize=0.2,)
        plt.plot(x, p(x), '-',  markersize=0.2,)
    #return z
    return y - p(x)


### regridding

#@jit(nopython=True)
def get_mode(y, bins = np.arange(-5,5,  0.1)):
    "returns modes of histogram of y defined by bins"
    hist, xbin = np.histogram(y, bins = bins )
    return xbin[hist.argmax()]

@jit(nopython=True, parallel= False)
def weighted_mean(x_rel, y):
    "returns the gaussian weighted mean for stencil"

    #@jit(nopython=True, parallel= False)
    def weight_fnk(x):
        "returns gaussian weight given the distance to the center x"
        return np.exp(- (x/.5)**2 )

    w = weight_fnk(x_rel)
    return np.sum(w*y)/np.sum(w)

# this function is applied to beam:
def get_stencil_stats_shift( T2, stencil_iter,  key_var , key_x_coord, stancil_width ,  Nphoton_min = 5, plot_flag= False):

    """
    T2              pd.Dataframe with beam data needs at least 'dist' and key
    stencil_iter    np.array that constains the stancil boundaries and center [left boundary, center, right boundary]
    key_var         coloumn index used in T2
    key_x_coord     coloumn index of x coordinate
    stancil_width   width of stencil. is used to normalize photon positions within each stancil.
    Nphoton_min     minimum required photots needed to return meaning full averages

    returns:
    pandas DataFrame with the same as T2 but not taken the median of each column
    the following columns are also added:
    key+ '_weighted_mean'   x-weighted gaussian mean of key for each stencil
    key+ '_mode'            mode of key for each stencil
    'N_photos'              Number of Photons for each stencil
    key+ '_std'             standard deviation for each stencil

    the column 'key' is rename to key+'_median'

    """
    import pandas as pd
    stencil_1       = stencil_iter[:, ::2]
    stencil_1half   = stencil_iter[:, 1::2]

    def calc_stencil_stats(group, key,  key_x_coord, stancil_width, stancils):

        "returns stats per stencil"
        #import time
        #tstart = time.time()
        Nphoton     = group.shape[0]
        istancil = group['x_bins'].iloc[int(Nphoton/2)]
        stencil_center = stancils[1, istancil-1]


        if Nphoton > Nphoton_min:

            x_rel   = (group[key_x_coord] - stencil_center)/ stancil_width
            y   = group[key]

            #Tmedian[key+ '_weighted_mean']
            key_weighted_mean = weighted_mean(np.array(x_rel), np.array(y))
            key_std           = y.std()
            key_mode         =  get_mode(y)

        else:

            #Nphoton           = 0
            key_weighted_mean = np.nan
            #Tmedian[key+ '_mode']           = np.nan
            key_std            = np.nan
            key_mode           = np.nan

        #Tweight = pd.DataFrame([key_weighted_mean, key_std, Nphoton], index= [key+ '_weighted_mean', key+ '_std', 'N_photos' ])
        Tweight = pd.Series([key_weighted_mean, key_std, Nphoton, key_mode], index= [key+ '_weighted_mean', key+ '_std', 'N_photos', key+ '_mode' ])


        #print ( str( istancil) + ' s' + str(time.time() - tstart))
        return Tweight.T

    T_sets = list()
    stancil_set = stencil_1
    for stancil_set in [stencil_1, stencil_1half]:

        # select photons that are in bins
        Ti_sel = T2[  (stancil_set[0,0] < T2['x']) &  (T2['x'] < stancil_set[2,-1]) ]

        # put each photon in a bin
        bin_labels  = np.searchsorted(stancil_set[0,:], Ti_sel['x'])
        #bin_labels2 = np.digitize( Ti_sel['x'], stancil_set[0,:], right = True )

        Ti_sel['x_bins'] =bin_labels
        # group data by this bin
        Ti_g = Ti_sel.groupby(Ti_sel['x_bins'], dropna= False , as_index = True )#.median()

        # take median of the data
        Ti_median = Ti_g.median()

        # apply weighted mean and count photons
        args = [ key_var, key_x_coord, stancil_width, stancil_set]

        #%timeit -r 1 -n 1 Ti_weight  = Ti_g.apply(calc_stencil_stats, *args)
        Ti_weight  = Ti_g.apply(calc_stencil_stats, *args)

        #merge both datasets
        T_merged = pd.concat( [Ti_median, Ti_weight], axis= 1)

        # rename columns
        T_merged             =  T_merged.rename(columns={key_var: key_var+'_median', key_x_coord: key_x_coord+ '_median'})
        T_merged[ key_var+  '_median'][ np.isnan(T_merged[key_var+ '_std']) ] = np.nan # replace median calculation with nans

        # set stancil center an new x-coodinate
        T_merged['x'] =  stancil_set[1, T_merged.index-1]

        T_sets.append(T_merged)

    # mergeboth stancils
    T3 = pd.concat(T_sets ).sort_values(by= 'x').reset_index()

    if plot_flag:
        Ti_1, Ti_1half =  T_sets

        plt.plot( Ti_1half.iloc[0:60].x, Ti_1half.iloc[0:60]['heights_c_median'], '.' )
        plt.plot( Ti_1.iloc[0:60].x, Ti_1.iloc[0:60]['heights_c_median'], '.' )
        plt.plot( T3.iloc[0:120].x, T3.iloc[0:120]['heights_c_median'], '-' )


    return T3


# this function is applied to beam:
#old version
def get_stencil_stats(T2, stencil_iter,  key , key_x_coord, stancil_width ,  Nphoton_min = 5, map_func=None):

    """
    T2              pd.DAtaframe with beam data needs at least 'dist' and key
    stencil_iter    iterable that constains the stancil boundaries and center [left boundary, center, right boundary]
    key             coloumn index used in T2
    stancil_width   width of stencil. is used to normalize photon positions within each stancil.
    Nphoton_min     minimum required photots needed to return meaning full averages
    map_func        (None) mapping function passed to method. can be a concurrent.futures.map object or similar.
                    If None, standard python map function is used.

    returns:
    pandas DataFrame with the same as T2 but not taken the median of each column
    the following columns are also added:
    key+ '_weighted_mean'   x-weighted gaussian mean of key for each stencil
    key+ '_mode'            mode of key for each stencil
    'N_photos'              Number of Photons for each stencil
    key+ '_std'             standard deviation for each stencil

    the column 'key' is rename to key+'_median'

    """
    import pandas as pd
    import time

    x_data = np.array(T2[key_x_coord])
    y_data = np.array(T2[key])

    # apply this funcion to each stancil
    def calc_stencil_stats(istencil):

        "returns stats per stencil"

        tstart = time.time()
        i_mask      = (x_data >= istencil[0])  & (x_data < istencil[2])
        Nphoton     = sum(i_mask)

        if Nphoton < Nphoton_min:

            Tmedian = T2[i_mask].median()

            Tmedian[key+ '_weighted_mean']  = np.nan
            Tmedian[key+ '_mode']           = np.nan
            Tmedian['N_photos']             = Nphoton
            Tmedian[key+ '_std']            = np.nan

            return istencil[1], Tmedian


        x_rel   = (x_data[i_mask] - istencil[1])/ stancil_width
        y   = y_data[i_mask]

        Tmedian                             = T2[i_mask].median()
        Tmedian[key+ '_weighted_mean']      = weighted_mean(x_rel, y)
        Tmedian[key+ '_mode']               = get_mode(y)
        Tmedian['N_photos']                 = Nphoton
        Tmedian[key+ '_std']                = y.std()
        #Tmedian[key+  '_median'][ np.isnan(Tmedian[key+ 'std']) ]= np.nan # replace median calculation with nans
        print ( str( istencil[1]) + ' s' + str(time.time() - tstart))
        return istencil[1], Tmedian

    # apply func to all stancils
    map_func = map if map_func is None else map_func
    D_filt   = dict(map_func(calc_stencil_stats, stencil_iter))

    DF_filt         = pd.DataFrame.from_dict(D_filt, orient='index')
    DF_filt         = DF_filt.rename(columns={key: key+'_median', key_x_coord: 'median_'+key_x_coord})
    DF_filt[ key+  '_median'][ np.isnan(DF_filt[key+ '_std']) ] = np.nan # replace median calculation with nans
    DF_filt[key_x_coord] = DF_filt.index
    DF_filt         = DF_filt.reset_index()

    return DF_filt

# %% old version
# define common dist_grid:
#dx= 5 # 2 * resolution in meters, datapoint +-dx are used to take the mean
#dist_grid = np.arange( np.nanmin(dist_list[:, 0], 0) , np.nanmax(dist_list[:, 1], 0), dx )

# derive bin means
def bin_means(T2, dist_grid):
    dF_mean = pd.DataFrame(index =T2.columns)
    ilim    = int(len(dist_grid))
    N_i     = list()

    for i in np.arange(1,ilim-1, 1):
        if i % 5000 ==0:
            print(i)
        i_mask=(T2['dist'] >= dist_grid[i-1])  & (T2['dist'] < dist_grid[i+1])
        #if ( (T2['dist'] >= dist_grid[i-1])  & (T2['dist'] < dist_grid[i+1]) ).sum() > 0:
        dF_mean[i] = T2[i_mask].mean()
        #dF_median[i] = T2[i_mask].median()
        N_i.append(i_mask.sum())

    dF_mean             = dF_mean.T
    dF_mean['N_photos'] = N_i
    dF_mean['dist'] = dist_grid[np.arange(1,ilim-1, 1)]

    return dF_mean
