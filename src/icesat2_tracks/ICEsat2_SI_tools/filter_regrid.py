import numpy as np
from numba import jit
import pandas as pd
import matplotlib as plt


def process_single_stencil_set(
    stancil_set, T2, key_var, key_x_coord, stancil_width, calc_stencil_stats
):
    # Select photons that are in bins
    Ti_sel = T2[(stancil_set[0, 0] < T2["x"]) & (T2["x"] < stancil_set[2, -1])]

    # Put each photon in a bin
    bin_labels = np.searchsorted(stancil_set[0, :], Ti_sel["x"])
    Ti_sel["x_bins"] = bin_labels

    # Group data by this bin
    Ti_g = Ti_sel.groupby("x_bins", dropna=False, as_index=True)

    # Take median of the data
    Ti_median = Ti_g.median()

    # Apply weighted mean and count photons
    Ti_weight = Ti_g.apply(
        calc_stencil_stats, key_var, key_x_coord, stancil_width, stancil_set
    )

    # Merge both datasets
    T_merged = pd.concat([Ti_median, Ti_weight], axis=1)

    # Rename columns
    T_merged = T_merged.rename(
        columns={key_var: key_var + "_median", key_x_coord: key_x_coord + "_median"}
    )
    T_merged[key_var + "_median"][np.isnan(T_merged[key_var + "_std"])] = np.nan

    # Set stencil center as new x-coordinate
    T_merged["x"] = stancil_set[1, T_merged.index - 1]

    return T_merged


def get_hemis(B, beams_list):
    return "SH" if B[beams_list[0]]["lats"].iloc[0] < 0 else "NH"


def correct_heights(T03, T03c, coord="delta_time"):
    """
    returns the corrected photon heigts in T03 given SSSH approxiamtion 'dem_h' in T03c
    """

    T03["heights_c"] = T03["heights"] - np.interp(
        T03[coord], T03c[coord], T03c["dem_h"]
    )
    return T03


def track_pole_ward_file(hdf5_file, product="ALT03"):
    """
    Returns true if track goes poleward
    hdf5_file is a an HFD5 object in read mode
    """

    if product == "ALT03":
        T_lat = hdf5_file["gt1r/geolocation/reference_photon_lat"][:]
        T_time = hdf5_file["gt1r/geolocation/delta_time"][:]
    elif product == "ALT10":
        T_lat = hdf5_file["gt1r/freeboard_beam_segment/latitude"][:]
        T_time = hdf5_file["gt1r/freeboard_beam_segment/delta_time"][:]

    print(
        "1st lat =" + str(abs(T_lat[T_time.argmin()])),
        ";last lat =" + str(abs(T_lat[T_time.argmax()])),
    )

    _lhs = abs(T_lat[T_time.argmax()])
    _rhs = abs(T_lat[T_time.argmin()])
    return _lhs > _rhs


def track_type(T):
    """
    Returns if track acending or desending
    T is a pandas table
    """

    max_lat = T["lats"].iloc[T["delta_time"].argmax()]
    min_lat = T["lats"].iloc[T["delta_time"].argmin()]
    delta_lat = max_lat - min_lat
    return delta_lat < 0


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

    accent = track_type(B[beams_list[0]]) if accent is None else accent

    hemis = get_hemis(B, beams_list)

    track_pos_start, track_pos_end = list(), list()
    for k in beams_list:
        if hemis == "SH":
            track_pos_start.append(B[k].loc[B[k]["lats"].argmax()][["lats", "lons"]])
            track_pos_end.append(B[k].loc[B[k]["lats"].argmin()][["lats", "lons"]])
        else:
            track_pos_start.append(B[k].loc[B[k]["lats"].argmin()][["lats", "lons"]])
            track_pos_end.append(B[k].loc[B[k]["lats"].argmax()][["lats", "lons"]])

    track_lat_start, track_lat_end = list(), list()
    track_lon_start, track_lon_end = list(), list()

    for ll in track_pos_start:
        track_lat_start.append(ll["lats"])
        track_lon_start.append(ll["lons"])

    for ll in track_pos_end:
        track_lat_end.append(ll["lats"])
        track_lon_end.append(ll["lons"])

    # Define a dictionary to map the conditions to the functions
    func_map = {
        ("SH", True): (
            max,
            min,
            max,
            min,
        ),  # accenting SH mean start is in the top right
        ("SH", False): (max, min, min, max),  # decent SH mean start is in the top left
        ("NH", True): (min, max, min, max),  # accent NH mean start is in the lower left
        ("NH", False): (
            min,
            max,
            max,
            min,
        ),  # decent NH mean start is in the lower right
    }
    # Get the functions based on the conditions
    funcs = func_map.get((hemis, accent))
    # If the key is not found in the dictionary, raise an error
    if funcs is None:
        raise ValueError("some definitions went wrong")
    lat_start_func, lat_end_func, lon_start_func, lon_end_func = funcs
    # Use the functions to calculate the start and end of latitude and longitude
    # Return these values along with accent
    return (
        [lat_start_func(track_lat_start), lat_end_func(track_lat_end)],
        [lon_start_func(track_lon_start), lon_end_func(track_lon_end)],
        accent,
    )


def lat_min_max(B, beams_list, accent=None):
    """
    defines common boundaries for beams_list in B
    inputs:
    beams_list list of considered beams
    B is dict of Pandas tables with beams
    accent if track is ascending or descending. if None, this will try to use the track time to get this

    returns:
    min_lat, max_lat, accent   min and max latitudes of the beams, (True/False) True if the track is ascending
    """

    accent = track_type(B[beams_list[0]]) if accent is None else accent

    hemis = get_hemis(B, beams_list)

    track_lat_mins, track_lat_maxs = zip(
        *[(B[k]["lats"].min(), B[k]["lats"].max()) for k in beams_list]
    )

    if hemis == "SH":
        return max(track_lat_maxs), min(track_lat_mins), accent
    else:
        return min(track_lat_mins), max(track_lat_maxs), accent


def derive_axis(TT, lat_lims=None):
    """
    returns TT distance along track 'dist' in meters
    input:
    TT pandas table with ICEsat2 track data
    lat_lims (None) tuple with the global latitude limits used to define local coodinate system
    returns:
    TT with x,y,dist and order by dist
    """

    # derive distances in meters
    r_e = 6.3710e6
    dy = r_e * 2 * np.pi / 360.0

    # either use position of the 1st photon or use defined start latitude
    if lat_lims is None:
        TT["y"] = (TT["lats"].max() - TT["lats"]) * dy
    else:
        TT["y"] = (lat_lims[0] - TT["lats"]) * dy

    if lat_lims[2] == True:
        # accending track
        lon_min = TT["lons"].max()
    else:
        # decending track
        lon_min = TT["lons"].min()

    TT["x"] = (TT["lons"] - lon_min) * np.cos(TT["lats"] * np.pi / 180.0) * dy
    TT["dist"] = np.sqrt(TT["x"] ** 2 + TT["y"] ** 2)

    # set 1st dist to 0, not used if global limits are used
    if lat_lims is None:
        TT["dist"] = TT["dist"] - TT["dist"].min()

    TT = TT.sort_values(by="dist")
    return TT


def reduce_to_height_distance(TT, key, dx=1, lat_lims=None):
    """
    interpolates key (photos heights) to regular grid using 'dist' in pandas table TT.
    dx          is the interpolation interval
    lat_lims    (None) tuple with the global latitude limits used to define local coodinate system
                if None 'dist' min and max are used

    returns:
    x1, y1     position, height
    """

    x1 = dx if isinstance(dx, np.ndarray) else np.arange(0, TT["dist"].max(), dx)

    y1 = np.interp(x1, TT["dist"], TT[key])

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
    z = np.polyfit(x, y, poly_order)
    p = np.poly1d(z)
    if plot_flag:
        plt.plot(
            x,
            y,
            ".",
            markersize=0.2,
        )
        plt.plot(
            x,
            p(x),
            "-",
            markersize=0.2,
        )

    return y - p(x)


### regridding


def get_mode(y, bins=np.arange(-5, 5, 0.1)):
    "returns modes of histogram of y defined by bins"
    hist, xbin = np.histogram(y, bins=bins)
    return xbin[hist.argmax()]


@jit(nopython=True, parallel=False)
def weighted_mean(x_rel, y):
    "returns the gaussian weighted mean for stencil"

    def weight_fnk(x):
        "returns gaussian weight given the distance to the center x"
        return np.exp(-((x / 0.5) ** 2))

    w = weight_fnk(x_rel)
    return np.sum(w * y) / np.sum(w)


# this function is applied to beam:
def get_stencil_stats_shift(
    T2,
    stencil_iter,
    key_var,
    key_x_coord,
    stancil_width,
    Nphoton_min=5,
    plot_flag=False,
):
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

    stencil_1 = stencil_iter[:, ::2]
    stencil_1half = stencil_iter[:, 1::2]

    def calc_stencil_stats(group, key, key_x_coord, stancil_width, stancils):

        "returns stats per stencil"
        Nphoton = group.shape[0]
        istancil = group["x_bins"].iloc[int(Nphoton / 2)]
        stencil_center = stancils[1, istancil - 1]

        if Nphoton > Nphoton_min:

            x_rel = (group[key_x_coord] - stencil_center) / stancil_width
            y = group[key]
            key_weighted_mean = weighted_mean(np.array(x_rel), np.array(y))
            key_std = y.std()
            key_mode = get_mode(y)

        else:
            key_weighted_mean = np.nan
            key_std = np.nan
            key_mode = np.nan

        Tweight = pd.Series(
            [key_weighted_mean, key_std, Nphoton, key_mode],
            index=[key + "_weighted_mean", key + "_std", "N_photos", key + "_mode"],
        )

        return Tweight.T

    T_sets = [
        process_single_stencil_set(
            stancil_set, T2, key_var, key_x_coord, stancil_width, calc_stencil_stats
        )
        for stancil_set in [stencil_1, stencil_1half]
    ]

    # mergeboth stancils
    T3 = pd.concat(T_sets).sort_values(by="x").reset_index()

    if plot_flag:
        Ti_1, Ti_1half = T_sets

        plt.plot(Ti_1half.iloc[0:60].x, Ti_1half.iloc[0:60]["heights_c_median"], ".")
        plt.plot(Ti_1.iloc[0:60].x, Ti_1.iloc[0:60]["heights_c_median"], ".")
        plt.plot(T3.iloc[0:120].x, T3.iloc[0:120]["heights_c_median"], "-")

    return T3


# this function is applied to beam:
# old version
def get_stencil_stats(
    T2, stencil_iter, key, key_x_coord, stancil_width, Nphoton_min=5, map_func=None
):
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
        i_mask = (x_data >= istencil[0]) & (x_data < istencil[2])
        Nphoton = sum(i_mask)

        if Nphoton < Nphoton_min:

            Tmedian = T2[i_mask].median()

            Tmedian[f"{key}_weighted_mean"] = np.nan
            Tmedian[f"{key}_mode"] = np.nan
            Tmedian["N_photos"] = Nphoton
            Tmedian[f"{key}_std"] = np.nan

            return istencil[1], Tmedian

        x_rel = (x_data[i_mask] - istencil[1]) / stancil_width
        y = y_data[i_mask]

        Tmedian = T2[i_mask].median()
        Tmedian[f"{key}_weighted_mean"] = weighted_mean(x_rel, y)
        Tmedian[f"{key}_mode"] = get_mode(y)
        Tmedian["N_photos"] = Nphoton
        Tmedian[f"{key}_std"] = y.std()

        print(f"{istencil[1]} s{time.time() - tstart}")
        return istencil[1], Tmedian

    # apply func to all stancils
    map_func = map if map_func is None else map_func
    D_filt = dict(map_func(calc_stencil_stats, stencil_iter))

    DF_filt = pd.DataFrame.from_dict(D_filt, orient="index")
    DF_filt = DF_filt.rename(
        columns={key: f"{key}_median", key_x_coord: f"median_{key_x_coord}"}
    )
    DF_filt[f"{key}_median"][
        np.isnan(DF_filt[f"{key}_std"])
    ] = np.nan  # replace median calculation with nans
    DF_filt[key_x_coord] = DF_filt.index
    DF_filt = DF_filt.reset_index()

    return DF_filt


# derive bin means
def bin_means(T2, dist_grid):
    dF_mean = pd.DataFrame(index=T2.columns)
    ilim = len(dist_grid)
    N_i = list()

    for i in np.arange(1, ilim - 1):
        i_mask = (T2["dist"] >= dist_grid[i - 1]) & (T2["dist"] < dist_grid[i + 1])
        dF_mean[i] = T2[i_mask].mean()
        N_i.append(i_mask.sum())

    dF_mean = dF_mean.T
    dF_mean["N_photos"] = N_i
    dF_mean["dist"] = dist_grid[np.arange(1, ilim - 1, 1)]

    return dF_mean
