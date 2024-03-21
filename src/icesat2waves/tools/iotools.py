import logging
import os
import re
import json
from pathlib import Path
import warnings
from datetime import datetime
from netrc import netrc
from lxml import etree
from posixpath import join as posixpath_join
from pandas import HDFStore
from pandas.io.pytables import PerformanceWarning
import pandas as pd
import h5py
from sliderule import icesat2
from icesat2waves.tools import sliderule_converter_tools as sct
import icesat2_toolkit.utilities
import icesat2waves.tools.convert_GPS_time as cGPS


_logger = logging.getLogger(__name__)


def init_from_input(arguments):
    """
    Initializes the variables track_name, batch_key, and test_flag based on the input arguments.

    Parameters:
    arguments (list): A list of input arguments.

    Returns:
    tuple: A tuple containing the values of track_name, batch_key, and test_flag.
    """

    if (len(arguments) <= 1) | ("-f" in set(arguments)):
        track_name = "20190605061807_10380310_004_01"
        batch_key = "SH_batch01"
        test_flag = True
        _logger.debug("use standard values")

    else:
        track_name = arguments[1]
        batch_key = arguments[2]

        if len(arguments) >= 4:
            if arguments[3] == "True":
                test_flag = True
            elif arguments[3] == "False":
                test_flag = False
            else:
                test_flag = arguments[3]

        else:
            test_flag = False

    return track_name, batch_key, test_flag


def init_data(ID_name, batch_key, ID_flag, ID_root, prefix="A01b_ID"):
    """
    Takes inputs and retrieves the ID, track_names that can be loaded, hemis, batch
    inputs: are the outputs from init_from_input, specifically
    ID_name     can be either a track_name of the form '20190101015140_00550210_005_01', or
                'NH_20190301_09560205'
    batch_key   batch key of the form 'SH_anystring'
    ID_flag     if True, ID_name has the form 'NH_20190301_09560205' and the ID's json file is loaded
    ID_root     root folder of the ID json files
    prefix      prefix of the processing stage. Used to construct the ID path

    """

    _logger.debug(
        "id name: %s, batch key: %s, id flag: %s", ID_name, batch_key, ID_flag
    )
    hemis, batch = batch_key.split("_")

    if ID_flag:
        ID_path = ID_root + "/" + batch_key + "/" + prefix + "/"
        ID = json_load(prefix + "_" + ID_name, ID_path)
        track_names = ID["tracks"]["ATL03"]

    else:
        track_names = ["ATL03_" + ID_name]
        ID = ID_name

    return ID, track_names, hemis, batch


def ID_to_str(ID_name):
    IDs = ID_name.split("_")
    date = datetime.strptime(IDs[1], "%Y%m%d").strftime("%Y-%m-%d")
    return IDs[0] + " " + date + " granule: " + IDs[2]


def get_atl06p(ATL03_track_name, params_yapc, maximum_height):
    """
    This method retrieves the ATL06 data from sliderule and returns a geodataframe. It also applies the corrections and removes the points above the maximum height. If the geodataframe is empty, an exception is raised.

    Parameters:
    ATL03_track_name (str): The name of the ATL03 track.
    params_yapc (dict): The parameters for the YAPC correction.
    maximum_height (float): The maximum height to filter out.

    Returns:
    geopandas.GeoDataFrame: The geodataframe containing the ATL06 data.
    """
    _logger.debug("Retrieving data from sliderule ...")
    gdf = icesat2.atl06p(params_yapc, resources=[ATL03_track_name])

    if gdf.empty:
        raise ValueError("Empty Geodataframe. No data could be retrieved.")

    _logger.debug("Initial data retrieved")
    gdf = sct.correct_and_remove_height(gdf, maximum_height)
    return gdf


class case_ID:
    """docstring for case_ID"""

    def __init__(self, track_name):
        track_name_pattern = r"(\D{2}|\d{2})_?(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?_(\d{4})(\d{2})(\d{2})_?(\d{3})?_?(\d{2})?"

        # Compile the regular expression pattern for track names
        track_name_rx = re.compile(track_name_pattern)

        # Use the compiled regular expression to find all matches in the track name
        # The pop() method is used to get the last (or only) match
        # The result is a tuple, which is unpacked into several properties of the current object
        (
            self.hemis,  # Hemisphere
            self.YY,  # Year
            self.MM,  # Month
            self.DD,  # Day
            self.HH,  # Hour
            self.MN,  # Minute
            self.SS,  # Second
            self.TRK,  # Track
            self.CYC,  # Cycle
            self.GRN,  # Granule
            self.RL,  # Release
            self.VRS,  # Version
        ) = track_name_rx.findall(track_name).pop()

        hemis_map = {"01": "NH", "02": "SH"}
        self.hemis = hemis_map.get(self.hemis, self.hemis)

        self.set()
        self.track_name_init = track_name

    def set(self):
        block1 = (self.YY, self.MM, self.DD)
        block2 = (self.TRK, self.CYC, self.GRN)

        self.ID = self.hemis + "_" + "".join(block1) + "_" + "".join(block2)
        return self.ID

    def get_granule(self):
        return "".join((self.TRK, self.CYC, self.GRN))

    def set_dummy(self):
        block1 = (self.YY, self.MM, self.DD)
        block2 = (self.TRK, self.CYC, self.GRN)

        self.ID_dummy = "".join(block1) + "_" + "".join(block2)
        return self.ID_dummy

    def set_ATL03_trackname(self):
        block1 = (self.YY, self.MM, self.DD)
        block1b = (self.HH, self.MN, self.SS)
        block2 = (self.TRK, self.CYC, self.GRN)
        if self.RL is "":
            raise ValueError("RL not set")
        if self.VRS is "":
            raise ValueError("VRS not set")

        block3 = (self.RL, self.VRS)

        self.ID_ATL03 = (
            "".join(block1)
            + "".join(block1b)
            + "_"
            + "".join(block2)
            + "_"
            + "_".join(block3)
        )
        return self.ID_ATL03

    def set_ATL10_trackname(self):
        block1 = (self.YY, self.MM, self.DD)
        block1b = (self.HH, self.MN, self.SS)
        block2 = (self.TRK, self.CYC, "01")  # granule is always '01' for ATL10
        if self.RL is "":
            raise ValueError("RL not set")
        if self.VRS is "":
            raise ValueError("VRS not set")

        block3 = (self.RL, self.VRS)

        if self.hemis == "NH":
            hemis = "01"
        elif self.hemis == "SH":
            hemis = "02"
        else:
            hemis = self.hemis

        self.ID_ATL10 = (
            hemis
            + "_"
            + "".join(block1)
            + "".join(block1b)
            + "_"
            + "".join(block2)
            + "_"
            + "_".join(block3)
        )
        return self.ID_ATL10


def nsidc_icesat2_get_associated_file(
    file_list, product, build=True, username=None, password=None
):
    """
    THis method returns assocociated files names and paths for files given
    in file_list for the "product" ICEsat2 product
    input:
    file_list:
    list of the form [ATL03_20190301004639_09560204_005_01, ..]
    or [processed_ATL03_20190301004639_09560204_005_01, ..]
    product:
    ATL03, (or, ATL10, ATL07, not tested)

    """

    AUXILIARY = False
    DIRECTORY = None
    FLATTEN = False
    TIMEOUT = 120

    if build and not (username or password):
        urs = "urs.earthdata.nasa.gov"
        username, login, password = netrc.netrc().authenticators(urs)
    # -- build urllib2 opener and check credentials
    if build:
        # -- build urllib2 opener with credentials
        icesat2_toolkit.utilities.build_opener(username, password)
        # -- check credentials
        icesat2_toolkit.utilities.check_credentials()

    parser = lxml.etree.HTMLParser()
    # -- remote https server for ICESat-2 Data
    HOST = "https://n5eil01u.ecs.nsidc.org"
    # -- regular expression operator for extracting information from files
    rx = re.compile(
        r"(processed_)?(ATL\d{2})(-\d{2})?_(\d{4})(\d{2})(\d{2})"
        r"(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})"
    )
    # -- regular expression pattern for finding specific files
    regex_suffix = "(.*?)$" if AUXILIARY else "(h5)$"
    remote_regex_pattern = (
        r"{0}(-\d{{2}})?_(\d{{4}})(\d{{2}})(\d{{2}})"
        r"(\d{{2}})(\d{{2}})(\d{{2}})_({1})({2})({3})_({4})_(\d{{2}})(.*?).{5}"
    )

    # -- build list of remote files, remote modification times and local files
    original_files = []
    remote_files = []
    remote_mtimes = []
    local_files = []
    remote_names = []

    for input_file in file_list:
        _logger.debug("input file: %s", input_file)
        # -- extract parameters from ICESat-2 ATLAS HDF5 file name
        SUB, PRD, HEM, YY, MM, DD, HH, MN, SS, TRK, CYC, GRN, RL, VRS = rx.findall(
            input_file
        ).pop()
        # -- get directories from remote directory
        product_directory = "{0}.{1}".format(product, RL)
        sd = "{0}.{1}.{2}".format(YY, MM, DD)
        PATH = [HOST, "ATLAS", product_directory, sd]
        # -- local and remote data directories
        remote_dir = posixpath.join(*PATH)
        temp = os.path.dirname(input_file) if (DIRECTORY is None) else DIRECTORY
        local_dir = os.path.expanduser(temp) if FLATTEN else os.path.join(temp, sd)
        # -- create output directory if not currently existing
        # if not os.access(local_dir, os.F_OK):
        #     os.makedirs(local_dir, MODE)
        # -- compile regular expression operator for file parameters
        args = (product, TRK, CYC, GRN, RL, regex_suffix)
        R1 = re.compile(remote_regex_pattern.format(*args), re.VERBOSE)
        # -- find associated ICESat-2 data file
        # -- find matching files (for granule, release, version, track)
        colnames, collastmod, colerror = icesat2_toolkit.utilities.nsidc_list(
            PATH, build=False, timeout=TIMEOUT, parser=parser, pattern=R1, sort=True
        )
        _logger.debug("colnames: %s", colnames)
        # -- print if file was not found
        if not colnames:
            _logger.debug("colerror: %s", colerror)
            continue
        # -- add to lists
        for colname, remote_mtime in zip(colnames, collastmod):
            # -- save original file to list (expands if getting auxiliary files)
            original_files.append(input_file)
            # -- remote and local versions of the file
            remote_files.append(posixpath.join(remote_dir, colname))
            local_files.append(os.path.join(local_dir, colname))
            remote_mtimes.append(remote_mtime)
            remote_names.append(colname)

    return original_files, remote_files, remote_names  # product_directory, sd,


def json_load(name, path, verbose=False):
    full_name = os.path.join(path, name + ".json")

    with open(full_name, "r") as ifile:
        data = json.load(ifile)
    if verbose:
        _logger.debug("loaded from: %s", full_name)
    return data


def ATL03_download(username, password, dpath, product_directory, sd, file_name):
    """
    inputs:
    username: username for https://urs.earthdata.nasa.gov
    password: your password
    dpath               path where the file should be saved
    product_directory   'ATL03.005' - remote directory on ATLAS
    sd                  '2019.03.01'- subdirectory on ATLAS
    file_name           'ATL03_20190301010737_09560204_005_01.h5' - filename in subdirectory
    """

    HOST = ["https://n5eil01u.ecs.nsidc.org", "ATLAS", product_directory, sd, file_name]
    _logger.debug("download to: %s", dpath + "/" + HOST[-1])
    buffer, error = icesat2_toolkit.utilities.from_nsidc(
        HOST,
        username=username,
        password=password,
        local=dpath + "/" + HOST[-1],
        verbose=True,
    )
    # -- raise exception if download error
    if not buffer:
        raise Exception(error)


def save_pandas_table(table_dict, name, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    with HDFStore(save_path / f"{name}.h5") as store:
        for name, table in table_dict.items():
            store[name] = table


def load_pandas_table_dict(name, save_path):
    warnings.filterwarnings("ignore", category=PerformanceWarning)

    return_dict = dict()
    with HDFStore(save_path + "/" + name + ".h5") as store:
        for k in store.keys():
            return_dict[k[1:]] = store.get(k)

    return return_dict


def get_beam_hdf_store(ATL03_k):
    DD = pd.DataFrame()  # columns = ATL03.keys())
    for ikey in ATL03_k.keys():
        DD[ikey] = ATL03_k[ikey]

    return DD


def get_beam_var_hdf_store(ATL03_k, ikey):
    DD = pd.DataFrame()  # columns = ATL03.keys())
    DD[ikey] = ATL03_k[ikey]
    return DD


def write_track_to_HDF5(data_dict, name, path, verbose=False, mode="w"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    full_name = path / (name + ".h5")
    with h5py.File(str(full_name), mode) as store:
        for k in data_dict.keys():
            store1 = store.create_group(k)
            for kk, I in list(data_dict[k].items()):
                store1[kk] = I

    if verbose:
        _logger.debug("saved at: %s", full_name)


def get_time_for_track(delta_time, atlas_epoch):
    "returns pandas dataframe"

    # Conversion of delta_time to a calendar date
    temp = cGPS.convert_GPS_time(atlas_epoch[0] + delta_time, OFFSET=0.0)

    year = temp["year"][:].astype("int")
    month = temp["month"][:].astype("int")
    day = temp["day"][:].astype("int")
    hour = temp["hour"][:].astype("int")
    second = temp["second"][:].astype("int")

    return pd.DataFrame(
        {"year": year, "month": month, "day": day, "hour": hour, "second": second}
    )


def getATL03_beam(fileT, numpy=False, beam="gt1l", maxElev=1e6):
    """
    returns 'beam' from fileT as pandas table.
    fillT   path of file
    numpy=  (False), or True. if True the method returns a list of numpy instances,
            if False it returns a pandas table
    beam    key of the iceSAT2 beam.
    """
    # Add in a proper description of the function here

    # Open the file
    ATL03 = h5py.File(fileT, "r")
    lons = ATL03[beam + "/heights/lon_ph"][:]
    lats = ATL03[beam + "/heights/lat_ph"][:]

    # Along track distance from equator i think.
    along_track_distance = ATL03[beam + "/heights/dist_ph_along"][:]
    across_track_distance = ATL03[beam + "/heights/dist_ph_across"][:]
    segment_dist_x = ATL03[beam + "/geolocation/segment_dist_x"][:]
    segment_length = ATL03[beam + "/geolocation/segment_length"][:]
    segment_id = ATL03[beam + "/geolocation/segment_id"][:]

    delta_time_geolocation = ATL03[beam + "/geolocation/delta_time"][:]
    reference_photon_index = ATL03[beam + "/geolocation/reference_photon_index"][:]
    ph_index_beg = ATL03[beam + "/geolocation/ph_index_beg"][:]

    ph_id_count = ATL03[beam + "/heights/ph_id_count"][:]
    #  Nathan says it's the number of seconds since the GPS epoch on midnight Jan. 6, 1980
    delta_time = ATL03[beam + "/heights/delta_time"][:]

    # Primary variables of interest

    # Photon height
    heights = ATL03[beam + "/heights/h_ph"][:]

    # Flag for signal confidence
    # column index:  0=Land; 1=Ocean; 2=SeaIce; 3=LandIce; 4=InlandWater
    # values:
    # -- -1: Events not associated with a specific surface type
    # --  0: noise
    # --  1: buffer but algorithm classifies as background
    # --  2: low
    # --  3: medium
    # --  4: high

    heighs_signal_conf_ph = "/heights/signal_conf_ph"
    quality_threshold = 2
    beam_data = ATL03[beam + heighs_signal_conf_ph]

    # ocean points  medium or high quality
    mask_ocean = beam_data[:, 1] > quality_threshold
    # sea ice points medium or high quality
    mask_seaice = beam_data[:, 2] > quality_threshold
    mask_total = mask_seaice | mask_ocean

    if sum(~mask_total) == beam_data[:, 1].size:
        _logger.debug("zero photons, lower photon quality to 2 or higher")
        # lower quality threshold and recompute
        quality_threshold = 1
        mask_ocean = beam_data[:, 1] > quality_threshold
        mask_seaice = beam_data[:, 2] > quality_threshold
        mask_total = mask_seaice | mask_ocean

    signal_confidence = ATL03[beam + heighs_signal_conf_ph][:, 1:3].max(1)

    ATL03.close()

    if numpy == True:
        # list the variables you want to output here..
        return along_track_dist, elev

    else:
        dF = pd.DataFrame(
            {
                "heights": heights,
                "lons": lons,
                "lats": lats,
                "signal_confidence": signal_confidence,
                "mask_seaice": mask_seaice,
                "delta_time": delta_time,
                "along_track_distance": along_track_distance,
                "across_track_distance": across_track_distance,
                "ph_id_count": ph_id_count,
            }
        )

        dF_seg = pd.DataFrame(
            {
                "delta_time": delta_time_geolocation,
                "segment_dist_x": segment_dist_x,
                "segment_length": segment_length,
                "segment_id": segment_id,
                "reference_photon_index": reference_photon_index,
                "ph_index_beg": ph_index_beg,
            }
        )
        # Filter out high elevation values
        _logger.debug("seg_dist shape %s", segment_dist_x.shape)
        _logger.debug("df shape %s", dF.shape)

        dF = dF[mask_total]
        return dF, dF_seg


def getATL03_height_correction(fileT, beam="gt1r"):
    """
    This method returns relevant data for wave estimates from ALT 07 tracks.
    returns: Pandas data frame
    """
    # Add in a proper description of the function here

    # Open the file
    ATL03 = h5py.File(fileT, "r")

    ### bulk positions and statistics
    vars_bulk = [
        "delta_time",  # referenc time since equator crossing
        "dem_h",  # best giod approxiamtion
    ]

    D_bulk = dict()
    for var in vars_bulk:
        D_bulk[var] = ATL03[beam + "/geophys_corr/" + var][:]
    dF_bulk = pd.DataFrame(D_bulk)

    ATL03.close()

    return dF_bulk


def getATL07_beam(fileT, beam="gt1r", maxElev=1e6):
    """
    This method returns relevant data for wave estimates from ALT 07 tracks.
    returns: Pandas data frame
    """
    # Add in a proper description of the function here

    # Open the file
    ATL07 = h5py.File(fileT, "r")

    ### bulk positions and statistics
    vars_bulk = [
        "longitude",
        "latitude",
        "height_segment_id",  # Height segment ID (10 km segments)
        "seg_dist_x",  # Along track distance from the equator crossing to the segment center.
    ]

    D_bulk = dict()
    for var in vars_bulk:
        D_bulk[var] = ATL07[beam + "/sea_ice_segments/" + var]
    dF_bulk = pd.DataFrame(D_bulk)

    #  Nathan says it's the number of seconds since the GPS epoch on midnight Jan. 6, 1980
    delta_time = ATL07[beam + "/sea_ice_segments/delta_time"][:]
    # #Add this value to delta time parameters to compute full gps_seconds
    atlas_epoch = ATL07["/ancillary_data/atlas_sdp_gps_epoch"][:]
    dF_time = get_time_for_track(delta_time, atlas_epoch)
    dF_time["delta_time"] = delta_time
    ### Primary variables of interest

    vars = [
        "across_track_distance",  # Across track distance of photons averaged over the sea ice height segment.
        "height_segment_asr_calc",  # Computed apparent surface reflectance for the sea ice segment.
        "height_segment_confidence",  # # Height segment confidence flag
        "height_segment_fit_quality_flag",  # Flag Values: ['-1', '1', '2', '3', '4', '5']
        # Flag Meanings: ['invalid', 'best', 'high', 'med', 'low', 'poor']
        "height_segment_height",  # Beam segment height
        "height_segment_length_seg",  # Along track length of segment
        "height_segment_ssh_flag",  # Flag for potential leads, 0=sea ice, 1 = sea surface
        "height_segment_surface_error_est",  # Error estimate of the surface height
        "height_segment_type",  # Flag Values: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # Flag Meanings: ['cloud_covered', 'other', 'specular_lead_low_w_bkg', 'specular_lead_low', 'specular_lead_high_w_bkg', 'specular_lead_high', 'dark_lead_smooth_w_bkg', 'dark_lead_smooth'
        "height_segment_w_gaussian",  # Width of Gaussian fit
        "height_segment_quality",  # Height quality flag, 1 for good fit, 0 for bad
    ]

    D_heights = dict()
    for var in vars:
        D_heights[var] = ATL07[beam + "/sea_ice_segments/heights/" + var][:]
    dF_heights = pd.DataFrame(D_heights)

    vars_env = {
        "mss": "geophysical/height_segment_mss",  # Mean sea surface height above WGS-84 reference ellipsoid (range: -105 to 87m), based on the DTU13 model.
        "t2m": "geophysical/height_segment_t2m",  # Temperature at 2m above the displacement height (K)
        "u2m": "geophysical/height_segment_u2m",  # Eastward wind at 2m above the displacement height (m/s-1)
        "v2m": "geophysical/height_segment_v2m",  # Northward wind at 2m above the displacement height (m/s-1)
        "n_photons_actual": "stats/n_photons_actual",  # Number of photons gathered
        "photon_rate": "stats/photon_rate",  # photon_rate
    }

    D_env = dict()
    for var, I in vars_env.items():
        D_env[var] = ATL07[beam + "/sea_ice_segments/" + I][:]
    dF_env = pd.DataFrame(D_env)

    DF = pd.concat(
        {"time": dF_time, "ref": dF_bulk, "heights": dF_heights, "env": dF_env}, axis=1
    )

    ATL07.close()

    # Filter out high elevation values
    DF = DF[(DF["heights"]["height_segment_height"] < maxElev)]
    # Reset row indexing
    DF = DF.reset_index(drop=True)
    return DF


def getATL10_beam(fileT, beam="gt1r", maxElev=1e6):
    """
    This method returns relevant data for wave estimates from ALT 10 tracks.
    returns: Pandas data frames one for sea ice heights and one for leads
    """
    # Add in a proper description of the function here

    # Open the file
    ATL07 = h5py.File(fileT, "r")

    ### bulk positions and statistics

    vars_bulk = [
        "seg_dist_x",
        "latitude",
        "longitude",
        "height_segment_id",
        "beam_fb_confidence",
        "beam_fb_height",
        "beam_fb_quality_flag",
        "beam_fb_sigma",
    ]

    D_bulk = dict()
    for var in vars_bulk:
        D_bulk[var] = ATL07[beam + "/freeboard_beam_segment/beam_freeboard/" + var]
    dF_bulk = pd.DataFrame(D_bulk)

    #  Nathan says it's the number of seconds since the GPS epoch on midnight Jan. 6, 1980
    delta_time = ATL07[beam + "/freeboard_beam_segment/height_segments/delta_time"][:]
    # #Add this value to delta time parameters to compute full gps_seconds
    atlas_epoch = ATL07["/ancillary_data/atlas_sdp_gps_epoch"][:]
    dF_time = get_time_for_track(delta_time, atlas_epoch)
    dF_time["delta_time"] = delta_time

    ### Primary variables of interest
    vars = [
        "height_segment_height",
        "height_segment_length_seg",
        "latitude",
        "longitude",
        "photon_rate",
        "height_segment_type",
        "height_segment_ssh_flag",
        "ice_conc",
    ]

    D_heights = dict()
    for var in vars:
        D_heights[var] = ATL07[beam + "/freeboard_beam_segment/height_segments/" + var][
            :
        ]
    dF_heights = pd.DataFrame(D_heights)

    # Df = pd.concat({k: pd.DataFrame(v).T for k, v in data.items()}, axis=0)
    DF = pd.concat({"time": dF_time, "ref": dF_bulk, "freeboard": dF_heights}, axis=1)

    # Filter out high elevation values
    DF = DF[(DF["freeboard"]["height_segment_height"] < maxElev)]
    # Reset row indexing
    DF = DF.reset_index(drop=True)

    # get leads as well
    vars_leads = [
        "delta_time",
        "latitude",
        "lead_dist_x",
        "lead_height",
        "lead_length",
        "lead_sigma",
        "longitude",
        "ssh_n",
        "ssh_ndx",
    ]

    D_leads = dict()
    for var in vars_leads:
        D_leads[var] = ATL07[beam + "/leads/" + var][:]
    DF_leads = pd.DataFrame(D_leads)

    return DF, DF_leads


def getATL07_height_corrections(fileT, beam="gt1r"):
    """
    This method returns relevant data for wave estimates from ALT 07 tracks.
    returns: Pandas data frame
    """
    # Add in a proper description of the function here

    # Open the file
    ATL07 = h5py.File(fileT, "r")

    ### bulk positions and statistics
    vars_bulk = [
        "longitude",
        "latitude",
        "height_segment_id",  # Height segment ID (10 km segments)
        "seg_dist_x",  # Along track distance from the equator crossing to the segment center.
    ]

    D_bulk = dict()
    for var in vars_bulk:
        D_bulk[var] = ATL07[beam + "/sea_ice_segments/" + var]
    dF_bulk = pd.DataFrame(D_bulk)

    #  Nathan says it's the number of seconds since the GPS epoch on midnight Jan. 6, 1980
    delta_time = ATL07[beam + "/sea_ice_segments/delta_time"][:]
    # #Add this value to delta time parameters to compute full gps_seconds
    atlas_epoch = ATL07["/ancillary_data/atlas_sdp_gps_epoch"][:]
    dF_time = get_time_for_track(delta_time, atlas_epoch)

    ### Primary variables of interest
    vars = [
        "height_segment_dac",
        "height_segment_ib",
        "height_segment_lpe",
        "height_segment_mss",
        "height_segment_ocean",
    ]
    D_heights = dict()
    for var in vars:
        D_heights[var] = ATL07[beam + "/sea_ice_segments/geophysical/" + var][:]
    dF_heights = pd.DataFrame(D_heights)

    DF = pd.concat(
        {
            "time": dF_time,
            "ref": dF_bulk,
            "corrections": dF_heights,
        },
        axis=1,
    )

    ATL07.close()
    # Filter out high elevation values
    # Reset row indexing
    DF = DF.reset_index(drop=True)
    return DF
