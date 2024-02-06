import os, sys

# execfile(os.environ['PYTHONSTARTUP'])

"""
This script opens an ATL07 track and tests if there is sufficient data and maybe waves:
1) disects each beam such that they start in the MIZ and end at the pole.
2) checks if the variance decays from the MIZ poleward if data density is high enough.
3) generates plausible ATL03 track_names
4) Saves dummy files tracknames and reporting A01b_success_'track_name'.json file

"""

# exec(open(os.environ['PYTHONSTARTUP']).read())
# exec(open(STARTUP_2019_DP).read())
sys.path
exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())

import datetime
import h5py
from random import sample
import imp
import ICEsat2_SI_tools.convert_GPS_time as cGPS
import ICEsat2_SI_tools.io as io

# Icesat2 Modules
from spectral_estimates import (
    create_chunk_boundaries_unit_lengths,
    create_chunk_boundaries,
)
import spectral_estimates as spec
import m_tools_ph3 as MT
import filter_regrid as regrid

import concurrent.futures as futures


# %%
# username= "mhell@ucsd.edu"
# pword   = "@[49[4tK\-qBWB%5"
# username, password = username,pword
# #build = True


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
    import netrc
    import lxml
    import re
    import posixpath
    import os
    import icesat2_toolkit.utilities

    AUXILIARY = False
    # product='ATL03'
    DIRECTORY = None
    FLATTEN = False
    TIMEOUT = 120
    MODE = 0o775
    file_list = ["ATL07-01_20210301023054_10251001_005_01"]

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

    # rx = re.compile(r'(processed_)?(ATL\d{2})(-\d{2})?_(\d{4})(\d{2})(\d{2})'
    #     r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    # #-- regular expression pattern for finding specific files
    # regex_suffix = '(.*?)$' if AUXILIARY else '(h5)$'
    # remote_regex_pattern = (r'{0}(-\d{{2}})?_(\d{{4}})(\d{{2}})(\d{{2}})'
    #     r'(\d{{2}})(\d{{2}})(\d{{2}})_({1})({2})({3})_({4})_(\d{{2}})(.*?).{5}')

    # -- build list of remote files, remote modification times and local files
    original_files = []
    remote_files = []
    remote_mtimes = []
    local_files = []
    remote_names = []

    for input_file in file_list:
        # print(input_file)
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
        print(colnames)
        # -- print if file was not found
        if not colnames:
            print(colerror)
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


# def nsidc_icesat2_get_associated_file(file_list, product):
#     """
#     THis method returns assocociated files names and paths for files given
#     in file_list for the "product" ICEsat2 product
#     input:
#     file_list:
#     list of the form [ATL03_20190301004639_09560204_005_01, ..]
#     or [processed_ATL03_20190301004639_09560204_005_01, ..]
#     product:
#     ATL03, (or, ATL10, ATL07, not tested)
#
#     """
#     file_list =['ATL03_20210301023054_10251001_005_01']
#     product ='ATL03'
#     import lxml
#     import re
#     import posixpath
#     import os
#     import icesat2_toolkit.utilities
#     AUXILIARY=False
#     #product='ATL03'
#     DIRECTORY= None
#     FLATTEN=False
#     TIMEOUT=120
#     MODE=0o775
#
#     if build:
#         #-- build urllib2 opener with credentials
#         icesat2_toolkit.utilities.build_opener(username, password)
#         #-- check credentials
#         icesat2_toolkit.utilities.check_credentials()
#
#
#     parser = lxml.etree.HTMLParser()
#     #-- remote https server for ICESat-2 Data
#     HOST = 'https://n5eil01u.ecs.nsidc.org'
#     #-- regular expression operator for extracting information from files
#     rx = re.compile(r'(processed_)?(ATL\d{2})(-\d{2})?_(\d{4})(\d{2})(\d{2})'
#         r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})')
#     #-- regular expression pattern for finding specific files
#     regex_suffix = '(.*?)$' if AUXILIARY else '(h5)$'
#     # remote_regex_pattern = (r'{0}(-\d{{2}})?_(\d{{4}})(\d{{2}})(\d{{2}})'
#     #     r'(\d{{2}})(\d{{2}})(\d{{2}})_({1})({2})({3})_({4})_(\d{{2}})(.*?).{5}')
#     remote_regex_pattern = (r'{0}_(\d{{4}})(\d{{2}})(\d{{2}})'
#         r'(\d{{2}})(\d{{2}})(\d{{2}})_({1})({2})({3})_({4})_(\d{{2}}).{5}')
#     # rx = re.compile(r'(processed_)?(ATL\d{2})(-\d{2})?_(\d{4})(\d{2})(\d{2})'
#     #     r'(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
#     # #-- regular expression pattern for finding specific files
#     # regex_suffix = '(.*?)$' if AUXILIARY else '(h5)$'
#     # remote_regex_pattern = (r'{0}(-\d{{2}})?_(\d{{4}})(\d{{2}})(\d{{2}})'
#     #     r'(\d{{2}})(\d{{2}})(\d{{2}})_({1})({2})({3})_({4})_(\d{{2}})(.*?).{5}')
#
#     #-- build list of remote files, remote modification times and local files
#     original_files = []
#     remote_files = []
#     remote_mtimes = []
#     local_files = []
#     remote_names =[]
#
#     for input_file in file_list:
#         #print(input_file)
#         #-- extract parameters from ICESat-2 ATLAS HDF5 file name
#         SUB,PRD,HEM,YY,MM,DD,HH,MN,SS,TRK,CYC,GRN,RL,VRS = \
#             rx.findall(input_file).pop()
#         #-- get directories from remote directory
#         product_directory = '{0}.{1}'.format(product,RL)
#         sd = '{0}.{1}.{2}'.format(YY,MM,DD)
#         PATH = [HOST,'ATLAS',product_directory,sd]
#         #-- local and remote data directories
#         remote_dir=posixpath.join(*PATH)
#         temp=os.path.dirname(input_file) if (DIRECTORY is None) else DIRECTORY
#         local_dir=os.path.expanduser(temp) if FLATTEN else os.path.join(temp,sd)
#         #-- create output directory if not currently existing
#         # if not os.access(local_dir, os.F_OK):
#         #     os.makedirs(local_dir, MODE)
#         #-- compile regular expression operator for file parameters
#         args = (product,TRK,CYC,GRN,RL,regex_suffix)
#         R1 = re.compile(remote_regex_pattern.format(*args), re.VERBOSE)
#         #-- find associated ICESat-2 data file
#         #-- find matching files (for granule, release, version, track)
#         sys.path
#
#         imp.reload(icesat2_toolkit.utilities)
#         colnames,collastmod,colerror=icesat2_toolkit.utilities.nsidc_list(PATH,
#             build=False,
#             timeout=TIMEOUT,
#             parser=parser,
#             pattern=R1,
#             sort=True)
#         print(colnames)
#         #-- print if file was not found
#         if not colnames:
#             print(colerror)
#             continue
#         #-- add to lists
#         for colname,remote_mtime in zip(colnames,collastmod):
#             #-- save original file to list (expands if getting auxiliary files)
#             original_files.append(input_file)
#             #-- remote and local versions of the file
#             remote_files.append(posixpath.join(remote_dir,colname))
#             local_files.append(os.path.join(local_dir,colname))
#             remote_mtimes.append(remote_mtime)
#             remote_names.append(colname)
#
#     return original_files, remote_files, remote_names #product_directory, sd,

org_files, remote_files, remote_names = nsidc_icesat2_get_associated_file(
    ["ATL07-01_20210301023054_10251001_005_01"], "ATL03"
)
