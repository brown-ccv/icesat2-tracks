# %%
import os, sys

"""
This file converts the kml files with the orbits to geopandas dataframes and saves them as shapefiles.
"""
exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())


from ICEsat2_SI_tools.read_ground_tracks import *

load_path = "/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/originals/"
# save_path ='/Users/Shared/Projects/2021_ICESat2_tracks/analysis_db/support_files/'
save_path = mconfig["paths"]["analysis"] + "../analysis_db/support_files/"

# for filename, hemis in zip(['ICESat2_groundtracks_EasternHem_small.zip' , 'ICESat2groundtracksWesternHem.zip'], ['EAST', 'WEST'] ):
for filename, hemis in zip(
    ["arcticallorbits.zip", "antarcticaallorbits.zip"], ["NH", "SH"]
):
    loadfile = load_path + filename
    save_basename = "IS2_mission_points_" + hemis + "_RGT_"

    G = ICESat2_mission_points(loadfile)
    G[(G["GT"] == "GT7")].to_file(save_path + save_basename + "all.shp")


# %%
