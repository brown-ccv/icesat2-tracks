import json
import sys
import numpy as np
import os
import icepyx as ipx
import datetime


def ATL03_to_ATL10(tracks, HH):
    length = len(tracks)
    tracks_ATL10 = np.array(["ATL10-HH_yyyymmddhhmmss_ttttccss_vvv_rr.h5"]*length)
    print("Beginning Conversion...")
    for i in range(length):
        #ATL03 convention: ATL03_yyyymmddhhmmss_ttttccss_vvv_rr.h5
        #ATL10 convention: ATL10-HH_yyyymmddhhmmss_ttttccss_vvv_rr.h5

        year = int(tracks[i][6:10])
        month = int(tracks[i][10:12])
        day = int(tracks[i][12:14])
        RGT = int(tracks[i][21:25])
        cycle = int(tracks[i][25:27])

        short_name = 'ATL03' #converting from ATL03
        spatial_extend = [180, -90, -180, 90] #search globally


        time_absolute = datetime.datetime(year = year, month = month, day = day)
        time_start = time_absolute - datetime.timedelta(days = 1)
        time_stop = time_absolute + datetime.timedelta(days = 1)

        time_start_str = str(time_start.year) + '-' + str(time_start.month) + '-'+ str(time_start.day)
        time_stop_str = str(time_stop.year) + '-' + str(time_stop.month) + '-'+ str(time_stop.day)

        date_range= [time_start_str, time_stop_str]

        region = ipx.Query(short_name, spatial_extend, date_range, cycles = cycle, tracks = RGT)

        avail_granules = region.avail_granules(ids = True)

        for granule_name in avail_granules[0]:
            if(granule_name[27:29] == '01'):
                seg_1 = granule_name

        main_name = seg_1[6:]
        tracks_ATL10[i] = "ATL10-0" + str(HH) + "_" + main_name
    print("Conversion Complete")
    unique_tracks_ATL10 = np.unique(tracks_ATL10)
    return tracks_ATL10.tolist(), unique_tracks_ATL10.tolist()

def main():
    argv = sys.argv[1:]

    if(len(argv) != 2):
        print("Error: The code requires two input, file.json and HH")
    else:
        file_json_path = argv[0]
        hemis = argv[1]
        if hemis == "NH":
            HH = 1
        elif hemis == "SH":
            HH = 2
        #HH = 2

        assert file_json_path[-5:] == ".json", "Error: the input file path must be a json file"


        file_json = open(file_json_path, "r")
        tracks = np.array(json.load(file_json))
        file_json.close()

        tracks_ATL10, unique_tracks_ATL10 = ATL03_to_ATL10(tracks, HH)

        ATL10_file_json_path = file_json_path[0:-5] + "_ATL10" + ".json"

        output = open(ATL10_file_json_path, "w+")
        json.dump(tracks_ATL10, output)
        output.close()

        ATL10_unique_file_json_path = file_json_path[0:-5] + "_ATL10_Unique" + ".json"

        output = open(ATL10_unique_file_json_path, "w+")
        json.dump(unique_tracks_ATL10, output)
        output.close()

if __name__ == "__main__":
    main()
