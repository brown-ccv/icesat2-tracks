import re
import zipfile
import argparse
import pathlib
import lxml.etree
import numpy as np
import geopandas
import osgeo.ogr
import matplotlib.pyplot as plt


# PURPOSE: read ICESat-2 ground tracks for TRACK and BEAM
def read_ICESat2_groundtrack(input_file):
    # decompress and parse KMZ file
    input_file = pathlib.Path(input_file).expanduser().absolute()
    kmls = zipfile.ZipFile(str(input_file), "r")
    parser = lxml.etree.XMLParser(recover=True, remove_blank_text=True)
    # list of geodataframes for all RGTs
    RGTS = []
    # for each kml in the zipfile (per RGT)
    for kml in kmls.filelist:
        tree = lxml.etree.parse(kmls.open(kml, "r"), parser)
        root = tree.getroot()
        # create list of rows
        rows = []
        # find documents within kmz file
        for document in root.iterfind(".//kml:Document//Folder//Placemark", root.nsmap):
            # extract laser name, satellite track and coordinates of line strings
            description = document.find("description", root.nsmap).text
            columns = {}
            (columns["RGT"],) = re.findall(r"RGT\s(\d+)", description)
            # get date and time
            date = re.findall(
                r"\d{2}\-\w{3}\-\d{4}\s\d{2}\:\d{2}\:\d{2}", description
            ).pop()
            columns["date"] = geopandas.pd.to_datetime(date, format="%d-%b-%Y %H:%M:%S")
            (columns["DOY"],) = re.findall(r"DOY\s(\d+)", description)
            (columns["cycle"],) = re.findall(r"Cycle\s(\d+)", description)
            coords = document.findall("Point/coordinates", root.nsmap)
            # for each set of coordinates
            for i, c in enumerate(coords):
                points = np.array(
                    [x.split(",")[:2] for x in c.text.split()], dtype="f8"
                )
                (columns["geometry"],) = geopandas.points_from_xy(
                    points[:, 0], points[:, 1]
                )
            rows.append(columns)
        # create geopandas geodataframe for points
        gdf = geopandas.GeoDataFrame(rows)
        RGTS.append(gdf)
    # return the concatenated geodataframe
    concatenated_df = geopandas.pd.concat(RGTS).set_index("date")
    return concatenated_df


# PURPOSE: read ICESat-2 ground tracks
def ICESat2_mission_groundtrack(input_file):
    # decompress and parse KMZ file
    input_file = pathlib.Path(input_file).expanduser().absolute()
    kmzs = zipfile.ZipFile(str(input_file), "r")
    parser = lxml.etree.XMLParser(recover=True, remove_blank_text=True)
    # for each kml in the zipfile (per GT)
    GTs = []
    for kmz in kmzs.filelist:
        kmls = zipfile.ZipFile(kmzs.open(kmz, "r"))
        for kml in kmls.filelist:
            tree = lxml.etree.parse(kmls.open(kml, "r"), parser)
            root = tree.getroot()
            # find documents within kmz file
            for document in root.iterfind(".//kml:Document", root.nsmap):
                # extract laser name, satellite track and coordinates of line strings
                name = document.find("name", root.nsmap).text
                placemarks = document.findall("Placemark/name", root.nsmap)
                coords = document.findall(
                    "Placemark/LineString/coordinates", root.nsmap
                )
                # create list of rows
                rows = []
                wkt = []
                # for each set of coordinates
                for i, c in enumerate(coords):
                    columns = {}
                    (columns["Laser"],) = re.findall(r"laser(\d+)", name)
                    (columns["GT"],) = re.findall(r"GT\d[LR]?", kmz.filename)
                    columns["RGT"] = int(placemarks[i].text)
                    coords = document.findall("Point/coordinates", root.nsmap)
                    # create LineString object
                    linestring = osgeo.ogr.Geometry(osgeo.ogr.wkbLineString)
                    line = np.array(
                        [x.split(",")[:2] for x in c.text.split()], dtype="f8"
                    )
                    for ln, lt in zip(line[:, 0], line[:, 1]):
                        linestring.AddPoint(ln, lt)
                    # convert to wkt and then add to geometry
                    wkt.append(linestring.ExportToWkt())
                    rows.append(columns)
                # create geopandas geodataframe for points
                gdf = geopandas.GeoDataFrame(
                    rows, geometry=geopandas.GeoSeries.from_wkt(wkt)
                )
                GTs.append(gdf)
    # return the concatenated geodataframe
    concatenated_df = geopandas.pd.concat(GTs)
    return concatenated_df


# PURPOSE: read ICESat-2 mission ground tracks as points
def ICESat2_mission_points(input_file):
    # decompress and parse KMZ file
    input_file = pathlib.Path(input_file).expanduser().absolute()
    kmzs = zipfile.ZipFile(str(input_file), "r")
    parser = lxml.etree.XMLParser(recover=True, remove_blank_text=True)
    # for each kml in the zipfile (per GT)
    GTs = []
    for kmz in kmzs.filelist:
        kmls = zipfile.ZipFile(kmzs.open(kmz, "r"))
        for kml in kmls.filelist:
            tree = lxml.etree.parse(kmls.open(kml, "r"), parser)
            root = tree.getroot()
            # find documents within kmz file
            for document in root.iterfind(".//kml:Document", root.nsmap):
                # extract laser name, satellite track and coordinates of line strings
                name = document.find("name", root.nsmap).text
                placemarks = document.findall("Placemark/name", root.nsmap)
                coords = document.findall(
                    "Placemark/LineString/coordinates", root.nsmap
                )
                # create list of rows
                rows = []
                x = []
                y = []
                # for each set of coordinates
                for i, c in enumerate(coords):
                    # create a line string of coordinates
                    line = np.array(
                        [x.split(",")[:2] for x in c.text.split()], dtype="f8"
                    )
                    for ln, lt in zip(line[:, 0], line[:, 1]):
                        columns = {}
                        (columns["Laser"],) = re.findall(r"laser(\d+)", name)
                        (columns["GT"],) = re.findall(r"GT\d[LR]?", kmz.filename)
                        columns["RGT"] = int(placemarks[i].text)
                        rows.append(columns)
                        x.append(ln)
                        y.append(lt)
                # create geopandas geodataframe for points
                gdf = geopandas.GeoDataFrame(
                    rows, geometry=geopandas.points_from_xy(x, y)
                )
                GTs.append(gdf)
    # return the concatenated and georefernced geodataframe
    G = geopandas.pd.concat(GTs)
    G.geometry.crs = {"init": "epsg:4326"}
    return G
