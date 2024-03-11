import logging

from ipyleaflet import (
    Map,
    basemaps,
    Polygon,
)
import numpy as np
from math import radians, cos, sin, asin, sqrt
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


# height correction tools
def correct_and_remove_height(Gi, height_limit):
    """
    corrects the height and removes the points with height +- height_limit
    """
    h_mean_corrected = Gi["h_mean"] - Gi["dem_h"]
    false_height_mask = abs(h_mean_corrected) > height_limit
    Gi["h_mean"] = h_mean_corrected
    Gi.drop(columns=["dem_h"], inplace=True)
    return Gi[~false_height_mask]


# polygon tools
def make_plot_polygon(poly_test, color="green"):
    """create a plot polygon from the given coordinates"""

    bb = [
        poly_test[0]["lon"],
        poly_test[0]["lat"],
        poly_test[2]["lon"],
        poly_test[2]["lat"],
    ]
    return Polygon(locations=poly_test, color=color)


# # make function to plot with basemaps
def plot_polygon(poly_test, basemap=basemaps.Esri.WorldImagery, zoom=3):
    """plots polygon in the map"""

    # icepx will want a bounding box with LL lon/lat, UR lon/lat

    polygon_plot = make_plot_polygon(poly_test, color="green")

    center = [
        np.mean(list(set([p["lat"] for p in poly_test]))),
        np.mean(list(set([p["lon"] for p in poly_test]))),
    ]
    m = Map(basemap=basemap, center=center, zoom=zoom)
    m.add_layer(polygon_plot)
    return m


# gemetric tools
def haversine(lon1, lat1, lon2, lat2, arc=False):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    arc     True: returns radians [0 , pi]
            False: returns km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    if arc:
        return c
    else:
        return c * r


# point closest to the equator in polygon
def get_min_eq_dist(ppoly):
    """
    returns the minimum distance of the polygon to the equator in meters
    """
    min_eq_dist = list()
    for point in ppoly:
        point["x_atc"] = haversine(
            point["lon"], 0, point["lon"], point["lat"], arc=False
        )
        min_eq_dist.append(point["x_atc"])
    return min(min_eq_dist) * 1e3  # in meters. needed for defining the x-axis


def create_polygons(latR, lonR):

    latR.sort()
    lonR.sort()
    poly_list = [
        {"lat": latR[ii], "lon": lonR[jj]}
        for ii, jj in zip([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    ]
    polygon_shapely = Polygon([(item["lon"], item["lat"]) for item in poly_list])
    return {"list": poly_list, "shapely": polygon_shapely, "lons": lonR, "lats": latR}


# RGT table manipulations:
def find_lowest_point_on_RGT(Gs, RGT):
    ipos = abs(Gs[Gs["RGT"] == RGT].geometry.y).argmin()
    return Gs[Gs["RGT"] == RGT].iloc[ipos]


def find_highest_point_on_RGT(Gs, RGT):
    ipos = abs(Gs[Gs["RGT"] == RGT].geometry.y).argmax()
    return Gs[Gs["RGT"] == RGT].iloc[ipos]


# find_lowest_point_on_RGT(Gs, 2).geometry
def get_RGT_start_points(Gs, RGT="RGT"):

    G_lowest = pd.concat(
        [find_lowest_point_on_RGT(Gs, rgt).T for rgt in Gs[RGT].unique()], axis=1
    ).T.reset_index()
    G_lowest = G_lowest.T.drop("index").T
    return gpd.GeoDataFrame(G_lowest, crs=Gs.crs)


def get_RGT_end_points(Gs, RGT="RGT"):

    G_lowest = pd.concat(
        [find_highest_point_on_RGT(Gs, rgt).T for rgt in Gs[RGT].unique()], axis=1
    ).T.reset_index()
    G_lowest = G_lowest.T.drop("index").T
    return gpd.GeoDataFrame(G_lowest, crs=Gs.crs)


def ascending_test(track_data):
    """
    test if the track is ascending or descending asuming the first and last point are the first and last point of the track
    """
    # get the first and last point
    first_point = abs(track_data.iloc[0].geometry.y)
    last_point = abs(track_data.iloc[-1].geometry.y)
    if first_point < last_point:
        return True
    else:
        return False


def ascending_test_distance(track_data):
    """
    test if the track is ascending or descending based on 'x_atc' column
    """
    # get the first and last point

    first_point = abs(track_data.iloc[track_data["x_atc"].argmin()].geometry.y)
    last_point = abs(track_data.iloc[track_data["x_atc"].argmax()].geometry.y)
    # test if ascending or descending
    if first_point < last_point:
        return True
    else:
        return False


## tools for calculating minimal distance to the equator
def define_reference_distance_with_RGT(
    Gtrack_lowest, rgt, acending, ground_track_length=20160e3
):
    """
    returns reference distance for the given 'rgt' groundtrack based on table of lowest points (Gtrack_lowest).
    This calculated the distance from the equator to the lowest point of the ground track.
    The distance from the equator is either the direct (latitudional) distance (acending) or the distance over the pole (decending).

    inputs:
        Gtrack_lowest: GeoDataFrame with the lowest point of each ground track
        rgt: the ground track number
        ascending: True/False
        ground_track_length: the length of the ground track in meters. Default is 20160e3 (the length of a half orbit) this is not very exact
    """

    start_point = Gtrack_lowest[Gtrack_lowest["RGT"] == rgt]
    if len(start_point) == 0:
        raise ValueError("RGT " + str(rgt) + ", no data in ground track")

    start_point_single = start_point.iloc[0]
    start_point_dist = 1e3 * haversine(
        start_point_single.geometry.x,
        0,
        start_point_single.geometry.x,
        start_point_single.geometry.y,
        arc=False,
    )

    if start_point.geometry.y.iloc[0] < 0:
        # SH
        if acending:
            start_point_dist = start_point_dist + ground_track_length
        else:
            start_point_dist = 2 * ground_track_length - start_point_dist
    else:
        # NH
        if acending:
            start_point_dist = start_point_dist
        else:
            start_point_dist = ground_track_length - start_point_dist

    return start_point_dist, start_point


def plot_reference_point_coordinates(tmp, start_point_dist, start_point):
    """
    plots the reference point and the data
    inputs:
        tmp: GeoDataFrame with the data
        start_point_dist: the distance from the equator to the reference point
        start_point: the reference point (GeoDataFrame)
    """

    rgt = tmp["rgt"].unique()[0]
    spoint_color = "black"
    data_col = "blue"
    accending_color = "black" if ascending_test_distance(tmp) else "red"

    fig, axx = plt.subplots(1, 2, figsize=(8, 4))

    ax = axx[0]
    ax.plot(tmp.geometry.x, tmp.geometry.y, ".", markersize=0.5, c=data_col)

    ax.plot(start_point.geometry.x, start_point.geometry.y, ".", c=spoint_color)
    ax.grid()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("RGT: " + str(rgt), loc="left")

    ax = axx[1]
    ax.plot(tmp["x_atc"], tmp.geometry.y, ".", markersize=0.5, c=data_col, label="data")
    ax.plot(
        start_point_dist,
        start_point.geometry.y,
        ".",
        c=spoint_color,
        label="reference point",
    )

    ax.set_xlabel("Distance from Equator")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "acending: " + str(ascending_test_distance(tmp)),
        loc="right",
        color=accending_color,
    )
    plt.legend()
    ax.grid()
    return fig, axx


def plot_data_in_domain(gdf2, polygon_list):
    """
    makes two panel figure:
    - on the left: plot the lon-lat photon postions for all track & the box
    - on the right: the hoffmoeller diagram with lon-time positions at the most southern points. Colors show ascending or descending tracks.
    inputs:
        gdf: GeoDataFrame with the down
    """

    # make two panel figure, on the left plot the photon postions, on the the hoffmoeller diagram
    fig, axx = plt.subplots(1, 2, figsize=(8, 4))

    plt.suptitle("Data overview and distribution over domain")

    ax = axx[0]
    ax.set_title("ATL Points")
    ax.set_aspect("equal")
    # FIX THIS TO SHOW EACH RGT value by color
    gdf2.plot(
        ax=ax, column="rgt", label="RGT", c=gdf2["rgt"], markersize=0.5, cmap="winter"
    )
    # Prepare coordinate lists for plotting the region of interest polygon
    region_lon = [e["lon"] for e in polygon_list]
    region_lat = [e["lat"] for e in polygon_list]
    ax.plot(region_lon, region_lat, linewidth=1, color="green")

    # labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # add a red dot for each 1st point in the RGT
    for rgt in gdf2["rgt"].unique()[0:20]:
        tmp = gdf2[gdf2["rgt"] == rgt]
        # 1st point
        ax.plot(
            tmp.iloc[0].geometry.x,
            tmp.iloc[0].geometry.y,
            "rv",
            markersize=5,
            label="RGT",
        )
        # last point
        ax.plot(
            tmp.iloc[-1].geometry.x,
            tmp.iloc[-1].geometry.y,
            "ks",
            markersize=5,
            label="RGT",
        )
        # line between first and last
        if ascending_test_distance(tmp):
            axx[1].plot(
                tmp.geometry.x.mean(),
                tmp.index.mean(),
                "ko",
                markersize=5,
                label="ascending",
            )
            ax.plot(
                [tmp.iloc[0].geometry.x, tmp.iloc[-1].geometry.x],
                [tmp.iloc[0].geometry.y, tmp.iloc[-1].geometry.y],
                "-",
                color="black",
                linewidth=1,
            )

        else:
            axx[1].plot(
                tmp.geometry.x.mean(),
                tmp.index.mean(),
                "o",
                color="orange",
                markersize=5,
                zorder=10,
                label="decending",
            )
            ax.plot(
                [tmp.iloc[0].geometry.x, tmp.iloc[-1].geometry.x],
                [tmp.iloc[0].geometry.y, tmp.iloc[-1].geometry.y],
                "-",
                color="orange",
                linewidth=2.5,
            )

    ax = axx[1]

    ax.plot(gdf2.geometry.x, gdf2.index, ".", markersize=0.5)
    # add labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("time")
    return fig, axx


def check_RGT_in_domain(Gtrack_lowest, gdf):
    """
    checks if the RGT in the gdf are found in Gtrack list
    inputs:
        Gtrack_lowest: GeoDataFrame with the ground tracks in the domain
        gdf: GeoDataFrame with the data
    returns:
        result: set of RGT that are in both Gtrack_lowest and gdf
    """

    gdf_list = list(gdf["rgt"].unique())
    result = set(gdf_list).intersection(set(Gtrack_lowest["RGT"]))

    interect_list = list(result)
    interect_list.sort()

    _logger.debug("RGT in domain: %s", len(Gtrack_lowest["RGT"].unique()))
    _logger.debug("RGT with data found: %s", len(gdf_list))
    _logger.debug("RGT in both: %s", len(interect_list))
    if len(interect_list) != len(gdf_list):
        _logger.debug("RGT not in both: %s", list(set(gdf_list) - result))
    return interect_list


def define_x_coordinate_in_polygon(table_data, polygon, round=True):
    """
    returns the 'x' coordinate for a given reference point.
    returns table with 'x' coordinate for a reference point based on the polygon minimum latitude.

    this is a dummy function for now.

    intputs:
        table_data: GeoDataFrame with the data
        polygon: the polygon
        round: True/False. If True the 'x' coordinate is rounded to the nearest 100m
    returns:
        table_data: GeoDataFrame with the data and the 'x' coordinate
    """
    if round:
        min_eq_dist = np.round(get_min_eq_dist(polygon) / 1e2) * 1e2
    else:
        min_eq_dist = np.round(get_min_eq_dist(polygon))

    if ascending_test(table_data):
        table_data["x"] = table_data["x_atc"] - min_eq_dist
    else:
        table_data["x"] = ((np.pi * 6371 * 1e3) - min_eq_dist) - table_data["x_atc"]

    return table_data


def define_x_coordinate_with_RGT(table_data, Gtrack_lowest):
    """
    returns table with 'x' coordinate for a reference point based on the RGT intersection with the polygon taken from 'Gtrack_lowest'
    the reference point is defined as the most equatorward point of the polygon.
    inputs:
        table_data: GeoDataFrame with the data
        Gtrack_lowest: GeoDataFrame with the lowest point of each ground track
    returns:
        table_data: GeoDataFrame with the data and the 'x' coordinate
    """
    if len(table_data["rgt"].unique()) != 1:
        raise ValueError("table_data must contain only one RGT")

    rgt = table_data["rgt"].unique()[0]
    acending = ascending_test_distance(table_data)

    start_point_dist, start_point = define_reference_distance_with_RGT(
        Gtrack_lowest, rgt, acending
    )

    if acending:
        table_data["x"] = table_data["x_atc"] - start_point_dist
    else:
        table_data["x"] = start_point_dist - table_data["x_atc"]

    return table_data


def define_x_coordinate_from_data(table_data):
    """
    define x coordinate from data
    inputs:
        table_data: GeoDataFrame with the data
    returns:
        table_data: GeoDataFrame with the data and x coordinate
    """
    acending = ascending_test_distance(table_data)

    if acending:
        start_point_dist = table_data["x_atc"].min()
        table_data["x"] = table_data["x_atc"] - start_point_dist
    else:
        start_point_dist = table_data["x_atc"].max()
        table_data["x"] = start_point_dist - table_data["x_atc"]
    table_data.sort_values(by="x", inplace=True)
    table_data.reset_index(inplace=True)
    return table_data


def create_ID_name(tt, segment=None):
    """
    creates a string with the ID name for each track.
    inputs:
        tt: 1 row of a GeoDataFrame from Sliderule
    returns:
        ID_name: string with the ID name
    """
    hemis = "NH" if tt.geometry.y > 0 else "SH"
    segment = "00" if segment is None else str(segment)
    return (
        hemis
        + "_"
        + str(tt.name.year)
        + str(tt.name.month).zfill(2)
        + str(tt.name.day).zfill(2)
        + "_"
        + str(tt.rgt).zfill(4)
        + str(tt.cycle).zfill(2)
        + segment
    )
