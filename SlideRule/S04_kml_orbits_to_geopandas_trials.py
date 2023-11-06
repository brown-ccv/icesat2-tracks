# %%
import os, sys

"""
This
"""
exec(open(os.environ["PYTHONSTARTUP"]).read())
exec(open(STARTUP_2021_IceSAT2).read())


from ICEsat2_SI_tools.read_ground_tracks import *
# %%
file  = '/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/originals/IS2_RGTs_cycle20_date_time.zip'
#file='/Users/Shared/Projects/2021_ICESat2_tracks/data/originals/IS2_RGTs_cycle20_date_time.zip'
G = read_ICESat2_groundtrack(file)
G[G['RGT']=='1'].plot()

G.plot(markersize=0.4)

# %%
G1 = G[G['RGT']=='1001']
G1 = G1[G1.geometry.y>0]
G1[0:-1].plot(markersize=1, linestyle='solid')
G1[1::].plot(markersize=1, marker='+')

plt.figure()
Gdist = G1[0:-1].reset_index().geometry.distance(G1[1::].reset_index().geometry)
Gdist.plot.hist()
#G1[1::].reset_index().geometry
print(Gdist.sum())
#Gdist.median()*93
#G2.plot.hist()
plt.show()
# %%
plt.figure()

Gdist.cumsum().plot()
plt.show()
# %%
plt.figure()
plt.plot(G1.geometry.y)
plt.show()

r = float(mconfig['constants']['radius'])
2 *np.pi*r/1e3

420*G1.shape[0]
# %%

file2 = '/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/originals/ICESat2_groundtracks_EasternHem.zip'
G = read_ICESat2_groundtrack(file2)


# %%

file  = '/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/originals/IS2_RGTs_cycle20_date_time.zip'
#file='/Users/Shared/Projects/2021_ICESat2_tracks/data/originals/IS2_RGTs_cycle20_date_time.zip'
G = ICESat2_mission_groundtrack(file)
# G[G['RGT']=='1'].plot()

# %%
# %%

load_path ='/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/originals/'
save_path ='/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/'

#for filename, hemis in zip(['arcticallorbits.zip' , 'antarcticaallorbits.zip'], ['NH', 'SH'] ):
for filename, hemis in zip(['ICESat2_groundtracks_EasternHem_small.zip' , 'ICESat2groundtracksWesternHem.zip'], ['EAST', 'WEST'] ):


    loadfile  = load_path + filename
    save_basename = 'IS2_mission_points_'+hemis+'_RGT_'

    G = ICESat2_mission_points(loadfile)
    G[ (G['GT']=='GT7')].to_file(save_path+save_basename +'all.shp')


# plotting tests
# G7[::500].plot(markersize=0.5)
# G7[::500].geometry.to_crs(epsg=6931).plot(markersize=0.5)


# %%

# save individual tracks
# def save_RGT_data(RGT, G, save_path, base_name):
#     G2 = G[(G['RGT']==RGT)]
#     G2.to_file(save_path+base_name+str(RGT).zfill(4)+'.shp')
#     return RGT

# for RGT in G7.RGT.unique():
#     save_RGT_data(RGT, G7, save_path ,save_basename )

# plotting examples
# G2.geometry.plot(markersize=0.5)
# plt.show()
# G2.geometry.to_crs(epsg=6931).plot(markersize=0.5)
# plt.axis('equal')


# %%

load_path ='/Users/Shared/Projects/2021_ICESat2_tracks/data/groundtracks/originals/ICESat2_groundtracks_EasternHem_small/'

flist = os.listdir(load_path)

#def ICESat2_mission_points2(input_file):
input_file = loadfile
# decompress and parse KMZ file
input_file = pathlib.Path(input_file).expanduser().absolute()
kmzs = zipfile.ZipFile(str(input_file), 'r')
parser = lxml.etree.XMLParser(recover=True, remove_blank_text=True)
# for each kml in the zipfile (per GT)
GTs = []
# %%
# for kmz in kmzs.filelist:
#     kmls = zipfile.ZipFile(kmzs.open(kmz, 'r'))
for kml in flist:
    tree = lxml.etree.parse(kml, parser)
    root = tree.getroot()
    # find documents within kmz file
    for document in root.iterfind('.//kml:Document', root.nsmap):
        # extract laser name, satellite track and coordinates of line strings
        name = document.find('name', root.nsmap).text
        placemarks = document.findall('Placemark/name', root.nsmap)
        coords = document.findall('Placemark/LineString/coordinates', root.nsmap)
        # create list of rows
        rows = []
        x = []; y = []
        # for each set of coordinates
        for i,c in enumerate(coords):
            # create a line string of coordinates
            line = np.array([x.split(',')[:2] for x in c.text.split()], dtype='f8')
            for ln,lt in zip(line[:,0], line[:,1]):
                columns = {}
                columns['Laser'], = re.findall(r'laser(\d+)', name)
                columns['GT'], = re.findall(r'GT\d[LR]?', kmz.filename)
                columns['RGT'] = int(placemarks[i].text)
                rows.append(columns)
                x.append(ln)
                y.append(lt)
        # create geopandas geodataframe for points
        gdf = geopandas.GeoDataFrame(rows,
            geometry=geopandas.points_from_xy(x,y)
        )
        GTs.append(gdf)
# return the concatenated and georefernced geodataframe
G = geopandas.pd.concat(GTs)
G.geometry.crs = {'init': 'epsg:4326'}
#return G

#G = ICESat2_mission_points2(loadfile)
# %%
