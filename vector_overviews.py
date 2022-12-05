import os
os.environ['USE_PYGEOS'] = '0'
import sys
from glob import glob
import warnings

sys.path.insert(1, os.path.abspath(".."))
from pairs_python.core import pairs_quadtree as pqt
from ibmpairs import paw, authentication, client, query, catalog
import vectorstore

import math
import numpy
import pandas
from datetime import datetime, timedelta
import pytz
import geopandas
import shapely
from shapely import wkb, wkt
import pyproj
import fiona
import zipfile
from functools import partial
import json
import geojson
import requests
import hashlib

class overviews(object):
    """
    Overviews Class for populating overviews and pyramids from Geolab vector data.
    A parquet vectorstore (cache) is populated in the process as well
    """
    
    # Default values for variables
    ISO_8601           = '%Y-%m-%dT%H:%M:%SZ'
    AOISTORE_DIRECTORY = '/data/vector/aoipolygons/'
    MIN_OVERVIEW_LEVEL = 7
        
    def __init__(self,
                 pairs_server,
                 base_uri,
                 pairs_credentials,
                 table_id, 
                 pairs_metadata_vector = None,
                 aoistore_directory = None,
                ):
        
        self.pairs_server                 = pairs_server
        self.base_uri                     = base_uri
        self.pairs_credentials            = pairs_credentials
        self.table_id                     = table_id
        
        if pairs_metadata_vector is None:
            pass
        else:
            self.pairs_metadata_vector    = pairs_metadata_vector

        self.aoistore_directory           = self.AOISTORE_DIRECTORY if aoistore_directory is None else aoistore_directory
        if not os.path.exists(self.aoistore_directory): os.makedirs(self.aoistore_directory)
        self.aoi_filepath                 = os.path.join(self.aoistore_directory, 'aoi_polygons.parquet')

    def query_pairs_vector_table(
        self, 
        pairs_box=[-89.9999, -179.9999, 89.9999, 179.9999], 
        starttime="1970-01-01T00:00:00Z", 
        endtime="2100-12-31T23:59:59Z",
    ):
        # Query entire vector table
        self.layerIDs = list(self.pairs_metadata_vector[self.pairs_metadata_vector['vector_table']==str(self.table_id)]['data_layer_id'])

        self.layers = [{"id": layerID} for layerID in self.layerIDs]
        print('layers', self.layers)

        #VECTOR QUERY USING PAW0.1
        # Query definition
        myQueryDef = {
            "layers": self.layers,
            "spatial": {
                #"type" : "poly",
                #"aoi" : "24", 
                "type" : "square",
                "coordinates" : pairs_box, 
            },
            "temporal": {
                "intervals": [
                    {
                        "start": starttime,
                        "end": endtime,
                    }
                ]
            },
            "outputType": "csv"
        }

        # create PAIRS query instance
        self.myQuery = paw.PAIRSQuery(
            myQueryDef,
            pairsHost = 'https://'+self.pairs_server,
            auth = self.pairs_credentials,
            baseURI = self.base_uri,
            inMemory = False,
            overwriteExisting=False, #debug
        )
        # submit and download modified query
        self.myQuery.submit()
        self.myQuery.poll_till_finished(printStatus=True)
        self.myQuery.download()
        #self.myQuery.create_layers()

        unzip_directory = os.path.join('downloads', self.myQuery.zipFilePath.split('.zip')[0])
        print(unzip_directory)
        with zipfile.ZipFile(os.path.join('downloads', self.myQuery.zipFilePath),"r") as zip_ref:
            zip_ref.extractall(unzip_directory)

        # This version works with the "outputType": "csv"
        filepath = os.path.join(unzip_directory, 'Vector_Data_Output.csv')
        self.df = pandas.read_csv(filepath)
        

    # RASTER QUERY USING IBMPAIRS0.2
    """
    # Query definition
    layerID = '82' # 'nir' MODIS TERRA

    delta = 1
    lon = -92
    lat = 42

    lonmin = lon - (delta/2)
    lonmax = lon + (delta/2)
    latmin = lat - (delta/2)
    latmax = lat + (delta/2)

    print([latmin, lonmin, latmax, lonmax])

    dt1 = datetime(2020, 5, 3)
    dt2 = datetime(2020, 5, 12) - timedelta(seconds=1)
    starttime = dt1.strftime(self.ISO_8601)
    endtime = dt2.strftime(self.ISO_8601)

    print(starttime, endtime)

    # Query Json
    myQueryDef = {
        "layers": [
            {"id": layerID},
        ],
        "spatial": {
            "type" : "square",
            "coordinates" : [latmin, lonmin, latmax, lonmax] #[-89, -179, 89, 179], #[38, -76, 43, -72], 
        },
        "temporal": {
            "intervals": [
                {
                    "start": starttime, #"2020-01-05T00:00:00Z",
                    "end": endtime, #"2020-01-12T23:59:59Z"
                }
            ]
        },
        #"outputType": "csv"
    }

    # Define the query, submit and download
    my_query = query.query_from_dict(myQueryDef)
    my_query = my_query.submit_check_status_and_download()

    # Find layer files to load from downloaded zip.
    filepath = os.path.join(my_query.download_folder, my_query.download_file_name)
    files = my_query.list_files()
    print(filepath)
    print('files', files)

    # Plot the raster data
    a = numpy.array(PIL.Image.open(files[0]))
    print('a', a.shape)
    plt.figure(figsize = (20, 12))
    plt.imshow(a, cmap = 'Greens', vmin = 0, vmax = 1)
    plt.title('NIR')
    plt.colorbar()
    plt.show()
    """

    # VECTOR QUERY USING IBMPAIRS0.2
    """
    # Query definition
    layerID = 'P74C377' # Total Population Zip Code.total_estimate --> Geometry references such as 33381:USA-ZIP-WY-Ralston-82440

    myQueryDef = {
        "layers": [
            {"id": layerID},
        ],
        "spatial": {
            "type" :        "square",
            "coordinates" : [-89.9999, -179.9999, 89.9999, 179.9999], #[38, -76, 43, -72], 
        },
        "temporal": {
            "intervals": [
                {
                    "start": "2010-03-01T00:00:00Z",
                    "end": "2030-03-10T23:59:59Z"
                }
            ]
        },
        "outputType": "csv"
    }

    # Define the query, submit and download
    my_query = query.query_from_dict(myQueryDef)
    my_query = my_query.submit_check_status_and_download()

    # Find layer files to load from downloaded zip.
    filepath = os.path.join(my_query.download_folder, my_query.download_file_name)
    files = my_query.list_files()
    print(filepath)
    print('files', files)

    df = pandas.read_csv(files[0])
    try:
        df['geometry'] = df['Region'].apply(wkt.loads)
    except:
        pass
    df.tail(2)
    """

    def get_polygon(self, aoi_id):
        """
        request the PAIRS polygons
        """
        GEOJSON_ENDPOINT = 'https://pairs.res.ibm.com/ws/queryaois/geojson/'
        https_string = GEOJSON_ENDPOINT + str(aoi_id)
        queryResponse = requests.get(https_string, auth = self.pairs_credentials)
        # There is a known issue with the /ws/queryaois/geojson/:id where
        # the response JSON is malformed with double encoding. The following string
        # manipulation of the queryResponse.text a tactical workaround until a patch is deployed
        geo_text = queryResponse.text
        geo_text = geo_text.rstrip()[1:-1].replace('""', '"')

        #poly = shapely.geometry.shape(geojson.loads(queryResponse.json()))
        poly = shapely.geometry.shape(geojson.loads(geo_text))
        return poly

    def request_polygons_add_to_aoi_cache(self, verbose=False):
        """
        Append to aoi cache
        """
        # Drop duplicates and already cached polygons
        try:
            gdf_existing = geopandas.read_parquet(self.aoi_filepath)
        except FileNotFoundError:
            gdf_existing = pandas.DataFrame(columns=['aoi_id'])
        self.df['aoi_id'] = self.df['aoi_id'].astype(int)
        df_aoi_id = self.df[['aoi_id']].drop_duplicates().reset_index(drop=True)
        df_missing = df_aoi_id[~df_aoi_id['aoi_id'].isin(gdf_existing['aoi_id'])].reset_index(drop=True)

        # Request the missing polyogns in chunks and send them to the cache
        ROWS_PER_CHUNK = 1000
        i = 0
        for i in range(math.ceil(len(df_missing)/ROWS_PER_CHUNK)):
            if verbose:
                print('Chunk', i)
            gdf_chunk = df_missing.loc[i*ROWS_PER_CHUNK:(i+1)*ROWS_PER_CHUNK-1, :].reset_index(drop=True)
            gdf_chunk['geometry'] = gdf_chunk['aoi_id'].apply(lambda x: self.get_polygon(x))
            gdf_chunk = geopandas.GeoDataFrame(gdf_chunk, geometry='geometry').set_crs(epsg=4326)
            try:
                # See if there is something already present 
                gdf_existing = geopandas.read_parquet(self.aoi_filepath)
            except FileNotFoundError:
                gdf_merged = gdf_chunk.sort_values(by='aoi_id').reset_index(drop=True)
            else:
                # Merge the two by concatenating and dropping duplicates
                gdf_merged = pandas.concat([gdf_existing, gdf_chunk]).drop_duplicates().sort_values(by='aoi_id').reset_index(drop=True)
            gdf_merged.to_parquet(self.aoi_filepath)

        return df_aoi_id

    def get_dimensions(self, layerID):
        """
        request PAIRS dimensions
        """
        queryResponse = requests.get(f'https://pairs.res.ibm.com/v2/datalayers/{layerID}/datalayer_dimensions', auth = self.pairs_credentials)
        return queryResponse.json()

    def expand_property_string_columns(self):
        # There may be problems with doubled quotation marks, so remove them
        self.gdf['PropertyString'] = self.gdf['PropertyString'].apply(lambda x: x.replace("'", "").replace('"', ''))
        # Expand property string columns
        df1 = self.gdf['PropertyString'].drop_duplicates().reset_index(drop=True)
        df2 = pandas.DataFrame(df1.apply(lambda x: dict(item.split(':', 1) for item in x.split(';') if len(item)>0)).tolist())
        # Making sure we expand the column only once
        if not any([(c in list(self.gdf.columns)) for c in list(df2.columns)]):
            df1 = pandas.concat([df1, df2], axis=1)
            self.gdf = pandas.merge(self.gdf, df1, on='PropertyString')

    def time_to_timestamp(self):
        df_timestamp = self.gdf[['Time']].drop_duplicates().reset_index(drop=True)
        df_timestamp['timestamp'] = df_timestamp['Time'].apply(
            lambda x: datetime.strptime(x, self.ISO_8601).replace(tzinfo=pytz.utc)
        )
        self.gdf = pandas.merge(self.gdf, df_timestamp, on='Time')
        
    def geom_id_hash(self, geoseries):
        """
        Using sha256 hash on wkb representation of geometry to get a nearly unique geometry id
        """
        return geoseries.to_wkb().apply(lambda x: hashlib.sha256(x).hexdigest())

    def vectorquery_to_geodataframe(self, verbose=False):
        self.df['Name'] = self.df['Name'].str.lower()
        # Build a geometry column
        try:
            # Get geometry from wkt string
            self.df['geometry'] = self.df['Region'].apply(wkt.loads)
        except:
            try:
                # Extract an aoi_id
                self.df['aoi_id'] = self.df['Region'].apply(lambda x: x.split(':', 1)[0]).astype(int)
            except:
                try:
                    # Pointdata from lat/lon columns
                    self.gdf = geopandas.GeoDataFrame(self.df, geometry=geopandas.points_from_xy(self.df.Longitude, self.df.Latitude)).set_crs(epsg=4326)
                except:
                    raise
            else:
                # Request polygons from PAIRS if not yet present in local cache
                df_aoi_id = self.request_polygons_add_to_aoi_cache(verbose=verbose)

                # Get the geometries from the local cach 
                gdf_existing = geopandas.read_parquet(self.aoi_filepath)
                gdf_unique = pandas.merge(gdf_existing, df_aoi_id, on='aoi_id')
                self.gdf = pandas.merge(gdf_unique, self.df, on='aoi_id')
        else:
            self.gdf = geopandas.GeoDataFrame(self.df, geometry='geometry').set_crs(epsg=4326)

        # Looks like sometimes we get (partially) empty rows
        self.gdf = self.gdf[~self.gdf['Time'].isnull()].reset_index(drop=True)

        # Time string to datetime object
        self.time_to_timestamp()

        # Dimensions from metadata
        # Get the data_layer_id from the Name
        df_meta = pandas.merge(self.pairs_metadata_vector[['data_layer_id', 'Name']], self.df[['Name']].drop_duplicates(), on='Name')
        # Get the dimensions from the data_layer_id
        df_meta['dimensions'] = df_meta['data_layer_id'].apply(self.get_dimensions)
        df_meta['dimensions_shortName'] = df_meta['dimensions'].apply(lambda x: [d['shortName'] for d in x])
        # Make sure the dimsensions are the same for all columns
        assert(len(df_meta['dimensions_shortName'].drop_duplicates())==1)
        self.dimensions = df_meta['dimensions_shortName'].drop_duplicates().reset_index(drop=True)[0]
        if verbose:
            print('dimensions:       ', self.dimensions)

        if len(self.dimensions)>0:
            # Expand the property string columns (there might be a dimension among them)
            self.expand_property_string_columns()

        # Extract the vector_column_name into its own column
        try:
            self.gdf[['vector_table_name', 'vector_column_name']] = self.gdf['Name'].str.split(':', expand=True)
        except:
            if 'vector_column_name' in self.gdf.columns:
                # looks like we ran this already
                pass
            else:
                raise

        self.vector_table_name = list(self.gdf['vector_table_name'].drop_duplicates())
        assert(len(self.vector_table_name)==1)
        self.vector_table_name = self.vector_table_name[0]
        if verbose:
            print('vector_table_name:', self.vector_table_name)

        # Limit to the columns we need
        index_cols = self.dimensions + ['timestamp', 'geometry']
        if 'aoi_id' in self.gdf.columns:
            index_cols = ['aoi_id'] + index_cols
        self.gdf = self.gdf[index_cols + ['vector_column_name', 'Value']]

        # Unstack the values for the different layers
        self.gdf = self.gdf.set_index(index_cols + ['vector_column_name']).unstack()['Value']
        # Recover the numeric column types
        self.layer_columns = list(self.gdf.columns)
        if verbose:
            print('layer_columns:    ', self.layer_columns)
        for col in self.layer_columns:
            try:
                self.gdf[col] = self.gdf[col].astype(float)
            except:
                pass
        self.gdf = self.gdf.reset_index()
        self.gdf = geopandas.GeoDataFrame(self.gdf)
        self.gdf.columns.name = None

        # Get the vectordata level
        df1 = self.pairs_metadata_vector[
            (self.pairs_metadata_vector['vector_column_name'].isin(self.layer_columns)) &
            (self.pairs_metadata_vector['vector_table_name']==self.vector_table_name)
        ]
        self.vector_level = df1['data_layer_level'].max()
        if verbose:
            print('vector_level:     ', self.vector_level)
            
        # Using sha256 hash on wkb representation of geometry to get a nearly unique geometry id       
        self.gdf['geom_id'] = self.geom_id_hash(self.gdf['geometry'])


    def quadtree(self, poly, level):
        polyQuadTree, _ = pqt.QuadTreePAIRS(poly, max_level=level)
        cells = pqt.QuadTreeCellsPAIRS(polyQuadTree, level)
        return cells

    def polyCells2geodataframe(self, cells, level):
        poly_cells=[]
        res = pqt.getResolution(level)
        for cell in cells:
            south, west = pqt.getLatLon(cell, level)
            north = south + res
            east = west + res
            poly_cells.append(shapely.geometry.box(west, south, east, north))
        gdf_cells = pandas.DataFrame(cells).rename(columns={0: 'spatial_key'})
        gdf_cells['pairs_level'] = level
        gdf_cells['geometry'] = poly_cells
        gdf_cells = geopandas.geodataframe.GeoDataFrame(gdf_cells, geometry='geometry').set_crs(epsg=4326)
        return gdf_cells

    def process_chunk(self, i, chunk, verbose=False):
        loop_start = time.time()
        gdf = prepare_data(chunk)
        partial_upload(gdf, verbose)
        print('Upload time for chunk', i, round(time.time()-loop_start, 3), 'seconds')

    def prepare_data(self, chunk):
        if verbose:
            stopwatch_start = time.time()
        # Read the geometry using wkt and cast to GeoDataFrame
        chunk['geometry'] = chunk['st_astext'].apply(wkt.loads)
        del chunk['st_astext']
        gdf = geopandas.GeoDataFrame(chunk, geometry='geometry', crs='4326')

        # Timestamp in datetime format
        gdf['timestamp'] = gdf['timestamp'].apply(lambda x: datetime.fromtimestamp(x).replace(tzinfo=pytz.utc)) 

        # fid does not seem to be unique, so generate a unique id
        gdf.index.name = 'geom_id'
        gdf = gdf.reset_index()

        # We want to use borocode as categorical variable, so cast to string
        gdf['borocode'] = gdf['borocode'].astype(str)
        if verbose:
            print('Time for preparing data', round(time.time()-stopwatch_start, 3), 'seconds')

        return gdf

    def partial_upload(self, gdf, verbose=False):
        # Create or initialize an existing Geolab vectorstore
        vs = vectorstore.Vectorstore(
            dataset=self.dataset, 
            temporal_keys=self.temporal_keys, 
            temporal_partitions=self.temporal_partitions, 
            overview_level=self.overview_level, 
            spatial_partition_levels=self.spatial_partition_levels, 
            filter_key_levels=self.filter_key_levels,
            dt_col=self.dt_col, 
            geom_col=self.geom_col, 
            id_col=self.id_col, 
            intersection_policy=self.intersection_policy,
        )

        # Upload data
        vs.load_geodataframe(gdf, verbose=verbose)

        # Write the vectorstore to parquet
        vs.to_parquet(append=True, verbose=verbose)

    # Upload to vectorstore by chunk
    """
    chunksize = 10**7
    verbose=True
    with pandas.read_csv(csv_path, chunksize=chunksize) as reader:
        for i, chunk in enumerate(reader):
            process_chunk(i, chunk, verbose)
            #if i>0:
            #    break
    """

    def guess_vectorstore_parameters(self, simplify_tolerance=0.001, verbose=True):
        # Vectorstore parameters
        self.dataset = 'P' + str(self.table_id) + '-' + self.vector_table_name
        self.dt_col = 'timestamp'
        self.geom_col = 'geometry'  # Needs to be called 'geometry'
        self.id_col = 'geom_id'
        if verbose:
            print('dataset:                 ', self.dataset)
            print('dt_col:                  ', self.dt_col)
            print('geom_col:                ', self.geom_col)
            print('id_col:                  ', self.id_col)

        # Intersection policy at overview cell boundaries
        self.intersection_policy = 'both' 
        if 'aoi_id' in self.gdf.columns:
            # If these are PAIRS aoi polygons, we should have the original ones in a cash
            self.intersection_policy = 'cut' #'oroginal' cut' 'both'
        if verbose:
            print('intersection_policy:     ', self.intersection_policy)

        # Temporal Keys (year, month, day, hour) are used to distribute data into separate parquet files.
        # Temporal Partitions (year, month, day, hour) organize parquet files in subfolders. 
        # A temporal key must exist for every temporal partition chosen
        self.temporal_keys = [] 
        self.temporal_partitions = [] 
        if verbose:
            print('temporal_keys:           ', self.temporal_keys)
            print('temporal_partitions:     ', self.temporal_partitions)
            
        # Dissolve all the geometries to estimate what overview_level will fit best
        self.dissolved = self.gdf[['geometry']].drop_duplicates().dissolve()
        if simplify_tolerance is not None:
            self.dissolved = self.dissolved.apply(lambda x: x.simplify(tolerance=simplify_tolerance))

        # Level at which the parquet files are stored. 
        # Choose so that each parquet file will have on order of 10**5 to 10**6 rows (taking into account possible temporal or dimension keys)
        for self.overview_level in range(min(self.MIN_OVERVIEW_LEVEL, self.vector_level), self.vector_level+1):
            self.cells = self.quadtree(self.dissolved['geometry'][0], self.overview_level)
            if len(self.cells)>500:
                break
        if verbose:
            print('overview_level:          ', self.overview_level)
            print('len(cells):              ', len(self.cells))

        # List of spatial partition levels (with levels < overview_level) organize parquet files in subfolders
        self.spatial_partition_levels = [self.overview_level-4, self.overview_level-2] 
        if verbose:
            print('spatial_partition_levels:', self.spatial_partition_levels)

        # Partition order determines the folder structure
        self.partition_order = 'temporal_before_spatial' #'spatial_before_temporal'
        if verbose:
            print('partition_order:         ', self.partition_order)

        # Filter key levels to store higher-resolution keys for every geometry that completely fits into the respective PAIRS cell.
        # levels must be > overview_level
        self.filter_key_levels = [self.overview_level+2, self.overview_level+4, self.overview_level+6, self.overview_level+8] 
        if verbose:
            print('filter_key_levels:       ', self.filter_key_levels)

        # Layer information needed to generate overview statistics. 
        # Each of these layers will get its own overview parquet file(s)
        self.timestamp_layers = ['timestamp']

        dtypes = self.gdf[self.layer_columns].dtypes
        self.numeric_layers = list(dtypes[(dtypes=='float') | (dtypes=='int')].index)
        self.categorical_layers = [l for l in self.layer_columns if l not in self.numeric_layers]

        self.quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        if verbose:
            print('timestamp_layers:        ', self.timestamp_layers)
            print('numeric_layers:          ', self.numeric_layers)
            print('categorical_layers:      ', self.categorical_layers)
            print('quantiles:               ', self.quantiles)

        # Create or initialize an existing Geolab vectorstore
        self.vs = vectorstore.Vectorstore(
            dataset=self.dataset, 
            temporal_keys=self.temporal_keys, 
            temporal_partitions=self.temporal_partitions, 
            overview_level=self.overview_level, 
            spatial_partition_levels=self.spatial_partition_levels, 
            filter_key_levels=self.filter_key_levels,
            dt_col=self.dt_col, 
            geom_col=self.geom_col, 
            id_col=self.id_col, 
            intersection_policy=self.intersection_policy,
        )

    def load_geodataframe(self, verbose=True):
        """
        Simply call the load_geodataframe method of the vectorstore module
        """
        self.vs.load_geodataframe(self.gdf, verbose=verbose)

    def to_parquet(self, append=True, verbose=True):
        """
        Write the vectorstore to parquet
        """
        self.vs.to_parquet(append=append, verbose=verbose)

    def calc_overview_statistics(self) :
        """
        Calculate the overview statistics
        """
        self.vs.calc_overview_statistics(
            numeric_layers=self.numeric_layers,
            timestamp_layers=self.timestamp_layers,
            categorical_layers=self.categorical_layers,
            quantiles=self.quantiles,
        )

    def calc_pyramids(self, verbose=False) :
        """
        Calculate the overview statistics
        """
        self.vs.calc_pyramids(
            verbose=verbose
        )
