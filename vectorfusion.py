import os
os.environ['USE_PYGEOS'] = '0'
import sys
import time

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
import zipfile
import json
import geojson
import requests
import hashlib

class Vectorfusion():
    """
    Vectorfusion class for populating a Vectorstore from Geolab vector data.
    """
    
    # Default values for variables
    ISO_8601           = '%Y-%m-%dT%H:%M:%SZ'
    AOISTORE_DIRECTORY = '/data/vector/aoipolygons/'
    MIN_SPATIAL_LEVEL = 7
    MAX_SPATIAL_LEVEL = 15
    COMPLETE_WORLD     = shapely.geometry.box(-180, -90, 180, 90)
    
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
        """
        Query entire vector table
        Use only with small or medium-sized Vector tables
        For large tables, use a full-table dump instead.
        """
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
        
    def get_polygon(self, aoi_id):
        """
        Request the PAIRS polygons
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

    def request_dimensions(self, layerID):
        """
        request PAIRS dimensions
        """
        queryResponse = requests.get(
            f'https://pairs.res.ibm.com/v2/datalayers/{layerID}/datalayer_dimensions', 
            auth = self.pairs_credentials
        )
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
        """
        Cast string to datetime
        """
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

    def vectorquery_to_geodataframe(self, temporal_keys=[], verbose=False):
        """
        Create the GeoDataFrame
        """
        self.temporal_keys = temporal_keys
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
                    self.gdf = geopandas.GeoDataFrame(
                        self.df, 
                        geometry=geopandas.points_from_xy(self.df.Longitude, self.df.Latitude)
                    ).set_crs(epsg=4326)
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

        before = len(self.gdf)
        # Looks like sometimes we get (partially) empty rows
        self.gdf = self.gdf[~self.gdf['Time'].isnull()].reset_index(drop=True)
        after = len(self.gdf)
        if (before!=after) and verbose:
            print(f'Warning: droppd {after-before} rows')

        # Time string to datetime object
        self.time_to_timestamp()
        
        # Dimensions from metadata
        """
        # Get the dimensions through an API call
        df_meta = pandas.merge(self.pairs_metadata_vector[['data_layer_id', 'Name']], self.df[['Name']].drop_duplicates(), on='Name')
        # Get the dimensions from the data_layer_id
        df_meta['dimensions'] = df_meta['data_layer_id'].apply(self.request_dimensions)
        df_meta['dimensions_shortName'] = df_meta['dimensions'].apply(lambda x: [d['shortName'] for d in x])
        # Make sure the dimsensions are the same for all columns
        df_dim_unique = df_meta['dimensions_shortName'].drop_duplicates()
        """
        # Get the dimensions from the pairs_metadata_vector file
        df_meta = pandas.merge(
            self.pairs_metadata_vector[['data_layer_id', 'Name', 'dimensions']], 
            self.df[['Name']].drop_duplicates(), 
            on='Name'
        )
        # Make sure the dimsensions are the same for all columns
        df_dim_unique = df_meta['dimensions'].dropna().drop_duplicates()
        if len(df_dim_unique)==0:
            self.dimension_keys=[]
        elif len(df_dim_unique)==1:
            self.dimension_keys = df_dim_unique.reset_index(drop=True)[0]
        else:
            raise
        if verbose:
            print('dimension_keys:   ', self.dimension_keys)

        if len(self.dimension_keys)>0:
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
        index_cols = self.dimension_keys + ['timestamp', 'geometry']
        print('index_cols', index_cols)
        if 'aoi_id' in self.gdf.columns:
            # Don't use the geometry column itself as an index
            index_cols = ['aoi_id'] + [i for i in index_cols if i!='geometry']
            # Remember the aoi_id - geometry relation
            gdf_unique = self.gdf[['aoi_id', 'geometry']].drop_duplicates(subset='aoi_id').reset_index(drop=True)
        else:
            gdf_unique = self.gdf[['geometry']].drop_duplicates().reset_index(drop=True)

        self.gdf = self.gdf[index_cols + ['vector_column_name', 'Value']]

        # Unstack the values for the different layers
        try:
            self.gdf = self.gdf.set_index(index_cols + ['vector_column_name']).unstack()['Value']
        except Exception as e:
            print(e)
            print('WARNING: FAILED TO UNSTACK THE DATAFRAME.')
            
            # We have several options to deal with duplicates here. Either:
            print('DROPPING DUPLICATE INDEX RETAINING FIRST VALUE')
            self.gdf = self.gdf.drop_duplicates(subset=index_cols+['vector_column_name'], keep='first').reset_index(drop=True)
            
            # Or:
            #print('DROPPING DUPLICATE INDEX ROWS COMPLETELY')
            #self.gdf = self.gdf.drop_duplicates(subset=index_cols+['vector_column_name'], keep=False).reset_index(drop=True)
            
            # Or we keep "first" duplicates where the value column also matches, 
            # and then do the above comparison dropping duplicate index rows completely
            #print('DROPPING DUPLICATE ROWS RETAINING FIRST and then DROPPING DUPLICATE INDEX ROWS COMPLETELY')
            #self.gdf = self.gdf.drop_duplicates(keep='first').reset_index(drop=True)
            #self.gdf = self.gdf.drop_duplicates(subset=index_cols+['vector_column_name'], keep=False).reset_index(drop=True)

            # Try unstacking again. This time it should work
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
        #if numpy.isnan(self.vector_level):
        if numpy.isnan(self.vector_level) or self.vector_level>self.MAX_SPATIAL_LEVEL:
            self.vector_level = self.MAX_SPATIAL_LEVEL
        if verbose:
            print('vector_level:     ', self.vector_level)

        # Make sure the geometries are valid
        gdf_unique['geometry'] = gdf_unique['geometry'].apply(shapely.validation.make_valid)
        # Using sha256 hash on wkb representation of geometry to get a nearly unique geometry id       
        gdf_unique['geom_id'] = self.geom_id_hash(gdf_unique['geometry'])
        
        # Merge-in the geom_id
        if 'aoi_id' in self.gdf.columns:
            self.gdf = pandas.merge(gdf_unique, self.gdf, on='aoi_id')
        else:
            self.gdf = pandas.merge(gdf_unique, self.gdf, on='geometry')

    def quadtree(self, poly, level):
        polyQuadTree, _ = pqt.QuadTreePAIRS(poly, max_level=level)
        cells = pqt.QuadTreeCellsPAIRS(polyQuadTree, level)
        return cells
    
    def quadtree2(self, poly, level):
        """
        version of quadtree that works better for complicated polygons
        """
        # Get all the cells within the total bounds
        bounds_cells = self.quadtree(shapely.box(*poly.bounds), level)
        gdf_bounds_cells = self.polyCells2geodataframe(bounds_cells, level)
        # Filter the cells that overlap the polygon
        mask = gdf_bounds_cells.intersects(poly)
        df_masked = gdf_bounds_cells[mask].reset_index(drop=True)
        cells = df_masked['spatial_key'].to_list()
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
        partial_ingest(gdf, verbose)
        print('Ingest time for chunk', i, round(time.time()-loop_start, 3), 'seconds')

    def prepare_data(self, chunk):
        if verbose:
            stopwatch_start = time.time()
        # Read the geometry using wkt and cast to GeoDataFrame
        chunk['geometry'] = chunk['st_astext'].apply(wkt.loads)
        del chunk['st_astext']
        gdf = geopandas.GeoDataFrame(chunk, geometry='geometry', crs='4326')

        # DEBUG: Timestamp in datetime format
        gdf['timestamp'] = gdf['timestamp'].apply(lambda x: datetime.fromtimestamp(x).replace(tzinfo=pytz.utc)) 
        #gdf['timestamp'] = gdf['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=pytz.utc)) 

        # fid does not seem to be unique, so generate a unique id
        gdf.index.name = 'geom_id'
        gdf = gdf.reset_index()

        # We want to use borocode as categorical variable, so cast to string
        gdf['borocode'] = gdf['borocode'].astype(str)
        if verbose:
            print('Time for preparing data', round(time.time()-stopwatch_start, 3), 'seconds')

        return gdf

    def partial_ingest(self, gdf, verbose=False):
        # Create or initialize an existing Geolab vectorstore
        vs = vectorstore.Vectorstore(
            dataset=self.dataset, 
            temporal_keys=self.temporal_keys, 
            temporal_partitions=self.temporal_partitions, 
            dimension_keys=self.dimension_keys,
            spatial_level=self.spatial_level, 
            spatial_partition_levels=self.spatial_partition_levels, 
            filter_key_levels=self.filter_key_levels,
            dt_col=self.dt_col, 
            geom_col=self.geom_col, 
            id_col=self.id_col, 
            intersection_policy=self.intersection_policy,
        )

        # Ingest data
        vs.ingest_geodataframe(gdf, verbose=verbose)

        # Write the vectorstore to parquet
        vs.to_parquet(append=True, verbose=verbose)

    # Ingest into vectorstore by chunk
    """
    chunksize = 10**7
    verbose=True
    with pandas.read_csv(csv_path, chunksize=chunksize) as reader:
        for i, chunk in enumerate(reader):
            process_chunk(i, chunk, verbose)
            #if i>0:
            #    break
    """

    def guess_vectorstore_parameters(self, simplify_tolerance=0.0005, verbose=True):
        """
        Calculate some reasononable parameters for vectorstore
        """
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

        # Intersection policy at spatial cell boundaries
        self.intersection_policy = 'both' 
        if 'aoi_id' in self.gdf.columns:
            # If these are PAIRS aoi polygons, we should have the original ones in a cash
            self.intersection_policy = 'cut' #'oroginal' cut' 'both'
        if verbose:
            print('intersection_policy:     ', self.intersection_policy)

        # Temporal Keys (year, month, day, hour) are used to distribute data into separate parquet files.
        # Temporal Partitions (year, month, day, hour) organize parquet files in subfolders. 
        # A temporal key must exist for every temporal partition chosen
        self.temporal_partitions = [] 
        if verbose:
            print('temporal_keys:           ', self.temporal_keys)
            print('temporal_partitions:     ', self.temporal_partitions)
            stopwatch_start = time.time()
            
        # Dissolve all the geometries to estimate what spatial_level will fit best
        gdf_unique = self.gdf[[self.id_col, self.geom_col]].drop_duplicates(
            subset=self.id_col).reset_index(drop=True)[[self.geom_col]]
        if simplify_tolerance is not None:
            print('debug buffering')
            gdf_unique[self.geom_col] = gdf_unique[self.geom_col].apply(lambda x: x.buffer(simplify_tolerance))
        try:
            self.gdf_dissolved = gdf_unique.dissolve()
        except:
            gdf_unique[self.geom_col] = gdf_unique[self.geom_col].apply(shapely.validation.make_valid)
            self.gdf_dissolved = gdf_unique.dissolve()
        if simplify_tolerance is not None:
            print('debug simplifying')
            self.gdf_dissolved[self.geom_col] = self.gdf_dissolved[self.geom_col].apply(
                lambda x: x.simplify(tolerance=simplify_tolerance)
            )
            self.gdf_dissolved[self.geom_col] = self.gdf_dissolved[self.geom_col].intersection(self.COMPLETE_WORLD)
        if verbose:
            print('Time for "dissolve" in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
            
        # Level at which the parquet files are stored. 
        # Choose so that each parquet file will have on order of 10**5 to 10**6 rows 
        # (taking into account possible temporal or dimension keys)
        num_geoms = len(self.gdf_dissolved.loc[0, 'geometry'].geoms)
        for self.spatial_level in range(min(self.MIN_SPATIAL_LEVEL, self.vector_level), self.vector_level+1):
            if num_geoms<1000:
                # Use sparse version of quadtree
                self.cells = self.quadtree(self.gdf_dissolved.loc[0, 'geometry'], self.spatial_level)
            else:
                # Use simplified version of quadtree
                self.cells = self.quadtree2(self.gdf_dissolved.loc[0, 'geometry'], self.spatial_level)
            if len(self.cells)>500:
                break

        """
        # Test the speed of two algorithms at a very low level in order to decide which one to use for the higher level(s)
        test_level = 1
        t0 = time.time()
        _ = self.quadtree(self.gdf_dissolved.loc[0, 'geometry'], test_level)
        quadtree_time = time.time()-t0
        t0 = time.time()
        _ = self.quadtree2(self.gdf_dissolved.loc[0, 'geometry'], test_level)
        quadtree2_time = time.time()-t0
        
        for self.spatial_level in range(min(self.MIN_SPATIAL_LEVEL, self.vector_level), self.vector_level+1):
            if quadtree_time<quadtree2_time:
                # Use sparse version of quadtree
                self.cells = self.quadtree(self.gdf_dissolved.loc[0, 'geometry'], self.spatial_level)
            else:
                # Use simplified version of quadtree
                self.cells = self.quadtree2(self.gdf_dissolved.loc[0, 'geometry'], self.spatial_level)
            if len(self.cells)>500:
                break
        """
        """
        bounds_area = shapely.box(*self.gdf_dissolved.loc[0, 'geometry'].bounds).area
        geom_area = self.gdf_dissolved.loc[0, 'geometry'].area
        for self.spatial_level in range(min(self.MIN_SPATIAL_LEVEL, self.vector_level), self.vector_level+1):
            if geom_area<bounds_area/10:
                # Use sparse version of quadtree
                self.cells = self.quadtree(self.gdf_dissolved.loc[0, 'geometry'], self.spatial_level)
            else:
                # Use simplified version of quadtree
                self.cells = self.quadtree2(self.gdf_dissolved.loc[0, 'geometry'], self.spatial_level)
            if len(self.cells)>500:
                break
        """
        
        if verbose:
            print('spatial_level:           ', self.spatial_level)
            print('len(cells):              ', len(self.cells))
            print('Time for "cells" in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # List of spatial partition levels (with levels < spatial_level) organize parquet files in subfolders
        self.spatial_partition_levels = [self.spatial_level-4, self.spatial_level-2] 
        if verbose:
            print('spatial_partition_levels:', self.spatial_partition_levels)

        # Partition order determines the folder structure
        self.partition_order = 'temporal_before_spatial' #'spatial_before_temporal'
        if verbose:
            print('partition_order:         ', self.partition_order)

        # Filter key levels to store higher-resolution keys for every geometry that completely fits into the respective PAIRS cell.
        # levels must be > spatial_level
        self.filter_key_levels = [self.spatial_level+2, self.spatial_level+4, self.spatial_level+6, self.spatial_level+8] 
        if verbose:
            print('filter_key_levels:       ', self.filter_key_levels)

        # Layer information needed to generate overview statistics. 
        # Each of these layers will get its own parquet file(s)
        self.timestamp_layers = ['timestamp']

        dtypes = self.gdf[self.layer_columns].dtypes
        self.numeric_layers = list(dtypes[(dtypes=='float') | (dtypes=='int')].index)
        self.categorical_layers = [l for l in self.layer_columns if l not in self.numeric_layers]

        self.quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        self.first = True
        
        # Count the timestamps. Individual records for each timestamp if there are fewer than 100 timestamps.
        timestamp_count = len(self.gdf[self.timestamp_layers].drop_duplicates())
        if timestamp_count<100:
            self.timestamp_aggregation = False
        else:
            self.timestamp_aggregation = True

        if verbose:
            print('timestamp_layers:        ', self.timestamp_layers)
            print('numeric_layers:          ', self.numeric_layers)
            print('categorical_layers:      ', self.categorical_layers)
            print('quantiles:               ', self.quantiles)
            print('first:                   ', self.first)
            print('timestamp_aggregation:   ', self.timestamp_aggregation)

    def ingest_geodataframe(self, verbose=True):
        """
        Call the ingest_geodataframe method of the vectorstore module
        """
        # Create or initialize an existing Geolab vectorstore
        self.vs = vectorstore.Vectorstore(
            dataset=self.dataset, 
            temporal_keys=self.temporal_keys, 
            temporal_partitions=self.temporal_partitions, 
            dimension_keys=self.dimension_keys,
            spatial_level=self.spatial_level, 
            spatial_partition_levels=self.spatial_partition_levels, 
            filter_key_levels=self.filter_key_levels,
            dt_col=self.dt_col, 
            geom_col=self.geom_col, 
            id_col=self.id_col, 
            intersection_policy=self.intersection_policy,
        )
        if hasattr(self, 'dissolved'):
            self.vs.ingest_geodataframe(self.gdf, dissolved=self.gdf_dissolved, verbose=verbose)
        else:
            self.vs.ingest_geodataframe(self.gdf, verbose=verbose)

    def to_parquet(self, append=True, geometry_area=False, geometry_length=False, verbose=True):
        """
        Write the vectorstore to parquet
        """
        self.vs.to_parquet(append=append, geometry_area=geometry_area, geometry_length=geometry_length, verbose=verbose)

    def write_vectorstore_settings(self):
        """
        Write the vectorstore settings to json
        """
        # Update overview-centric attributes
        self.vs.numeric_layers = self.numeric_layers
        self.vs.timestamp_layers = self.timestamp_layers
        self.vs.categorical_layers = self.categorical_layers
        self.vs.quantiles = self.quantiles
        self.vs.first = self.first
        self.vs.timestamp_aggregation = self.timestamp_aggregation
        
        self.vs.write_vectorstore_settings()

        
def vectorfusion_worker(
    table_id,
    PAIRS_SERVER,
    BASE_URI,
    PAIRS_CREDENTIALS,
    pairs_metadata_vector,
    temporal_keys = [], #['year']
    simplify_tolerance = None, #0 #0.0005 #None
    geometry_area = True,
    geometry_length = True,
    pairs_box = [-89.9999, -179.9999, 89.9999, 179.9999],
    starttime = "1970-01-01T00:00:00Z",
    endtime = "2100-12-31T23:59:59Z",
    verbose = True,
):
    # Initialize
    fused = Vectorfusion(
        pairs_server=PAIRS_SERVER, 
        base_uri=BASE_URI,
        pairs_credentials=PAIRS_CREDENTIALS,
        table_id=table_id, 
        pairs_metadata_vector=pairs_metadata_vector,
    )

    if verbose:
        stopwatch_start = time.time()

    # Query PAIRS
    fused.query_pairs_vector_table(pairs_box)

    if verbose:
        print('Time for "query_pairs_vector_table" in seconds', round(time.time()-stopwatch_start, 3))
        stopwatch_start = time.time()

    #fused.df = fused.df.loc[::100, :].reset_index(drop=True)

    # Vectorquery to GeoDataframe
    fused.vectorquery_to_geodataframe(verbose=verbose, temporal_keys=temporal_keys)

    if verbose:
        print('Time for "vectorquery_to_geodataframe" in seconds', round(time.time()-stopwatch_start, 3))
        stopwatch_start = time.time()

    # Use DataFrame statistics to guess good vectorstore parameters
    fused.guess_vectorstore_parameters(simplify_tolerance=simplify_tolerance, verbose=verbose)

    if verbose:
        print('Time for "guess_vectorstore_parameters" in seconds', round(time.time()-stopwatch_start, 3))
        stopwatch_start = time.time()

    # Complete ingest of gdf into vectorstore in one go
    fused.ingest_geodataframe(verbose=verbose)

    if verbose:
        print('Time for "ingest_geodataframe" in seconds', round(time.time()-stopwatch_start, 3))
        stopwatch_start = time.time()

    # Write the vectorstore to parquet
    fused.to_parquet(append=False, geometry_area=geometry_area, geometry_length=geometry_length, verbose=verbose)

    # Update the settings in the json file (existing will be overwritten)
    fused.write_vectorstore_settings()

    if verbose:
        print('Time for "to_parquet" in seconds', round(time.time()-stopwatch_start, 3))
        stopwatch_start = time.time()
        
    return fused
