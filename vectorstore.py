import os
import sys
import warnings
from glob import glob

sys.path.insert(1, os.path.abspath(".."))
from pairs_python.core import pairs_quadtree as pqt

import numpy
import pandas
from datetime import datetime, timedelta
import pytz
import geopandas
import shapely
import pyproj
from functools import partial
import json

class Vectorstore(object):
    """
    Vectorstore Class for handling vector data in Geolab-compatible parquet files
    """
    
    # Default values for variables
    VECTORSTORE_DIRECTORY     = '/data/vector/vectorstore/'
    OVERVIEWSTORE_DIRECTORY   = '/data/vector/overviews/'

    TEMPORAL_KEYS             = ['year']
    TEMPORAL_PARTITIONS       = []

    # Overview resolution layer for the Parquet file
    # (PAIRS level 6 ~ 1000km, level 9 ~ 100km, level 13 ~ 10km, level 16 ~ 1km, level 23 ~ 10m)
    OVERVIEW_LEVEL            = 8
    SPATIAL_PARTITION_IDENTIFIER = 'partition_level'
    SPATIAL_PARTITION_LEVELS  = [6]
    
    # Order of the partitions in the stored directory structure
    PARTITION_ORDER           = 'temporal_before_spatial' #'spatial_before_temporal'

    # Filter-key columns at various resolution levels (above the overview cell level)
    FILTER_KEY_LEVELS         = []
    
    # Policy how to deal with polygons that intersect overview cells ("cut", "original", "both")
    INTERSECTION_POLICY       = 'cut' # 'original', 'both'
    INTERSECTION_FLAG_COL     = 'intersection_flag'
    
    # GeoDataFrame colum conventions (used when loading data from gdf or vectorstore)
    DT_COL                    = 'timestamp'
    GEOM_COL                  = 'geometry'  # Geopandas relies on this column being named 'geometry', so enforce this for all tables
    ID_COL                    = 'geometry_id'
    OVERVIEW_LEVEL_COL        = 'overview_level'
    SPATIAL_KEY_COL           = 'spatial_key'
    OVERVIEW_KEY_COL          = 'overview_key'
    
    # Overview statistics
    NUMERIC_LAYERS            = []
    TIMESTAMP_LAYERS          = []
    CATEGORICAL_LAYERS        = []
    PERCENTILES               = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
    
    # Pyramids
    PYRAMID_LEVELS            = []  # Initialize as empty list. Will be populated when pyramids are generated
    
    # Min and max datetime conventions (used when querying data)
    MIN_DT = datetime(1, 1, 1).replace(tzinfo=pytz.utc)
    MAX_DT = datetime(9999, 12, 31).replace(tzinfo=pytz.utc)
    
    # Other query conventions
    COMPLETE_WORLD            = shapely.geometry.box(-180, -90, 180, 90)
    QUERY_INTERSECTION_POLICY = 'cut' #'original'
    
    def __init__(self,
                 dataset,
                 vectorstore_directory = None,
                 overviewstore_directory = None,
                 temporal_keys = None,
                 temporal_partitions = None,
                 overview_level = None,
                 spatial_partition_identifier = None,
                 spatial_partition_levels = None,
                 partition_order = None,
                 filter_key_levels = None,
                 intersection_policy = None,
                 intersection_flag_col = None,
                 dt_col = None,
                 geom_col = None, 
                 id_col = None,
                 overview_level_col = None, 
                 spatial_key_col = None, 
                 overview_key_col = None,
                 numeric_layers = None,
                 timestamp_layers = None,
                 categorical_layers = None,
                 percentiles = None,
                ):
        
        # Dataset name
        self.dataset                      = dataset
        
        # Vectorstore base directory
        self.vectorstore_directory        = self.VECTORSTORE_DIRECTORY if vectorstore_directory is None else vectorstore_directory
        # Overview base directory
        self.overviewstore_directory      = self.OVERVIEWSTORE_DIRECTORY if overviewstore_directory is None else overviewstore_directory
        
        # Dataset directory is subfolder in vectorstore_directory and/or overviewstore_directory
        self.dataset_directory            = os.path.join(self.vectorstore_directory, self.dataset)
        if not os.path.exists(self.dataset_directory): os.makedirs(self.dataset_directory)
        self.overview_directory            = os.path.join(self.overviewstore_directory, self.dataset)
        if not os.path.exists(self.overview_directory): os.makedirs(self.overview_directory)

        # Temporal keys and partitions
        self.temporal_keys                = self.TEMPORAL_KEYS if temporal_keys is None else temporal_keys
        self.temporal_partitions          = self.TEMPORAL_PARTITIONS if temporal_partitions is None else temporal_partitions
        # Make sure that the temporal partiting keys are present in temporal_keys and used to define parquet file ranges
        missing_keys                      = [k for k in self.temporal_partitions if k not in self.temporal_keys]
        assert(len(missing_keys)==0)
        
        # Spatial keys and partitions
        self.overview_level               = self.OVERVIEW_LEVEL if overview_level is None else overview_level
        self.spatial_partition_identifier = self.SPATIAL_PARTITION_IDENTIFIER if spatial_partition_identifier is None else spatial_partition_identifier
        self.spatial_partition_levels     = self.SPATIAL_PARTITION_LEVELS if spatial_partition_levels is None else spatial_partition_levels
        self.partition_order              = self.PARTITION_ORDER if partition_order is None else partition_order
        assert(self.partition_order in ['temporal_before_spatial', 'spatial_before_temporal'])
        # Make sure that it's sorted and the highest level is <= overview level 
        self.spatial_partition_levels     = sorted(self.spatial_partition_levels)
        if len(self.spatial_partition_levels)>0:
            assert(max(self.spatial_partition_levels)<=self.overview_level)
        self.spatial_partitions           = [self.spatial_partition_identifier + str(l) for l in self.spatial_partition_levels]
        
        # Filter-key columns at various resolution levels (above the overview cell level)
        self.filter_key_levels            = self.FILTER_KEY_LEVELS if filter_key_levels is None else filter_key_levels 
        self.filter_key_levels            = sorted(self.filter_key_levels)

        # Policy how to deal with polygons that intersect overview cells ("cut", "original", "both")
        self.intersection_policy = self.INTERSECTION_POLICY if intersection_policy is None else intersection_policy
        assert(self.intersection_policy in ['cut', 'original', 'both'])
        self.intersection_flag_col = self.INTERSECTION_FLAG_COL if intersection_flag_col is None else intersection_flag_col
        
        # Table specific column information
        self.dt_col = self.DT_COL if dt_col is None else dt_col
        if (geom_col is not None) and (geom_col!=self.GEOM_COL):
            raise ValueError("Due to geopandas dependencies, we require the geometry column (geom_col) to be named geometry." )
        else:
            self.geom_col = self.GEOM_COL
        self.id_col = self.ID_COL if id_col is None else id_col
        self.overview_level_col = self.OVERVIEW_LEVEL_COL if overview_level_col is None else overview_level_col
        self.spatial_key_col = self.SPATIAL_KEY_COL if spatial_key_col is None else spatial_key_col
        self.overview_key_col = self.OVERVIEW_KEY_COL if overview_key_col is None else overview_key_col
        
        # Overview statistics variables
        self.numeric_layers = self.NUMERIC_LAYERS if numeric_layers is None else numeric_layers
        self.timestamp_layers = self.TIMESTAMP_LAYERS if timestamp_layers is None else timestamp_layers
        self.categorical_layers = self.CATEGORICAL_LAYERS if categorical_layers is None else categorical_layers
        self.percentiles = self.PERCENTILES if percentiles is None else percentiles

        # Pyramid variables
        self.pyramid_levels = self.PYRAMID_LEVELS  # Will be set when pyramids are generated

        # If the vectorstore already exists, read the settings and metadata
        try:
            self.read_vectorstore_settings()
            self.metadata_from_parquet()
        except:
            pass

    @staticmethod
    def polygons2geodataframe(lst_polygons, geom_col, crs=4326):
        gdf_poly = pandas.DataFrame(lst_polygons)
        gdf_poly.columns=[geom_col]
        gdf_poly = geopandas.geodataframe.GeoDataFrame(gdf_poly, geometry=geom_col)
        gdf_poly = gdf_poly.set_crs(epsg=crs)
        return gdf_poly
    
    @staticmethod
    def quadtree(poly, level):
        # Get the quadtree representation 
        polyQuadTree, _ = pqt.QuadTreePAIRS(poly, max_level=level)
        # Get all the keys on the same resolution level
        polyCells = pqt.QuadTreeCellsPAIRS(polyQuadTree, level)
        return polyQuadTree, polyCells
    
    def _quadtree2geodataframe(self, quadtree):
        poly_squares=[]
        for square in quadtree:
            south, west = pqt.getLatLon(square[1], square[0])
            res = pqt.getResolution(square[0])
            north = south + res
            east = west + res
            poly_squares.append(shapely.geometry.box(west, south, east, north))

        df_squares = pandas.DataFrame(poly_squares)
        df_squares.columns=[self.geom_col]
        df_squares = pandas.concat([
            pandas.DataFrame(quadtree).rename(columns={0: self.overview_level_col, 1:self.spatial_key_col}), 
            df_squares
        ], axis=1)
        gdf_squares = geopandas.geodataframe.GeoDataFrame(df_squares, geometry=self.geom_col).set_crs(epsg=4326)
        return gdf_squares

    def _polyCells2geodataframe(self, cells):
        poly_cells=[]
        res = pqt.getResolution(self.overview_level)
        for cell in cells:
            south, west = pqt.getLatLon(cell, self.overview_level)
            north = south + res
            east = west + res
            poly_cells.append(shapely.geometry.box(west, south, east, north))

        gdf_cells = pandas.DataFrame(cells).rename(columns={0: self.spatial_key_col})
        gdf_cells[self.overview_level_col] = self.overview_level
        gdf_cells[self.geom_col] = poly_cells
        gdf_cells = geopandas.geodataframe.GeoDataFrame(gdf_cells, geometry=self.geom_col).set_crs(epsg=4326)
        return gdf_cells
    
    def _spatialPartitionLevel(self, sp):
        return int(sp.split(self.spatial_partition_identifier)[-1])
    
    def _create_spatial_partition_columns(self, gdf):
        for sp in self.spatial_partitions:
            levelsUp = self.overview_level - self._spatialPartitionLevel(sp)
            getParentKey_part = partial(pqt.getParentKey, levelsUp=levelsUp)
            gdf[sp] = gdf[self.spatial_key_col].apply(getParentKey_part)
    
    def _get_partitions(self):
        if self.partition_order=='spatial_before_temporal':
            self.partitions = self.spatial_partitions + self.temporal_partitions
        elif self.partition_order=='temporal_before_spatial':
            self.partitions = self.temporal_partitions + self.spatial_partitions
        else:
            raise ValueError("Invalid partition_order %s" % repr(self.partition_order))
            
    def _create_filter_key_column(self, gdf, level): 
        # Bottom left keys
        gdf[f'filter_key_level{level}'] = pqt.getKey_vect(gdf['bb_miny'], gdf['bb_minx'], level)
        # Top_right_keys
        top_right_keys = pqt.getKey_vect(gdf['bb_maxy'], gdf['bb_maxx'], level)
        # Overwrite in cases where the would be more than one cell
        gdf.loc[gdf[f'filter_key_level{level}']!=top_right_keys, f'filter_key_level{level}'] = numpy.nan

    def _create_filter_key_columns(self, gdf): 
        # Getting the bounds seems to require most of the time
        gdf[['bb_minx', 'bb_miny', 'bb_maxx', 'bb_maxy']] = gdf[self.geom_col].apply(lambda x: x.bounds).to_list()
        
        # Efficient way of getting the bounding-box keys 
        for level in self.filter_key_levels:
            self._create_filter_key_column(gdf, level)
            
        # Decide here if we want to keep the geometry bounds or delete them
        del gdf['bb_minx']
        del gdf['bb_miny']
        del gdf['bb_maxx']
        del gdf['bb_maxy']

    def _generate_overview_keys(self, df):
        """
        Combine the temporal and spatial keys into one "overview_key"
        """
        df[self.overview_key_col] = 'level' + df[self.overview_level_col].astype(str) + '_' + df[self.spatial_key_col].astype(str) 
        for k in self.temporal_keys:
            df[self.overview_key_col] = df[self.overview_key_col] + '_' + k + '_' + df[k].astype(str)
        return df[self.overview_key_col]
    
    def _select_partition(self, gdf, overview_key):
        df_part = gdf[gdf[self.overview_key_col]==overview_key].reset_index(drop=True)
        filepath = self.dataset_directory
        for partition_name in self.partitions:
            partition_value = df_part.loc[0, partition_name]
            filepath = os.path.join(filepath, partition_name + '_' + str(partition_value))
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = '_'.join([self.dataset, overview_key]) + '.parquet'
        filepath = os.path.join(filepath, filename)
        return df_part, filepath

    def _read_parquet(self, filepath, dt_start=None, dt_end=None, verbose=False):
        """
        Query the parquet vector store using temporal filters if applicable.
        Temporal and spatial partitioning is done by hand
        Temporal filtering is happening in pyarrow.
        Spatial filtering is done after the data has been received.
        """
        if verbose:
            print(filepath)
        filters = []
        if dt_start is not None:
            filters.append((self.dt_col, '>=', dt_start))
        if dt_end is not None:
            filters.append((self.dt_col, '<=', dt_end))
        try:
            gdf = geopandas.read_parquet(
                filepath, 
                filters = filters,
            )
        except Exception as e:
            if verbose:
                print(e)
                print('File not found', filepath)
            gdf = geopandas.geodataframe.GeoDataFrame()
        return gdf

    
    def load_geodataframe(
        self, gdf,
    ):
        """
        Load a geopandas geoDataFrame and transform it into a DataFrame with spatio-temporal overview keys, aligned with other geolab data
        """
        self.gdf = gdf
        assert(self.dt_col in self.gdf.columns)
        assert(self.geom_col in self.gdf.columns)
        assert(self.id_col in self.gdf.columns)
        
        # (A) CREATE COLUMNS FOR THE TEMPORAL KEYS USING ATTRIBUTES SUCH AS year, month, day
        for k in self.temporal_keys:
            self.gdf[k] = self.gdf[self.dt_col].apply(lambda x: getattr(x, k))
            
        # (B) CREATE A "spatial_key" COLUMN, ALIGNED WITH THE GEOLAB GRID
        # (B1) GET THE GEOLAB GRID FOR THE AREA OF INTEREST
        # Calculate aoi (total bounds) in order to create the quadtree keys
        self.total_bounds_polygon = shapely.geometry.box(*self.gdf[self.geom_col].total_bounds)
        
        # For plotting purposes we may want the boundary polygon in geodataframe form
        #self.gdf_total_bounds = self.polygons2geodataframe([self.total_bounds_polygon], self.geom_col)
        
        # Get guadtree and cell keys at overview_level resolution
        self.total_bounds_quadtree, self.total_bounds_cells = self.quadtree(self.total_bounds_polygon, self.overview_level)
        
        # Currently we don't use the variable-level quadtree representation
        """
        # Geodataframe of PAIRS squares (quadtree)
        self.gdf_total_bounds_squares = self._quadtree2geodataframe(self.total_bounds_quadtree)
        """

        # Geodataframe of geolab cells (all on the same overview_level resolution)
        self.gdf_total_bounds_cells = self._polyCells2geodataframe(self.total_bounds_cells)
        
        # (B2) JOIN ORIGINAL GEODATAFRAME TO GET THE SPATIAL KEY FROM THE OVERVIEW CELLS 
        # Joining original GeoDataFram with overview cells may result in duplication of rows. 
        # intersection_policy specifies how to deal with the geometry in such cases
        if self.intersection_policy=='cut':
            # Use geopandas.overlay (intersection) to cut the polygons at their intersection
            self.gdf_intersection = geopandas.overlay(self.gdf ,self.gdf_total_bounds_cells , how='intersection')
        elif self.intersection_policy=='original':
            # Use geopandas.sjoin (spatial join) to keep geometries intact
            self.gdf_intersection = geopandas.sjoin(self.gdf, self.gdf_total_bounds_cells, how='left', predicate='intersects').dropna(
                subset=['index_right']).drop(columns=['index_right'])
        elif self.intersection_policy=='both':
            # Cut the intersecting polygons but keep originals in another column: 'geometry_original'
            self.gdf_intersection = pandas.merge(
                geopandas.overlay(self.gdf ,self.gdf_total_bounds_cells , how='intersection'), 
                self.gdf[[self.id_col, self.geom_col]].rename(columns={self.geom_col: self.geom_col + '_original'}), 
                on=self.id_col
            )
            
        # (C) KEEP TRACK OF THE POLYGONS THAT INTERSECT THE OVERVIEW CELLS
        self.gdf_intersection[self.intersection_flag_col] = False
        self.gdf_intersection.loc[self.gdf_intersection[self.id_col].duplicated(keep=False), self.intersection_flag_col] = True
        
        # (D) COMBINE THE TEMPORAL AND SPATIAL KEYS INTO ONE "overview_key".
        self.gdf_intersection[self.overview_key_col] = self._generate_overview_keys(self.gdf_intersection)
        
        # (E) CREATE COLUMNS FOR SPATIAL PARTITIONING
        self._create_spatial_partition_columns(self.gdf_intersection)
        
        # (F) CREATE FILTER-KEY COLUMNS WITH KEYS FULLY CONTAINING THE GEOMETRY BOUNDING BOX AT VARIOUS RESOLUTION LEVELS
        self._create_filter_key_columns(self.gdf_intersection)
        
        
    def to_parquet(self, append=False, verbose=False):
        """
        Write GeoDataFrame to parquet
        """
        self._get_partitions()
        
        # Save the parquet files by overview_key in a partitioned directory structure
        self.overview_keys = sorted(self.gdf_intersection.overview_key.unique())
        for overview_key in self.overview_keys:
            df_part, filepath = self._select_partition(self.gdf_intersection, overview_key)
            if verbose:
                print(filepath)
            if len(df_part)==0:
                print('WARNING: no data to write')
            else:
                if append:
                    try:
                        # See if there is something already present under this overview key
                        df_existing = geopandas.read_parquet(filepath)
                    except:
                        pass
                    else:
                        # Merge the two by concatenating and dropping duplicates
                        df_part = pandas.concat([df_existing, df_part]).drop_duplicates().reset_index(drop=True)
                df_part.to_parquet(
                    path=filepath,
                    engine='pyarrow',
                    compression='snappy',
                    #partition_cols=self.partitions
                )
                
        # Write the settings to a json file
        self.write_vectorstore_settings()
        
            
    def write_vectorstore_settings(self):
        """
        Dump the settings to a json file
        """
        vs_settings = {}
        vs_settings['dt_col'] = self.dt_col
        vs_settings['geom_col'] = self.geom_col
        vs_settings['id_col'] = self.id_col
        vs_settings['overview_level_col'] = self.overview_level_col
        vs_settings['spatial_key_col'] = self.spatial_key_col
        vs_settings['overview_key_col'] = self.overview_key_col
        vs_settings['overview_level'] = self.overview_level
        vs_settings['temporal_keys'] = self.temporal_keys
        vs_settings['temporal_partitions'] = self.temporal_partitions
        vs_settings['spatial_partition_identifier'] = self.spatial_partition_identifier
        vs_settings['spatial_partition_levels'] = self.spatial_partition_levels
        vs_settings['spatial_partitions'] = self.spatial_partitions
        vs_settings['partitions'] = self.partitions
        vs_settings['intersection_policy'] = self.intersection_policy
        vs_settings['dataset'] = self.dataset
        vs_settings['vectorstore_directory'] = self.vectorstore_directory
        vs_settings['dataset_directory'] = self.dataset_directory
        vs_settings['overviewstore_directory'] = self.overviewstore_directory
        vs_settings['overview_directory'] = self.overview_directory
        vs_settings['partition_order'] = self.partition_order
        vs_settings['numeric_layers'] = self.numeric_layers
        vs_settings['timestamp_layers'] = self.timestamp_layers
        vs_settings['categorical_layers'] = self.categorical_layers
        vs_settings['percentiles'] = self.percentiles
        vs_settings['pyramid_levels'] = self.pyramid_levels

        json_path = os.path.join(self.dataset_directory, self.dataset+'.json')
        with open(json_path, 'w') as f:
            json.dump(vs_settings, f)
        
    def read_vectorstore_settings(self):
        json_path = os.path.join(self.dataset_directory, self.dataset+'.json')
        with open(json_path) as f:
            vs_settings = json.load(f)
        for k in vs_settings:
            setattr(self, k, vs_settings[k])
            
    def _calc_write_intersected_geometry(self):
        # Determine the geometries that intersect overview cells
        df_intersected = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            df1 = pandas.read_parquet(row['filepath'], columns=[self.overview_key_col, self.id_col, self.intersection_flag_col])
            df_intersected.append(df1[df1[self.intersection_flag_col]])
        df_intersected = pandas.concat(df_intersected).reset_index(drop=True)
        
        # Generate a two column table (id_col and lists of overview_key_col) 
        df_intersected = df_intersected.groupby(self.id_col)[self.overview_key_col].apply(list).reset_index()

        filepath = os.path.join(self.overview_directory, 'intersected_geometry.parquet')
        df_intersected = df_intersected.to_parquet(path=filepath, engine='pyarrow', compression='snappy')

    def read_intersected_geometry(self):
        filepath = os.path.join(self.overview_directory, 'intersected_geometry.parquet')
        try:
            df = pandas.read_parquet(filepath)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
            
    def _calc_write_overview_statistics_numeric(self):
        df_cell_statistics = []
        # Numeric layers
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            df = pandas.read_parquet(row['filepath'], columns=self.numeric_layers)
            df = df.describe(percentiles=self.percentiles)
            df.columns = pandas.MultiIndex.from_product([df.columns, [row[self.overview_key_col]]])
            df_cell_statistics.append(df)
        df_cell_statistics = pandas.concat(df_cell_statistics, axis=1)

        for numeric_layer in self.numeric_layers:
            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + numeric_layer + '.parquet')
            df_cell_statistics_layer = df_cell_statistics[numeric_layer].T.rename_axis(self.overview_key_col).reset_index()
            df_cell_statistics_layer = df_cell_statistics_layer.set_index(self.overview_key_col)
            df_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
            
    def _calc_write_overview_statistics_temporal(self):
        """
        Timestamp treated as numeric separately from the other numeric layers. (pandas .describe does not work for the combined layers)
        """
        df_cell_statistics = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            df = pandas.read_parquet(row['filepath'], columns=self.timestamp_layers)
            df = df.describe(percentiles=self.percentiles, datetime_is_numeric=True)
            df.columns = pandas.MultiIndex.from_product([df.columns, [row[self.overview_key_col]]])
            df_cell_statistics.append(df)
        df_cell_statistics = pandas.concat(df_cell_statistics, axis=1)

        for timestamp_layer in self.timestamp_layers:
            df_cell_statistics_layer = df_cell_statistics[timestamp_layer].T.rename_axis(self.overview_key_col)
            for col in [c for c in df_cell_statistics_layer.columns if c!='count']:
                # From object to datetime requires some serious data wrangling
                df_cell_statistics_layer[col] = df_cell_statistics_layer[col].apply(
                    lambda x: x.tz_localize(None)
                ).astype('datetime64[us]').apply(lambda x: x.tz_localize(pytz.utc))
            df_cell_statistics_layer['count'] = df_cell_statistics_layer['count'].astype(int)
            df_cell_statistics_layer = df_cell_statistics_layer.reset_index()

            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + timestamp_layer + '.parquet')
            df_cell_statistics_layer = df_cell_statistics_layer.set_index(self.overview_key_col)
            df_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')

    def _calc_write_overview_statistics_categorical(self):
        """
        Categorical layers (count, unique, top, frequency)
        For object data (e.g. strings), the result’s index will include count, unique, top, and freq. 
        The top is the most common value. The freq is the most common value’s frequency. 
        """
        df_cell_statistics = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            df = pandas.read_parquet(row['filepath'], columns=self.categorical_layers)
            df = df.astype(object).describe()
            df.columns = pandas.MultiIndex.from_product([df.columns, [row[self.overview_key_col]]])
            df_cell_statistics.append(df)
        df_cell_statistics = pandas.concat(df_cell_statistics, axis=1)

        for categorical_layer in self.categorical_layers:
            df_cell_statistics_layer = df_cell_statistics[categorical_layer].T.rename_axis(self.overview_key_col).reset_index()
            df_cell_statistics_layer['count'] = df_cell_statistics_layer['count'].astype(int)
            df_cell_statistics_layer['unique'] = df_cell_statistics_layer['unique'].astype(int)
            df_cell_statistics_layer['freq'] = df_cell_statistics_layer['freq'].astype(int)
            
            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + categorical_layer + '.parquet')
            df_cell_statistics_layer = df_cell_statistics_layer.set_index(self.overview_key_col)
            df_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
            
    def _calc_write_overview_histogram(self, histogram_layer):
        # Categorical and timestamp layers only (full histogram)
        assert(histogram_layer in (self.categorical_layers + self.timestamp_layers))

        df_cell_hist = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            df1 = pandas.read_parquet(row['filepath'], columns=[histogram_layer])
            df2 = df1[histogram_layer].value_counts()
            df2.name = row[self.overview_key_col]
            df_cell_hist.append(df2)
        df_cell_hist = pandas.concat(df_cell_hist, axis=1)
        df_cell_hist = df_cell_hist.sort_index().fillna(0).astype(int)

        filepath = os.path.join(self.overview_directory, 'overview_histogram_' + histogram_layer + '.parquet')
        df_cell_hist.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
        
    def calc_overview_statistics(self, numeric_layers=None, timestamp_layers=None, categorical_layers=None, percentiles=None):
        """
        Caluculate the overview layer statistics and write them out to parquet
        """
        if numeric_layers is not None:
            self.numeric_layers = numeric_layers 
        if timestamp_layers is not None:
            self.timestamp_layers = timestamp_layers 
        if categorical_layers is not None:
            self.categorical_layers = categorical_layers 
        if percentiles is not None:
            self.percentiles = percentiles 
        
        # Dump the settings to a json file (existing will be overwritten)
        self.write_vectorstore_settings()
        
        # Assert that we don't have layers in multiple categories at once
        all_layers = self.numeric_layers + self.timestamp_layers + self.categorical_layers
        assert(len(all_layers)==len(list(set(all_layers))))
        
        if len(self.numeric_layers)>0:
            self._calc_write_overview_statistics_numeric()
        if len(self.timestamp_layers)>0:
            self._calc_write_overview_statistics_temporal()
            for timestamp_layer in self.timestamp_layers:
                self._calc_write_overview_histogram(timestamp_layer)
        if len(self.categorical_layers)>0:
            self._calc_write_overview_statistics_categorical()
            for categorical_layer in self.categorical_layers:
                self._calc_write_overview_histogram(categorical_layer)
                    
        # Summary of intersected geometries (at overview cell boundaries)
        self._calc_write_intersected_geometry()
            
    def read_overview_statistics(self, layer):
        filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
        try:
            # If needed, transpose after reading the parquet instead of in the parquet file itself, 
            # because parquet has limitations in storing mixed datatypes in columns.
            df = pandas.read_parquet(filepath)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df

    def read_overview_histogram(self, layer):
        filepath = os.path.join(self.overview_directory, 'overview_histogram_' + layer + '.parquet')
        try:
            df = pandas.read_parquet(filepath)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
    
    def _numeric_pyramids(self, numeric_layer, verbose=False):
        if verbose:
            print('numeric_layer', numeric_layer)
        # Get the cell statistics at the overview level
        df_cell_statistics = self.read_overview_statistics(numeric_layer)

        # Keep a limited number of statistics and add the pyramid keys
        df = df_cell_statistics[['count', 'mean', 'min', 'max']]
        df = df.join(self.df_pyramid_keys)

        # Calculate the sum
        df['sum'] = df['count'] * df['mean']

        for pyramid_level in self.pyramid_levels:
            if verbose:
                print('pyramid_level', pyramid_level)
            # Naively aggregate count, sum, min, max
            df1 = []
            df1.append(df[['pyramid_level' + str(pyramid_level), 'count']].groupby('pyramid_level' + str(pyramid_level)).sum())
            df1.append(df[['pyramid_level' + str(pyramid_level), 'sum']].groupby('pyramid_level' + str(pyramid_level)).sum())
            df1.append(df[['pyramid_level' + str(pyramid_level), 'min']].groupby('pyramid_level' + str(pyramid_level)).min())
            df1.append(df[['pyramid_level' + str(pyramid_level), 'max']].groupby('pyramid_level' + str(pyramid_level)).max())
            df1 = pandas.concat(df1, axis=1)

            # Correct the count and sum statistics for polygons intersecting multiple overview cells 
            # (otherwise they would be counted multiple times)
            df3 = []
            for i, row in self.df_intersected_T.iterrows():
                overview_key = row[self.overview_key_col]
                filepath = row['filepath']
                lst_filter_id = row[self.id_col]
                df2 = pandas.read_parquet(filepath, columns=[self.overview_key_col, self.id_col, numeric_layer])
                df2 = df2[df2[self.id_col].isin(lst_filter_id)]
                df3.append(df2)

            df3 = pandas.concat(df3).reset_index(drop=True)
            df3 = df3.set_index(self.overview_key_col)
            df3 = df3.join(self.df_pyramid_keys)

            # Groupby id and pyramid level together to find relevant overcounts
            df4 = []
            df4.append(df3[[self.id_col, 'pyramid_level' + str(pyramid_level), numeric_layer]].groupby(
                [self.id_col, 'pyramid_level' + str(pyramid_level)]
            ).count().rename(columns={numeric_layer: 'count'}))
            df4.append(df3[[self.id_col, 'pyramid_level' + str(pyramid_level), numeric_layer]].groupby(
                [self.id_col, 'pyramid_level' + str(pyramid_level)]
            ).first().rename(columns={numeric_layer: 'first'}))
            df4.append(df3[[self.id_col, 'pyramid_level' + str(pyramid_level), numeric_layer]].groupby(
                [self.id_col, 'pyramid_level' + str(pyramid_level)]
            ).sum().rename(columns={numeric_layer: 'sum'}))
            df4 = pandas.concat(df4, axis=1)

            # Prepare for summing up by pyramid key
            df4 = df4[df4['count']>1]
            df4 = df4.reset_index()
            df4['overcount'] = df4['count']-1
            df4['oversum'] = df4['overcount'] * df4['first'] 

            # Sum up the overcount and oversum by pyramid level key
            df5 = df4[['pyramid_level' + str(pyramid_level), 'overcount', 'oversum']].groupby('pyramid_level' + str(pyramid_level)).sum()

            # Join with the original statistics and correct them
            df6 = df1.join(df5)
            df6['count'] = df6['count'] - df6['overcount'].fillna(0)
            df6['sum'] = df6['sum'] - df6['oversum'].fillna(0)
            #del df6['overcount']
            #del df6['oversum']
            df6['mean'] = df6['sum'] / df6['count']
            #del df6['sum']
            df6['count'] = df6['count'].astype(int)
            df6 = df6[['count', 'mean', 'min', 'max']]

            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + numeric_layer + '.parquet')
            df6.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
            
    def _histogram_pyramids(self, categorical_layer, verbose=False):
        if verbose:
            print('histogram_layer', categorical_layer)
            
        # Get the histogram at the overview level
        df = self.read_overview_histogram(categorical_layer)
        df = df.T
        histogram_categories = df.columns
        df = df.join(self.df_pyramid_keys)

        for pyramid_level in self.pyramid_levels:
            if verbose:
                print('pyramid_level', pyramid_level)
            # Naively aggregate the counts for each historgram category
            df1 = []
            for category in histogram_categories:
                df1.append(df[['pyramid_level' + str(pyramid_level), category]].groupby('pyramid_level' + str(pyramid_level)).sum())
            df1 = pandas.concat(df1, axis=1)

            # Correct the statistics for polygons intersecting multiple overview cells 
            # (otherwise they would be counted multiple times)
            df3 = []
            for i, row in self.df_intersected_T.iterrows():
                overview_key = row[self.overview_key_col]
                filepath = row['filepath']
                lst_filter_id = row[self.id_col]
                df2 = pandas.read_parquet(filepath, columns=[self.overview_key_col, self.id_col, categorical_layer])
                df2 = df2[df2[self.id_col].isin(lst_filter_id)]
                df3.append(df2)

            df3 = pandas.concat(df3).reset_index(drop=True)
            df3 = df3.set_index(self.overview_key_col)
            df3 = df3.join(self.df_pyramid_keys)

            # Groupby id and pyramid level together to find relevant overcounts
            df4 = []
            df4.append(df3[[self.id_col, 'pyramid_level' + str(pyramid_level), categorical_layer]].groupby(
                [self.id_col, 'pyramid_level' + str(pyramid_level)]
            ).count().rename(columns={categorical_layer: 'count'}))
            df4.append(df3[[self.id_col, 'pyramid_level' + str(pyramid_level), categorical_layer]].groupby(
                [self.id_col, 'pyramid_level' + str(pyramid_level)]
            ).first().rename(columns={categorical_layer: 'category'}))

            df4 = pandas.concat(df4, axis=1)

            # Prepare for summing up by pyramid key
            df4 = df4[df4['count']>1]
            df4 = df4.reset_index()
            df4['overcount'] = df4['count']-1

            # Sum up the overcount and oversum by pyramid level key
            df5 = df4[['pyramid_level' + str(pyramid_level), 'category', 'overcount']].groupby(
                ['pyramid_level' + str(pyramid_level), 'category']
            ).sum()

            # Join with the original statistics and correct them
            df1.columns.name = 'category'
            df1_stacked = pandas.DataFrame(df1.stack().rename('count'))
            df6 = df1_stacked.join(df5)
            df6['count'] = df6['count'] - df6['overcount'].fillna(0)
            df6 = df6[['count']]
            df6 = df6.unstack()['count'].astype(int)
            df6.columns.name = None

            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_histogram_' + categorical_layer + '.parquet')
            df6.to_parquet(path=filepath, engine='pyarrow', compression='snappy')            
            
    def _categorical_pyramids(self, categorical_layer, verbose=False):
        if verbose:
            print('categorical_layer', categorical_layer)
            for pyramid_level in self.pyramid_levels:
                print('pyramid_level', pyramid_level)
                filepath_histogram = os.path.join(
                    self.overview_directory, 
                    'pyramid_level' + str(pyramid_level) + '_histogram_' + categorical_layer + '.parquet'
                )
                try:
                    df = pandas.read_parquet(filepath_histogram)
                except:
                    # Generate histogram pyramid first
                    print('WARNING: (Re-)generating histogram needed for categorical pyramid')
                    self._histogram_pyramids(categorical_layer, verbose=verbose)
                    df = pandas.read_parquet(filepath_histogram)
                    
                df1 = df[[]].copy()
                df1['count'] = df.sum(axis=1)
                df_stacked = df[df!=0].stack().reset_index()
                df1['unique'] = df_stacked.groupby('pyramid_level' + str(pyramid_level)).count()['level_1']
                df1['top'] = df.idxmax(axis=1)
                df1['freq'] = df_stacked.groupby('pyramid_level' + str(pyramid_level)).max()[0].astype(int)

                # Save to parquet file
                filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + categorical_layer + '.parquet')
                df1.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
                
    def _create_pyramid_columns(self, gdf):
        self.pyramid_levels = list(range(self.overview_level + 1))
        for pyramid_level in self.pyramid_levels:
            levelsUp = self.overview_level - pyramid_level
            getParentKey_part = partial(pqt.getParentKey, levelsUp=levelsUp)
            gdf['pyramid_level' + str(pyramid_level)] = gdf[self.spatial_key_col].apply(getParentKey_part)
            
    def pyramid_keys(self):
        self.df_pyramid_keys = self.gdf_meta.copy()
        self._create_pyramid_columns(self.df_pyramid_keys)
        self.df_pyramid_keys = self.df_pyramid_keys[
            [self.overview_key_col] + ['pyramid_level' + str(l) for l in self.pyramid_levels]
        ].set_index(self.overview_key_col)
                
    def calc_pyramids(self, verbose=False):
        if verbose:
            print('Calculating pyramids')
            
        # Generate the pyramid keys
        self.pyramid_keys()
        
        # Find the geometries intersecting overview cells
        self.df_intersected = self.read_intersected_geometry()

        # Transpose df_intersected so we can make quick filter queries using the id_col
        self.df_intersected_T = self.df_intersected.set_index(self.id_col)[self.overview_key_col].explode().reset_index()
        self.df_intersected_T = self.df_intersected_T.groupby(self.overview_key_col)[self.id_col].apply(list).reset_index()
        self.df_intersected_T = pandas.merge(self.df_intersected_T, self.gdf_meta[[self.overview_key_col, 'filepath']], on=self.overview_key_col)
        
        for numeric_layer in self.numeric_layers:
            self._numeric_pyramids(numeric_layer, verbose=verbose)
            
        for categorical_layer in self.categorical_layers:
            self._histogram_pyramids(categorical_layer, verbose=verbose)
            self._categorical_pyramids(categorical_layer, verbose=verbose)
            
        # Update the vectorstore settings
        self.write_vectorstore_settings()
        
    def read_pyramid(self, pyramid_level, layer, histogram=False):
        if histogram:
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_histogram_' + layer + '.parquet')
        else:
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + layer + '.parquet')
        try:
            # If needed, transpose after reading the parquet instead of in the parquet file itself, 
            # because parquet has limitations in storing mixed datatypes in columns.
            df = pandas.read_parquet(filepath)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
    
    def metadata_from_parquet(self, verbose=False):
        glob_wildcard_path = self.dataset_directory
        for partition_name in self.partitions:
            #partition_value = df_part.loc[0, partition_name]
            glob_wildcard_path = os.path.join(glob_wildcard_path, partition_name + '*')
        glob_wildcard_path = os.path.join(glob_wildcard_path, self.dataset+'*.parquet')
        if verbose:
            print('glob_wildcard_path', glob_wildcard_path)
        globbed_filepaths = sorted(glob(glob_wildcard_path))
        
        # Remove the dataset directory and the filetype ('.parquet') from the paths
        globbed = [g.split(self.dataset_directory)[-1] for g in globbed_filepaths]
        globbed = [g.split('.parquet')[0] for g in globbed]
        
        # Cast to pandas and split into several columns
        df_meta = pandas.DataFrame([g.split('/')[1:] for g in globbed], columns=self.partitions+[self.overview_key_col])
        df_meta['filepath'] = globbed_filepaths
        
        # Remove parts of the strings and cast the partitions to integer 
        for p in self.partitions:
            df_meta[p] = df_meta[p].apply(lambda x: int(x.split(p+'_')[-1]))
        df_meta[self.overview_key_col] = df_meta[self.overview_key_col].apply(lambda x: x.split(self.dataset+'_')[-1])
        
        # Split the overview key column
        df_meta[[self.overview_level_col, self.spatial_key_col]] = df_meta[self.overview_key_col].apply(
            lambda x: x.split('level', 1)[1].split('_', 2)[:2]).to_list()
        df_meta[self.overview_level_col] = df_meta[self.overview_level_col].astype(int)
        df_meta[self.spatial_key_col] = df_meta[self.spatial_key_col].astype(int)
        for k in self.temporal_keys:
            df_meta[k] = df_meta[self.overview_key_col].apply(lambda x: int(x.split(k+'_')[1].split('_')[0]))
            
        # Cast to geopandas by infering the geometry column (cell boxes) from the overview level and spatial key
        self.gdf_meta = self._polyCells2geodataframe(df_meta[self.spatial_key_col].drop_duplicates().to_numpy())
        self.gdf_meta = pandas.merge(self.gdf_meta, df_meta, on=[self.spatial_key_col, self.overview_level_col], how='right')

            
    def query_single_parquet(self, query_latitude, query_longitude, query_dt, spatial_filter=False, temporal_filter=False):
        """
        Query a point in time and space
        Either retrieve the matching record
        Or retrieve the entire parquet file
        """
        self.read_vectorstore_settings()
        self.query_df_meta = {}
        for k in self.temporal_keys:
            self.query_df_meta[k] = getattr(query_dt, k)
        self.query_df_meta[self.overview_level_col] = self.overview_level
        self.query_df_meta[self.spatial_key_col] = pqt.getKey(query_latitude, query_longitude, self.overview_level)
        self.query_df_meta = pandas.DataFrame([self.query_df_meta])

        # Spatial key for the partition
        for sp in self.spatial_partitions:
            levelsUp = self.overview_level-self._spatialPartitionLevel(sp)
            getParentKey_part = partial(pqt.getParentKey, levelsUp=levelsUp)
            self.query_df_meta[sp] = self.query_df_meta[self.spatial_key_col].astype(int).apply(getParentKey_part)

        self.query_df_meta[self.overview_key_col] = self._generate_overview_keys(self.query_df_meta)
        overview_key = self.query_df_meta[self.overview_key_col].values[0]
        _, filepath = self._select_partition(self.query_df_meta, overview_key)

        # Load a complete single parquet file
        try:
            self.gdf_query = geopandas.read_parquet(filepath)
        except FileNotFoundError as e:
            self.gdf_query = geopandas.geodataframe.GeoDataFrame()

        # Filter the query if requested
        if temporal_filter:
            self.gdf_query=self.gdf_query[self.gdf_query[self.dt_col]==query_dt]

        if spatial_filter:
            self.gdf_query = self.gdf_query[self.gdf_query.intersects(shapely.geometry.Point(query_longitude, query_latitude))]
            
        self.gdf_query=self.gdf_query.reset_index(drop=True)
    
    def query_vectorstore(
        self,
        query_polygon=None, 
        query_dt_start=None,  
        query_dt_end=None, 
        query_intersection_policy=None, 
        verbose=False
    ):
        """
        Query the parquet vector store (intersecting in time and space)
        May be incommensurate with cells, span multiple cells, or may be incommensurate with temporal key
        """
        self.query_polygon             = self.COMPLETE_WORLD if query_polygon is None else query_polygon
        self.query_dt_start            = self.MIN_DT if query_dt_start is None else query_dt_start
        self.query_dt_end              = self.MAX_DT if query_dt_end is None else query_dt_end
        self.query_intersection_policy = self.QUERY_INTERSECTION_POLICY if query_intersection_policy is None else query_intersection_policy

        self.read_vectorstore_settings()
        self.query_gdf_polygon = self.polygons2geodataframe([self.query_polygon], self.geom_col)
        
        # Temporal part of the overview key
        self.query_df_meta = pandas.DataFrame()
        if len(self.temporal_keys)>0:
            if 'year' in self.temporal_keys:
                start_year = self.query_dt_start.year
                # For partition purposes, start time may be different
                freq = 'YS'
            else:
                start_year = 1970
            if 'month' in self.temporal_keys:
                start_month = self.query_dt_start.month
                freq = 'MS'
            else:
                start_month = 1
            if 'day' in self.temporal_keys:
                start_day = self.query_dt_start.day
                freq = 'D'
            else:
                start_day = 1
            if 'hour' in self.temporal_keys:
                start_hour = self.query_dt_start.hour
                freq = 'H'
            else:
                start_hour = 0
            dt_start_partition = datetime(start_year, start_month, start_day, start_hour, tzinfo=pytz.utc)

            # Generate the list of temporal key combinations for the partitions
            date_range = pandas.date_range(dt_start_partition, self.query_dt_end, freq=freq) #'2014-10-10','2016-01-07'
            df_date_range = pandas.DataFrame(date_range).rename(columns={0: self.dt_col})
            for k in self.temporal_keys:
                df_date_range[k] = df_date_range[self.dt_col].apply(lambda x: getattr(x, k))
                
            # Temporal part of the overview keys
            for i, row in df_date_range[self.temporal_keys].iterrows():
                for k in self.temporal_keys:
                    self.query_df_meta.loc[i, k] = row[k]
                    
        # Spatial part of the overview key
        query_quadtree, query_spatial_keys = self.quadtree(self.query_polygon, self.overview_level)
        df_spatial = pandas.DataFrame()
        for i, query_spatial_key in enumerate(query_spatial_keys):
            df_spatial.loc[i, self.spatial_key_col] = query_spatial_key
            
        # Cartesian product of spatial and temporal parts
        if len(self.query_df_meta)>0:
            self.query_df_meta = self.query_df_meta.merge(df_spatial, how='cross')
        else:
            self.query_df_meta = df_spatial

        # Pairs level of the overview key
        self.query_df_meta[self.overview_level_col] = self.overview_level

        # Convert the floats to int
        self.query_df_meta = self.query_df_meta.astype(int)
        
        # Generate the overview keys
        self.query_df_meta[self.overview_key_col] = self._generate_overview_keys(self.query_df_meta)
        
        # Generate the spatial partition columns
        self._create_spatial_partition_columns(self.query_df_meta)

        lst_gdf = []
        overview_keys_requested = set(self.query_df_meta[self.overview_key_col])
        try:
            overview_keys_present = set(self.gdf_meta[self.overview_key_col])
        except:
            # gdf_meta likely not present yet, so generate it
            self.metadata_from_parquet()
            overview_keys_present = set(self.gdf_meta[self.overview_key_col])
        overview_keys = sorted(overview_keys_requested & overview_keys_present)
        for overview_key in overview_keys:
            _, filepath = self._select_partition(self.query_df_meta, overview_key)
            gdf = self._read_parquet(filepath, self.query_dt_start, self.query_dt_end, verbose=False)
            if len(gdf)>0:
                if not self.query_polygon.contains(shapely.geometry.box(*gdf.total_bounds)):
                    # Spatial filtering may be necessary
                    if self.query_intersection_policy=='cut':
                        # return intersection (cut polygons)
                        gdf = geopandas.overlay(
                            gdf,
                            self.query_gdf_polygon, 
                            how='intersection'
                        ).reset_index(drop=True)
                    elif self.query_intersection_policy=='original':
                        # Return intersecting polygons intact
                        if self.intersection_policy=='cut':
                            # To do: use geometry_id to find all locations of this polygon and stitch together.
                            print('NOT IMPLEMENTED WARNING: Query asks for original polygons but we are returning the cut ones.')
                            gdf = geopandas.sjoin(
                                gdf, 
                                self.query_gdf_polygon, 
                                how='left', 
                                predicate='intersects'
                            ).dropna(subset=['index_right']).drop(columns=['index_right'])
                        elif self.intersection_policy=='original':
                            # Data has been saved as complete polygons
                            gdf = geopandas.sjoin(
                                gdf, 
                                self.query_gdf_polygon, 
                                how='left', 
                                predicate='intersects'
                            ).dropna(subset=['index_right']).drop(columns=['index_right'])
                        elif self.intersection_policy=='both':
                            # Pick the original geometry column with complete polygons
                            gdf = geopandas.sjoin(
                                gdf.set_geometry(self.geom_col + '_original', crs=4326), 
                                self.query_gdf_polygon, 
                                how='left', 
                                predicate='intersects'
                            ).dropna(subset=['index_right']).drop(columns=['index_right'])
                if len(gdf)>0:
                    lst_gdf.append(gdf)
            else:
                if verbose:
                    print(f"WARNING: no data found in {overview_key}")

        if len(lst_gdf)==0:
            self.gdf_query = geopandas.geodataframe.GeoDataFrame()
            print('WARNING: no data found')
        else:
            self.gdf_query = pandas.concat(lst_gdf).reset_index(drop=True)
            if verbose:
                print('lst_gdf', len(lst_gdf))
                print('gdf_query before merging duplicated', len(self.gdf_query))
            
            # Some objects may be duplicated because they are saved in multiple overview cells. 
            # If we saved intersections of polygons, we need to stitch those together
            duplicated = self.gdf_query.groupby(self.id_col).count()
            duplicated = list(duplicated[duplicated[self.geom_col]>1].index)
            if len(duplicated)>0:
                gdf_duplicated = self.gdf_query[self.gdf_query[self.id_col].isin(duplicated)]
                # Removing the annoying buffer warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Drop the duplicated rows and dissolve the geometris
                    gdf_duplicated = pandas.merge(
                        gdf_duplicated.drop_duplicates(subset=self.id_col).drop(columns=self.geom_col), 
                        gdf_duplicated.dissolve(by=self.id_col).buffer(1e-10).buffer(-1e-10).reset_index().rename(columns={0: self.geom_col}), 
                        on=self.id_col,
                        how='left'
                    )
                self.gdf_query = pandas.concat([
                    self.gdf_query[~self.gdf_query[self.id_col].isin(duplicated)], 
                    gdf_duplicated,
                ]).sort_values(by=[self.dt_col, self.overview_key_col]).reset_index(drop=True)

            if verbose:
                print('gdf_query after merging duplicated ', len(self.gdf_query))
