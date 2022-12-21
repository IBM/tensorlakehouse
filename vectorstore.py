import os
import sys
import warnings
from glob import glob

sys.path.insert(1, os.path.abspath(".."))
from pairs_python.core import pairs_quadtree as pqt

import time
import math
import numpy
import pandas
from datetime import datetime, timedelta
import pytz
import geopandas
import shapely
import pyproj
from functools import partial
import json
from multiprocessing import Pool
from pathos.pools import ProcessPool


class Vectorstore(object):
    """
    Vectorstore Class for handling vector data in Geolab-compatible parquet files
    """
    
    # Default values for variables
    VECTORSTORE_DIRECTORY      = '/data/vector/vectorstore/'
    OVERVIEWSTORE_DIRECTORY    = '/data/vector/overviews/'

    TEMPORAL_KEYS              = ['year']
    TEMPORAL_PARTITIONS        = []
    
    DIMENSION_KEYS             = []

    # Overview resolution layer for the Parquet file
    # (PAIRS level 6 ~ 1000km, level 9 ~ 100km, level 13 ~ 10km, level 16 ~ 1km, level 23 ~ 10m)
    OVERVIEW_LEVEL             = 8
    SPATIAL_PARTITION_IDENTIFIER = 'partition_level'
    SPATIAL_PARTITION_LEVELS   = [6]
    
    # Order of the partitions in the stored directory structure
    PARTITION_ORDER            = 'temporal_before_spatial' #'spatial_before_temporal'

    # Filter-key columns at various resolution levels (above the overview cell level)
    FILTER_KEY_LEVELS          = []
    
    # Policy how to deal with polygons that intersect overview cells ("cut", "original", "both")
    INTERSECTION_POLICY        = 'cut' # 'original', 'both'
    INTERSECTION_FLAG_COL      = 'intersection_flag'
    
    # GeoDataFrame colum conventions (used when loading data from gdf or vectorstore)
    DT_COL                     = 'timestamp'
    GEOM_COL                   = 'geometry'  # Geopandas relies on this column being named 'geometry', so enforce this for all tables
    ID_COL                     = 'geometry_id'
    OVERVIEW_LEVEL_COL         = 'overview_level'
    SPATIAL_KEY_COL            = 'spatial_key'
    OVERVIEW_KEY_COL           = 'overview_key'
    GEOM_AREA_COL              = 'geom_area'
    GEOM_LENGTH_COL            = 'geom_length'
    
    # Overview statistics
    NUMERIC_LAYERS             = []
    TIMESTAMP_LAYERS           = []
    CATEGORICAL_LAYERS         = []
    QUANTILES                  = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    FIRST                      = True # First statistics for vector overviews
    # Set to True for timeseries data, where we have too many timestamps to calculate individual statistics (e.g. >100)
    TIMESTAMP_AGGREGATION      = True 
    
    # Pyramids
    PYRAMID_LEVELS             = []  # Initialize as empty list. Will be populated when pyramids are generated
    
    # Min and max datetime conventions (used when querying data)
    MIN_DT = datetime(1, 1, 1).replace(tzinfo=pytz.utc)
    MAX_DT = datetime(9999, 12, 31).replace(tzinfo=pytz.utc)
    
    # Other query conventions
    COMPLETE_WORLD             = shapely.geometry.box(-180, -90, 180, 90)
    QUERY_INTERSECTION_POLICY  = 'cut' #'original'
    
    def __init__(self,
                 dataset,
                 vectorstore_directory = None,
                 overviewstore_directory = None,
                 temporal_keys = None,
                 temporal_partitions = None,
                 dimension_keys = None,
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
                 geom_area_col = None, 
                 geom_length_col = None, 
                 numeric_layers = None,
                 timestamp_layers = None,
                 categorical_layers = None,
                 quantiles = None,
                 first = None,
                 timestamp_aggregation=None,
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
        
        # Dimensions
        self.dimension_keys               = self.DIMENSION_KEYS if dimension_keys is None else dimension_keys

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
        self.geom_area_col = self.GEOM_AREA_COL if geom_area_col is None else geom_area_col
        self.geom_length_col = self.GEOM_LENGTH_COL if geom_length_col is None else geom_length_col
        
        # Overview statistics variables
        self.numeric_layers = self.NUMERIC_LAYERS if numeric_layers is None else numeric_layers
        self.timestamp_layers = self.TIMESTAMP_LAYERS if timestamp_layers is None else timestamp_layers
        self.categorical_layers = self.CATEGORICAL_LAYERS if categorical_layers is None else categorical_layers
        self.quantiles = self.QUANTILES if quantiles is None else quantiles
        self.first = self.FIRST if first is None else first
        self.timestamp_aggregation = self.TIMESTAMP_AGGREGATION if timestamp_aggregation is None else timestamp_aggregation

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
        # Get the quadtree cells only
        polyQuadTree, _ = pqt.QuadTreePAIRS(poly, max_level=level)
        # Get all the keys on the same resolution level
        polyCells = pqt.QuadTreeCellsPAIRS(polyQuadTree, level)
        return polyCells
    
    def _quadtreeGDF(self, poly):
        """
        version of quadtree that works better for complicated polygons and returns a GeoDataFrame
        """
        # Get all the cells within the total bounds
        bounds_cells = self.quadtree(shapely.box(*poly.bounds), self.overview_level)
        gdf_bounds_cells = self._polyCells2geodataframe(bounds_cells)
        # Filter the cells that overlap the polygon
        mask = gdf_bounds_cells.intersects(poly)
        gdf_masked = gdf_bounds_cells[mask].reset_index(drop=True)
        #polyCells = gdf_masked['spatial_key'].to_list()
        return gdf_masked

    
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
        gdf_unique = gdf[[self.id_col, self.geom_col]].drop_duplicates(subset=self.id_col).reset_index(drop=True)
        del gdf_unique[self.id_col]
        gdf_unique[['bb_minx', 'bb_miny', 'bb_maxx', 'bb_maxy']] = gdf_unique[self.geom_col].apply(lambda x: x.bounds).to_list()
        
        # Efficient way of getting the bounding-box keys 
        for level in self.filter_key_levels:
            self._create_filter_key_column(gdf_unique, level)
            
        # Decide here if we want to keep the geometry bounds or delete them
        del gdf_unique['bb_minx']
        del gdf_unique['bb_miny']
        del gdf_unique['bb_maxx']
        del gdf_unique['bb_maxy']
        
        gdf = pandas.merge(gdf, gdf_unique, on=self.geom_col)

    def _generate_overview_keys(self, df):
        """
        Combine the temporal, spatial and dimension keys into one "overview_key"
        """
        df[self.overview_key_col] = 'level' + df[self.overview_level_col].astype(str) + '_' + df[self.spatial_key_col].astype(str) 
        for k in self.temporal_keys + self.dimension_keys:
            df[self.overview_key_col] = df[self.overview_key_col] + '_' + k + '_' + df[k].astype(str)
        return df[self.overview_key_col]

    def _decompose_overview_keys(self, df):
        """
        Decompose the overview_key into temporal, spatial, and dimension keys
        """
        cols = [self.overview_level_col, self.spatial_key_col]
        df[cols] = df[self.overview_key_col].apply(lambda x: x.split('level', 1)[1].split('_', 2)[:2]).to_list()
        df[self.overview_level_col] = df[self.overview_level_col].astype(int)
        df[self.spatial_key_col] = df[self.spatial_key_col].astype(int)
        for k in self.temporal_keys + self.dimension_keys:
            cols = cols + [k]
            df[k] = df[self.overview_key_col].apply(lambda x: int(x.split(k+'_')[1].split('_')[0]))
        #for k in self.dimension_keys: 
        #    cols = cols=[k]
        #    df[k] = df[self.overview_key_col].apply(lambda x: x.split(k+'_')[1].split('_')[0]) # Dimension keys are of type str
        return df[cols]

    def _select_parquet_file(self, gdf, overview_key):
        gdf_part = gdf[gdf[self.overview_key_col]==overview_key].reset_index(drop=True)
        filepath = self.dataset_directory
        for partition_name in self.partitions:
            partition_value = gdf_part.loc[0, partition_name]
            filepath = os.path.join(filepath, partition_name + '_' + str(partition_value))
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = '_'.join([self.dataset, overview_key]) + '.parquet'
        filepath = os.path.join(filepath, filename)
        return gdf_part, filepath
    
    def _all_the_same(self, s):
        # Quick test if column values (pandas series s) are all the same
        a = s.to_numpy()
        return (a[0] == a).all()
    
    def _reproject_to_local_equal_area_grid(self, gdf):
        """
        All geometries must belong to the same overview cell
        """
        assert(self._all_the_same(gdf[self.spatial_key_col]))
        
        # Get the center lat/lon of the overview tile
        key = gdf.loc[0, self.spatial_key_col]
        level = gdf.loc[0, self.overview_level_col]
        lat, lon = pqt.getCenterLatLon(key, level)

        # local_azimuthal_projection preserves angle (e.g. circles stay circles)
        #local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={lat} +lon_0={lon}"

        # Lambert Azimuthal Equal Area preserves area 
        lambert_azimuthal_ea = f"+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

        #gdf['area_local_azimuthal_projection'] = gdf.to_crs(local_azimuthal_projection).area
        #gdf['area_lambert_azimuthal_ea'] = gdf.to_crs(lambert_azimuthal_ea).area
        
        return gdf.to_crs(lambert_azimuthal_ea)
        
    def _read_parquet(self, filepath, dt_start=None, dt_end=None, columns=None, geometry_area=False, geometry_length=False, verbose=False):
        """
        Query the parquet vector store using temporal filters if applicable.
        Temporal and spatial partitioning is done by hand
        Temporal filtering is happening in pyarrow.
        Spatial filtering is done after the data has been received.
        """        
        filters = []
        if dt_start is not None:
            filters.append((self.dt_col, '>=', dt_start))
        if dt_end is not None:
            filters.append((self.dt_col, '<=', dt_end))
        if verbose:
            print('filepath', filepath)
            #print('filters', filters)
        try:
            if len(filters)>0:
                gdf = geopandas.read_parquet(filepath, columns=columns, filters=filters)
            else:
                gdf = geopandas.read_parquet(filepath, columns=columns)
        except Exception as e:
            if verbose:
                print(e)
                print('Problem loading file', filepath)
            gdf = geopandas.geodataframe.GeoDataFrame()
            
        # Add area and/or length columns if requested
        if geometry_area or geometry_length:
            gdf_lambert_ea = self._reproject_to_local_equal_area_grid(gdf)
            if geometry_area:
                gdf[self.geom_area_col] = gdf_lambert_ea.area
            if geometry_length:
                gdf[self.geom_length_col] = gdf_lambert_ea.length
                
        return gdf

    def read_selected_parquet(
        self, overview_key, dt_start=None, dt_end=None, columns=None, geometry_area=False, geometry_length=False, verbose=False
    ):
        _, filepath = self._select_parquet_file(self.gdf_meta, overview_key)
        gdf = self._read_parquet(
            filepath, dt_start=dt_start, dt_end=dt_end, columns=columns, geometry_area=geometry_area, geometry_length=geometry_length, verbose=verbose
        )
        return gdf
    
    def ingest_geodataframe(self, gdf, gdf_dissolved=None, verbose=False):
        """
        Load a geopandas geoDataFrame and transform it into a DataFrame with spatio-temporal overview keys, aligned with other geolab data
        """
        if verbose:
            stopwatch_start = time.time()
        self.gdf = gdf
        assert(self.dt_col in self.gdf.columns)
        assert(self.geom_col in self.gdf.columns)
        assert(self.id_col in self.gdf.columns)
        
        # (A) CREATE COLUMNS FOR THE TEMPORAL KEYS USING ATTRIBUTES SUCH AS year, month, day
        for k in self.temporal_keys:
            self.gdf[k] = self.gdf[self.dt_col].apply(lambda x: getattr(x, k))
        if verbose:
            print('Time for (A) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
            
        # (B) CREATE A "spatial_key" COLUMN, ALIGNED WITH THE GEOLAB GRID
        # (B1) GET THE GEOLAB GRID FOR THE AREA OF INTEREST
        # Calculate dissolved aoi in order to create the quadtree keys
        if gdf_dissolved is None:
            self.gdf_dissolved = self.gdf[[self.id_col, self.geom_col]].drop_duplicates(
                subset=self.id_col)[[self.geom_col]].dissolve()
        else:
            self.gdf_dissolved = gdf_dissolved
        
        # Get Geodataframe of geolab cells (all on the same overview_level resolution)
        self.gdf_dissolved_cells = self._quadtreeGDF(self.gdf_dissolved.loc[0, self.geom_col])
        
        if verbose:
            print('gdf_dissolved_cells', len(self.gdf_dissolved_cells))
            print('Time for (B1) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
        # (B2) JOIN ORIGINAL GEODATAFRAME TO GET THE SPATIAL KEY FROM THE OVERVIEW CELLS 
        # Joining original GeoDataFram with overview cells may result in multiple copies of a row. 
        # intersection_policy specifies how to deal with the geometry in such cases

        # Drop duplicate geometries before calculating spatial overlays (e.g. when we have multiple timestamps for the same geometry) 
        self.gdf_unique = self.gdf[[self.id_col, self.geom_col]].drop_duplicates(subset=self.id_col).reset_index(drop=True)
        if verbose:
            print('len(gdf_unique)    ', len(self.gdf_unique))

        if self.intersection_policy=='cut':
            # Use geopandas.overlay (intersection) to cut the polygons at their intersection
            self.gdf_intersection = geopandas.overlay(self.gdf_unique ,self.gdf_dissolved_cells , how='intersection')
        elif self.intersection_policy=='original':
            # Use geopandas.sjoin (spatial join) to keep geometries intact
            self.gdf_intersection = geopandas.sjoin(self.gdf_unique, self.gdf_dissolved_cells, how='left', predicate='intersects').dropna(
                subset=['index_right']).drop(columns=['index_right'])
        elif self.intersection_policy=='both':
            # Cut the intersecting polygons but keep originals in another column: 'geometry_original'
            self.gdf_intersection = pandas.merge(
                geopandas.overlay(self.gdf_unique ,self.gdf_dissolved_cells , how='intersection'), 
                self.gdf_unique.rename(columns={self.geom_col: self.geom_col + '_original'}), 
                on=self.id_col
            )
        
        if verbose:
            print('Time for (B2) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
            
        # (C1) KEEP TRACK OF THE POLYGONS THAT INTERSECT THE OVERVIEW CELLS
        self.gdf_intersection[self.intersection_flag_col] = False
        self.gdf_intersection.loc[self.gdf_intersection[self.id_col].duplicated(keep=False), self.intersection_flag_col] = True
        
        # (C2) JOIN IN THE MANY COLUMNS WE DROPPED WHEN FORMING "self.gdf_unique".
        del self.gdf[self.geom_col]
        self.gdf_intersection = pandas.merge(self.gdf_intersection, self.gdf, on=self.id_col)
                
        # (C3) COMBINE THE TEMPORAL AND SPATIAL KEYS INTO ONE "overview_key".
        self.gdf_intersection[self.overview_key_col] = self._generate_overview_keys(self.gdf_intersection)
        
        # (C4) CREATE COLUMNS FOR SPATIAL PARTITIONING
        self._create_spatial_partition_columns(self.gdf_intersection)
        if verbose:
            print('Time for (C) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
        # (D) CREATE FILTER-KEY COLUMNS WITH KEYS FULLY CONTAINING THE GEOMETRY BOUNDING BOX AT VARIOUS RESOLUTION LEVELS
        self._create_filter_key_columns(self.gdf_intersection)
        if verbose:
            print('Time for (D) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
    def to_parquet(self, append=False, geometry_area=False, geometry_length=False, verbose=False):
        """
        Write GeoDataFrame to parquet
        """
        if verbose:
            stopwatch_start = time.time()
        self._get_partitions()
        
        # Save the parquet files by overview_key in a partitioned directory structure
        self.overview_keys = sorted(self.gdf_intersection.overview_key.unique())
        for overview_key in self.overview_keys:
            gdf_part, filepath = self._select_parquet_file(self.gdf_intersection, overview_key)
            if len(gdf_part)==0:
                print('WARNING: no data to write')
            else:
                # Add area and/or length columns if requested
                if geometry_area or geometry_length:
                    gdf_unique = gdf_part[
                        [self.id_col, self.geom_col, self.spatial_key_col, self.overview_level_col]
                    ].drop_duplicates(subset=self.id_col).reset_index(drop=True)
                    gdf_lambert_ea = self._reproject_to_local_equal_area_grid(gdf_unique)
                    if geometry_area:
                        gdf_unique[self.geom_area_col] = gdf_lambert_ea.area / 1e6 # units: [km2]
                    if geometry_length:
                        gdf_unique[self.geom_length_col] = gdf_lambert_ea.length / 1e3 # units: [km]
                    del gdf_unique[self.geom_col]
                    gdf_part = pandas.merge(gdf_part, gdf_unique, on=[self.id_col, self.spatial_key_col, self.overview_level_col])

                if append:
                    try:
                        # See if there is something already present under this overview key
                        df_existing = geopandas.read_parquet(filepath)
                    except:
                        pass
                    else:
                        # Merge the two by concatenating and dropping duplicates
                        gdf_part = pandas.concat([df_existing, gdf_part]).drop_duplicates().reset_index(drop=True)
                gdf_part.to_parquet(
                    path=filepath,
                    engine='pyarrow',
                    compression='snappy',
                    #partition_cols=self.partitions
                )
                
        # Write the settings to a json file
        self.write_vectorstore_settings()
        if verbose:
            print('Time for to_parquet in seconds', round(time.time()-stopwatch_start, 3))
            
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
        vs_settings['dimension_keys'] = self.dimension_keys
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
        vs_settings['quantiles'] = self.quantiles
        vs_settings['first'] = self.first
        vs_settings['timestamp_aggregation'] = self.timestamp_aggregation
        vs_settings['pyramid_levels'] = self.pyramid_levels

        json_path = os.path.join(self.vectorstore_directory, self.dataset+'.json')
        with open(json_path, 'w') as f:
            json.dump(vs_settings, f)
        
    def read_vectorstore_settings(self):
        json_path = os.path.join(self.vectorstore_directory, self.dataset+'.json')
        with open(json_path) as f:
            vs_settings = json.load(f)
        for k in vs_settings:
            setattr(self, k, vs_settings[k])
            
    def _calc_write_intersected_geometry(self):
        # Determine the geometries that intersect overview cells
        df_intersected = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            df1 = pandas.read_parquet(row['filepath'], columns=[self.overview_key_col, self.id_col, self.intersection_flag_col])
            df1 = df1.drop_duplicates().reset_index(drop=True)
            df_intersected.append(df1[df1[self.intersection_flag_col]])
        df_intersected = pandas.concat(df_intersected).reset_index(drop=True)
        
        # Generate a two column table (id_col and lists of overview_key_col) 
        df_intersected = df_intersected.groupby(self.id_col, group_keys=False)[self.overview_key_col].apply(list).reset_index()

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
        if self.timestamp_aggregation:
            timestamp_agg_cols = ['timestamp_start', 'timestamp_end']
        else:
            timestamp_agg_cols = []
        df_cell_statistics = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            # Read one-by-one so we won't have any memory issues
            df_parquet = pandas.read_parquet(row['filepath'], columns=self.timestamp_layers+self.numeric_layers)
            if self.timestamp_aggregation:
                # Groupby spatial only
                df_parquet['dummy_key'] = 1
                grp = df_parquet.groupby('dummy_key')
            else:
                # Groupby spatio-temporal
                grp = df_parquet.groupby(self.timestamp_layers)

            # Most of the descriptive statistics calculated here
            df1 = grp.describe(percentiles=self.quantiles).reset_index()
            if self.first:
                # "First" statistics requested
                df2 = grp.first().reset_index()
                # df1 has a multi-index, so we need one for df2 too
                df2.columns = pandas.MultiIndex.from_product([df2.columns, ['first']])
                df1 = df1.join(df2)
            if self.timestamp_aggregation:
                # Add start and end timestamp columns
                df1['timestamp_start'] = df_parquet[self.timestamp_layers[0]].min()
                df1['timestamp_end'] = df_parquet[self.timestamp_layers[0]].max()
            df1[self.overview_key_col] = row[self.overview_key_col]
            df_cell_statistics.append(df1)

        df_cell_statistics = pandas.concat(df_cell_statistics, axis=0).reset_index(drop=True)
        self._decompose_overview_keys(df_cell_statistics)

        for layer in self.numeric_layers: 
            # Pick relevant overviews by layer
            df_cell_statistics_layer = pandas.concat([df_cell_statistics[layer], df_cell_statistics[
                timestamp_agg_cols + self.temporal_keys + self.dimension_keys + [
                    self.spatial_key_col, self.overview_level_col, self.overview_key_col
                ]].droplevel(1, axis=1)
            ], axis=1)

            # Add geometry column
            gdf_cell_statistics_layer = pandas.merge(
                self._polyCells2geodataframe(df_cell_statistics_layer['spatial_key'].drop_duplicates()),
                df_cell_statistics_layer, 
                on=['spatial_key', 'overview_level'],
                how='right',
            )

            # Save to geo-parquet
            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
            gdf_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
 
    def _sanitize_dt_column(self, dt64):
        dt64 = dt64.apply(lambda x: x.tz_localize(None))
        ts = numpy.round((dt64 - numpy.datetime64('1970-01-01T00:00:00')) / numpy.timedelta64(1, 'us')) /1e6
        ts = ts.apply(lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=pytz.utc))
        return ts

    def _calc_write_overview_statistics_categorical(self):
        """
        Categorical layers (count, unique, top, frequency)
        For object data (e.g. strings), the result’s index will include count, unique, top, and freq. 
        The top is the most common value. The freq is the most common value’s frequency. 
        """
        if self.timestamp_aggregation:
            timestamp_agg_cols = ['timestamp_start', 'timestamp_end']
        else:
            timestamp_agg_cols = []
        df_cell_statistics = []
        for i, row in self.gdf_meta[[self.overview_key_col, 'filepath']].iterrows():
            # Read one-by-one so we won't have any memory issues
            df_parquet = pandas.read_parquet(row['filepath'], columns=self.timestamp_layers+self.categorical_layers)
            df_parquet = df_parquet.astype(object) # Make sure the columns will be interpreted as categorical
            if self.timestamp_aggregation:
                # Groupby spatial only
                df_parquet['dummy_key'] = 1
                grp = df_parquet.groupby('dummy_key')
            else:
                # Groupby spatio-temporal
                grp = df_parquet.groupby(self.timestamp_layers)

            # Most of the descriptive statistics calculated here
            df1 = grp.describe().reset_index()
            if self.first:
                # "First" statistics requested
                df2 = grp.first().reset_index()
                # df1 has a multi-index, so we need one for df2 too
                df2.columns = pandas.MultiIndex.from_product([df2.columns, ['first']])
                df1 = df1.join(df2)

            if self.timestamp_aggregation:
                # Add start and end timestamp columns
                df1['timestamp_start'] = df_parquet[self.timestamp_layers[0]].min()
                df1['timestamp_end'] = df_parquet[self.timestamp_layers[0]].max()
            df1[self.overview_key_col] = row[self.overview_key_col]
            df_cell_statistics.append(df1)

        df_cell_statistics = pandas.concat(df_cell_statistics, axis=0).reset_index(drop=True)
        self._decompose_overview_keys(df_cell_statistics)

        for layer in self.categorical_layers: 
            # Pick relevant overviews by layer
            df_cell_statistics_layer = pandas.concat([df_cell_statistics[layer], df_cell_statistics[
                timestamp_agg_cols + self.temporal_keys + self.dimension_keys + [
                    self.spatial_key_col, self.overview_level_col, self.overview_key_col
                ]].droplevel(1, axis=1)
            ], axis=1)

            # Add geometry column
            gdf_cell_statistics_layer = pandas.merge(
                self._polyCells2geodataframe(df_cell_statistics_layer['spatial_key'].drop_duplicates()),
                df_cell_statistics_layer, 
                on=['spatial_key', 'overview_level'],
                how='right',
            )

            # Save to geo-parquet
            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
            gdf_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
    
    def _calc_write_overview_histogram(self, histogram_layer):
        # Categorical and timestamp layers only
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
        
    def calc_overview_statistics(
        self, 
        numeric_layers=None, 
        timestamp_layers=None, 
        categorical_layers=None, 
        quantiles=None, 
        first=None,
        timestamp_aggregation=None,
    ):
        """
        Caluculate the overview layer statistics and write them out to parquet
        """
        if numeric_layers is not None:
            self.numeric_layers = numeric_layers 
        if timestamp_layers is not None:
            self.timestamp_layers = timestamp_layers 
        if categorical_layers is not None:
            self.categorical_layers = categorical_layers 
        if quantiles is not None:
            self.quantiles = quantiles 
        if first is not None:
            self.first = first 
        if timestamp_aggregation is not None:
            self.timestamp_aggregation = timestamp_aggregation 
        
        # Dump the settings to a json file (existing will be overwritten)
        self.write_vectorstore_settings()
        
        # Assert that we don't have layers in multiple categories at once
        all_layers = self.numeric_layers + self.timestamp_layers + self.categorical_layers
        assert(len(all_layers)==len(list(set(all_layers))))
        
        # Make sure gdf_meta is present
        try:
            self.gdf_meta
        except:
            self.metadata_from_parquet()
        
        if len(self.numeric_layers)>0:
            self._calc_write_overview_statistics_numeric()
        if len(self.categorical_layers)>0:
            self._calc_write_overview_statistics_categorical()
            for categorical_layer in self.categorical_layers:
                self._calc_write_overview_histogram(categorical_layer)
        if len(self.timestamp_layers)>0:
            for timestamp_layer in self.timestamp_layers:
                self._calc_write_overview_histogram(timestamp_layer)
                    
        # Summary of intersected geometries (at overview cell boundaries)
        self._calc_write_intersected_geometry()
            
    def read_overview_statistics(self, layer, **kwargs):
        filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
        try:
            # If needed, transpose after reading the parquet instead of in the parquet file itself, 
            # because parquet has limitations in storing mixed datatypes in columns.
            #df = pandas.read_parquet(filepath, **kwargs)
            df = geopandas.read_parquet(filepath, **kwargs) # Note reading using geopandas is slightly slower
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df

    def read_overview_histogram(self, layer, **kwargs):
        filepath = os.path.join(self.overview_directory, 'overview_histogram_' + layer + '.parquet')
        try:
            df = pandas.read_parquet(filepath, **kwargs)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
    
    def _pyramid_groupby_cols(self, pyramid_level):
        cols = ['pyramid_level' + str(pyramid_level)] + self.temporal_keys + self.dimension_keys
        return cols
    
    def _numeric_pyramids(self, numeric_layer, verbose=False):
        if verbose:
            print('numeric_layer', numeric_layer)
            
        if self.timestamp_aggregation:
            timestamp_keys = ['timestamp_start', 'timestamp_end']
        else:
            timestamp_keys = self.temporal_keys
            
        # Read the detailed overview statistics at the overview level
        df_cell_statistics = self.read_overview_statistics(numeric_layer)

        # Keep a limited number of statistics and add the pyramid keys
        df = df_cell_statistics[['count', 'mean', 'min', 'max'] + timestamp_keys] # + self.dimension_keys
        df = df.join(self.df_pyramid_keys)

        # Calculate the sum from the count and mean
        df['sum'] = df['count'] * df['mean']

        # Correct the count and sum statistics for polygons intersecting multiple overview cells 
        # (otherwise they would be counted multiple times)
        df_multiple = []
        for i, row in self.df_intersected_T.iterrows():
            overview_key = row[self.overview_key_col]
            filepath = row['filepath']
            lst_filter_id = row[self.id_col]
            df_tmp = pandas.read_parquet(
                filepath, 
                columns=[self.overview_key_col, self.id_col, numeric_layer] + timestamp_keys # + self.dimension_keys
            )  
            df_tmp = df_tmp[df_tmp[self.id_col].isin(lst_filter_id)]
            df_multiple.append(df_tmp)

        df_multiple = pandas.concat(df_multiple).reset_index(drop=True)
        df_multiple = df_multiple.set_index(self.overview_key_col)

        for pyramid_level in self.pyramid_levels:
            cols = self._pyramid_groupby_cols(pyramid_level)

            # Naively aggregate count, sum, min, max
            df_naive = []
            if verbose:
                print('pyramid grouping columns', cols)
            df_naive.append(df[cols + ['count']].groupby(cols).sum())
            df_naive.append(df[cols + ['sum']].groupby(cols).sum())
            df_naive.append(df[cols + ['min']].groupby(cols).min())
            df_naive.append(df[cols + ['max']].groupby(cols).max())
            df_naive = pandas.concat(df_naive, axis=1).reset_index()

            # First group the intersecting geometries by overview cell level 
            grp = df_multiple[[self.id_col, numeric_layer]].reset_index().groupby([self.id_col, self.overview_key_col])
            df1_count = grp.count().rename(columns={numeric_layer: 'count'}).reset_index()
            df1_sum = grp.sum().rename(columns={numeric_layer: 'sum'}).reset_index()
            df1 = pandas.merge(df1_sum, df1_count, on=['geom_id', self.overview_key_col])
            df1 = df1.set_index(self.overview_key_col).join(self.df_pyramid_keys)

            # Groupby pyramid cols (level and partition keys) and individual geometries
            grp = df1[[self.id_col, 'sum'] + cols].groupby([self.id_col] + cols)
            df2_count = grp.count().rename(columns={'sum': 'count'}).reset_index()
            df2_sum = grp.sum().reset_index()
            df2 = pandas.merge(df2_count, df2_sum, on=[self.id_col] + cols)
            df2['overcount'] = df2['count'] - 1
            df2['oversum'] = df2['overcount'] * df2['sum'] / df2['count']
            df2 = df2.groupby(cols)[['overcount', 'oversum']].sum().reset_index()
            df_corrected = pandas.merge(df_naive, df2, on=cols, how='left').fillna(0)
            df_corrected['count'] = df_corrected['count'] - df_corrected['overcount']
            df_corrected['sum'] = df_corrected['sum'] - df_corrected['oversum']
            df_corrected['mean'] = df_corrected['sum'] / df_corrected['count']
            #del df_corrected['sum']
            df_corrected['count'] = df_corrected['count'].astype(int)
            df_corrected = df_corrected[cols + ['count', 'mean', 'min', 'max']].sort_values(by=cols).set_index(cols)

            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + numeric_layer + '.parquet')
            df_corrected.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
            
    def _histogram_pyramids(self, categorical_layer, verbose=False):
        if verbose:
            print('histogram_layer', categorical_layer)
            
        # Get the histogram at the overview level
        df = self.read_overview_histogram(categorical_layer)
        df = df.T
        histogram_categories = df.columns

        # Get the temporal (and dimension) keys from the overview key
        df.index.name = self.overview_key_col
        df = df.reset_index()
        self._decompose_overview_keys(df)
        #del df[self.overview_level_col]
        #del df[self.spatial_key_col]
        df = df.set_index(self.overview_key_col)

        df = df.join(self.df_pyramid_keys)

        for pyramid_level in self.pyramid_levels:
            # Naively aggregate the counts for each historgram category
            df1 = []
            cols = self._pyramid_groupby_cols(pyramid_level)
            if verbose:
                print('pyramid grouping columns', cols)
            for category in histogram_categories:
                df1.append(df[cols + [category]].groupby(cols).sum())
            df1 = pandas.concat(df1, axis=1)

            # Correct the statistics for polygons intersecting multiple overview cells 
            # (otherwise they would be counted multiple times)
            df3 = []
            for i, row in self.df_intersected_T.iterrows():
                overview_key = row[self.overview_key_col]
                filepath = row['filepath']
                lst_filter_id = row[self.id_col]
                df2 = pandas.read_parquet(
                    filepath, 
                    columns=[self.overview_key_col, self.id_col, categorical_layer] + self.temporal_keys + self.dimension_keys
                )
                df2 = df2[df2[self.id_col].isin(lst_filter_id)]
                df3.append(df2)

            if len(df3)>0:
                df3 = pandas.concat(df3).reset_index(drop=True)
                df3 = df3.set_index(self.overview_key_col)
                df3 = df3.join(self.df_pyramid_keys)

                # Groupby id and pyramid level together to find relevant overcounts
                df4 = []
                df4.append(df3[cols + [self.id_col, categorical_layer]].groupby([self.id_col] + cols).count().rename(
                    columns={categorical_layer: 'count'}
                ))
                df4.append(df3[cols + [self.id_col, categorical_layer]].groupby([self.id_col] + cols).first().rename(
                    columns={categorical_layer: 'category'}
                ))

                df4 = pandas.concat(df4, axis=1)

                # Prepare for summing up by pyramid key
                df4 = df4[df4['count']>1]
                df4 = df4.reset_index()
                df4['overcount'] = df4['count']-1

                # Sum up the overcount and oversum by pyramid level key
                df5 = df4[cols + ['category', 'overcount']].groupby(cols + ['category']).sum()

                # Join with the original statistics and correct them
                df1.columns.name = 'category'
                df1_stacked = pandas.DataFrame(df1.stack().rename('count'))
                df6 = df1_stacked.join(df5)
                df6['count'] = df6['count'] - df6['overcount'].fillna(0)
                df6 = df6[['count']]
                df6 = df6.unstack()['count'].astype(int)
                df6.columns.name = None
                #df6 = df6.reset_index()
            else:
                df6=df1

            # Integer or float value column names are not allowed in parquet
            df6.columns = [str(c) for c in df6.columns]
            
            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_histogram_' + categorical_layer + '.parquet')
            df6.to_parquet(path=filepath, engine='pyarrow', compression='snappy')            
            
    def _categorical_pyramids(self, categorical_layer, verbose=False):
        if verbose:
            print('categorical_layer', categorical_layer)
        for pyramid_level in self.pyramid_levels:
            cols = self._pyramid_groupby_cols(pyramid_level)
            if verbose:
                print('pyramid grouping columns', cols)
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

            df.columns.name = 'categories'
            df1 = df[[]].copy()
            df1['count'] = df.sum(axis=1)
            df_stacked = df[df!=0].stack().reset_index()
            df1['unique'] = df_stacked.groupby(cols).count()['categories']
            df1['top'] = df.idxmax(axis=1)
            df1['freq'] = df_stacked.groupby(cols).max()[0].astype(int)
            df1.columns.name = None

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
        self.df_intersected_T = self.df_intersected_T.groupby(self.overview_key_col, group_keys=False)[self.id_col].apply(list).reset_index()
        self.df_intersected_T = pandas.merge(self.df_intersected_T, self.gdf_meta[[self.overview_key_col, 'filepath']], on=self.overview_key_col)
        
        for numeric_layer in self.numeric_layers:
            self._numeric_pyramids(numeric_layer, verbose=verbose)
            
        for categorical_layer in self.categorical_layers:
            self._histogram_pyramids(categorical_layer, verbose=verbose)
            self._categorical_pyramids(categorical_layer, verbose=verbose)
            
        # Update the vectorstore settings
        self.write_vectorstore_settings()
        
    def read_pyramid(self, pyramid_level, layer, histogram=False, **kwargs):
        if histogram:
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_histogram_' + layer + '.parquet')
        else:
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + layer + '.parquet')
        try:
            # If needed, transpose after reading the parquet instead of in the parquet file itself, 
            # because parquet has limitations in storing mixed datatypes in columns.
            df = pandas.read_parquet(filepath, **kwargs)
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
        self._decompose_overview_keys(df_meta)
            
        # Cast to geopandas by infering the geometry column (cell boxes) from the overview level and spatial key
        self.gdf_meta = self._polyCells2geodataframe(df_meta[self.spatial_key_col].drop_duplicates().to_numpy())
        self.gdf_meta = pandas.merge(self.gdf_meta, df_meta, on=[self.spatial_key_col, self.overview_level_col], how='right')

            
    def query_single_parquet(
        self, query_latitude, query_longitude, query_dt, spatial_filter=False, temporal_filter=False, columns=None,
        geometry_area=False, geometry_length=False
    ):
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
        _, filepath = self._select_parquet_file(self.query_df_meta, overview_key)

        # Load a complete single parquet file
        try:
            gdf_query = self._read_parquet(filepath, columns=columns, geometry_area=geometry_area, geometry_length=geometry_length)
        except FileNotFoundError as e:
            gdf_query = geopandas.geodataframe.GeoDataFrame()

        # Filter the query if requested
        if temporal_filter:
            gdf_query=gdf_query[gdf_query[self.dt_col]==query_dt]

        if spatial_filter:
            gdf_query = gdf_query[gdf_query.intersects(shapely.geometry.Point(query_longitude, query_latitude))]
            
        return gdf_query.reset_index(drop=True)
        
    def _query_worker(self, overview_key, columns=None):
        """
        Partial query using specific overview_key, corresponding to one parquet file.
        """
        _, filepath = self._select_parquet_file(self.query_df_meta, overview_key)
        gdf = self._read_parquet(
            filepath, self.query_dt_start, self.query_dt_end, columns=columns, 
            geometry_area=self.geometry_area, geometry_length=self.geometry_length, verbose=False
        )
        if len(gdf)!=0:
            if not self.query_polygon.contains(shapely.geometry.box(*gdf.total_bounds)):
                # Spatial filtering may be necessary
                gdf_unique = gdf[[self.id_col, self.geom_col, self.overview_key_col]].drop_duplicates(
                    subset=[self.id_col, self.overview_key_col]
                )
                if self.query_intersection_policy=='cut':
                    # return intersection (cut polygons)
                    gdf_unique = geopandas.overlay(
                        gdf_unique,
                        self.query_gdf_polygon, 
                        how='intersection'
                    ).reset_index(drop=True)
                elif self.query_intersection_policy=='original':
                    # Return intersecting polygons intact
                    if self.intersection_policy=='cut':
                        # To do: use geometry_id to find all locations of this polygon and stitch together.
                        print('NOT IMPLEMENTED WARNING: Query asks for original polygons but we are returning the cut ones.')
                        gdf_unique = geopandas.sjoin(
                            gdf_unique, 
                            self.query_gdf_polygon, 
                            how='left', 
                            predicate='intersects'
                        ).dropna(subset=['index_right']).drop(columns=['index_right'])
                    elif self.intersection_policy=='original':
                        # Data has been saved as complete polygons
                        gdf_unique = geopandas.sjoin(
                            gdf_unique, 
                            self.query_gdf_polygon, 
                            how='left', 
                            predicate='intersects'
                        ).dropna(subset=['index_right']).drop(columns=['index_right'])
                    elif self.intersection_policy=='both':
                        # Pick the original geometry column with complete polygons
                        gdf_unique = geopandas.sjoin(
                            gdf_unique.set_geometry(self.geom_col + '_original', crs=4326), 
                            self.query_gdf_polygon, 
                            how='left', 
                            predicate='intersects'
                        ).dropna(subset=['index_right']).drop(columns=['index_right'])
                        
                # Get back the full width of the dataframe
                del gdf[self.geom_col]
                gdf = pandas.merge(gdf_unique, gdf, on=[self.id_col, self.overview_key_col])
                        
        if len(gdf)>0:
            return gdf
        else:
            return None

    def query_vectorstore(
        self,
        query_polygon=None, 
        query_dt_start=None,  
        query_dt_end=None, 
        dimensions=None,
        query_intersection_policy=None, 
        columns=None,
        geometry_area=False, 
        geometry_length=False,
        n_workers=1,
        verbose=False,
    ):
        """
        Query the parquet vector store (intersecting in time and space)
        May be incommensurate with cells, span multiple cells, or may be incommensurate with temporal key
        """
        # DEBUG: NEED TO IMPLEMENT DIMENSIONS

        if verbose:
            stopwatch_start = time.time()

        self.query_polygon             = self.COMPLETE_WORLD if query_polygon is None else query_polygon
        self.query_dt_start            = self.MIN_DT if query_dt_start is None else query_dt_start
        self.query_dt_end              = self.MAX_DT if query_dt_end is None else query_dt_end
        self.query_intersection_policy = self.QUERY_INTERSECTION_POLICY if query_intersection_policy is None else query_intersection_policy
        self.geometry_area             = geometry_area
        self.geometry_length           = geometry_length

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
                    
        if verbose:
            print('Time for (A) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # Spatial part of the overview key
        query_spatial_keys = self.quadtree(self.query_polygon, self.overview_level)
        df_spatial = pandas.DataFrame()
        for i, query_spatial_key in enumerate(query_spatial_keys):
            df_spatial.loc[i, self.spatial_key_col] = query_spatial_key
            
        if verbose:
            print('Time for (B) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # Cartesian product of spatial and temporal parts
        if len(self.query_df_meta)>0:
            self.query_df_meta = self.query_df_meta.merge(df_spatial, how='cross')
        else:
            self.query_df_meta = df_spatial

        # Pairs level of the overview key
        self.query_df_meta[self.overview_level_col] = self.overview_level

        # Convert the floats to int
        self.query_df_meta = self.query_df_meta.astype(int)
        
        if verbose:
            print('Time for (C) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # Generate the overview keys
        self.query_df_meta[self.overview_key_col] = self._generate_overview_keys(self.query_df_meta)
        
        # Generate the spatial partition columns
        self._create_spatial_partition_columns(self.query_df_meta)

        if verbose:
            print('Time for (D) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        overview_keys_requested = set(self.query_df_meta[self.overview_key_col])
        try:
            overview_keys_present = set(self.gdf_meta[self.overview_key_col])
        except:
            # gdf_meta likely not present yet, so generate it
            self.metadata_from_parquet()
            overview_keys_present = set(self.gdf_meta[self.overview_key_col])
        overview_keys = sorted(overview_keys_requested & overview_keys_present)
        
        if verbose:
            print('Time for (E) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        if n_workers>1:
            query_worker_part = partial(self._query_worker, columns=columns)
            with ProcessPool(nodes=n_workers) as p:
                lst_gdf = p.map(query_worker_part, overview_keys)
                lst_gdf = [x for x in lst_gdf if x is not None]
            """
            with Pool(n_workers) as p:
                lst_gdf = p.map(query_worker_part, overview_keys)
                lst_gdf = [x for x in lst_gdf if x is not None]
            """
        else:
            lst_gdf = []
            for overview_key in overview_keys:
                gdf = self._query_worker(overview_key, columns=columns)
                if gdf is not None:
                    lst_gdf.append(gdf)
                    
        if verbose:
            print('Time for (F) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
        if len(lst_gdf)==0:
            gdf_query = geopandas.geodataframe.GeoDataFrame()
            print('WARNING: no data found in query area')
        else:
            gdf_query = pandas.concat(lst_gdf).reset_index(drop=True)
            if verbose:
                print('lst_gdf', len(lst_gdf))
                print('gdf_query before merging duplicated', len(gdf_query))
            
            # Some objects may be duplicated because they are saved in multiple overview cells. 
            # If we saved intersections of polygons, we need to stitch those together
            gdf_unique = gdf_query[[self.id_col, self.geom_col, self.overview_key_col]].drop_duplicates(
                subset=[self.id_col, self.overview_key_col]
            )
            duplicated = gdf_unique[[self.id_col, self.geom_col]].groupby(self.id_col).count()
            duplicated = list(duplicated[duplicated[self.geom_col]>1].index)
            if len(duplicated)>0:
                gdf_duplicated = gdf_unique[gdf_unique[self.id_col].isin(duplicated)]
                if self.intersection_policy=='cut':
                    # Removing the annoying buffer warning
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Drop the duplicated rows and dissolve the geometries
                        gdf_duplicated = pandas.merge(
                            gdf_duplicated.drop_duplicates(subset=self.id_col).drop(columns=self.geom_col), 
                            gdf_duplicated.dissolve(by=self.id_col).buffer(1e-10).buffer(-1e-10).reset_index().rename(columns={0: self.geom_col}), 
                            on=self.id_col,
                            how='left'
                        )
                elif self.intersection_policy=='original':
                    gdf_duplicated = gdf_duplicated.drop_duplicates(subset=self.id_col)
                elif self.intersection_policy=='both':
                    gdf_duplicated = gdf_duplicated.drop_duplicates(subset=self.id_col)
                    gdf_duplicated[self.geom_col] = gdf_duplicated[self.geom_col + '_original'] 
                    if self.query_intersection_policy=='cut':
                        # return intersection with query polygon
                        gdf_duplicated = geopandas.overlay(
                            gdf_duplicated,
                            self.query_gdf_polygon, 
                            how='intersection'
                        ).reset_index(drop=True)
                        
                # Get back the full width of the dataframe
                gdf_tmp = gdf_query[gdf_query[self.id_col].isin(duplicated)]
                del gdf_tmp[self.geom_col]
                gdf_duplicated = pandas.merge(
                    gdf_duplicated,
                    gdf_tmp,
                    on=[self.id_col, self.overview_key_col],
                )
                
                gdf_query = pandas.concat([
                    gdf_query[~gdf_query[self.id_col].isin(duplicated)], 
                    gdf_duplicated,
                ])
            gdf_query = gdf_query.sort_values(by=[self.dt_col, self.spatial_key_col]).reset_index(drop=True)

            if verbose:
                print('gdf_query after merging duplicated ', len(gdf_query))

        if verbose:
            print('Time for (G) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
            
        return gdf_query
    
    def _is_unique(self, s):
        a = s.to_numpy()
        return a[0], (a[0] == a).all()
    
    def _overlay_worker(self, gdf_right, columns=None, verbose=False):
        """
        Partial overlay using specific overview_key, corresponding to one parquet file.
        """
        # Get the unique overview_key
        overview_key, is_unique = self._is_unique(gdf_right['overview_key'])
        assert(is_unique)
        _, filepath = self._select_parquet_file(gdf_right, overview_key)
        gdf_left = self._read_parquet(
            filepath, self.query_dt_start, self.query_dt_end, columns=columns, 
            geometry_area=self.geometry_area, geometry_length=self.geometry_length, verbose=verbose,
        )
        if len(gdf_left)>0:
            if self.how=='intersection':
                # return intersection (cut polygons)
                gdf = geopandas.overlay(
                    gdf_left,
                    gdf_right[['polygon_id', 'geometry']], 
                    how='intersection'
                ).reset_index(drop=True)
            elif self.how=='left':
                # Return intersecting polygons intact
                if self.intersection_policy=='cut':
                    # To do: use geometry_id to find all locations of this polygon and stitch together.
                    print('NOT IMPLEMENTED WARNING: Asking for original polygons but we are returning the cut ones.')
                    gdf = geopandas.sjoin(
                        gdf_left, 
                        gdf_right[['polygon_id', 'geometry']], 
                        how='left', 
                        predicate='intersects'
                    ).dropna(subset=['index_right']).drop(columns=['index_right'])
                elif self.intersection_policy=='original':
                    # Data has been saved as complete polygons
                    gdf = geopandas.sjoin(
                        gdf_left, 
                        gdf_right[['polygon_id', 'geometry']], 
                        how='left', 
                        predicate='intersects'
                    ).dropna(subset=['index_right']).drop(columns=['index_right'])
                elif self.intersection_policy=='both':
                    # Pick the original geometry column with complete polygons
                    gdf = geopandas.sjoin(
                        gdf_left.set_geometry(self.geom_col + '_original', crs=4326), 
                        gdf_right[['polygon_id', 'geometry']], 
                        how='left', 
                        predicate='intersects'
                    ).dropna(subset=['index_right']).drop(columns=['index_right'])
            if len(gdf)>0:
                return gdf
        return None

    def overlay(
        self,
        gdf_right, 
        how=None,
        query_polygon=None, 
        query_dt_start=None,  
        query_dt_end=None, 
        columns=None,
        geometry_area=False, 
        geometry_length=False,
        n_workers=1,
        verbose=False,
    ):
        # DEBUG: NEED TO IMPLEMENT DIMENSIONS

        self.how             = 'intersection' if how is None else how
        assert(self.how in ['intersection', 'left', 'right'])
        self.query_polygon   = self.COMPLETE_WORLD if query_polygon is None else query_polygon
        self.query_dt_start  = self.MIN_DT if query_dt_start is None else query_dt_start
        self.query_dt_end    = self.MAX_DT if query_dt_end is None else query_dt_end
        self.geometry_area   = geometry_area
        self.geometry_length = geometry_length

        self.read_vectorstore_settings()
        self.query_gdf_polygon = self.polygons2geodataframe([self.query_polygon], self.geom_col)
        
        # Find the relevant overview cells and cut up the gdf_right geometries into parts commensurate with overview_cell bounds
        gdf_right2 = geopandas.overlay(self.gdf_meta, gdf_right, how='intersection')
        overview_keys = list(gdf_right2['overview_key'].unique())
        
        if verbose:
            stopwatch_start = time.time()
        lst_gdf2 = []
        lst_gdf_bypass = []
        for i, overview_key in enumerate(overview_keys):
            gdf2 = gdf_right2[gdf_right2['overview_key']==overview_key]
            
            # We may be able to save some time in circumstances where the gdf_right geometries are larger than an overview cell
            if self.intersection_policy in ['cut', 'both']:
                # Check if any (or all) of the intersected geometries completely fill an overview cell
                full_containment = gdf2.contains(self.gdf_meta.loc[self.gdf_meta['overview_key']==overview_key, 'geometry'].values[0])
                # Currently only considering a special case if full_containment.all() and the overlay step can be skipped completely.
                if full_containment.all():
                    if verbose:
                        print('full_containment @ overview_key', overview_key)
                    _, filepath = self._select_parquet_file(gdf2, overview_key)
                    gdf_left = self._read_parquet(
                        filepath, self.query_dt_start, self.query_dt_end, columns=columns, 
                        geometry_area=self.geometry_area, geometry_length=self.geometry_length, verbose=verbose,
                    )
                    gdf_bypass = gdf_left.merge(gdf2['polygon_id'], how='cross')
                    lst_gdf_bypass.append(gdf_bypass)
                else:
                    lst_gdf2.append(gdf2)
            else:
                lst_gdf2.append(gdf2)
        if verbose:
            print('Time for full_containment in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        lst_gdf = []
        if n_workers==1:
            #gdf = gdf2.groupby('overview_key').apply(lambda x: self._overlay_worker(x, columns=columns, verbose=verbose)).reset_index(drop=True)
            # Work on each overview cell sequentially
            for gdf2 in lst_gdf2:
                gdf = self._overlay_worker(gdf2, columns=columns, verbose=verbose)
                if gdf is not None:
                    lst_gdf.append(gdf)
        else:
            # Do the heavy lifting in parallel
            _overlay_worker_part = partial(self._overlay_worker, columns=columns)
            with Pool(n_workers) as p:
                lst_gdf = p.map(_overlay_worker_part, lst_gdf2)
                lst_gdf = [x for x in lst_gdf if x is not None]
        
        gdf = pandas.concat(lst_gdf + lst_gdf_bypass).reset_index(drop=True)

        # Deal with multiple rows for same geom_id, polygon_id combination
        if self.intersection_policy=='original':
            # Simply drop the duplicates
            gdf = gdf.drop_duplicates(subset=[self.id_col, 'polygon_id']).reset_index(drop=True)
        elif self.intersection_policy=='cut':
            # Find out which geometries need to be dissolved
            gdf_duplicated = gdf.groupby([self.id_col, 'polygon_id']).count()['overview_key'].reset_index()
            gdf_duplicated = gdf_duplicated[gdf_duplicated['overview_key']>1]
            if len(gdf_duplicated)>0:
                gdf_duplicated = gdf[gdf[self.id_col].isin(gdf_duplicated[self.id_col])].sort_values(by=self.id_col).reset_index(drop=True)
                # Removing the annoying buffer warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Dissolve the geometries
                    gdf_duplicated = pandas.merge(
                        gdf_duplicated.drop_duplicates(subset=[self.id_col, 'polygon_id']).drop(columns=self.geom_col), 
                        gdf_duplicated.dissolve(by=self.id_col).buffer(1e-10).buffer(-1e-10).reset_index().rename(columns={0: self.geom_col}), 
                        on=self.id_col,
                        how='left'
                    )
                # Now we drop all of the duplicated rows ...
                if verbose:
                    print('gdf before dropping duplicates: ', len(gdf))
                gdf = gdf.drop_duplicates(subset=[self.id_col, 'polygon_id'], keep=False)
                if verbose:
                    print('gdf after dropping duplicates:  ', len(gdf))
                # ... and concat the dissolved ones
                gdf = pandas.concat([gdf, gdf_duplicated])
                if verbose:
                    print('gdf after adding dissolved ones:', len(gdf))

        # Final sort
        return gdf.sort_values(by=['polygon_id', self.id_col]).reset_index(drop=True)