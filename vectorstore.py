import os
import warnings
from glob import glob
import time
import math
import numpy
import pandas
from datetime import datetime, timedelta
import pytz
import geopandas
import shapely
from functools import partial
import json
from multiprocessing import Pool
from pathos.pools import ProcessPool

import pairs_quadtree


class Vectorstore():
    """
    Vectorstore Class for handling indexed (e.g. fused to the PAIRS grid) vector data in parquet files
    """
    
    # Default values for variables
    VECTORSTORE_DIRECTORY      = '/data/vector/vectorstore/'

    TEMPORAL_KEYS              = []  #['year']
    TEMPORAL_PARTITIONS        = []
    
    DIMENSION_KEYS             = []

    # PAIRS resolution level for the Parquet file
    # (PAIRS level 6 ~ 1000km, level 9 ~ 100km, level 13 ~ 10km, level 16 ~ 1km, level 23 ~ 10m)
    SPATIAL_LEVEL              = 8
    SPATIAL_PARTITION_IDENTIFIER = 'partition_level'
    SPATIAL_PARTITION_LEVELS   = [] #[6]
    
    # Order of the partitions in the stored directory structure
    PARTITION_ORDER            = 'temporal_before_spatial' #'spatial_before_temporal'

    # Filter-key columns at various resolution levels (above the cell level)
    FILTER_KEY_LEVELS          = []
    
    # Policy how to deal with polygons that intersect spatial cells ("cut", "original", "both")
    INTERSECTION_POLICY        = 'cut' # 'original', 'both'
    INTERSECTION_FLAG_COL      = 'intersection_flag'
    
    # GeoDataFrame colum conventions (used when loading data from gdf or vectorstore)
    DT_COL                     = 'timestamp'
    GEOM_COL                   = 'geometry'  # Geopandas relies on this column being named 'geometry', so enforce this for all tables
    ID_COL                     = 'geometry_id'
    SPATIAL_LEVEL_COL          = 'spatial_level'
    SPATIAL_KEY_COL            = 'spatial_key'
    COMPOSITE_KEY_COL          = 'composite_key'
    GEOM_AREA_COL              = 'geom_area'
    GEOM_LENGTH_COL            = 'geom_length'
    
    # Min and max datetime conventions (used when querying data)
    MIN_DT = datetime(1, 1, 1).replace(tzinfo=pytz.utc)
    MAX_DT = datetime(9999, 12, 31).replace(tzinfo=pytz.utc)
    
    # Other query conventions
    COMPLETE_WORLD             = shapely.geometry.box(-180, -90, 180, 90)
    QUERY_INTERSECTION_POLICY  = 'cut' #'original'
    
    def __init__(
        self,
        dataset,
        vectorstore_directory = None,
        temporal_keys = None,
        temporal_partitions = None,
        dimension_keys = None,
        spatial_level = None,
        spatial_partition_identifier = None,
        spatial_partition_levels = None,
        partition_order = None,
        filter_key_levels = None,
        intersection_policy = None,
        intersection_flag_col = None,
        dt_col = None,
        geom_col = None, 
        id_col = None,
        spatial_level_col = None, 
        spatial_key_col = None, 
        composite_key_col = None,
        geom_area_col = None, 
        geom_length_col = None, 
    ):
        
        # Dataset name
        self.dataset                      = dataset
        
        # Vectorstore base directory
        self.vectorstore_directory        = self.VECTORSTORE_DIRECTORY if vectorstore_directory is None else vectorstore_directory
        
        # Dataset directory is subfolder in vectorstore_directory 
        self.dataset_directory            = os.path.join(self.vectorstore_directory, self.dataset)
        if not os.path.exists(self.dataset_directory): os.makedirs(self.dataset_directory)
 
        # Temporal keys and partitions
        self.temporal_keys                = self.TEMPORAL_KEYS if temporal_keys is None else temporal_keys
        self.temporal_partitions          = self.TEMPORAL_PARTITIONS if temporal_partitions is None else temporal_partitions
        # Make sure that the temporal partiting keys are present in temporal_keys and used to define parquet file ranges
        missing_keys                      = [k for k in self.temporal_partitions if k not in self.temporal_keys]
        assert(len(missing_keys)==0)
        
        # Dimensions
        self.dimension_keys               = self.DIMENSION_KEYS if dimension_keys is None else dimension_keys

        # Spatial keys and partitions
        self.spatial_level                = self.SPATIAL_LEVEL if spatial_level is None else spatial_level
        self.spatial_partition_identifier = self.SPATIAL_PARTITION_IDENTIFIER if spatial_partition_identifier is None else spatial_partition_identifier
        self.spatial_partition_levels     = self.SPATIAL_PARTITION_LEVELS if spatial_partition_levels is None else spatial_partition_levels
        self.partition_order              = self.PARTITION_ORDER if partition_order is None else partition_order
        assert(self.partition_order in ['temporal_before_spatial', 'spatial_before_temporal'])
        # Make sure that it's sorted and the highest level is <= spatial level 
        self.spatial_partition_levels     = sorted(self.spatial_partition_levels)
        if len(self.spatial_partition_levels)>0:
            assert(max(self.spatial_partition_levels)<=self.spatial_level)
        self.spatial_partitions           = [self.spatial_partition_identifier + str(l) for l in self.spatial_partition_levels]
        
        # Filter-key columns at various resolution levels (above the spatial cell level)
        self.filter_key_levels            = self.FILTER_KEY_LEVELS if filter_key_levels is None else filter_key_levels 
        self.filter_key_levels            = [int(f) for f in self.filter_key_levels] # in order to be json serializable
        self.filter_key_levels            = sorted(self.filter_key_levels)

        # Policy how to deal with polygons that intersect spatial cells ("cut", "original", "both")
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
        self.spatial_level_col = self.SPATIAL_LEVEL_COL if spatial_level_col is None else spatial_level_col
        self.spatial_key_col = self.SPATIAL_KEY_COL if spatial_key_col is None else spatial_key_col
        self.composite_key_col = self.COMPOSITE_KEY_COL if composite_key_col is None else composite_key_col
        self.geom_area_col = self.GEOM_AREA_COL if geom_area_col is None else geom_area_col
        self.geom_length_col = self.GEOM_LENGTH_COL if geom_length_col is None else geom_length_col
        
        # Overview statistics variables (will be set later)
        self.numeric_layers = None
        self.timestamp_layers = None
        self.categorical_layers = None
        self.quantiles = None
        self.first = None
        self.timestamp_aggregation = None

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
        polyQuadTree, _ = pairs_quadtree.quadTreePAIRS_dfs(poly, max_level=level)
        # Get all the keys on the same resolution level
        polyCells = pairs_quadtree.quadTreeCellsPAIRS_dfs(polyQuadTree, level)
        return polyCells
    
    def quadtree2(self, poly):
        """
        version of quadtree that works better for complicated polygons
        """
        # Get all the cells within the total bounds
        bounds_cells = self.quadtree(shapely.box(*poly.bounds), self.spatial_level)
        gdf_bounds_cells = self._polyCells2geodataframe(bounds_cells)
        # Filter the cells that overlap the polygon
        mask = gdf_bounds_cells.intersects(poly)
        df_masked = gdf_bounds_cells[mask].reset_index(drop=True)
        polyCells = df_masked['spatial_key'].to_list()
        return polyCells

    def _quadtreeGDF(self, poly):
        """
        version of quadtree that works better for sparse polygons and returns a GeoDataFrame
        """
        # Get all the cells within the total bounds
        polyCells = self.quadtree(poly, self.spatial_level)
        gdf_cells = self._polyCells2geodataframe(polyCells)
        return gdf_cells

    def _quadtreeGDF2(self, poly):
        """
        version of quadtree that works better for complicated polygons and returns a GeoDataFrame
        """
        # Get all the cells within the total bounds
        bounds_cells = self.quadtree(shapely.box(*poly.bounds), self.spatial_level)
        gdf_bounds_cells = self._polyCells2geodataframe(bounds_cells)
        # Filter the cells that overlap the polygon
        mask = gdf_bounds_cells.intersects(poly)
        gdf_masked = gdf_bounds_cells[mask].reset_index(drop=True)
        #polyCells = gdf_masked[self.spatial_key_col].to_list()
        return gdf_masked

    def _quadtree2geodataframe(self, quadtree):
        poly_squares=[]
        for square in quadtree:
            south, west = pairs_quadtree.getLatLon(square[1], square[0])
            res = pairs_quadtree.getResolution(square[0])
            north = south + res
            east = west + res
            poly_squares.append(shapely.geometry.box(west, south, east, north))

        df_squares = pandas.DataFrame(poly_squares)
        df_squares.columns=[self.geom_col]
        df_squares = pandas.concat([
            pandas.DataFrame(quadtree).rename(columns={0: self.spatial_level_col, 1:self.spatial_key_col}), 
            df_squares
        ], axis=1)
        gdf_squares = geopandas.geodataframe.GeoDataFrame(df_squares, geometry=self.geom_col).set_crs(epsg=4326)
        return gdf_squares
    
    def _polyCells2geodataframe(self, cells):
        poly_cells=[]
        res = pairs_quadtree.getResolution(self.spatial_level)
        for cell in cells:
            south, west = pairs_quadtree.getLatLon(cell, self.spatial_level)
            north = south + res
            east = west + res
            poly_cells.append(shapely.geometry.box(west, south, east, north))

        gdf_cells = pandas.DataFrame(cells).rename(columns={0: self.spatial_key_col})
        gdf_cells[self.spatial_level_col] = self.spatial_level
        gdf_cells[self.geom_col] = poly_cells
        gdf_cells = geopandas.geodataframe.GeoDataFrame(gdf_cells, geometry=self.geom_col).set_crs(epsg=4326)
        return gdf_cells
    
    def _spatialPartitionLevel(self, sp):
        return int(sp.split(self.spatial_partition_identifier)[-1])
    
    def _create_spatial_partition_columns(self, gdf):
        for sp in self.spatial_partitions:
            levelsUp = self.spatial_level - self._spatialPartitionLevel(sp)
            getParentKey_part = partial(pairs_quadtree.getParentKey, levelsUp=levelsUp)
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
        gdf[f'filter_key_level{level}'] = pairs_quadtree.getKey(gdf['bb_miny'], gdf['bb_minx'], level)

        # Top_right_keys
        top_right_keys = pairs_quadtree.getKey(gdf['bb_maxy'], gdf['bb_maxx'], level)
        # Overwrite in cases where there would be more than one cell
        gdf.loc[gdf[f'filter_key_level{level}']!=top_right_keys, f'filter_key_level{level}'] = numpy.nan

    def _create_filter_key_columns(self, gdf, epsilon=None):
        gdf_unique = gdf[[self.id_col, self.geom_col]].drop_duplicates(subset=self.id_col).reset_index(drop=True)
        del gdf_unique[self.id_col]
        gdf_unique[['bb_minx', 'bb_miny', 'bb_maxx', 'bb_maxy']] = gdf_unique[self.geom_col].apply(lambda x: x.bounds).to_list()
        
        # Rounding errors can lead to duplication of geometries with edges along cell boundaries
        if epsilon is not None:
            # Negative buffer by something like epsilon=1e-13
            gdf_unique['bb_minx'] += epsilon
            gdf_unique['bb_miny'] += epsilon
            gdf_unique['bb_maxx'] -= epsilon
            gdf_unique['bb_maxy'] -= epsilon
        
        # Efficient way of getting the bounding-box keys 
        for level in self.filter_key_levels:
            self._create_filter_key_column(gdf_unique, level)
            
        # Decide here if we want to keep the geometry bounds or delete them
        del gdf_unique['bb_minx']
        del gdf_unique['bb_miny']
        del gdf_unique['bb_maxx']
        del gdf_unique['bb_maxy']
        
        return pandas.merge(gdf, gdf_unique, on=self.geom_col)

    def _generate_composite_keys(self, df):
        """
        Combine the temporal, spatial and dimension keys into one "composite_key"
        """
        df[self.composite_key_col] = 'level' + df[self.spatial_level_col].astype(str) + '_' + df[self.spatial_key_col].astype(str) 
        for k in self.temporal_keys + self.dimension_keys:
            df[self.composite_key_col] = df[self.composite_key_col] + '_' + k + '_' + df[k].astype(str)
        return df[self.composite_key_col]

    def _decompose_composite_keys(self, df):
        """
        Decompose the composite_key into temporal, spatial, and dimension keys
        """
        cols = [self.spatial_level_col, self.spatial_key_col]
        df[cols] = df[self.composite_key_col].apply(lambda x: x.split('level', 1)[1].split('_', 2)[:2]).to_list()
        df[self.spatial_level_col] = df[self.spatial_level_col].astype(int)
        df[self.spatial_key_col] = df[self.spatial_key_col].astype(int)
        for k in self.temporal_keys:
            cols = cols + [k]
            df[k] = df[self.composite_key_col].apply(lambda x: int(x.split(k+'_')[1].split('_')[0]))
        for k in self.dimension_keys: 
            cols = cols + [k]
            df[k] = df[self.composite_key_col].apply(lambda x: x.split(k+'_')[1].split('_')[0]) # Dimension keys are of type str
        return df[cols]

    def _select_parquet_file(self, gdf, composite_key):
        gdf_part = gdf[gdf[self.composite_key_col]==composite_key].reset_index(drop=True)
        filepath = self.dataset_directory
        for partition_name in self.partitions:
            partition_value = gdf_part.loc[0, partition_name]
            filepath = os.path.join(filepath, partition_name + '_' + str(partition_value))
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        filename = '_'.join([self.dataset, composite_key]) + '.parquet'
        filepath = os.path.join(filepath, filename)
        return gdf_part, filepath
    
    def _all_the_same(self, s):
        # Quick test if column values (pandas series s) are all the same
        a = s.to_numpy()
        return (a[0] == a).all()
    
    def _reproject_to_local_equal_area_grid(self, gdf):
        """
        All geometries must belong to the same spatial cell
        """
        assert(self._all_the_same(gdf[self.spatial_key_col]))
        
        # Get the center lat/lon of the spatial cell
        key = gdf.loc[0, self.spatial_key_col]
        level = gdf.loc[0, self.spatial_level_col]
        lat, lon = pairs_quadtree.getCenterLatLon(key, level)

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
        self, composite_key, dt_start=None, dt_end=None, columns=None, geometry_area=False, geometry_length=False, verbose=False
    ):
        try:
            _, filepath = self._select_parquet_file(self.gdf_meta, composite_key)
        except:
            # Make sure gdf_meta is present
            self.metadata_from_parquet()
            _, filepath = self._select_parquet_file(self.gdf_meta, composite_key)
        gdf = self._read_parquet(
            filepath, 
            dt_start=dt_start, 
            dt_end=dt_end, 
            columns=columns, 
            geometry_area=geometry_area, 
            geometry_length=geometry_length, 
            verbose=verbose,
        )
        return gdf
    
    def ingest_geodataframe(self, gdf, gdf_dissolved=None, epsilon=None, verbose=False):
        """
        Load a geopandas geoDataFrame and transform it into a DataFrame with spatio-temporal keys, aligned with other geolab data
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
        
        # Get Geodataframe of geolab cells (all on the same spatial_level resolution)
        #bounds_area = shapely.box(*self.gdf_dissolved.loc[0, 'geometry'].bounds).area
        #geom_area = self.gdf_dissolved.loc[0, 'geometry'].area
        #if geom_area<bounds_area/10:
        if (not hasattr(self.gdf_dissolved.loc[0, self.geom_col], 'geoms')) or (len(self.gdf_dissolved.loc[0, self.geom_col].geoms)<1000):
            # Use sparse version of quadtree
            self.gdf_dissolved_cells = self._quadtreeGDF(self.gdf_dissolved.loc[0, self.geom_col])
        else:
            # Use simplified version of quadtree
            self.gdf_dissolved_cells = self._quadtreeGDF2(self.gdf_dissolved.loc[0, self.geom_col])
        
        # Rounding errors can lead to duplication of geometries with edges along cell boundaries
        if epsilon is not None:
            # Negative buffer by something like epsilon=1e-13
            self.gdf_dissolved_cells[self.geom_col] = self.gdf_dissolved_cells[self.geom_col].buffer(-epsilon)
        
        if verbose:
            print('gdf_dissolved_cells', len(self.gdf_dissolved_cells))
            print('Time for (B1) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
        # (B2) JOIN ORIGINAL GEODATAFRAME TO GET THE SPATIAL KEY  
        # Joining original GeoDataFram with spatial cells may result in multiple copies of a row. 
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
            
        # (C1) KEEP TRACK OF THE POLYGONS THAT INTERSECT THE SPATIAL CELLS
        self.gdf_intersection[self.intersection_flag_col] = False
        self.gdf_intersection.loc[self.gdf_intersection[self.id_col].duplicated(keep=False), self.intersection_flag_col] = True
        
        # (C2) JOIN IN THE MANY COLUMNS WE DROPPED WHEN FORMING "self.gdf_unique".
        del self.gdf[self.geom_col]
        self.gdf_intersection = pandas.merge(self.gdf_intersection, self.gdf, on=self.id_col)
                
        # (C3) COMBINE THE TEMPORAL AND SPATIAL KEYS INTO ONE "composite_key".
        self.gdf_intersection[self.composite_key_col] = self._generate_composite_keys(self.gdf_intersection)
        
        # (C4) CREATE COLUMNS FOR SPATIAL PARTITIONING
        self._create_spatial_partition_columns(self.gdf_intersection)
        if verbose:
            print('Time for (C) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
        # (D) CREATE FILTER-KEY COLUMNS WITH KEYS FULLY CONTAINING THE GEOMETRY BOUNDING BOX AT VARIOUS RESOLUTION LEVELS
        self.gdf_intersection = self._create_filter_key_columns(self.gdf_intersection, epsilon=epsilon)
        if verbose:
            print('Time for (D) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()
        
    def to_parquet(
        self, 
        append=False, 
        geometry_area=False, 
        geometry_length=False, 
        remove_metadata_cols=False, 
        rename_cols={}, 
        verbose=False
    ):
        """
        Write GeoDataFrame to parquet
        """
        if verbose:
            stopwatch_start = time.time()
        self._get_partitions()
        
        # Save the parquet files by composite_key in a partitioned directory structure
        self.composite_keys = sorted(self.gdf_intersection.composite_key.unique())
        if len(self.composite_keys)>10000:
            # Currently each dimension value becomes part of the composite key, which determines the parquet partition filename
            # Raising an error when we get too many combinations
            # Often this is due to bad dimension_keys definition
            # To Do: allow different dimension values to be stored in the same parquet file
            print('self.gdf_intersection', len(self.gdf_intersection))
            print('self.composite_keys', len(self.composite_keys))
            print('self.dimension_keys', self.dimension_keys)
            raise ValueError('Too many composite_keys. Maybe we have a wrong dimension_keys definition?')

        for composite_key in self.composite_keys:
            gdf_part, filepath = self._select_parquet_file(self.gdf_intersection, composite_key)
            if len(gdf_part)==0:
                print('WARNING: no data to write')
            else:
                # Add area and/or length columns if requested
                if geometry_area or geometry_length:
                    gdf_unique = gdf_part[
                        [self.id_col, self.geom_col, self.spatial_key_col, self.spatial_level_col]
                    ].drop_duplicates(subset=self.id_col).reset_index(drop=True)
                    gdf_lambert_ea = self._reproject_to_local_equal_area_grid(gdf_unique)
                    if geometry_area:
                        gdf_unique[self.geom_area_col] = gdf_lambert_ea.area / 1e6 # units: [km2]
                    if geometry_length:
                        gdf_unique[self.geom_length_col] = gdf_lambert_ea.length / 1e3 # units: [km]
                    del gdf_unique[self.geom_col]
                    gdf_part = pandas.merge(gdf_part, gdf_unique, on=[self.id_col, self.spatial_key_col, self.spatial_level_col])
                    
                if remove_metadata_cols:
                    del gdf_part[self.spatial_key_col]
                    del gdf_part[self.spatial_level_col]
                    del gdf_part[self.composite_key_col]
                    del gdf_part[self.intersection_flag_col]
                    
                if len(rename_cols)>0:
                    gdf_part = gdf_part.rename(columns=rename_cols)

                if append:
                    try:
                        # See if there is something already present under this composite key
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
        vs_settings['spatial_level_col'] = self.spatial_level_col
        vs_settings['spatial_key_col'] = self.spatial_key_col
        vs_settings['composite_key_col'] = self.composite_key_col
        vs_settings['spatial_level'] = self.spatial_level
        vs_settings['temporal_keys'] = self.temporal_keys
        vs_settings['temporal_partitions'] = self.temporal_partitions
        vs_settings['dimension_keys'] = self.dimension_keys
        vs_settings['spatial_partition_identifier'] = self.spatial_partition_identifier
        vs_settings['spatial_partition_levels'] = self.spatial_partition_levels
        vs_settings['spatial_partitions'] = self.spatial_partitions
        vs_settings['partitions'] = self.partitions
        vs_settings['filter_key_levels'] = self.filter_key_levels
        vs_settings['intersection_policy'] = self.intersection_policy
        vs_settings['dataset'] = self.dataset
        vs_settings['vectorstore_directory'] = self.vectorstore_directory
        vs_settings['dataset_directory'] = self.dataset_directory
        vs_settings['partition_order'] = self.partition_order
        vs_settings['numeric_layers'] = self.numeric_layers
        vs_settings['timestamp_layers'] = self.timestamp_layers
        vs_settings['categorical_layers'] = self.categorical_layers
        vs_settings['quantiles'] = self.quantiles
        vs_settings['first'] = self.first
        vs_settings['timestamp_aggregation'] = self.timestamp_aggregation

        json_path = os.path.join(self.vectorstore_directory, self.dataset+'.json')
        with open(json_path, 'w') as f:
            json.dump(vs_settings, f)
        
    def read_vectorstore_settings(self):
        json_path = os.path.join(self.vectorstore_directory, self.dataset+'.json')
        with open(json_path) as f:
            vs_settings = json.load(f)
        for k in vs_settings:
            setattr(self, k, vs_settings[k])
            
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
        df_meta = pandas.DataFrame([g.split('/')[1:] for g in globbed], columns=self.partitions+[self.composite_key_col])
        df_meta['filepath'] = globbed_filepaths
        
        # Remove parts of the strings and cast the partitions to integer 
        for p in self.partitions:
            df_meta[p] = df_meta[p].apply(lambda x: int(x.split(p+'_')[-1]))
        df_meta[self.composite_key_col] = df_meta[self.composite_key_col].apply(lambda x: x.split(self.dataset+'_')[-1])
        
        # Split the composite key column
        self._decompose_composite_keys(df_meta)
            
        # Cast to geopandas by infering the geometry column (cell boxes) from the spatial level and spatial key
        self.gdf_meta = self._polyCells2geodataframe(df_meta[self.spatial_key_col].drop_duplicates().to_numpy())
        self.gdf_meta = pandas.merge(self.gdf_meta, df_meta, on=[self.spatial_key_col, self.spatial_level_col], how='right')

            
    def query_single_parquet(
        self, 
        query_latitude, 
        query_longitude, 
        query_dt, 
        spatial_filter=False, 
        temporal_filter=False, 
        columns=None,
        geometry_area=False, 
        geometry_length=False,
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
        self.query_df_meta[self.spatial_level_col] = self.spatial_level
        self.query_df_meta[self.spatial_key_col] = pairs_quadtree.getKey(query_latitude, query_longitude, self.spatial_level)
        self.query_df_meta = pandas.DataFrame([self.query_df_meta])

        # Spatial key for the partition
        for sp in self.spatial_partitions:
            levelsUp = self.spatial_level-self._spatialPartitionLevel(sp)
            getParentKey_part = partial(pairs_quadtree.getParentKey, levelsUp=levelsUp)
            self.query_df_meta[sp] = self.query_df_meta[self.spatial_key_col].astype(int).apply(getParentKey_part)

        self.query_df_meta[self.composite_key_col] = self._generate_composite_keys(self.query_df_meta)
        composite_key = self.query_df_meta[self.composite_key_col].values[0]
        _, filepath = self._select_parquet_file(self.query_df_meta, composite_key)

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
        
    def _query_worker(self, composite_key, columns=None):
        """
        Partial query using specific composite_key, corresponding to one parquet file.
        """
        _, filepath = self._select_parquet_file(self.query_df_meta, composite_key)
        gdf = self._read_parquet(
            filepath, self.query_dt_start, self.query_dt_end, columns=columns, 
            geometry_area=self.geometry_area, geometry_length=self.geometry_length, verbose=False
        )
        if len(gdf)!=0:
            if not self.query_polygon.contains(shapely.geometry.box(*gdf.total_bounds)):
                # Spatial filtering may be necessary
                gdf_unique = gdf[[self.id_col, self.geom_col, self.composite_key_col]].drop_duplicates(
                    subset=[self.id_col, self.composite_key_col]
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
                gdf = pandas.merge(gdf_unique, gdf, on=[self.id_col, self.composite_key_col])
                        
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
        
        # Temporal part of the composite key
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
                
            # Temporal part of the composite keys
            for i, row in df_date_range[self.temporal_keys].iterrows():
                for k in self.temporal_keys:
                    self.query_df_meta.loc[i, k] = row[k]
                    
        if verbose:
            print('Time for (A) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # Spatial part of the composite key
        #bounds_area = shapely.box(*self.query_polygon.bounds).area
        #geom_area = self.query_polygon.area
        #if geom_area<bounds_area/10:
        if len(self.query_polygon.loc[0, self.geom_col].geoms)<1000:
            # Use sparse version of quadtree
            query_spatial_keys = self.quadtree(self.query_polygon, self.spatial_level)
        else:
            # Use simplified version of quadtree
            query_spatial_keys = self.quadtree2(self.query_polygon)

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

        # Pairs level of the composite key
        self.query_df_meta[self.spatial_level_col] = self.spatial_level

        # Convert the floats to int
        self.query_df_meta = self.query_df_meta.astype(int)
        
        if verbose:
            print('Time for (C) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # Generate the composite keys
        self.query_df_meta[self.composite_key_col] = self._generate_composite_keys(self.query_df_meta)
        
        # Generate the spatial partition columns
        self._create_spatial_partition_columns(self.query_df_meta)

        if verbose:
            print('Time for (D) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        composite_keys_requested = set(self.query_df_meta[self.composite_key_col])
        try:
            composite_keys_present = set(self.gdf_meta[self.composite_key_col])
        except:
            # gdf_meta likely not present yet, so generate it
            self.metadata_from_parquet()
            composite_keys_present = set(self.gdf_meta[self.composite_key_col])
        composite_keys = sorted(composite_keys_requested & composite_keys_present)
        
        if verbose:
            print('Time for (E) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        if n_workers>1:
            query_worker_part = partial(self._query_worker, columns=columns)
            with ProcessPool(nodes=n_workers) as p:
                lst_gdf = p.map(query_worker_part, composite_keys)
                lst_gdf = [x for x in lst_gdf if x is not None]
            """
            with Pool(n_workers) as p:
                lst_gdf = p.map(query_worker_part, composite_keys)
                lst_gdf = [x for x in lst_gdf if x is not None]
            """
        else:
            lst_gdf = []
            for composite_key in composite_keys:
                gdf = self._query_worker(composite_key, columns=columns)
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
            
            # Some objects may be duplicated because they are saved in multiple spatial cells. 
            # If we saved intersections of polygons, we need to stitch those together
            gdf_unique = gdf_query[[self.id_col, self.geom_col, self.composite_key_col]].drop_duplicates(
                subset=[self.id_col, self.composite_key_col]
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
                    on=[self.id_col, self.composite_key_col],
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
        Partial overlay using specific composite_key, corresponding to one parquet file.
        """
        # Get the unique composite_key
        composite_key, is_unique = self._is_unique(gdf_right[self.composite_key_col])
        assert(is_unique)
        _, filepath = self._select_parquet_file(gdf_right, composite_key)
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
        
        # Find the relevant spatial cells and cut up the gdf_right geometries into parts commensurate with spatial cell bounds
        gdf_right2 = geopandas.overlay(self.gdf_meta, gdf_right, how='intersection')
        composite_keys = list(gdf_right2[self.composite_key_col].unique())
        
        if verbose:
            stopwatch_start = time.time()
        lst_gdf2 = []
        lst_gdf_bypass = []
        for i, composite_key in enumerate(composite_keys):
            gdf2 = gdf_right2[gdf_right2[self.composite_key_col]==composite_key]
            
            # We may be able to save some time in circumstances where the gdf_right geometries are larger than a spatial cell
            if self.intersection_policy in ['cut', 'both']:
                # Check if any (or all) of the intersected geometries completely fill a spatial cell
                full_containment = gdf2.contains(
                    self.gdf_meta.loc[self.gdf_meta[self.composite_key_col]==composite_key, 'geometry'].values[0]
                )
                # Currently only considering a special case if full_containment.all() and the overlay step can be skipped completely.
                if full_containment.all():
                    if verbose:
                        print('full_containment @ composite_key', composite_key)
                    _, filepath = self._select_parquet_file(gdf2, composite_key)
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
            #gdf = gdf2.groupby(self.composite_key_col).apply(lambda x: self._overlay_worker(x, columns=columns, verbose=verbose)).reset_index(drop=True)
            # Work on each spatial cell sequentially
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
            gdf_duplicated = gdf.groupby([self.id_col, 'polygon_id']).count()[self.composite_key_col].reset_index()
            gdf_duplicated = gdf_duplicated[gdf_duplicated[self.composite_key_col]>1]
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