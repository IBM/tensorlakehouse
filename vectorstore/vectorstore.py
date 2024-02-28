"""Generate vectorstore.

    Classes

        Vectorstore    Generating vector store.
"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
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
from rasterio import CRS
import hashlib
from functools import partial
import json
from multiprocessing import Pool
#from pathos.pools import ProcessPool

sys.path.insert(1, os.path.abspath(".."))
from qtree_index import nestedgrid, mortoncurve, qtree


class Vectorstore():
    """Generates quadtree-based indices for Vector data and persist them in parquet files.
    
    Enables fused (e.g. fused to the PAIRS grid) vector data and fast raster-vector queries.

    Attributes:

        vectorstore_directory   Base directory where the vectorstore is persisted.
        dimension_values        Dictionary of valid dimension values indexed by dimension_names.
        
    Methods

        register_geodataframe   Load and transform GeoDataFrame into a spatially indexed GeoDataFrame.
        create_partitions       Create spatial partitions of roughly equal size.
        to_parquet              Write entire GeoDataFrame (all partitions) to parquet.
        partial_upload          Register GeoDataFrame, create partition, and save parquet to disk.
        repartition             Repartrition parquet files to balance number of rows in each file.
        reindex                 (Re-)create a qtree spatial index for the entire dataset.

    """

    # Default values for class attributes
    VECTORSTORE_DIRECTORY      = 'data/vectorstore/'

    # GeoDataFrame colum conventions (used when loading data from gdf or vectorstore)
    DT_COL                     = 'time'
    
    # Geopandas relies on this column being named 'geometry', so enforce this
    GEOM_COL                   = 'geometry'
    ID_COL                     = '_geom_hash'
    KEY_COL                    = '_root' #'q_key'
    IDX_COL                    = '_idx'
    IDX_BOX_COL                = 'idx_bounds'
    GEOM_AREA_COL              = '_geom_area'
    GEOM_LENGTH_COL            = '_geom_length'

    # Target level for qtree (only reached when max_depth allows it).
    TARGET_LEVEL               = 8

    # Maximum depth of qtree index, measured from the root (head) level.
    # Set to target_level to ensure the generation of a "complete" qtree index for every geometry.
    # Set to 0 in order to skip generation of q_tree index (we will be relying on root node only).
    # Set to value in between 0 and target_level to limit the size of the q_tree index.
    MAX_DEPTH                  = 5
    
    # MAX_INFLATION factor is another way to limit the depth of the qtree index.
    # We limit the length of gdf_merge to roughly a factor of the length of df_root.
    # Once gdf_merge is larger than max_inflation*len(df_root) we stop going deeper into the tree.
    MAX_INFLATION              = None  # 10 , 2, 1.1

    # Partition target rows
    TARGET_ROWS                = 1e6
    
    CHILDREN                   = ['child_0', 'child_1', 'child_2', 'child_3']
    
    # Default grid
    GRID = nestedgrid.PAIRS()
    

    def __init__(
        self,
        dataset,
        vectorstore_directory = None,
        dimension_values = {},
        dt_col = None,
        geom_col = None, 
        id_col = None,
        key_col = None,
        idx_col = None,
        idx_box_col = None,
        geom_area_col = None, 
        geom_length_col = None, 
        max_level = None,
        target_level = None,
        max_depth = None,
        max_inflation = None,
        target_rows = None,
        grid = None,
        valid_range = None,
        verbose = False,
    ):
        
        # Nested grid to overwrite the default PAIRS grid
        self.grid = self.GRID if grid is None else grid

        # Definition of the morton curve on top of the nested grid
        self.morton = mortoncurve.Morton(self.grid)

        # Opportunity to limit the valid range here
        self.valid_range = self.morton.valid_range if valid_range is None else valid_range

        # Dataset name
        self.dataset                      = dataset

        # List of the timestamp hierarchy levels ('year', 'month', ...)
        self.temporal_levels = []

        # Dataset dimensions other than space and time
        self.dimension_values = dimension_values

        # Vectorstore base directory
        self.vectorstore_directory = self.VECTORSTORE_DIRECTORY if vectorstore_directory is None else vectorstore_directory
        
        # Dataset directory is subfolder in vectorstore_directory
        self.dataset_directory = os.path.join(self.vectorstore_directory, self.dataset).replace('\\', '/')
        if not os.path.exists(self.dataset_directory): os.makedirs(self.dataset_directory)

        # Grid directory (where the index is defined on) is subfolder in dataset_directory
        self.grid_directory = os.path.join(
            self.dataset_directory,
            'grid=' + self.grid.__repr__()
        ).replace('\\', '/')
        if not os.path.exists(self.grid_directory): os.makedirs(self.grid_directory)

        # Parquet directory is subfolder in grid_directory
        self.geoparquet_directory = os.path.join(self.grid_directory, 'geoparquet').replace('\\', '/')
        if not os.path.exists(self.geoparquet_directory): os.makedirs(self.geoparquet_directory)

        # Index directory is subfolder in grid_directory 
        self.index_directory = os.path.join(self.grid_directory, 'index').replace('\\', '/')
        if not os.path.exists(self.index_directory): os.makedirs(self.index_directory)

        # Table specific column information
        self.dt_col = self.DT_COL if dt_col is None else dt_col
        self.key_col = self.KEY_COL if key_col is None else key_col
        self.idx_col = self.IDX_COL if idx_col is None else idx_col
        if (geom_col is not None) and (geom_col!=self.GEOM_COL):
            raise ValueError("Geopandas dependencies require geom_col to be named geometry." )
        self.geom_col = self.GEOM_COL
        self.id_col = self.ID_COL if id_col is None else id_col
        self.idx_box_col = self.IDX_BOX_COL if idx_box_col is None else idx_box_col
        self.geom_area_col = self.GEOM_AREA_COL if geom_area_col is None else geom_area_col
        self.geom_length_col = self.GEOM_LENGTH_COL if geom_length_col is None else geom_length_col

        # Primary keys
        self.primary_keys = [self.dt_col, self.id_col]+list(self.dimension_values)

        # Indexing/partitioning parameters
        self.max_level = self.grid.max_levels if max_level is None else max_level
        self.target_level = self.TARGET_LEVEL if target_level is None else target_level
        self.max_depth = self.MAX_DEPTH if max_depth is None else max_depth
        self.max_inflation = self.MAX_INFLATION if max_inflation is None else max_inflation
        self.target_rows = self.TARGET_ROWS if target_rows is None else target_rows

        self.verbose = verbose


    def _geometry_hash(self, geoseries):
        """
        Using sha256 hash on wkb representation of geometry to get a nearly unique geometry id
        """
        return geoseries.to_wkb().apply(lambda x: hashlib.sha256(x).hexdigest())

    
    def _index_root(self, gdf):
        """Calculate the qtree index root for each geometry.

        The root of the qtree index is the smallest z-order square that still contains the entire geometry.
        :param gdf:  Geopandas GeoDataFrame to be indexed.
        """
        # Numpy arrays of the geometry bounds
        minx = numpy.array(gdf.bounds.minx)
        miny = numpy.array(gdf.bounds.miny)
        maxx = numpy.array(gdf.bounds.maxx)
        maxy = numpy.array(gdf.bounds.maxy)

        # Bottom-up calc. of qtree roots (smallest z-order square that still contains the entire geometry)
        level = self.max_level
        while level>=0:
            # Calculate the sw and ne corner pixel keys based on level
            sw_keys = self.morton.get_key(miny, minx, level)
            ne_keys = self.morton.get_key(maxy, maxx, level)
        
            # If the keys are the same, we have a good key. Otherwise mask (with -1)
            keys = sw_keys.copy()
            levels = sw_keys * 0 + level
            # Skip masking at level 0 to accomodate geometries that touch the top or left valid bounds.
            if level>0:
                keys[sw_keys!=ne_keys] = -1 # -1 interpreted as nan
                levels[sw_keys!=ne_keys] = -1 # -1 interpreted as nan
        
            # Fill masked values by the next lower level if possible
            if level==self.max_level:
                agg_keys = keys
                agg_levels = levels
            else:
                agg_keys[agg_keys==-1] = keys[agg_keys==-1]
                agg_levels[agg_levels==-1] = levels[agg_levels==-1]
            
            level-=1

        gdf[self.key_col] = self.morton.encode(agg_keys, agg_levels)
        return gdf
        

    def _base4_to_box(self, arr):
        """Translate keys to shapely boxes.
    
        Speed up and harden bulk-operarions of the base4_to_box method of mortoncurve using pandas.
        :param arr:  1D array of quaternary keys.
        :returns: 1D array of shapely geometries (boxes) that correspond to the keys.
        """

        df = pandas.DataFrame({self.key_col: numpy.array(arr)})
        
        # Do the expensive operation on as few rows as possible
        df_unique = df[[self.key_col]].drop_duplicates()
    
        # zeroth level key needs special attention when vectorizing with numpy (TypeError).
        df_zero = None
        if '0q' in df_unique[self.key_col].values:
            df_unique = df_unique[df_unique[self.key_col]!='0q']
            df_zero = pandas.DataFrame({self.key_col: ['0q']})
            df_zero[self.geom_col] = df_zero[self.key_col].apply(self.morton.base4_to_box)
    
        # Try vectorized (numpy) version of base4_to_box first
        try:
            df_unique[self.geom_col] = self.morton.base4_to_box(df_unique[self.key_col])
        except (OverflowError, ValueError) as e:
            print(e)
            print('Failed to use numpy. Switching to pandas')
            df_unique[self.geom_col] = df_unique[self.key_col].apply(self.morton.base4_to_box)
    
        if df_zero is not None:
            df_unique = pandas.concat ([df_zero, df_unique])
            
        df = pandas.merge(df, df_unique, on=self.key_col, how='left')
        
        return numpy.array(df[self.geom_col])


    def _children(self, q_key):
        """Generate list of children keys."""
        q0 = q_key + "0"
        q1 = q_key + "1"
        q2 = q_key + "2"
        q3 = q_key + "3"
        return [q0, q1, q2, q3]


    def _nan_subscribe(self, x, level):
        """restrict the q_key to level"""
        try:
            return x[:2+level]
        except:
            return None
    
    
    def _depth_to_level(self, level=None):
        """Translate the key by depth columns to key by level columns."""

        if level is None:
            # We only need the target level. Low-res can be generated using subscription [:level+2]
            level = self.target_level

        for depth in numpy.arange(self.max_depth, -1, -1):
            if 'depth_'+str(depth) in self.gdf_idx.columns:
                if self.idx_col not in self.gdf_idx.columns:
                    self.gdf_idx[self.idx_col] = self.gdf_idx['depth_'+str(depth)].apply(
                        lambda x: self._nan_subscribe(x, level)
                    )
                else:
                    self.gdf_idx[self.idx_col] = self.gdf_idx[self.idx_col].fillna(
                        self.gdf_idx['depth_'+str(depth)].apply(lambda x: self._nan_subscribe(x, level))
                    )

    
    def _preprocess_dimension_column(self, s):
        """Preprocess dimension column. We require dimension columns to be of type int, datetime, or str.
    
        :param s: pandas Series
        """
        if pandas.api.types.is_float_dtype(s):
            # Convert to string. Remove insignificant trailing zeros from the significand, and 
            # remove the decimal point if there are no remaining digits.
            s =  s.apply(lambda x: f'{x:g}')
    
        return s


    def _warning_duplicate_primary_key(self, len_before, len_after):
        """Warn the user when dropping rows."""
        if len_before>len_after:
            print('WARNING: -------------------------------------------------------------------')
            print('dropping', len_before-len_after, 'rows with duplicate primary keys, composed of:')
            print(self.primary_keys)
            print('Keeping last version of the data')
            print('----------------------------------------------------------------------------')

    
    def register_geodataframe(self, gdf, validate_geometries=True):
        """Calculate the root index for each unique geometry.

        :param gdf:  Geopandas GeoDataFrame to be ingested.
        """
        if self.verbose:
            stopwatch_start = time.time()
            
        self.gdf_ingest = gdf
        assert(self.dt_col in self.gdf_ingest.columns)
        assert(self.geom_col in self.gdf_ingest.columns)

        # We require dimension columns to be of type int, datetime, or str.
        if len(self.dimension_values)>0:
            for dimension in self.dimension_values:
                gdf[dimension] = self._preprocess_dimension_column(gdf[dimension])
    
                # Register dimension values
                self.dimension_values[dimension] = list(set(
                    self.dimension_values[dimension]+list(gdf[dimension].drop_duplicates())
                ))
            self.write_metadata()

            if self.verbose:
                print('Time for preprocessing dimension columns in seconds', round(time.time()-stopwatch_start, 3))
                stopwatch_start = time.time()

        if validate_geometries:
            # Make sure the geometries are valid
            self.gdf_ingest[self.geom_col] = self.gdf_ingest[self.geom_col].apply(shapely.validation.make_valid)
            
            if self.verbose:
                print('Time for validating geometries in seconds', round(time.time()-stopwatch_start, 3))
                stopwatch_start = time.time()

        # If an id_col was not privided, we calculate a geometry hash for each unique geometry
        if self.id_col==self.ID_COL:
            # Using sha256 hash on wkb representation of geometry to get a nearly unique id_col
            self.gdf_ingest[self.id_col] = self._geometry_hash(self.gdf_ingest[self.geom_col])
            
            if self.verbose:
                print('Time for calculating a geometry hash in seconds', round(time.time()-stopwatch_start, 3))
                stopwatch_start = time.time()

        # Drop duplicate primary keys (keeping the last version of the data)
        len_before = len(self.gdf_ingest)
        self.gdf_ingest = self.gdf_ingest.drop_duplicates(
            subset=self.primary_keys,
            keep='last',
        ).reset_index(drop=True)
        len_after = len(self.gdf_ingest)
        if len_before>len_after:
            self._warning_duplicate_primary_key(len_before, len_after)
        
        # Drop duplicate geometries before calculating spatial index (e.g. when we have multiple timestamps for the same geometry) 
        gdf_unique = self.gdf_ingest[[self.id_col, self.geom_col]].drop_duplicates(subset=self.id_col).reset_index(drop=True)

        # Fast algorithm for smallest z-order squares that still contain the entire geometries.
        gdf_unique = self._index_root(gdf_unique)

        # Remember all root keys for each geometry
        self.df_root = gdf_unique[[self.id_col, self.key_col]]

        if self.verbose:
            print('df_root', len(self.df_root))
            print('Time for indexing the root of each unique geometry in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

    
    def create_partitions(self, target_rows=None):
        """Create spatial partitions of roughly equal size.

        :param target_rows:  Approximate number of rows per parquet file.
        """
        if self.verbose:
            stopwatch_start = time.time()
            
        if target_rows is not None: self.target_rows = target_rows
            
        self.gdf_partition = pandas.DataFrame()
        df_count = self.df_root.groupby(self.key_col)[self.id_col].count().rename('root_count').reset_index()
        
        # Initialize partition key with root at target_level+1 so maximum partition becomes target_level
        df_count['partition'] = df_count[self.key_col].apply(lambda x: x[:self.target_level+1+2])

        df_count['count'] = df_count['root_count']

        # Breadth First Search starting at highest resolution level
        df_count['level'] = df_count['partition'].apply(lambda x: len(x)-2)
        level = df_count['level'].max()
        n_rows = self.target_rows/2
        while level>=0 and len(df_count)>0:

            # Check if any of the counts at specific level have reached target_rows
            df_level = df_count[df_count['level']==level]
            df_count = df_count[df_count['level']!=level]

            if len(df_level)>0:
                if level==0:
                    # Reached the coarsest level
                    n_rows = 0
                    # Assign all remaining records to the finest possible partition
                    stop_l = min(
                        df_level[self.key_col].apply(len).min()-2,
                        self.target_level
                    )
                    l = 0
                    while l<stop_l:
                        l+=1
                        # Check potential partition level
                        p = df_level.loc[0, self.key_col][:l+2]
                        if not df_level[self.key_col].apply(lambda x: x.startswith(p)).all():
                            # mixed keys at this level, so stop and go back to previous level
                            l-=1
                            break

                    df_level['partition'] = df_level[self.key_col].iloc[0][:l+2]

                # Separate the keys with number of rows above target_rows
                self.gdf_partition = pandas.concat([
                    self.gdf_partition,
                    df_level[df_level['count']>=n_rows].reset_index(drop=True),
                ])
                df_level = df_level[df_level['count']<n_rows].reset_index(drop=True)

                # Get the parent keys and decrease the level for the next iteration
                df_level['partition'] = df_level['partition'].apply(lambda x: x[:-1])
                df_level['level'] = level-1
                df_count = pandas.concat([df_count, df_level])

            df_count['count'] = df_count.groupby('partition')['root_count'].transform('sum')
            if len(df_count['partition'].drop_duplicates())==1:
                # Nothing to gain by decreasing level further
                n_rows = 0
            level-=1
            
        assert len(df_count)==0

        self.gdf_partition = self.gdf_partition.reset_index(drop=True)
        self.gdf_partition[self.geom_col] = self._base4_to_box(self.gdf_partition['partition'])
        self.gdf_partition = geopandas.GeoDataFrame(self.gdf_partition)

        # Add the partition column to the root
        self.df_idx = pandas.merge(
            self.df_root,
            self.gdf_partition[[self.key_col, 'partition']],
            on=self.key_col,
        ).reset_index(drop=True)
        
        if self.verbose:
            print('distributed over', len(self.gdf_partition['partition'].drop_duplicates()), 'partitions')
            print('Time for creating partitions in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()


    def _glob_partitions(self):
        """Parse the geoparquet directory to find existing partitions."""
        
        partitions = glob(os.path.join(self.geoparquet_directory, '**/*.parquet').replace('\\', '/'), recursive=True)
        partitions = [os.path.splitext(os.path.basename(f))[0] for f in partitions]
        df_partitions = pandas.DataFrame({
            'partition': partitions,
            'partition_level': [len(p)-2 for p in partitions],
        }).sort_values(by=['partition_level', 'partition']).reset_index(drop=True)
        return df_partitions


    def _glob_indexfiles(self):
        """Parse the index directory to find existing indices."""
        
        indices = glob(os.path.join(self.index_directory, '**/*.parquet').replace('\\', '/'), recursive=True)
        indices = [os.path.splitext(os.path.basename(f))[0] for f in indices]
        df_indices = pandas.DataFrame({
            'indices': indices,
            'index_level': [len(p)-2 for p in indices],
        }).sort_values(by=['index_level', 'indices']).reset_index(drop=True)
        return df_indices

    
    def _shard(self, temporal_partition, source_partition, target_partitions):
        """Shard the contents of source_partition to higher-resolution existing partitions whenever possible."""

        source_filepath = self._filepath(temporal_partition, source_partition)
        if os.path.exists(source_filepath):
            gdf_source = geopandas.read_parquet(source_filepath)
            initial_length = len(gdf_source)
    
            for target_partition in target_partitions:
                gdf_shard = gdf_source[gdf_source[self.key_col].apply(
                    lambda x: x.startswith(target_partition)
                )].sort_values(by=self.key_col).reset_index(drop=True)
                
                if len(gdf_shard)>0:
                    # Move the source records to the target partition
                    self._to_parquet(temporal_partition, target_partition, gdf_part=gdf_shard, append=True)
    
                    # Remove the sharded records from the source DataFrame
                    gdf_source = gdf_source[~gdf_source[self.key_col].apply(
                        lambda x: x.startswith(target_partition)
                    )]
    
                if self.verbose and len(gdf_shard)>0:
                    print(f"{source_partition:25} {target_partition:25} {str(len(gdf_source)):15} {str(len(gdf_shard)):15}")
                    
            # Remove the sharded records from the source partition
            if len(gdf_source)==0:
                # Erase the source partition
                os.remove(source_filepath)
            elif len(gdf_source)<initial_length:
                # Overwrite the source partition
                self._to_parquet(temporal_partition, source_partition, gdf_part=gdf_source, append=False)
                
            return gdf_source
        else:
            return geopandas.GeoDataFrame()


    def _split(self, temporal_partition, source_partition, gdf_source=None):
        """Split contents of partition into new children partitions if enough records exist."""
        
        source_filepath = self._filepath(temporal_partition, source_partition)
        if gdf_source is None:
            gdf_source = self._read_parquet(source_filepath)
        initial_length = len(gdf_source)
        children = self._children(source_partition)
        
        # Keep partition flag: Parent records that can't be distributed to children are present.
        keep_partition = any(gdf_source[self.key_col].apply(lambda x: x==source_partition))
        
        for child in children:
            gdf_child = gdf_source[gdf_source[self.key_col].apply(lambda x: x.startswith(child))]
            n_records = len(gdf_child)

            if (n_records>self.target_rows/2) or (
                (os.path.exists(self._filepath(temporal_partition, child)) and (n_records>0) and keep_partition)
            ):
                # Split the records off and assign to the child
    
                # Move the source records to the child partition
                self._to_parquet(temporal_partition, child, gdf_part=gdf_child, append=True)
    
                # Remove the child records from the source DataFrame
                gdf_source = gdf_source[~gdf_source[self.key_col].apply(lambda x: x.startswith(child))]

                if self.verbose:
                    print(f"{source_partition:25} {child:25} {str(len(gdf_source)):15} {str(len(gdf_child)):15}")
                if len(gdf_source)==0:
                    break
    
        # Remove the child records from the source partition
        if len(gdf_source)==0:
            # Erase the source partition
            os.remove(source_filepath)
            return geopandas.GeoDataFrame()
        else:
            # Overwrite the source partition
            self._to_parquet(temporal_partition, source_partition, gdf_part=gdf_source, append=False)
            return gdf_source


    def _combine(self, temporal_partition, df_rows):
        """Combine the contents of tiny partitions at their parent levels."""

        df_small = df_rows[df_rows['n_rows']<=self.target_rows/2].sort_values(
            by='partition_level', ascending=False).reset_index(drop=True)
        
        merge_partitions = list(df_small['partition'])
        while len(merge_partitions)>0:
            merge_partition = merge_partitions[0]
            if len(merge_partition)==2:
                # The root level cannot be merged any further
                break
        
            parent = merge_partition[:-1]
            ancestor = parent
            ancestors = []
            while len(ancestor)>=2:
                ancestors.append(ancestor)
                ancestor = ancestor[:-1]
                
            # Potential siblings or cousing to merge with on the parent or grandparent node
            # Note we are not merging more distant cousins
            df1 = df_rows[(
                (
                    df_rows['partition'].apply(lambda x: x.startswith(merge_partition[:-2]))
                ) & (
                    df_rows['n_rows']<=self.target_rows/2
                ) & (
                    df_rows['partition']!=merge_partition
                )
            )]
            
            # Direct ancestor partitions already present
            df2 = df_rows[(df_rows['partition'].isin(ancestors))]
            
            df3 = pandas.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
            
            if len(df3)>0:
                if os.path.exists(self._filepath(temporal_partition, parent)):
                    # Merge the child and parent partitions on the parent node
                    print('Merge: ', merge_partition, '->', parent)
                    source_filepath = self._filepath(temporal_partition, merge_partition)
                    if os.path.exists(source_filepath):
                        gdf_source = geopandas.read_parquet(source_filepath)
                    self._to_parquet(temporal_partition, parent, gdf_part=gdf_source, append=True)
                    os.remove(source_filepath)
                    
                    # Assign the rows from the child to the parent
                    n_rows = df_rows.loc[df_rows['partition']==merge_partition, 'n_rows'].values[0]
                    df_rows.loc[df_rows['partition']==parent, 'n_rows']+=n_rows
                    df_rows = df_rows[df_rows['partition']!=merge_partition].reset_index(drop=True)
                    
                else:
                    # Move the child partition to the parent
                    print('Rename:', merge_partition, '->', parent)
                    os.rename(
                        self._filepath(temporal_partition, merge_partition),
                        self._filepath(temporal_partition, parent)
                    )
                    
                    # Assign the rows from the child to the parent
                    df_rows.loc[df_rows['partition']==merge_partition, 'partition_level']-=1
                    df_rows.loc[df_rows['partition']==merge_partition, 'partition'] = parent
            else:
                # Remove the child record (even if the partition stays)
                df_rows = df_rows[df_rows['partition']!=merge_partition].reset_index(drop=True)
        
            # Re-calculate the small partitions
            df_small = df_rows[df_rows['n_rows']<=self.target_rows/2].sort_values(
                by='partition_level', ascending=False).reset_index(drop=True)
            merge_partitions = list(df_small['partition'])


    def repartition(self, temporal_partition, target_rows=None):
        """Repartrition parquet files to balance number of rows in each file as much as possible."""
        
        if self.verbose:
            stopwatch_start = time.time()
                    
        if target_rows is not None: self.target_rows = target_rows

        # Potential source partitions for sharding and/or splitting
        df_partitions = self._glob_partitions()
        df_source_partitions = df_partitions[df_partitions['partition_level']<=self.target_level].sort_values(
            by='partition_level').reset_index(drop=True)

        # Keep track of the number or rows within each source partition after sharding and splitting
        dict_n_rows = {}
        
        # Start with the coarsest source partitions
        for level in numpy.arange(min(df_source_partitions['partition_level']), self.max_level):
            if level>min(df_source_partitions['partition_level']):
                # Update source partitions, since partitions may have changed
                df_partitions = self._glob_partitions()
                df_source_partitions = df_partitions[df_partitions['partition_level']<=self.target_level].sort_values(
                    by='partition_level').reset_index(drop=True)

            if self.verbose:
                print('level', level)
                print(f"{'source partition':25} {'target partition':25} {'records stay':15} {'records move':15}")
    
            # Source partitions of specific level
            source_partitions = sorted(set(df_source_partitions[df_source_partitions['partition_level']==level]['partition']))
            for source_partition in source_partitions:

                # (1) Sharding into existing high-resolution partitions.
                # Get the target partitions for this source_partition.
                df = df_partitions[(df_partitions['partition_level']>level) & (
                    df_partitions['partition'].apply(lambda x: x.startswith(source_partition))
                )
                ].sort_values(by='partition_level', ascending=False).reset_index(drop=True)
                target_partitions = list(df['partition'])

                # Shard the contents of source_partition to higher-resolution existing partitions whenever possible.
                gdf_source = self._shard(temporal_partition, source_partition, target_partitions)

                # (2) Split contents of partition into new children partitions if enough records exist.
                if len(gdf_source)>0:
                    gdf_source = self._split(temporal_partition, source_partition, gdf_source=gdf_source)
                    dict_n_rows[source_partition] = len(gdf_source)

        # (3) Combine multiple tiny files at high resolution to lower-resolution combined files
        df_rows = pandas.DataFrame({'partition': dict_n_rows.keys(), 'n_rows': dict_n_rows.values()})
        df_rows['partition_level'] = df_rows['partition'].apply(lambda x: len(x)-2)
        df_rows = df_rows.sort_values(by=['partition_level'], ascending=False)

        self._combine(temporal_partition, df_rows)

        if self.verbose:
            print('Time for repartitioning in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

    
    def _filepath_idx(self, temporal_partition, spatial_partition, create_path=True):
        """Composing the filepath from metadata."""
        filepath = self.index_directory
        for temporal_level in self.temporal_levels:
            filepath = os.path.join(
                filepath,
                temporal_level+'='+str(temporal_partition[temporal_level])
            ).replace('\\', '/')
        # Note, currently not building any sub-directories associated with spatial levels
        if create_path and not os.path.exists(filepath): os.makedirs(filepath)

        filename = spatial_partition + '.parquet'
        return os.path.join(filepath, filename).replace('\\', '/')


    def _filepath(self, temporal_partition, spatial_partition, create_path=True):
        """Composing the filepath from metadata."""
        filepath = self.geoparquet_directory
        for temporal_level in self.temporal_levels:
            filepath = os.path.join(
                filepath,
                temporal_level+'='+str(temporal_partition[temporal_level])
            ).replace('\\', '/')
        # Note, currently not building any sub-directories associated with spatial levels
        if create_path and not os.path.exists(filepath): os.makedirs(filepath)

        filename = spatial_partition + '.parquet'
        return os.path.join(filepath, filename).replace('\\', '/')

    
    def _decode_filepath(self, filepath):
        dirname, filename = os.path.split(filepath)
        spatial_partition = os.path.splitext(filename)[0]
        
        temporal_partition = {}
        while True:
            # Continue splitting
            dirname, tail = os.path.split(dirname)
            try:
                key, value = tail.split('=')
            except ValueError:
                break
            if key in self.temporal_levels:
                # Found another temporal partition
                temporal_partition[key] = value
            elif key=='grid':
                assert value==self.grid.__repr__()
            else:
                raise ValueError("Non-compliant filepath." )

        # Now we should arrive at the dataset directory
        assert tail==self.dataset

        # Finally we should be left with the vectorstore_directory
        assert dirname.rstrip('/')==self.vectorstore_directory.rstrip('/')

        return temporal_partition, spatial_partition

    
    def _filter_partition(self, temporal_partition, spatial_partition):
        gdf_part = pandas.merge(
            self.gdf_ingest,
            self.df_idx[self.df_idx['partition']==spatial_partition][[self.id_col, self.key_col]],
            on=self.id_col,
        )
        return gdf_part


    def reindex(self, temporal_partition):
        """(Re-)create a qtree spatial index for the entire dataset."""

        if self.verbose:
            stopwatch_start = time.time()

        # Start from a clean slate
        indices = sorted(self._glob_indexfiles()['indices'])
        for index in indices:
            filepath = self._filepath_idx({}, index)
            os.remove(filepath)

        # Reindex each partition
        spatial_partitions = sorted(self._glob_partitions()['partition'])
        for spatial_partition in spatial_partitions:
            
            print('spatial_partition', spatial_partition)
            filepath = self._filepath(temporal_partition, spatial_partition)
            gdf_part = geopandas.read_parquet(filepath, columns=[self.id_col, self.geom_col, self.key_col])
            self._index_to_parquet(temporal_partition, spatial_partition, gdf_part=gdf_part)

        if self.verbose:
            print('Time to index entire dataset', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()


    def _recursive_bfs(self, gdf_tree, depth, initial_length=None, final_iteration=False, key_col=None, id_col=None):
        """Calculate the next level of children keys."""
        if key_col is None:
            key_col = self.key_col
        if id_col is None:
            id_col = self.id_col
    
        # Generate 4 child columns representing the 4 quadrants we need to check for intersections
        df_children = pandas.DataFrame(
            gdf_tree[key_col].apply(lambda x: self._children(x)).to_list(),
            columns=self.CHILDREN
        )
        
        # Create the children bounding boxes
        for child in self.CHILDREN:
            df_children[child + '_bounds'] = self._base4_to_box(df_children[child])

        # Check for intersection of original geometry with children
        for child in self.CHILDREN:
            mask1 = gdf_tree.intersects(
                geopandas.GeoSeries(df_children[child + '_bounds']).set_crs(self.grid.crs)
            )
            df_children[child] = df_children[child].where(mask1)
        
        # Stack the child spatial keys and bounding geometries
        df1 = pandas.concat([gdf_tree[id_col], df_children], axis=1)
        df_children = pandas.DataFrame()
        for child in self.CHILDREN:
            df2 = df1[[id_col, child, child+'_bounds']].dropna().rename(columns={
                child: key_col,
                child+'_bounds': self.idx_box_col,
            })
            df_children = pandas.concat([df_children, df2])
        df_children = df_children.reset_index(drop=True)

        # spatial index as broad DataFrame
        df = df_children[[id_col, key_col]].rename(columns={key_col: 'depth_'+str(depth)})
        df['depth_'+str(depth-1)] = df['depth_'+str(depth)].apply(lambda x: x[:-1])
        self.gdf_idx = pandas.merge(self.gdf_idx, df, on=[id_col, 'depth_'+str(depth-1)], how='outer')

        # Update the keys we need to work on in the next iteration
        mask2 = df_children[key_col].apply(len)>=self.target_level+2
        gdf_tree = pandas.merge(
            gdf_tree[[id_col, self.geom_col]].drop_duplicates(),
            df_children[~mask2].reset_index(drop=True),
            on=id_col,
        )

        # Assign fully contained nodes from further splitting considerations.
        mask3 = gdf_tree.contains(geopandas.GeoSeries(gdf_tree[self.idx_box_col]).set_crs(self.grid.crs))
        
        gdf1 = gdf_tree[mask3].reset_index(drop=True)
        gdf_tree = gdf_tree[~mask3].reset_index(drop=True)

        # This step is only needed for _index_geodataframe_point_or_line(), not _index_geodataframe_aerial()
        if (self.max_inflation is not None) and (initial_length is not None):
            if len(self.gdf_idx) > initial_length * self.max_inflation:
                # Break even befor max_depth is reached
                final_iteration = True

        return gdf_tree, df_children, final_iteration


    def _index_geodataframe_point_or_line(self, gdf):
        """Calculate the qtree index, and target keys at target_level for each geometry where possible.

        Create spatial index, aligned with GeoDN raster data.
        This method is typically run on each vectorstore partition separately.
        Optimized to deal with Points, LineStrings, or small polygons.
        :param gdf:  Geopandas GeoDataFrame to be indexed.
        """
        if self.verbose:
            stopwatch_start = time.time()

        assert(self.geom_col in gdf.columns)
        assert(self.id_col in gdf.columns)

        cols = [self.id_col, self.geom_col]
        if self.key_col in gdf.columns:
            cols += [self.key_col]
            
        # Drop duplicate geometries before calculating spatial index (e.g. when we have multiple timestamps for the same geometry) 
        gdf_unique = gdf[cols].drop_duplicates(subset=self.id_col).reset_index(drop=True)
        if self.verbose:
            print('gdf_unique', len(gdf_unique))

        if self.key_col not in gdf.columns:
            # Fast algorithm for smallest z-order squares that still contain the entire geometries.
            gdf_unique = self._index_root(gdf_unique)
            if self.verbose:
                print('Time for _index_root (A1) in seconds', round(time.time()-stopwatch_start, 3))
                stopwatch_start = time.time()

        # gdf_idx builds upon the root geometries
        self.gdf_idx = gdf_unique.copy()
        self.gdf_idx['depth_0'] = self.gdf_idx[self.key_col]
        
        # Determine the geometries for which we have to build an entire quadtree, rather than just the root node.
        gdf_tree = gdf_unique[gdf_unique[self.key_col].apply(len)<self.target_level+2].reset_index(drop=True)

        depth = 0
        initial_length = len(gdf_unique)
        final_iteration = False
        while (depth < self.max_depth) and (not final_iteration):
            depth+=1
            if depth==self.max_depth:
                final_iteration = True
            if len(gdf_tree)>0:
                gdf_tree, df_children, final_iteration = self._recursive_bfs(gdf_tree, depth, initial_length, final_iteration)
                #debug
                self.gdf_tree = gdf_tree
                self.df_children = df_children
            else:
                break
            if self.verbose:
                print('depth:', depth, '   gdf_idx:', len(self.gdf_idx))

        if self.verbose:
            print('Time for recursive BFS (A) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        # Finalize the quadtree index
        self._depth_to_level(level = self.target_level)
        self.gdf_idx = self.gdf_idx[[self.id_col, self.idx_col, self.key_col]]

        # Add a geometry column for the index bounds
        self.gdf_idx[self.idx_box_col] = self._base4_to_box(self.gdf_idx[self.idx_col])
        self.gdf_idx = geopandas.GeoDataFrame(self.gdf_idx, geometry=self.idx_box_col).set_crs(self.grid.crs)

        # Add the original geometry column (we will cut the polygons below)
        self.gdf_idx = gdf_unique[[self.id_col, self.geom_col]].merge(
            self.gdf_idx,
            on=self.id_col,
            how='right',            
        )

        # Negative buffer by something like epsilon=1e-13
        # Removing the annoying buffer warning
        s_bounds = self.gdf_idx[self.idx_box_col]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_bounds_buffer = s_bounds.buffer(-self.morton.epsilon)
        # debug: maybe this needs to be made permanent in column self.idx_box_col ?

        # Check for spatial containment of quadtree boxes within the original geometries
        self.gdf_idx['contained'] = self.gdf_idx.contains(s_bounds_buffer)
        
        # Add a geometry column containing the intersection of index bounds and original geometry
        self.gdf_idx[self.geom_col] = self.gdf_idx.intersection(s_bounds)

        if self.verbose:
            print('Time for finalizing gdf_idx (B) in seconds', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()

        return self.gdf_idx

    
    def _step_index_geodataframe(self, gdf, depth, key_col=None, id_col=None):
        """Calculate the next level of qtree index where possible.

        :param gdf:  Geopandas GeoDataFrame to be indexed.
        """
        if key_col is None:
            key_col = self.key_col
        if id_col is None:
            id_col = self.id_col
            
        cols = [id_col, self.geom_col]
        if key_col in gdf.columns:
            cols += [key_col]
            
        # Drop duplicate geometries before calculating spatial index (e.g. when we have multiple timestamps for the same geometry) 
        gdf_unique = gdf[cols].drop_duplicates(subset=id_col).reset_index(drop=True)

        if key_col not in gdf.columns:
            # Fast algorithm for smallest z-order squares that still contain the entire geometries.
            gdf_unique = self._index_root(gdf_unique)

        # gdf_idx builds upon the root geometries
        self.gdf_idx = gdf_unique.copy()
        self.gdf_idx['depth_'+str(depth-1)] = self.gdf_idx[key_col]
        
        # Determine the geometries for which we have to build an entire quadtree, rather than just the root node.
        gdf_tree = gdf_unique[gdf_unique[key_col].apply(len)<self.target_level+2].reset_index(drop=True)

        initial_length = len(gdf_unique)
        final_iteration = False
        if (depth < self.max_depth):
            if depth==self.max_depth:
                final_iteration = True
            if len(gdf_tree)>0:
                gdf_tree, df_children, final_iteration = self._recursive_bfs(
                    gdf_tree, depth, initial_length, final_iteration, key_col=key_col, id_col=id_col)
                #debug
                self.gdf_tree = gdf_tree
                self.df_children = df_children
            if self.verbose:
                print('depth:', depth, '   gdf_idx:', len(self.gdf_idx))

        # Finalize the quadtree index
        self._depth_to_level(level = self.target_level)
        self.gdf_idx = self.gdf_idx[[id_col, self.idx_col, key_col]]

        # Add a geometry column for the index bounds
        self.gdf_idx[self.idx_box_col] = self._base4_to_box(self.gdf_idx[self.idx_col])
        self.gdf_idx = geopandas.GeoDataFrame(self.gdf_idx, geometry=self.idx_box_col).set_crs(self.grid.crs)

        # Add the original geometry column (we will cut the polygons below)
        self.gdf_idx = gdf_unique[[id_col, self.geom_col]].merge(
            self.gdf_idx,
            on=id_col,
            how='right',            
        )

        # Negative buffer by something like epsilon=1e-13
        # Removing the annoying buffer warning
        s_bounds = self.gdf_idx[self.idx_box_col]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_bounds_buffer = s_bounds.buffer(-self.morton.epsilon)
        # debug: maybe this needs to be made permanent in column self.idx_box_col ?

        # Check for spatial containment of quadtree boxes within the original geometries
        self.gdf_idx['contained'] = self.gdf_idx.contains(s_bounds_buffer)
        
        # Add a geometry column containing the intersection of index bounds and original geometry
        self.gdf_idx[self.geom_col] = self.gdf_idx.intersection(s_bounds)

        return self.gdf_idx


    def _index_geodataframe_aerial(self, gdf):
        """Calculate the qtree index, and target keys at target_level for each geometry where possible.

        Create spatial index, aligned with GeoDN raster data.
        This method is typically run on each vectorstore partition separately.
        Optimized to deal with complicated polygons
        :param gdf:  Geopandas GeoDataFrame to be indexed.
        """
        initial_length = len(gdf)
        depth = 1
        # First iteration returns a dataframe at depth 1
        gdf2 = self._step_index_geodataframe(gdf, depth=depth) 
        
        gdf_finished = pandas.DataFrame()
        while depth<self.max_depth:
            depth+=1
            # Create the id_col for the next iteration
            gdf2 = gdf2.reset_index().rename(columns={'index': 'tmp_id'})
            
            # Use the generated index column as the key (root) for the next iteration
            gdf2 = gdf2[[c for c in gdf2.columns if c!=self.key_col]]
            gdf2 = gdf2.rename(columns={self.idx_col: self.key_col})
            
            # We can stop iterating when 
            # (1) the geometry fully contains the cell associated with the key
            mask1 = gdf2['contained']
            # or (2) we ave reached the target level
            mask2 = gdf2[self.key_col].apply(len)-2>=self.target_level
            
            gdf_finished = pandas.concat([gdf_finished, gdf2[mask1 | mask2]])
            gdf2 = gdf2[(~mask1) & (~mask2)]
        
            if len(gdf2)==0:
                # Done iterating
                break
        
            if (
                depth==self.max_depth
            ) or (
                (self.max_inflation is not None) and (len(gdf_finished)+len(gdf2) > initial_length * self.max_inflation)
            ):
                gdf_finished = pandas.concat([gdf_finished, gdf2])
                break
            else:
                # Iterate
                gdf_part3 = self._step_index_geodataframe(gdf2, depth=depth, id_col='tmp_id')
                
                # Keep track of the original geometry id
                gdf2 = pandas.merge(gdf_part3, gdf2[['tmp_id', self.id_col]], on='tmp_id')
                del gdf2['tmp_id']
        
        del gdf_finished['tmp_id']
        gdf_finished = gdf_finished.rename(columns={self.key_col: self.idx_col})
        self.gdf_idx = gdf_finished.reset_index(drop=True)

        return self.gdf_idx


    def _index_to_parquet(self, temporal_partition, spatial_partition, gdf_part=None, append=False):
        """Create a qtree spatial index for a specific partition."""

        if gdf_part is None:
            # Creating gdf_part from latest self.gdf_ingest
            gdf_part = self._filter_partition(temporal_partition, spatial_partition)

        # Only a few columns are needed for the index
        gdf_part = gdf_part[[self.id_col, self.geom_col, self.key_col]]

        # Removing the annoying buffer warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if gdf_part.loc[::100, self.geom_col].area.mean()>self.morton.resolution(max(0, self.max_level-4))[0]:
                if self.verbose:
                    print('Using algorithm optimized for Polygon geometries')
                # Create the index
                gdf_idx = self._index_geodataframe_aerial(gdf_part)
            else:
                if self.verbose:
                    print('Using algorithm optimized for Point or LineString geometries')
                # Create the index
                gdf_idx = self._index_geodataframe_point_or_line(gdf_part)

        filepath_idx = self._filepath_idx(temporal_partition, spatial_partition)
        if append:
            try:
                # See if there is something already present
                gdf_idx_existing = geopandas.read_parquet(filepath_idx)
            except IOError:
                pass
            else:
                # Merge by concatenating and dropping duplicates (keeping the newer version)
                gdf_idx = pandas.concat([gdf_idx_existing, gdf_idx]).drop_duplicates(
                    subset=[self.id_col, self.idx_col],
                    keep='last',
                )

        # ToDo: maybe we need to allow appending similar to _to_parquet method
        gdf_idx.to_parquet(
            path=filepath_idx,
            row_group_size=100000,
            engine='pyarrow',
            compression='snappy',
            #partition_cols=self.partition_cols
        )

    
    def _to_parquet(self, temporal_partition, spatial_partition, gdf_part=None, append=False):
        """Write GeoDataFrame partition to parquet."""
        
        if gdf_part is None:
            # Creating gdf_part from latest self.gdf_ingest
            gdf_part = self._filter_partition(temporal_partition, spatial_partition)

        filepath = self._filepath(temporal_partition, spatial_partition)
        if append:
            try:
                # See if there is something already present
                gdf_existing = geopandas.read_parquet(filepath)
            except IOError:
                pass
            else:
                # Merge by concatenating and dropping duplicates (keeping the newer version)
                gdf_part = pandas.concat([gdf_existing, gdf_part])
                len_before = len(gdf_part)
                gdf_part = gdf_part.drop_duplicates(
                    subset=self.primary_keys,
                    keep='last',
                )
                len_after = len(gdf_part)
                if len_before>len_after:
                    self._warning_duplicate_primary_key(len_before, len_after)

        if len(gdf_part)>0:
            # Sorting so that these columns are used as indices in parquet file
            gdf_part = gdf_part.sort_values(
                by=[self.dt_col]+list(self.dimension_values)+[self.key_col]
            ).reset_index(drop=True)

            gdf_part.to_parquet(
                path=filepath,
                row_group_size=100000,
                engine='pyarrow',
                compression='snappy',
                #partition_cols=self.partition_cols
            )


    def to_parquet(self, append=False, index=False):
        """Write entire GeoDataFrame (all partitions) to parquet."""
        if self.verbose:
            stopwatch_start = time.time()

        for spatial_partition in sorted(self.gdf_partition['partition'].drop_duplicates()):
            # To do: loop through temporal partitions (and other dimensions that we choose to make into partitions.
            temporal_partition = {}

            # Write the data to parquet partitions
            self._to_parquet(temporal_partition, spatial_partition, append=append)
            
            if index:
                # Write the index to parquet
                self._index_to_parquet(temporal_partition, spatial_partition, append=append)

        if self.verbose:
            print('Time for writing GeoDataFrame to GeoParquet partitions', round(time.time()-stopwatch_start, 3))
            stopwatch_start = time.time()


    def partial_upload(self, gdf, target_rows=None, index=False, validate_geometries=False):
        """Register GeoDataFrame, create partition, and save parquet to disk.

        :param gdf:         Geopandas GeoDataFrame to be indexed and written to disk.
        :param index:       Flag whether to create spatial index.
        """
        # We will be appending to existing partitions
        APPEND = True

        # Register data
        self.register_geodataframe(gdf, validate_geometries=validate_geometries)

        # Create suggestions for parquet file partitions
        self.create_partitions(target_rows)

        # Write the dataset to parquet
        self.to_parquet(append=APPEND, index=index)


    def _initialize_reproject(
        self,
        grid,
        max_level = None,
        target_level = None,
        max_depth = None,
        max_inflation = None,
        target_rows = None,
        valid_range = None,
    ):
        """Initialize a reprojected vectorstore.
        
        :param grid: target nestedgrid
        """
        if max_level is None: max_level = self.max_level
        if target_level is None: target_level = self.target_level
        if max_depth is None: max_depth = self.max_depth
        if max_inflation is None: max_inflation = self.max_inflation
        if target_rows is None: target_rows = self.target_rows

        vs2 = Vectorstore(
            dataset=self.dataset,
            dimension_values=self.dimension_values,
            grid=grid,
            dt_col=self.dt_col,
            geom_col=self.geom_col,
            id_col=self.id_col,
            max_level=max_level,
            target_level=target_level,
            max_depth=max_depth,
            max_inflation=max_inflation,
            target_rows=target_rows,
            valid_range=valid_range,
            verbose=self.verbose,
        )
        
        vs2.write_metadata()
        vs2.read_metadata()

        return vs2

    
    def _reproject_geodataframe(self, gdf, crs):
        """Reproject a single in-memory geodataframe"""
        try:
            return gdf.drop([self.key_col], axis=1).to_crs(crs)
        except:
            return gdf.to_crs(crs)
            
    
    def reproject(
        self,
        grid,
        max_level = None,
        target_level = None,
        max_depth = None,
        max_inflation = None,
        valid_range = None,
        target_rows = None, 
        index = False,
    ):
        """Reproject vectorstore.
        
        :param grid: target nestedgrid
        """
        vs2 = self._initialize_reproject(
            grid=grid,
            max_level=max_level,
            target_level=target_level,
            max_depth=max_depth,
            max_inflation=max_inflation,
            valid_range=valid_range,
            target_rows=target_rows,
        )

        df_partitions = self._glob_partitions()
        source_partitions = list(df_partitions['partition'])
        for source_partition in source_partitions:
            gdf = geopandas.read_parquet(self._filepath({}, source_partition))

            # Reproject
            gdf2 = self._reproject_geodataframe(gdf, grid.crs)
            
            # Upload to new vectorstore
            vs2.partial_upload(
                gdf2,
                target_rows=target_rows,
                index=index,
                validate_geometries=False, #No need to validate since reprojecting from good geometries
            )

        return vs2


    def _read_parquet(
        self,
        filepath,
        columns=None,
        filters=None,  # List[Tuple] or List[List[Tuple]]
        q_key=None,    # inclusive all children
        dt_start=None, # inclusive
        dt_end=None,   # exclusive
    ):
        """Query the parquet vector store using spatial, temporal and/or dimension filters."""
        if filters is None:
            filters = []
        if q_key is not None:
            key, level = self.morton.decode(q_key)
            q_key_end = self.morton.encode(key+1, level)[0]
            filters.append((self.key_col, '>=', q_key))
            filters.append((self.key_col, '<', q_key_end))
            
        if dt_start is not None:
            filters.append((self.dt_col, '>=', dt_start))
        if dt_end is not None:
            filters.append((self.dt_col, '<', dt_end))
            
        if self.verbose:
            print('filepath', filepath)
            print('columns ', columns)
            print('filters ', filters)
        try:
            if len(filters)>0:
                if (columns is None) or (self.geom_col in columns):
                    gdf = geopandas.read_parquet(filepath, columns=columns, filters=filters)
                else:
                    gdf = pandas.read_parquet(filepath, columns=columns, filters=filters)
            else:
                if (columns is None) or (self.geom_col in columns):
                    gdf = geopandas.read_parquet(filepath, columns=columns)
                else:
                    gdf = pandas.read_parquet(filepath, columns=columns)
        except Exception as e:
            if self.verbose:
                print(e)
                print('Problem loading file', filepath)
            gdf = geopandas.GeoDataFrame()
            
        return gdf


    def write_metadata(self):
        """
        Dump the settings to a json file
        """
        vs_settings = {}
        vs_settings['dataset'] = self.dataset
        vs_settings['vectorstore_directory'] = self.vectorstore_directory
        vs_settings['dataset_directory'] = self.dataset_directory
        vs_settings['grid_directory'] = self.grid_directory
        vs_settings['geoparquet_directory'] = self.geoparquet_directory
        vs_settings['index_directory'] = self.index_directory
        vs_settings['temporal_levels'] = self.temporal_levels
        vs_settings['dimension_values'] = self.dimension_values
        vs_settings['dt_col'] = self.dt_col
        vs_settings['geom_col'] = self.geom_col
        vs_settings['id_col'] = self.id_col
        vs_settings['key_col'] = self.key_col
        vs_settings['idx_col'] = self.idx_col
        vs_settings['idx_box_col'] = self.idx_box_col
        vs_settings['geom_area_col'] = self.geom_area_col
        vs_settings['geom_length_col'] = self.geom_length_col
        vs_settings['primary_keys'] = self.primary_keys
        vs_settings['max_level'] = self.max_level
        vs_settings['target_level'] = self.target_level
        vs_settings['max_depth'] = self.max_depth
        vs_settings['max_inflation'] = self.max_inflation
        vs_settings['target_rows'] = self.target_rows
        vs_settings['grid'] = self.grid.__repr__()
        vs_settings['valid_range'] = shapely.to_geojson(self.valid_range)

        json_path = os.path.join(self.grid_directory, 'metadata.json').replace('\\', '/')
        with open(json_path, 'w') as f:
            json.dump(vs_settings, f)

    
    def read_metadata(self):
        json_path = os.path.join(self.grid_directory, 'metadata.json').replace('\\', '/')
        with open(json_path) as f:
            vs_settings = json.load(f)
        for k in vs_settings:
            if k=='valid_range':
                vs_settings[k] = shapely.from_geojson(vs_settings[k])
            elif k=='grid':
                # Recreate a grid object from __repr__()
                vs_settings[k] = eval("nestedgrid." + vs_settings[k])
            setattr(self, k, vs_settings[k])
            


    
    # def _generate_composite_keys(self, df):
    #     """
    #     Combine the temporal, spatial and dimension keys into one "composite_key"
    #     """
    #     df[self.composite_key_col] = 'level' + df[self.spatial_level_col].astype(str) + '_' + df[self.spatial_key_col].astype(str) 
    #     for k in self.temporal_keys + self.dimension_keys:
    #         df[self.composite_key_col] = df[self.composite_key_col] + '_' + k + '_' + df[k].astype(str)
    #     return df[self.composite_key_col]

    # def _decompose_composite_keys(self, df):
    #     """
    #     Decompose the composite_key into temporal, spatial, and dimension keys
    #     """
    #     cols = [self.spatial_level_col, self.spatial_key_col]
    #     df[cols] = df[self.composite_key_col].apply(lambda x: x.split('level', 1)[1].split('_', 2)[:2]).to_list()
    #     df[self.spatial_level_col] = df[self.spatial_level_col].astype(int)
    #     df[self.spatial_key_col] = df[self.spatial_key_col].astype(int)
    #     for k in self.temporal_keys:
    #         cols = cols + [k]
    #         df[k] = df[self.composite_key_col].apply(lambda x: int(x.split(k+'_')[1].split('_')[0]))
    #     for k in self.dimension_keys: 
    #         cols = cols + [k]
    #         df[k] = df[self.composite_key_col].apply(lambda x: x.split(k+'_')[1].split('_')[0]) # Dimension keys are of type str
    #     return df[cols]

    # def _all_the_same(self, s):
    #     # Quick test if column values (pandas series s) are all the same
    #     a = s.to_numpy()
    #     return (a[0] == a).all()
    
    # def _reproject_to_local_equal_area_grid(self, gdf):
    #     """
    #     All geometries must belong to the same spatial cell
    #     """
    #     assert(self._all_the_same(gdf[self.spatial_key_col]))
        
    #     # Get the center lat/lon of the spatial cell
    #     key = gdf.loc[0, self.spatial_key_col]
    #     level = gdf.loc[0, self.spatial_level_col]
    #     lat, lon = self.morton.center_coordinates(key, level)

    #     # local_azimuthal_projection preserves angle (e.g. circles stay circles)
    #     #local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={lat} +lon_0={lon}"

    #     # Lambert Azimuthal Equal Area preserves area 
    #     lambert_azimuthal_ea = f"+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"

    #     #gdf['area_local_azimuthal_projection'] = gdf.to_crs(local_azimuthal_projection).area
    #     #gdf['area_lambert_azimuthal_ea'] = gdf.to_crs(lambert_azimuthal_ea).area
        
    #     return gdf.to_crs(lambert_azimuthal_ea)
        

    # def read_selected_parquet(
    #     self, composite_key, dt_start=None, dt_end=None, columns=None, geometry_area=False, geometry_length=False, verbose=False
    # ):
    #     try:
    #         _, filepath = self._select_parquet_file(self.gdf_meta, composite_key)
    #     except:
    #         # Make sure gdf_meta is present
    #         self.metadata_from_parquet()
    #         _, filepath = self._select_parquet_file(self.gdf_meta, composite_key)
    #     gdf = self._read_parquet(
    #         filepath, 
    #         dt_start=dt_start, 
    #         dt_end=dt_end, 
    #         columns=columns, 
    #         geometry_area=geometry_area, 
    #         geometry_length=geometry_length, 
    #         verbose=verbose,
    #     )
    #     return gdf
    #
    #    
    # def to_parquet(
    #     self, 
    #     append=False, 
    #     geometry_area=False, 
    #     geometry_length=False, 
    #     remove_metadata_cols=False, 
    #     rename_cols={}, 
    #     verbose=False
    # ):
    #     """
    #     Write GeoDataFrame to parquet
    #     """
    #     if verbose:
    #         stopwatch_start = time.time()
    #     self._get_partitions()
        
    #     # Save the parquet files by composite_key in a partitioned directory structure
    #     self.composite_keys = sorted(self.gdf_intersection.composite_key.drop_duplicates())
    #     if len(self.composite_keys)>10000:
    #         # Currently each dimension value becomes part of the composite key, which determines the parquet partition filename
    #         # Raising an error when we get too many combinations
    #         # Often this is due to bad dimension_keys definition
    #         # To Do: allow different dimension values to be stored in the same parquet file
    #         print('self.gdf_intersection', len(self.gdf_intersection))
    #         print('self.composite_keys', len(self.composite_keys))
    #         print('self.dimension_keys', self.dimension_keys)
    #         raise ValueError('Too many composite_keys. Maybe we have a wrong dimension_keys definition?')

    #     for composite_key in self.composite_keys:
    #         gdf_part, filepath = self._select_parquet_file(self.gdf_intersection, composite_key)
    #         if len(gdf_part)==0:
    #             print('WARNING: no data to write')
    #         else:
    #             # Add area and/or length columns if requested
    #             if geometry_area or geometry_length:
    #                 gdf_unique = gdf_part[
    #                     [self.id_col, self.geom_col, self.spatial_key_col, self.spatial_level_col]
    #                 ].drop_duplicates(subset=self.id_col).reset_index(drop=True)
    #                 gdf_lambert_ea = self._reproject_to_local_equal_area_grid(gdf_unique)
    #                 if geometry_area:
    #                     gdf_unique[self.geom_area_col] = gdf_lambert_ea.area / 1e6 # units: [km2]
    #                 if geometry_length:
    #                     gdf_unique[self.geom_length_col] = gdf_lambert_ea.length / 1e3 # units: [km]
    #                 del gdf_unique[self.geom_col]
    #                 gdf_part = pandas.merge(gdf_part, gdf_unique, on=[self.id_col, self.spatial_key_col, self.spatial_level_col])
                    
    #             if remove_metadata_cols:
    #                 del gdf_part[self.spatial_key_col]
    #                 del gdf_part[self.spatial_level_col]
    #                 del gdf_part[self.composite_key_col]
    #                 del gdf_part[self.intersection_flag_col]
                    
    #             if len(rename_cols)>0:
    #                 gdf_part = gdf_part.rename(columns=rename_cols)

    #             if append:
    #                 try:
    #                     # See if there is something already present under this composite key
    #                     df_existing = geopandas.read_parquet(filepath)
    #                 except:
    #                     pass
    #                 else:
    #                     # Merge the two by concatenating and dropping duplicates
    #                     gdf_part = pandas.concat([df_existing, gdf_part]).drop_duplicates().reset_index(drop=True)
    #             gdf_part.to_parquet(
    #                 path=filepath,
    #                 engine='pyarrow',
    #                 compression='snappy',
    #                 #partition_cols=self.partitions
    #             )
                
    #     # Write the settings to a json file
    #     self.write_vectorstore_settings()
    #     if self.verbose:
    #         print('Time for to_parquet in seconds', round(time.time()-stopwatch_start, 3))
            
    def metadata_from_parquet(self, verbose=False):
        glob_wildcard_path = self.grid_directory
        for partition_name in self.partitions:
            #partition_value = df_part.loc[0, partition_name]
            glob_wildcard_path = os.path.join(glob_wildcard_path, partition_name + '*').replace('\\', '/')
        glob_wildcard_path = os.path.join(glob_wildcard_path, self.dataset+'*.parquet').replace('\\', '/')
        if verbose:
            print('glob_wildcard_path', glob_wildcard_path)
        globbed_filepaths = sorted(glob(glob_wildcard_path))
        
        # Remove the dataset directory and the filetype ('.parquet') from the paths
        globbed = [g.split(self.grid_directory)[-1] for g in globbed_filepaths]
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
        self.query_df_meta[self.spatial_key_col] = self.morton.get_key(query_latitude, query_longitude, self.spatial_level)
        self.query_df_meta = pandas.DataFrame([self.query_df_meta])

        # Spatial key for the partition
        for sp in self.spatial_partitions:
            levels_up = self.spatial_level-self._spatialPartitionLevel(sp)
            self.query_df_meta[sp] = self.morton.parent_key(numpy.array(self.query_df_meta[self.spatial_key_col].astype(int)), levels_up)

        self.query_df_meta[self.composite_key_col] = self._generate_composite_keys(self.query_df_meta)
        composite_key = self.query_df_meta[self.composite_key_col].values[0]
        _, filepath = self._select_parquet_file(self.query_df_meta, composite_key)

        # Load a complete single parquet file
        try:
            gdf_query = self._read_parquet(filepath, columns=columns, geometry_area=geometry_area, geometry_length=geometry_length)
        except FileNotFoundError as e:
            gdf_query = geopandas.GeoDataFrame()

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
            gdf_query = geopandas.GeoDataFrame()
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
                    gdf_right[['polygon_id', self.geom_col]], 
                    how='intersection'
                ).reset_index(drop=True)
            elif self.how=='left':
                # Return intersecting polygons intact
                if self.intersection_policy=='cut':
                    # To do: use geometry_id to find all locations of this polygon and stitch together.
                    print('NOT IMPLEMENTED WARNING: Asking for original polygons but we are returning the cut ones.')
                    gdf = geopandas.sjoin(
                        gdf_left, 
                        gdf_right[['polygon_id', self.geom_col]], 
                        how='left', 
                        predicate='intersects'
                    ).dropna(subset=['index_right']).drop(columns=['index_right'])
                elif self.intersection_policy=='original':
                    # Data has been saved as complete polygons
                    gdf = geopandas.sjoin(
                        gdf_left, 
                        gdf_right[['polygon_id', self.geom_col]], 
                        how='left', 
                        predicate='intersects'
                    ).dropna(subset=['index_right']).drop(columns=['index_right'])
                elif self.intersection_policy=='both':
                    # Pick the original geometry column with complete polygons
                    gdf = geopandas.sjoin(
                        gdf_left.set_geometry(self.geom_col + '_original', crs=4326), 
                        gdf_right[['polygon_id', self.geom_col]], 
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
        composite_keys = list(gdf_right2[self.composite_key_col].drop_duplicates())
        
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
                    self.gdf_meta.loc[self.gdf_meta[self.composite_key_col]==composite_key, self.geom_col].values[0]
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

        # Deal with multiple rows for same id_col, polygon_id combination
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