"""Partitioning tables so they can be accessed rapidly and appended to easily.

Classes:

    TemporalPartition:  Temporal partitions based on attributes such as year, month, day, hour
    SpatialPartition:   Spatial partition are split on demand, when the data becomes too big.
    UniformSpatialPartition: Spatial partitions at uniform partition level throughout.
    
"""

import numpy
import pandas
import geopandas

import mortoncurve
import qtree


class TemporalPartition():
    """Partitioning tables into separate files.

    Distribute data (rows) into separate files for each partition.
    Temporal partitions based on attributes such as year, month, day, hour

    Attributes:

        raster_or_vector   Are we dealing with raster or vector data?
        dt_col             Name of timestamp column in DataFrame.
        verbose            Set True for debugging

    Methods

        get_temporal_partition_levels  Collect temporal statistics and infer appropriate
                                       partition levels.
        get_temporal_partitions        Based on timestamps and temporal levels calculate
                                       a dictionary of temporal partitions (keys) and the
                                       timestamps belonging to them (values are lists).

    """
    # Default values for class attributes
    DT_COL                     = 'timestamp'
    VALID_TEMPORAL_LEVELS      = ['year', 'month', 'day', 'hour']

    def __init__(
        self,
        raster_or_vector,
        timestamps,
        dt_col = None,
        verbose = False,
    ):
        # Are we dealing with raster or vector data?
        self.raster_or_vector = raster_or_vector
        assert self.raster_or_vector in ['raster', 'vector']

        # List of available timestamps
        self.timestamps = timestamps

        # Dict of statistics about timestamps and temporal partitions
        self.stats = {}
        self.stats['number_global_timestamps'] = len(self.timestamps)

        # Timestamp column name in DataFrames
        self.dt_col = self.DT_COL if dt_col is None else dt_col

        self.valid_temporal_levels = self.VALID_TEMPORAL_LEVELS
        
        # For later use
        self.temporal_levels = None
        self.temporal_partitions = {}

        self.verbose = verbose

    def get_temporal_partition_levels(self, temporal_levels=None):
        """Collect temporal statistics and infer appropriate partition levels."""

        # Get attributes (year, month, day) for each timestamp
        df_timestamps = pandas.DataFrame(self.timestamps).rename(columns={0: self.dt_col})
        df_timestamps['year'] = df_timestamps[self.dt_col].dt.year
        df_timestamps['month'] = df_timestamps[self.dt_col].dt.month
        if temporal_levels is not None:
            if ('day' in temporal_levels):
                df_timestamps['day'] = df_timestamps[self.dt_col].dt.day
            if ('hour' in temporal_levels):
                df_timestamps['hour'] = df_timestamps[self.dt_col].dt.hour

        # Maximum timestamps per year
        self.stats['max_y_count'] = df_timestamps.groupby(
            'year'
        )[self.dt_col].count().max()
        # Maximum timestamps per month
        self.stats['max_ym_count'] = df_timestamps.groupby(
            ['year', 'month']
        )[self.dt_col].count().max()

        if temporal_levels is None:
        # Choose appropriate temporal levels
            if self.stats['max_ym_count']>=10000:
                # We don't have any datasets with high enough frequency to justify daily partitions
                self.temporal_levels = ['year', 'month']
                #self.temporal_levels = ['year', 'month', 'day']
            elif self.stats['max_y_count']>=10000:
                self.temporal_levels = ['year', 'month']
            elif self.stats['max_y_count']>=1000:
                self.temporal_levels = ['year']
            else:
                self.temporal_levels = []
        else:
            assert set(temporal_levels) <= set(self.valid_temporal_levels)
            self.temporal_levels = temporal_levels
        if self.verbose:
            print('temporal_levels                 ', self.temporal_levels)

        # Count the number of timestamps per temporal partition
        if len(self.temporal_levels)>0:
            df_grouped = df_timestamps.groupby(self.temporal_levels).count()[self.dt_col]
            self.stats['max_ts_per_temp_part'] = df_grouped.max()
            self.stats['number_temporal_partitions'] = len(df_grouped)
        else:
            self.stats['max_ts_per_temp_part'] = len(df_timestamps)
            self.stats['number_temporal_partitions'] = 1
        if self.verbose:
            print('number_temporal_partitions      ', self.stats['number_temporal_partitions'])
            print('max_ts_per_temp_part            ', self.stats['max_ts_per_temp_part'])

    def get_temporal_partitions(self):
        """Based on timestamps and temporal levels calculate a list of temporal partitions.

        :param df_timestamps:   Pandas DataFrame with the timestamp information
        :param temporal_levels: List of temporal levels, such as ['year', 'month', 'day']
        """
        if len(self.temporal_levels)==0:
            self.temporal_partitions = {} # No temporal partitions
            return

        # Get the the unique values of the temporal levels
        df_timestamps = pandas.DataFrame(self.timestamps).rename(columns={0: self.dt_col})
        for t_level in self.temporal_levels:
            self.temporal_partitions[t_level] = sorted(
                df_timestamps[self.dt_col].apply(
                    lambda x: getattr(x, t_level)
                ).drop_duplicates()
            )

        if self.verbose:
            print()
            print('temporal_partitions ', self.temporal_partitions)
            print()


class SpatialPartition():
    """Partitioning tables into separate files.

    Distribute data (rows) into separate files for each partition.
    Spatial partitions identified by the quaternary key in the filename.
    Partition are split on demand, when the data becomes too big.

    Attributes:

        raster_or_vector    Are we dealing with raster or vector data?
        max_spatial_level   Maximum spatial level to consider as split partition.
        split_n             Maximum number of geometries in partition before a split is initiated
        id_col              ID column name. Typically 'q_key' for raster and 'geom_id' for vector
        max_depth           Vector only: maximum depth of trees as measured from head node
        simplify_n          Vector only: simplify the trees
        verbose             Set True for debugging

    Methods

        split_partition     Splitting a partition into 2-4 children partitions.
        _qtree_columns      Create a separate quadtree for every geometry in gdf.
        planned_partitions  Partition planning for an existing (historical) raster layer.

    """
    # Default values for class attributes
    # Geopandas relies on the geometry column being named 'geometry', so enforce this
    GEOM_COL       = 'geometry'
    PARTITION_COL  = 'partition'
    RASTER_KEY_COL = 'q_key'
    VECTOR_KEY_COL = 'geom_id'

    def __init__(
        self,
        raster_or_vector,
        max_spatial_level,
        split_n,
        id_col = None,
        key_col = None,
        max_depth = None,
        simplify_n = None,
        grid = None,
        verbose = False,
    ):

        # Nested grid to overwrite the default PAIRS grid
        self.grid = grid

        # Definition of the morton curve on top of the nested grid
        self.morton = mortoncurve.Morton(self.grid)

        # Are we dealing with raster or vector data?
        self.raster_or_vector = raster_or_vector
        assert self.raster_or_vector in ['raster', 'vector']

        # Maximum spatial level to consider as split partition.
        # Typically equal to the overview level.
        self.max_spatial_level = max_spatial_level

        # Maximum number of geometries in partition before a split is initiated
        self.split_n = split_n

        # Geometry id column name
        # typically 'q_key' for raster and 'geom_id' for vector
        if id_col is None:
            if self.raster_or_vector=='raster':
                self.id_col = 'q_key'
            elif self.raster_or_vector=='vector':
                self.id_col = 'geom_id'
            else:
                raise NotImplementedError
        else:
            self.id_col = id_col

        # Geometry column name
        self.geom_col = self.GEOM_COL
        # Partition column name
        self.partition_col = self.PARTITION_COL
        # Key column name
        if key_col is None:
            if self.raster_or_vector=='raster':
                self.key_col = self.RASTER_KEY_COL
            elif self.raster_or_vector=='vector':
                self.key_col = self.VECTOR_KEY_COL
            else:
                raise ValueError('raster_or_vector not understood.')
        else:
            self.key_col = key_col

        # Vector only: Additional QTree attributes used to grow the individual trees
        # for each geometry. These are stored in dedicated columns of the DataFrame.
        # They are only needed for vector overviews, not the simpler raster overviews,
        # where the trees would all be very narrow (exactly 1 child at all levels).
        # Maximum depth of tree (measured from level before first split (head)).
        self.max_depth = max_depth
        # Simplify the quadtree by merging n=4 or n=3+ quadrants
        self.simplify_n = simplify_n

        # For later use
        self.spatial_partitions = {}

        # Dict of statistics about timestamps, overview keys, and spatio-temporal partitions
        self.stats = {}

        self.verbose = verbose

    def split_partition(
        self, gdf: geopandas.GeoDataFrame, parent_partition: str
    ) -> geopandas.GeoDataFrame:
        """Splitting a partition into 2-4 children partitions.

        Operationally it's faster to partition by overlaying (intersection) polygon with 
        partition boxes first. Then grow full qtrees again
        :param gdf:        (partial) GeoDataFrame with parent geometry and id_col
        :parent_partition: quaternary hash of the parent node that will be split
        """
        # The 4 possible children keys
        partitions = numpy.array([parent_partition + q for q in ['0', '1', '2', '3']])
        print('partitions', partitions)

        # GeoDataFrame of 4 boxes
        gdf_boxes = pandas.DataFrame([partitions, self.morton.base4_to_box(partitions)]).T.rename(
            columns={0: self.partition_col, 1: self.geom_col})
        gdf_boxes = geopandas.GeoDataFrame(
            gdf_boxes, geometry=self.geom_col
        ).set_crs(self.morton.grid.crs)

        # Vectorized intersections of all polygons with all boxes
        gdf_parts = geopandas.overlay(gdf[[self.id_col, self.geom_col]], gdf_boxes)

        return gdf_parts

    def _qtree_columns(self, gdf: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
        """Create a separate quadtree for every geometry in gdf."""
        # initialize QTree and find the root node for each gemetry
        gdf['qt'] = gdf[self.geom_col].apply(lambda x: qtree.QTree(
            x,
            self.max_spatial_level,
            max_depth=self.max_depth,
            simplify_n=self.simplify_n,
            grid=self.grid,
        ))
        # apply the quadtree algorithm
        gdf['qt'].apply(lambda x: x.quadtree_dfs())

        # get a breadth-first view
        gdf['qt_dict'] = gdf['qt'].apply(lambda x: x.walker())
        return gdf

    def planned_partitions(self, gdf_unique):
        """Partition planning for an existing (historical) raster layer.

        Try to make the partitions hold similar numbers of rows
        :param gdf_unique: Geopandas dataframe with the unique geometries
        """
        # Starting partition planning from scratch
        gdf_partition = gdf_unique.copy()
        gdf_partition[self.partition_col] = '0q' # Level 0 has empty quaternary hash

        # Statistics of the key distribution (count of geoms per key) at successive levels
        df_dict = {}
        level = -1
        parents_to_split = []
        while ((level<1) or len(parents_to_split)>0) and (level<self.max_spatial_level):
            level+=1
            if 'q_key' in gdf_partition.columns:
                # When dealing with raster data, we usually have a q_key column already. (Fast.)
                key_instances = list(gdf_partition['q_key'].str[:level+2])
            else:
                # When dealing with vector data, we instead rely on the quadtrees for each geometry.
                key_instances = list(gdf_partition['qt_dict'].apply(
                    lambda x: [n.encode() for n in x[level]]
                ))
                key_instances = [item for sublist in key_instances for item in sublist]

            # Groupby counts
            df_dict[level] = pandas.DataFrame(
                key_instances).reset_index().rename(columns={0: 'q_key'})
            df_dict[level] = df_dict[level].groupby(
                'q_key').count().rename(columns={'index': 'count'})
            df_dict[level] = df_dict[level].reset_index()
            df_dict[level]['parent_partition'] = df_dict[level]['q_key'].str[:-1]

            # The first round (l=0) we only collected some stats in df[0] for later comparison
            if level>0:
                # Compare statistics on parent level (x) and child level (y)
                df_count = pandas.merge(
                    df_dict[level-1],
                    df_dict[level],
                    left_on='q_key',
                    right_on='parent_partition'
                )

                # Drop small parent partitions from consideration
                df_count = df_count[df_count['count_x']>self.split_n]

                df_count['reduction'] = df_count['count_x'] - df_count['count_y']

                # Groupby sum up to see how many geometries will be affected by split
                df_sum = df_count[['q_key_x', 'count_x', 'count_y']].groupby(
                    ['q_key_x', 'count_x']).sum().reset_index()

                df_sum['increase'] = df_sum['count_y'] - df_sum['count_x']

                # List of parents to split on this level
                parents_to_split = list(df_sum['q_key_x'].values)

                if self.verbose:
                    print()
                    print('LEVEL', level)
                    print('Reduction in size of biggest partition(s) due to splits')
                    print(df_count[['q_key_y', 'count_x', 'count_y', 'reduction']])
                    print()
                    print('Increase in the number of geometries due to splits')
                    print(df_sum)
                    print()
                    print('Parents to split: ', parents_to_split)

                # The part of the DataFrame that does not split
                gdf_keep = gdf_partition[
                    ~gdf_partition[self.partition_col].isin(df_count['q_key_x'])]

                # The part of the DataFrame that does the split
                gdf_split = gdf_partition[
                    gdf_partition[self.partition_col].isin(df_count['q_key_x'])]

                # Do the split
                for parent_partition in parents_to_split:
                    gdf = self.split_partition(gdf_split, parent_partition)
                    if self.raster_or_vector=='vector':
                        # Add the forest of trees
                        gdf = self._qtree_columns(gdf)
                    gdf_keep = pandas.concat([gdf_keep, gdf])

                gdf_partition = gdf_keep.reset_index(drop=True)

        # Spatial partitions
        self.spatial_partitions = sorted(gdf_partition[self.partition_col].unique())

        # Counts per spatial partition
        grouper = gdf_partition[[self.partition_col, self.key_col]].groupby(
            self.partition_col
        )[self.key_col]
        df_partition_count = grouper.count().sort_index()
        df_partition_count = df_partition_count.reset_index().rename(
            columns={self.key_col: 'count'}
        )
        df_partition_count = df_partition_count.reset_index(drop=True)

        # Spatial stats
        self.stats['number_spatial_partitions'] = len(self.spatial_partitions)
        self.stats['max_ovw_keys_per_spatial_part'] = df_partition_count['count'].max()

        return df_partition_count

class UniformSpatialPartition():
    """Partitioning tables into separate files.

    Distribute data (rows) into separate files for each partition.
    Spatial partitions identified by the quaternary key in the filename.
    Spatial partitions at uniform partition level throughout.

    Attributes:

        raster_or_vector    Are we dealing with raster or vector data?
        spatial_level       Spatial level of the partition keys.
        id_col              ID column name. Typically 'q_key' for raster and 'geom_id' for vector

    Methods

        planned_partitions  Partition planning for an existing (historical) raster layer.

    """
    # Default values for class attributes
    # Geopandas relies on the geometry column being named 'geometry', so enforce this
    GEOM_COL       = 'geometry'
    PARTITION_COL  = 'partition'
    RASTER_KEY_COL = 'q_key'
    VECTOR_KEY_COL = 'geom_id'

    def __init__(
        self,
        raster_or_vector,
        spatial_level,
        id_col = None,
        key_col = None,
    ):
        # Are we dealing with raster or vector data?
        self.raster_or_vector = raster_or_vector
        assert self.raster_or_vector in ['raster', 'vector']

        # Uniform spatial level of the partitions and query keys
        self.spatial_level = spatial_level

        # Geometry id column name
        # typically 'q_key' for raster and 'geom_id' for vector
        if id_col is None:
            if self.raster_or_vector=='raster':
                self.id_col = 'q_key'
            elif self.raster_or_vector=='vector':
                self.id_col = 'geom_id'
            else:
                raise NotImplementedError
        else:
            self.id_col = id_col

        # Geometry column name
        self.geom_col = self.GEOM_COL
        # Partition column name
        self.partition_col = self.PARTITION_COL
        # Key column name
        if key_col is None:
            if self.raster_or_vector=='raster':
                self.key_col = self.RASTER_KEY_COL
            elif self.raster_or_vector=='vector':
                self.key_col = self.VECTOR_KEY_COL
            else:
                raise ValueError('raster_or_vector not understood.')
        else:
            self.key_col = key_col

        # For later use
        self.spatial_partitions = {}

        # Dict of statistics about timestamps, overview keys, and spatio-temporal partitions
        self.stats = {}

    def planned_partitions(self, gdf_unique):
        """Partition planning for an existing (historical) raster layer.

        Make all partitions the same spatial_level
        :param gdf_unique: Geopandas dataframe with the unique geometries
        """
        # Starting partition planning from scratch
        gdf_partition = gdf_unique.copy()
        gdf_partition[self.partition_col] = gdf_partition[self.id_col].str[:self.spatial_level+2]

        # Spatial partitions
        self.spatial_partitions = sorted(gdf_partition[self.partition_col].unique())
        print('len(spatial_partitions) ', len(self.spatial_partitions))
        if len(self.spatial_partitions)<100:
            print('spatial_partitions      ', self.spatial_partitions)

        # Counts per spatial partition
        grouper = gdf_partition[[self.partition_col, self.key_col]].groupby(
            self.partition_col
        )[self.key_col]
        df_partition_count = grouper.count().sort_index()
        df_partition_count = df_partition_count.reset_index().rename(
            columns={self.key_col: 'count'}
        )
        df_partition_count = df_partition_count.reset_index(drop=True)

        # Spatial stats
        self.stats['number_spatial_partitions'] = len(self.spatial_partitions)
        self.stats['max_ovw_keys_per_spatial_part'] = df_partition_count['count'].max()

        return df_partition_count
