"""Generate raster overviews (Hierarchical Spatial Index HSI)

    Classes

        Rasteroverview    Generating raster overviews.
"""
import os
os.environ['USE_PYGEOS'] = '0'
import sys
from glob import glob
import json
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool
import numpy
import pandas
import geopandas
import shapely
import pytz
import xarray
import dask
import itertools

import dataservice.query

sys.path.insert(1, os.path.abspath(".."))
from qtree_index import nestedgrid, mortoncurve, qtree


class Rasteroverview():
    """Generating raster overviews.

    Generates Hirarchical Spatial Inices and persist them in spatially and/or temporally partitioned parquet files.

    Attributes:

        pixel_level             Pixel level
        delta_pixel_hsi         Difference between pixel level and hsi level.
        hsi_directory           Base directory where dataset hsi are stored.
        dset_id                 Dataset ID
        layer_id                Layer ID
        dimension_values        Dictionary of valid dimension values indexed by dimension_names
        dataservice_type        Currently supported: 'hbase', 'local_filesystem', 'remote_filesystem'
        dt_col                  Name of timestamp column in DataFrame.
        geom_col                Name of geometry column in DataFrame.
        key_col                 Name of spatial key column in DataFrame.
        numeric_or_categorical  Flag to indicate numeric or categorical type of statistics.
        stats                   Statistics (metadata) concerning available timestamps,
                                hsi keys, and spatio-temporal partitions.
        hsi_keys                List of the unique hsi keys
        temporal_levels         List of the timestamp hierarchy levels ('year', 'month', ...)
        temporal_partitions     Dictionary of the temporal partitions
        spatial_partition_level Level of the spatial partitions
        spatial_partitions      List of the spatial partitions
        timestamps              available timestamps in datetime format
        
    Methods

        available_timestamps    Query the dataservice for global timestamps.
        qtree_hsi               Determine hsi cells using qtree algorithm.
        partitions              Let raster object know about the partitions.
        xarray_stats_numeric    Area-weighted statistics for the spatial dimensions of an xarray.
        xarray_stats_categorical  Unweighted categorical statistics for the spatial dimensions 
                                of an xarray.
        hsi_statistics          Calculate HSI and append or write to file.
        to_parquet              Write GeoDataFrame partition to parquet.
        to_dataframe            Compose DataFrame from the layer's parquet files.
        to_zarr                 Write entire GeoDataFrame to zarr.
        to_csv                  Write entire GeoDataFrame to csv.
        to_json                 Dump attributes to a json file.
        from_json               Load attributes from a json file.

    """
    # Default values for class attributes
    DATASERVICE_TYPE           = 'hbase'
    HSI_DIRECTORY              = 'data/raster/hsi/'
    TMP_DIRECTORY              = 'data/raster/tmp'
    DELTA_PIXEL_HSI            = 5
    MAX_QUERY_PIXELS           = 5e8
    DT_COL                     = 'time'
    # Geopandas relies on the geometry column being named 'geometry', so enforce this
    GEOM_COL                   = 'geometry'
    KEY_COL                    = 'q_key'
    NUMERIC_OR_CATEGORICAL     = 'numeric'

    QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    ISO_8601 = '%Y-%m-%dT%H:%M:%SZ'

    def __init__(
        self,
        pixel_level,
        delta_pixel_hsi = None,
        hsi_directory = None,
        tmp_directory = None,
        dset_id = None,
        layer_id = None,
        dimension_values = {},
        dataservice_type = None,
        valid_range = None,
        dt_col = None,
        geom_col = None,
        key_col = None,
        numeric_or_categorical = None,
        grid = None,
        verbose = False,
    ):

        # Nested grid to overwrite the default PAIRS grid
        self.grid = grid

        # Definition of the morton curve on top of the nested grid
        self.morton = mortoncurve.Morton(self.grid)

        # layer attributes
        self.dset_id          = dset_id
        self.layer_id         = layer_id
        self.pixel_level      = pixel_level
        self.dimension_values = dimension_values

        # Overview level calculated relative to pixel level
        if delta_pixel_hsi is None:
            self.delta_pixel_hsi = self.DELTA_PIXEL_HSI
        else:
            self.delta_pixel_hsi = delta_pixel_hsi
        self.hsi_level   = self.pixel_level - self.delta_pixel_hsi

        # Type of dataservice (e.g. 'hbase', 'local_filesystem', 'remote_filesystem')
        if dataservice_type is None:
            self.dataservice_type = self.DATASERVICE_TYPE
        else:
            self.dataservice_type = dataservice_type

        # hsi_directory base directory
        if hsi_directory is None:
            self.hsi_directory = self.HSI_DIRECTORY
        else:
            self.hsi_directory = hsi_directory

        # tmp_directory
        if tmp_directory is None:
            self.tmp_directory = self.TMP_DIRECTORY
        else:
            self.tmp_directory = tmp_directory

        # Table specific column information
        self.dt_col = self.DT_COL if dt_col is None else dt_col
        self.key_col = self.KEY_COL if key_col is None else key_col
        if (geom_col is not None) and (geom_col!=self.GEOM_COL):
            raise ValueError("Geopandas dependencies require geom_col to be named geometry." )
        self.geom_col = self.GEOM_COL

        # Opportunity to limit the valid range here
        self.valid_range = self.morton.valid_range if valid_range is None else valid_range

        if numeric_or_categorical is None:
            self.numeric_or_categorical = self.NUMERIC_OR_CATEGORICAL
        else:
            self.numeric_or_categorical = numeric_or_categorical

        # Dict of statistics about timestamps, hsi keys, and spatio-temporal partitions
        self.stats = {}

        self.verbose = verbose

        # For later use
        self.timestamps = None
        self.temporal_levels = None
        self.temporal_partitions = None
        self.hsi_keys = None
        self.spatial_partition_level = None
        self.spatial_partitions = None

    
    def available_timestamps(
        self,
        dt_start = datetime(1970, 1, 1, tzinfo=pytz.utc),
        dt_end = datetime(2100, 12, 31, tzinfo=pytz.utc)-timedelta(seconds=1),
        **kwargs
    ):
        """Query the dataservice for global timestamps.

        :param dt_start: starttime
        :param dt_end:   endtime
        :param kwargs:   dataserviceendpoint
        """
        epochtimes = dataservice.query.get_global_timestamps(
            self.layer_id, dt_start, dt_end, **kwargs
        )

        # The dataservice has a limit on the number of epochtimes returned: 100,000
        # So in case we get that many epochtimes, query by year
        if len(epochtimes)>=100000:
            epochtimes = []
            years = numpy.arange(dt_start.year, dt_end.year+1)
            for year in years:
                dt0 = datetime(year, 1, 1, tzinfo=pytz.utc)
                dt1 = datetime(year, 12, 31, tzinfo=pytz.utc)-timedelta(seconds=1)
                epochtimes.extend(
                    dataservice.query.get_global_timestamps(
                        self.layer_id, dt0, dt1, **kwargs
                    )
                )

        # We may get duplicate epochtimes if the data is distributed in multiple federated instances
        len_epochtimes = len(epochtimes)
        epochtimes = sorted(set(epochtimes))

        if len_epochtimes!=len(epochtimes):
            print('WARNING:')
            print('len(timestamps) before set operation ', len_epochtimes)
            print('len(timestamps) after set operation  ', len(epochtimes))
        elif self.verbose:
            print('len(timestamps)                 ', len(epochtimes))

        # Translate to datetime
        timestamps = [
            datetime.utcfromtimestamp(e).replace(tzinfo=pytz.utc) for e in epochtimes
        ]

        return timestamps

    
    def qtree_hsi(self, key_col=None, level_col=None, geom_col=None):
        """Determine hsi cells using qtree algorithm."""
        # Get the gridded geometries (boxes) at the hsi level
        qt = qtree.QTree(self.valid_range, self.hsi_level, grid=self.grid) # initialize
        qt.quadtree_dfs() # build the quadtree (depth first search)
        gdf_grid = qt.gridded_to_geodataframe(
            key_col=key_col, level_col=level_col, hash_col='q_key', geom_col=geom_col
        ) # grid

        # Set of hsi keys
        self.hsi_keys = gdf_grid[self.key_col].to_list()
        self.stats['number_hsi_keys'] = len(self.hsi_keys)
        if self.verbose:
            print('number_hsi_keys                 ', self.stats['number_hsi_keys'])

        # # Building separate (very narrow) trees for each hsi cell
        # gdf_grid['qt'] = gdf_grid[self.geom_col].apply(
        #     lambda x: qtree.QTree(x, self.hsi_level, grid=self.grid)
        # )
        # # apply the quadtree algorithm
        # gdf_grid['qt'].apply(lambda x: x.quadtree_dfs())
        # gdf_grid['qt_dict'] = gdf_grid['qt'].apply(lambda x: x.walker())

        return gdf_grid

    
    def partitions(self, t_part=None, s_part=None):
        """Let raster object know about the partitions.
        
        :params t_part:  partition.TemporalPartition object
        :params s_part:  partition.SpatialPartition object
        """
        if t_part is not None:
            self.timestamps = t_part.timestamps
            self.temporal_levels = t_part.temporal_levels
            self.temporal_partitions = t_part.temporal_partitions
            self.stats = self.stats | t_part.stats
        else:
            self.temporal_levels = []
            self.temporal_partitions = {}

        # if self.hsi_keys is None:
        #     self.qtree_hsi()

        if s_part is not None:
            self.spatial_partitions = s_part.spatial_partitions
            self.spatial_partition_level = s_part.spatial_level
            self.stats = self.stats | s_part.stats
        else:
            self.spatial_partitions = ['0q']
            self.spatial_partition_level = 0

        if (t_part is not None) and (s_part is not None):
            # Calculate additional spatio-temporal stats
            try:
                self._partition_stats()
            except:
                pass

    
    def _partition_stats(self):
        """Collect spatio-temporal partition statistics (counts per partition)."""
        # Add spatio-temporal stats
        self.stats['max_rows_per_temp_part'] = self.stats[
            'number_hsi_keys'] * self.stats['max_ts_per_temp_part']
        self.stats['max_total_records'] = (self.stats['number_global_timestamps']
                                           * self.stats['number_hsi_keys'])
        self.stats['max_rows_per_spatial_part'] = (self.stats['number_global_timestamps']
                                                   * self.stats['max_ovw_keys_per_spatial_part'])
        self.stats['max_rows_per_part'] = (self.stats['max_ts_per_temp_part']
                                           * self.stats['max_ovw_keys_per_spatial_part'])
        self.stats['total_number_partitions'] = (self.stats['number_temporal_partitions']
                                                 * self.stats['number_spatial_partitions'])

        if self.verbose:
            print('number_global_timestamps        ', self.stats['number_global_timestamps'])
            print('number_hsi_keys                 ', self.stats['number_hsi_keys'])
            print('max_total_records (Million)     ', self.stats['max_total_records'] * 1e-6)
            print()
            print('number_temporal_partitions      ', self.stats['number_temporal_partitions'])
            print('max_ts_per_temp_part            ', self.stats['max_ts_per_temp_part'])
            print('max_rows_per_temp_part (Million)', self.stats['max_rows_per_temp_part'] * 1e-6)
            print()
            print('number_spatial_partitions       ', self.stats['number_spatial_partitions'])
            print('max_ovw_keys_per_spatial_part   ', self.stats['max_ovw_keys_per_spatial_part'])
            print('max_rows_per_spatial_part (Mill)', self.stats['max_rows_per_spatial_part']*1e-6)
            print()
            print('max_rows_per_part (Million)     ', self.stats['max_rows_per_part'] * 1e-6)
            print('total_number_partitions         ', self.stats['total_number_partitions'])

    
    def xarray_stats_numeric(self, arr, weights):
        """Area-weighted statistics for the spatial dimensions of an xarray.
        
        :param arr:  xarray of lateral dimensions equaling exactly one hsi cell.
                     multiple timestamps may be included 
        """
        mod_ys = numpy.array(arr.mod_y)
        mod_xs = numpy.array(arr.mod_x)

        count = arr.count(("mod_x", "mod_y"))
        count.name = 'count'
        first = arr.sel(mod_x=min(mod_xs), mod_y=min(mod_ys))
        first.name = 'first'
        minimum = arr.min(("mod_x", "mod_y"))
        minimum.name = 'min'
        maximum = arr.max(("mod_x", "mod_y"))
        maximum.name = 'max'

        arr_weighted = arr.weighted(weights)
        self.weights = weights #debug

        weighted_mean = arr_weighted.mean(("mod_x", "mod_y")).astype(numpy.float32, casting='same_kind')
        weighted_mean.name = 'mean'
        weighted_std = arr_weighted.std(("mod_x", "mod_y")).astype(numpy.float32, casting='same_kind')
        weighted_std.name = 'std'
        weighted_quantiles = arr_weighted.quantile(
            self.QUANTILES, dim=("mod_x", "mod_y")
        ).astype(numpy.float32, casting='same_kind')
        weighted_quantiles.name = 'quantiles'

        stats = xarray.merge([count, first, minimum, maximum, weighted_mean, weighted_std])
        ds_wq = weighted_quantiles.to_dataset(dim='quantile')
        ds_wq = ds_wq.rename(
            dict(zip(self.QUANTILES, [str(int(q*100))+'%' for q in self.QUANTILES]))
        )
        stats = xarray.merge([stats, ds_wq])

        stats = stats.drop(['mod_y', 'mod_x'])
        dims = ['dimension_' + d for d in self.dimension_values]
        if len(stats.dims)>1:
            df = stats.to_dataframe().reset_index()
        else:
            df = stats.to_pandas().reset_index()

        df = df[df['count']>0]
        
        return df.set_index(dims + ['time'])

    
    def xarray_stats_categorical(self, arr, histogram=True):
        """Unweighted categorical statistics for the spatial dimensions of an xarray.
        
        If requested, also get the full histogram of value_counts.
        :param arr:  xarray of lateral dimensions equaling exactly one overview cell.
                     multiple timestamps may be included.
        :histogram:  Flag to indicate whether histogram is requested.
        """
        arr.name = 'value'
        df = arr.to_dataframe().reset_index()
        df = df.dropna(subset='value').reset_index(drop=True)
        if len(df)==0:
            return pandas.DataFrame()

        # For categorical data we cast to int and then string
        df['value'] = df['value'].astype(int).astype(str)

        # Sort (needed for first statistic)
        dims = ['dimension_' + d for d in self.dimension_values]
        df = df.sort_values(by = dims + ['time', 'ovw_x', 'ovw_y', 'mod_x', 'mod_y'])
        grp = df[['ovw_x', 'ovw_y'] + dims + ['time', 'value']].groupby(
            ['ovw_x', 'ovw_y'] + dims + ['time']
        )

        # Get count, unique, top, freq statistics in one shot
        stats = grp.describe()['value']

        # Get "first" statistic (first non-nan value starting from bottom-left corner pixel)
        first = grp.first()
        first = first.rename(columns={'value': 'first'})

        stats = pandas.concat([stats, first], axis=1)

        if histogram:
            # Get the full histogram of value counts
            hist = df.set_index(['mod_x', 'mod_y']).value_counts()
            hist.name='value_counts'
            hist = hist.reset_index()

            # The values will be column identifiers, so cast to integer then to string
            hist['value'] = hist['value'].astype(int).astype(str)

            # Unstack and fill the counts of missing values with zero.
            hist = hist.sort_values(by = dims + ['time', 'value']).set_index(
                ['ovw_x', 'ovw_y'] + dims + ['time', 'value']
            ).unstack()
            hist = hist.fillna(0).astype(int)

            # Histogram columns may be identified by a prefix: 'count_'.
            hist = hist['value_counts']
            hist.columns.name = None
            hist.columns = ['count_' + h for h in hist.columns]

            # Histogram columns will be merged with the other statistics.
            stats = pandas.concat([stats, hist], axis=1)

        return stats.reset_index().set_index(dims + ['time'])

    
    def hsi_statistics(
        self, temporal_partition={}, spatial_partition=None, chunk_n=None,
        probe_local_timestamps=False, skip_existing=True, debug=False, **kwargs
    ):
        """Calculate HSIs and append or write to file.
        
        temporal_partition calculating query timestamps based on a specific
                           temporal_partition. (E.g. specific year, month)
        spatial_partition  calculating query box based on spatial_partition
        chunk_n            limit number of timestamps queried at a time
        :param kwargs:     cluster, dataserviceendpoint
        """          
        if len(temporal_partition)==0:
            if self.dataservice_type=='hbase':
                # Query all timestamps
                query_epochtimes = [int(t.timestamp()) for t in sorted(self.available_timestamps())]
            else:
                if self.timestamps is None:
                    s = 'available timestamps need to be supplied when dataservice_type is not hbase'
                    raise ValueError(s)
                else:
                    query_epochtimes = [int(t.timestamp()) for t in sorted(self.timestamps)]
        else:
            # Query partition timestamps
            if self.timestamps is None:
                query_epochtimes = self._get_query_timestamps(temporal_partition)
            else:
                query_epochtimes = self._get_query_timestamps(temporal_partition, self.timestamps)

        dimensions_lst = []  # List of all valid dimensions dictionaries
        if len(self.dimension_values)>0: # Debug consider dimension partitions
            for elements in itertools.product(*self.dimension_values.values()):
                # Looping through the cartesian product
                dimensions={}
                for i, dimension_name in enumerate(self.dimension_values):
                    dimensions[dimension_name] = elements[i]
                dimensions_lst.append(dimensions)

        if spatial_partition is None:
            # Query global
            spatial_partition =  '0q'
            query_key = '0q'
            aggregation_keys = self.hsi_keys
        else:
            # Query spatial partition
            query_key        = spatial_partition
            aggregation_keys = [o for o in self.hsi_keys if o.startswith(query_key)]

            if probe_local_timestamps:
                # probe the query_key area in the four corners as well as the center.
                query_epochtimes = self._probe_query_timestamps(query_key, query_epochtimes, **kwargs)

        # check for existing timestamps and coverage in parquet files
        if skip_existing:
            filepath = self._filepath(spatial_partition, temporal_partition, create_path=False)
            try:
                df1 = pandas.read_parquet(filepath, columns=[self.dt_col, self.key_col])
            except IOError:
                # Likely nothing there yet
                pass
            else:
                # debug: consider dimensions here
                #df1 = df1.groupby(self.dt_col)[self.key_col].apply(set).reset_index()
                #df1 = df1[df1[self.key_col]==set(aggregation_keys)]
                exclude_timestamps = set(df1[self.dt_col].apply(
                    lambda x: x.timestamp()
                ).astype(int))
                query_epochtimes = sorted(set(query_epochtimes) - exclude_timestamps)

        if len(query_epochtimes)==0:
            if self.verbose:
                print('No new timestamps available')
        else:
            if self.verbose:
                print('Found', len(query_epochtimes), 'new timestamps')
                
            if debug:
                return self._hsi_statistics(
                    query_epochtimes, query_key, aggregation_keys, dimensions_lst,
                    chunk_n=chunk_n, debug=debug,
                    temporal_partition=temporal_partition, **kwargs
                )

            gdf_hsi = self._hsi_statistics(
                query_epochtimes, query_key, aggregation_keys, dimensions_lst,
                chunk_n=chunk_n, temporal_partition=temporal_partition, **kwargs
            )
            
            if len(gdf_hsi)>0:
                # Append (if existing rows skipped)
                self.to_parquet(
                    gdf_hsi, temporal_partition, spatial_partition, append=True
                )
                # Free memory
                del gdf_hsi

    
    def _probe_query_timestamps(self, query_key, query_epochtimes, **kwargs):
        """Given a query key (partition), what timestamps are likely available?

        :param query_key:        spatial key on the partition level
        :param query_epochtimes: "global" timestamps (for current temporal partition)
        :param kwargs:           cluster, dataserviceendpoint
        """
        delta = self.hsi_level - self.spatial_partition_level
        probe_keys = []
        probe_keys.append(query_key + '0' * delta) #sw
        probe_keys.append(query_key + '1' * delta) #se
        probe_keys.append(query_key + '2' * delta) #nw
        probe_keys.append(query_key + '3' * delta) #ne
        if delta>1:
            # near center
            probe_keys.append(query_key + '0' + '3' * (delta-1))
            # perimiter
            probe_keys.append(query_key + '0' + '1' * (delta-1))
            probe_keys.append(query_key + '1' + '3' * (delta-1))
            probe_keys.append(query_key + '2' + '0' * (delta-1))
            probe_keys.append(query_key + '3' + '2' * (delta-1))

        dt_lst = []
        for probe_key in probe_keys:
            x_coord, y_coord = self.morton.base4_to_center_coords(probe_key)
            # Get the data from the dataservice
            arr = dataservice.query.to_xarray(
                layer_id=self.layer_id,
                level=self.pixel_level,
                latmin=y_coord,
                lonmin=x_coord,
                latmax=y_coord,
                lonmax=x_coord,
                timestamps=query_epochtimes,
                **kwargs
            )
            count = arr.count(dim=['lat', 'lon'])
            dt_lst += list(count[count>0]['time'].to_pandas())
        dt_lst = sorted(set(dt_lst))
        return [int(t.timestamp()) for t in dt_lst]

    
    def _get_query_timestamps(self, temporal_partition, timestamps=None):
        """Given a temporal partition, what timestamps need to be queried?

        :param temporal_partition: temporal partition
        """
        if timestamps is None:
            timestamps = self.available_timestamps()

        # get the timestamp attributes (year, month) specified in temporal_levels
        df_timestamps = pandas.DataFrame(timestamps).rename(columns={0: self.dt_col})
        for t_level in self.temporal_levels:
            df_timestamps[t_level] = df_timestamps[self.dt_col].apply(
                lambda x: getattr(x, t_level)
            )

        # Filter by partiton values
        for temporal_level in self.temporal_levels:
            df_timestamps = df_timestamps[df_timestamps[temporal_level]==temporal_partition[temporal_level]]

        # Sort and translate back to epoch time
        query_epochtimes = [int(t.timestamp()) for t in sorted(df_timestamps[self.dt_col])]
        return query_epochtimes


    def _weights_chunk(self, weights, idx_y, idx_x):
        """Get the weights correctly chunked."""
        if 'ovw_y' in weights.dims:
            assert weights.dims[0]=='ovw_y'
            weights_chunk = weights[idx_y[0]:idx_y[-1]+1]
        else:
            weights_chunk = weights[::]

        if 'ovw_x' in weights_chunk.dims:
            assert 'ovw_x' in [weights_chunk.dims[0], weights_chunk.dims[1]]
            if weights_chunk.dims[0]=='ovw_x':
                weights_chunk = weights_chunk[idx_x[0]:idx_x[-1]+1]
            elif weights_chunk.dims[1]=='ovw_x':
                weights_chunk = weights_chunk[:, idx_x[0]:idx_x[-1]+1]
                
        return weights_chunk

    
    def _hsi_statistics(
        self, query_epochtimes, query_key, aggregation_keys, dimensions_lst, 
        chunk_n=None, debug=False, temporal_partition=None, **kwargs
    ):
        """
        query_epochtimes list of timestamps
        query_key        quaternary key to define the query area. (E.g., one spatial partition)
                         Usually called as: query_key = spatial_partition
        aggregation_keys list of q_keys to collect hsi stats for. 
                         (E.g., belonging to the spatial partition).
        chunk_n          limit number of timestamps queried at a time
        """
        HISTOGRAM = True

        # query box 
        lonmin, latmin, lonmax, latmax = self.valid_range.bounds
        west, south, east, north = self.morton.base4_to_box(query_key).bounds
        latmin = max(latmin, south)
        latmax = min(latmax, north)
        lonmin = max(lonmin, west)
        lonmax = min(lonmax, east)

        if self.dataservice_type=='hbase':
            # reduce query box by 1/2 pixel because the dataservice will buffer to the full cell
            res_x, res_y = self.morton.resolution(self.pixel_level)
            latmin = max(latmin, south + res_y/2)
            latmax = min(latmax, north - res_y/2)
            lonmin = max(lonmin, west + res_x/2)
            lonmax = min(lonmax, east - res_x/2)

        # Number of timestamps queried at a time
        if chunk_n is None:
            max_query_pixels_per_ts = (
                self.stats['max_ovw_keys_per_spatial_part']
                * 4**self.delta_pixel_hsi
            )
            chunk_n = int(self.MAX_QUERY_PIXELS // max_query_pixels_per_ts)

        if chunk_n==0:
            if self.stats['number_global_timestamps']>0:
                s = 'Too many query pixels per timestamp. Define smaller spatial partitions'
                raise ValueError(s)
            else:
                raise RuntimeError('Nothing found.')
                
        if self.dataservice_type in ['local_filesystem', 'remote_filesystem']:
            gdf_local_meta = kwargs.get('gdf_local_meta')
            # Filter by query area
            gdf_local_meta = gdf_local_meta[
                gdf_local_meta.intersects(self.morton.base4_to_box(query_key))
            ].reset_index(drop=True)
            # Find timestamps of filtered data
            filtered_epochtimes = [int(t.timestamp()) for t in sorted(set(gdf_local_meta[self.dt_col]))]
            query_epochtimes = sorted(set(query_epochtimes) & set(filtered_epochtimes))

        df_stats = []
        for i, chunk in enumerate(self._chunks(query_epochtimes, chunk_n)):
            print(query_key, temporal_partition, '; chunk', i, ': ', len(chunk))
 
            if self.dataservice_type=='hbase':
                # Get the data from the hbase dataservice
                if len(dimensions_lst)==0:
                    # No dimensions
                    arr = dataservice.query.to_xarray(
                        layer_id=self.layer_id,
                        level=self.pixel_level,
                        latmin=latmin,
                        lonmin=lonmin,
                        latmax=latmax,
                        lonmax=lonmax,
                        timestamps=chunk,
                        **kwargs,
                    )
                else:
                    lst = []
                    for dimensions in dimensions_lst:
                        arr = dataservice.query.to_xarray(
                            layer_id=self.layer_id,
                            level=self.pixel_level,
                            latmin=latmin,
                            lonmin=lonmin,
                            latmax=latmax,
                            lonmax=lonmax,
                            timestamps=chunk,
                            dimensions=dimensions,
                            **kwargs,
                        )
                        lst.append(arr)
                    arr = xarray.merge(lst)[self.layer_id]

                # HSI grid is automatically aligned with pixel grid
                delta_x = 0
                delta_y = 0

            elif self.dataservice_type in ['local_filesystem', 'remote_filesystem']:
                lst = []
                for elements in itertools.product(*({'time':chunk} | self.dimension_values).values()):
                    # Filtering the metadata using specific epochtime and dimension values
                    flt = {
                        'time': datetime.utcfromtimestamp(elements[0]).replace(tzinfo=pytz.utc)
                    } | {
                        d:elements[i+1] for i, d in enumerate(self.dimension_values)
                    }
                    df_filtered = gdf_local_meta.loc[
                        (gdf_local_meta[list(flt)] == pandas.Series(flt)).all(axis=1)
                    ]
                    
                    if len(df_filtered)==0:
                        if self.verbose:
                            print('Warning: dimension combination not found', elements)
                    else:
                        if self.dataservice_type=='local_filesystem':
                            # Data is present locally
                            local_paths = df_filtered['filepath'].values
                        elif self.dataservice_type=='remote_filesystem':
                            # Download the remote data to a temporary directory
                            local_paths = []
                            for remote_path in df_filtered['filepath'].values:
                                try:
                                    local_path = os.path.join(
                                        self.tmp_directory,
                                        os.path.split(remote_path)[1]
                                    ).replace('\\', '/')
                                    remote_fs = kwargs.get('remote_fs')
                                    remote_fs.download(remote_path, local_path)
                                except IOerror as e:
                                    print(e)
                                    return
                                else:
                                    local_paths.append(local_path)

                        if len(local_paths)>1:
                            print('Warning: combining multiple files with the same dimensions', elements)
                        i=0
                        for local_path in local_paths:
                            if self.verbose:
                                print('local_path', local_path)
                            if i==0:
                                arr = xarray.load_dataarray(local_path)
                            else:
                                arr.fillna(xarray.load_dataarray(local_path))
                            if (self.dataservice_type=='remote_filesystem') and os.path.isfile(local_path):
                                # clean up
                                os.remove(local_path)
                            i+=1
    
                        try:
                            # Clip according to the query bounds
                            arr = arr.where(
                                (arr.x>lonmin) &
                                (arr.x<lonmax) &
                                (arr.y>latmin) &
                                (arr.y<latmax),
                                drop=True
                            )
                        except ValueError as e:
                            print('Warning at elements ', elements, ': ', e)
                        else:
                            # Build a multiindex for all the timestamps and dimension values,
                            # so that we can simpliy concat the arrays.
                            midx = pandas.MultiIndex.from_arrays(
                                [[v] for v in list(flt.values())], names=list(flt)
                            )
        
                            # Recast to 3D using the multiindex
                            arr = xarray.DataArray(
                                numpy.array(arr),
                                coords = [midx, arr.y, arr.x],
                                dims =  ['midx', 'y', 'x'],
                            )
                            
                            # Asserting that the y axis values are always in descending order
                            assert arr.y.values[1]<arr.y.values[0]
                            
                            lst.append(arr)

                if len(lst)>0:
                    arr = xarray.concat(lst, dim='midx')

                    # Concat apparently does the padding of non-commensurate arrays correctly, 
                    # but it may reverse the y-axis direction, so switch back here if needed.
                    if arr.y.values[1]>arr.y.values[0]:
                        arr = arr.reindex(y=arr.y[::-1])

                    # Unstacking the multiindex
                    arr = arr.set_index(midx=['time'] + list(self.dimension_values)).unstack('midx')

                    # Align hsi grid with pixel grid
                    delta_pixel_partition = self.pixel_level - self.spatial_partition_level

                    x0 = arr.x.min().item()-self.morton.resolution(self.pixel_level)[0]/2
                    x0_idx_px = self.morton.x_coord_to_idx(x0, self.pixel_level)
                    delta_x = int(x0_idx_px % (2**delta_pixel_partition))

                    y0 = arr.y.min().item()-self.morton.resolution(self.pixel_level)[1]/2
                    y0_idx_px = self.morton.y_coord_to_idx(y0, self.pixel_level)
                    delta_y = int(y0_idx_px % (2**delta_pixel_partition))
                else:
                    arr = None
                
            else:
                raise NotImplementedError(self.dataservice_type)

            if not arr is None:
                self.arr = arr #debug
                # Multiindex in order to select all the lat/lon values belonging to hsi cells
                ovw_x = [
                    l//2**self.delta_pixel_hsi for l in range(delta_x, len(arr.x)+delta_x)
                ]
                mod_x = [
                    l%2**self.delta_pixel_hsi for l in range(delta_x, len(arr.x)+delta_x)
                ]
                ovw_y = list(reversed([
                    l//2**self.delta_pixel_hsi for l in range(delta_y, len(arr.y)+delta_y)
                ]))
                mod_y = list(reversed([
                    l%2**self.delta_pixel_hsi for l in range(delta_y, len(arr.y)+delta_y)
                ]))
                
                midx_x = pandas.MultiIndex.from_arrays(
                    [mod_x, ovw_x], names=("mod_x", "ovw_x")
                )
                midx_y = pandas.MultiIndex.from_arrays(
                    [mod_y, ovw_y], names=("mod_y", "ovw_y")
                )

                # 1D array of x-coordinates indexed by mod_x
                arr_xs = xarray.DataArray(
                    numpy.array(arr.x),
                    coords = {'midx_x': midx_x},
                    dims =  ['midx_x'],
                )
                
                # 1D array of y-coordinates indexed by mod_y
                arr_ys = xarray.DataArray(
                    numpy.array(arr.y),
                    coords = {'midx_y': midx_y},
                    dims =  ['midx_y'],
                ) 
                
                # Area weight dynamically indexed dependent on the grid implementation
                weights = self.morton.grid.area_weights(arr_xs, arr_ys).unstack().fillna(0)
                weights.name = "weights"
            
                # Using __getattr__ instead of getattr because there can be a clash between
                # an attribute name like "quantile" and a bound xarray method "quantile".
                coords = {name: arr.__getattr__(name) for name in self.dimension_values}
                coords = coords | {'time': arr.time, 'y': midx_y, 'x': midx_x}
                
                # Getting the first three array dimensions in the right order
                dims = [d.replace('mod_y', 'y').replace('mod_x', 'x') for d in arr.dims[:3]]
                dims += list(arr.dims[3:])
                
                arr = xarray.DataArray(
                    numpy.array(arr),
                    coords = coords,
                    dims =  dims,
                )
            
                # Renaming the dimension names by perpending 'dimension_' since there could be 
                # clashes with keywords such as "quantile".
                for name in self.dimension_values:
                    arr = arr.rename({name: 'dimension_' + name})
            
                # Unstack the xarray raster to create separate axis for ovw_x and ovw_y
                # Unstacking in a two for-loops because the built-in arr.unstack(dim=['x', 'y']) is slow.
                arr_x_lst = []
                for x in sorted(set(arr.ovw_x.values)):
                    arr_x_lst.append(arr.sel(ovw_x=x))
                arr = xarray.concat(arr_x_lst, dim='ovw_x')
                
                arr_y_lst = []
                for y in sorted(set(arr.ovw_y.values)):
                    arr_y_lst.append(arr.sel(ovw_y=y))
                
                arr = xarray.concat(arr_y_lst, dim='ovw_y')
                # Order of arr dimensions is now: ovw_y, ovw_x, mod_y, mod_x, time, other dimensions...

                # get weights dims in the same order as arr dims for fast positional indexing
                weights = weights.transpose(*[d for d in arr.dims if d in weights.dims])
            
                # # Debug: Apply the area weights
                # arr_weighted = arr.weighted(weights)
            
                # Pandas is fastest at a length around 1Mio rows, so target aerial chunks to that size
                # Making chunks small also allows us to remove all-nan chunks before heavy calculations
                TARGET = 1e7
                
                target_chunks = numpy.ceil(numpy.prod(arr.shape) / TARGET) # ceil assures at least one chunk
                target_len_x = int(numpy.ceil(arr.ovw_x.shape[0]/numpy.sqrt(target_chunks)))
                target_len_y = int(numpy.ceil(arr.ovw_y.shape[0]/numpy.sqrt(target_chunks)))
                if self.verbose:
                    print('ovw_x.min, ovw_x.max, ovw_y.min, ovw_y.max')
                    print(int(arr.ovw_x.min()), int(arr.ovw_x.max()), int(arr.ovw_y.min()), int(arr.ovw_y.max()))
                    print('--------------')
                for ovw_x_range in self._chunks(list(arr.ovw_x.to_numpy()), target_len_x):
                    for ovw_y_range in self._chunks(list(arr.ovw_y.to_numpy()), target_len_y):
                        # Work on each chunk sequentially
                        idx_x = numpy.array(ovw_x_range) - int(arr.ovw_x[0])
                        idx_y = numpy.array(ovw_y_range) - int(arr.ovw_y[0])
                        arr_chunk = arr[idx_y[0]:idx_y[-1]+1, idx_x[0]:idx_x[-1]+1,::]
                        if not arr_chunk.isnull().all():
                            if self.verbose:
                                print(int(arr_chunk.ovw_x.min()), int(arr_chunk.ovw_x.max()), int(arr_chunk.ovw_y.min()), int(arr_chunk.ovw_y.max()))
                            if self.numeric_or_categorical == 'numeric':
                                weights_chunk = self._weights_chunk(weights, idx_y, idx_x)
                                df_stats.append(self.xarray_stats_numeric(arr_chunk, weights_chunk))
                            elif self.numeric_or_categorical == 'categorical':
                                df_stats.append(self.xarray_stats_categorical(arr_chunk, histogram=HISTOGRAM))
                            
        if len(df_stats)>0:
            df_stats = pandas.concat(df_stats)
            
        if len(df_stats)==0:
            print('Nothing Found')
            return pandas.DataFrame()

        if self.numeric_or_categorical == 'categorical' and HISTOGRAM:
            # Histogram columns may not be present in all parts
            hist_cols = [c for c in df_stats.columns if c.startswith('count_')]
            df_stats[hist_cols] = df_stats[hist_cols].fillna(0).astype(int)

        # Get the quaternary key from the ovw_x and ovw_y positions within the xarray
        #x_min, y_min = self.morton.base4_to_xy_indices(min(aggregation_keys))
        x_min, y_min = self.morton.base4_to_xy_indices(
            query_key + '0' * (self.hsi_level-self.spatial_partition_level)
        )
        gdf_unique = df_stats[['ovw_y', 'ovw_x']].drop_duplicates().reset_index(drop=True)
        xs = self.morton.x_idx_to_center_coord(gdf_unique['ovw_x'] + x_min, self.hsi_level)
        ys = self.morton.y_idx_to_center_coord(gdf_unique['ovw_y'] + y_min, self.hsi_level)
        keys = self.morton.get_key(numpy.array(ys), numpy.array(xs), [self.hsi_level]*len(xs))
        q_keys = self.morton.encode(keys, [self.hsi_level]*len(xs))
        gdf_unique[self.key_col] = q_keys

        # Convert to geopandas GeoDataFrame
        gdf_unique[self.geom_col] = self.morton.base4_to_box(gdf_unique[self.key_col])
        gdf_unique = geopandas.GeoDataFrame(gdf_unique, geometry=self.geom_col)
        gdf_unique[self.geom_col] = gdf_unique.intersection(self.valid_range)

        # Catch the case when the valid range intersects with the query partition
        gdf_unique = gdf_unique[~gdf_unique[self.geom_col].is_empty].reset_index(drop=True)
        
        # Debug: There may be another problem when the valid range intersects, where gdf_unique has additional keys
        # print('debug gdf_unique', gdf_unique)
        # print('debug set(aggregation_keys)', set(aggregation_keys))
        # assert set(gdf_unique[self.key_col]).issubset(set(aggregation_keys))

        # Merge keys and stats
        gdf_hsi = pandas.merge(
            gdf_unique,
            df_stats.reset_index().rename(
                # Keeping the dimension prefix due to possible clashes with statistic columns
                #columns={'time': self.dt_col} | {'dimension_' + d: d for d in self.dimension_values}
                columns={'time': self.dt_col}
            ),
            on=['ovw_x', 'ovw_y']
        )
        del gdf_hsi['ovw_x']
        del gdf_hsi['ovw_y']
        
        # Debug: Strangely we are getting epochtimes here instead of datetimes, so catching here
        if gdf_hsi[self.dt_col].dtype == numpy.dtype('int64'):
            print('WARNING: ', self.dt_col, 'of dtype int64 detected. Translating to datetime')
            try:
                gdf_hsi[self.dt_col] = gdf_hsi[self.dt_col].apply(
                    lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=pytz.utc)
                )
            except OSError as e:
                # Maybe value was too large (nano second timestamp definition)
                gdf_hsi[self.dt_col] = gdf_hsi[self.dt_col].apply(
                    lambda x: datetime.utcfromtimestamp(x/1e9).replace(tzinfo=pytz.utc)
                )

        gdf_hsi = gdf_hsi.set_crs(self.grid.crs)
        return gdf_hsi

    
    def _chunks(self, lst, chunk_n):
        """
        Yield successive n-sized chunks from lst.
        """
        for i in range(0, len(lst), chunk_n):
            yield lst[i:i + chunk_n]

    
    def _parquet_directory(self):
        """Composing the parquet directory."""
        parquet_directory = self.hsi_directory
        if self.dset_id is not None:
            parquet_directory = os.path.join(parquet_directory, 'dset_id=' + self.dset_id).replace('\\', '/')
        if self.layer_id is not None:
            parquet_directory = os.path.join(parquet_directory, 'layer_id=' + self.layer_id).replace('\\', '/')
        parquet_directory = os.path.join(
            parquet_directory,
            'hsi_level=' + str(self.hsi_level)
        ).replace('\\', '/')
        if not os.path.exists(parquet_directory):
            os.makedirs(parquet_directory)
        return parquet_directory

    
    def _filepath(self, spatial_partition, temporal_partition={}, create_path=True):
        """Composing the filepath from metadata."""
        filepath = self._parquet_directory()
        for temporal_level in self.temporal_levels:
            filepath = os.path.join(
                filepath,
                temporal_level+'='+str(temporal_partition[temporal_level])
            ).replace('\\', '/')
        filepath = os.path.join(filepath, 'spatial_partition'+'='+spatial_partition).replace('\\', '/')
        if create_path and not os.path.exists(filepath):
            os.makedirs(filepath)

        filename = 'hsi.parquet'
        return os.path.join(filepath, filename).replace('\\', '/')

    
    def _decode_filepath(self, filepath):
        dirname, filename = os.path.split(filepath)
        filename = os.path.splitext(filename)[0]
        
        # split off the spatial partition folder from the dirname
        dirname, tail = os.path.split(dirname)
        key, value = tail.split('=')
        assert key=='spatial_partition'
        spatial_partition = value
        
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
            elif key=='hsi_level':
                hsi_level = int(value)
                assert hsi_level==self.hsi_level
            elif key=='layer_id':
                layer_id = int(value)
                assert layer_id==self.layer_id
            elif key=='dset_id':
                dset_id = int(value)
                assert dset_id==self.dset_id
            else:
                raise ValueError("Non-compliant filepath." )

        # Finally we should be left with the hsi_directory
        assert dirname.rstrip('/')==self.hsi_directory.rstrip('/')

        return temporal_partition, spatial_partition

    
    def to_parquet(self, gdf, temporal_partition, spatial_partition, append=False):
        """Write GeoDataFrame partition to parquet."""
        filepath = self._filepath(spatial_partition, temporal_partition)
        if append:
            try:
                # See if there is something already present
                gdf_existing = geopandas.read_parquet(filepath)
            except IOError:
                pass
            else:
                # Merge by concatenating and dropping duplicates (keeping the newer version)
                gdf = pandas.concat([gdf_existing, gdf]).drop_duplicates(
                    subset=[self.dt_col, self.key_col]+['dimension_' + d for d in self.dimension_values],
                    keep='last',
                )
                
        if len(gdf)>0:
            # Histogram columns may not be present in all parts
            # Cast back to integer after filling with 0 since these may be float now.
            hist_cols = [c for c in gdf.columns if c.startswith('count_')]
            if len(hist_cols)>0:
                gdf[hist_cols] = gdf[hist_cols].fillna(0).astype(int)

            # Sorting so that these columns are used as indices in parquet file
            gdf = gdf.sort_values(
                by=[self.dt_col]+['dimension_' + d for d in self.dimension_values]+[self.key_col]
            ).reset_index(drop=True)

            gdf.to_parquet(
                path=filepath,
                row_group_size=100000,
                engine='pyarrow',
                compression='snappy',
                #partition_cols=self.partition_cols
            )

    
    def to_dataframe(self, use_dask=True, columns=None):
        """Compose DataFrame from the layer's parquet files."""
        if columns is not None:
            # E.g., columns=['count', 'unique', 'top', 'freq', 'first']
            if not self.key_col in columns:
                columns = [self.key_col] + columns
            if not self.dt_col in columns:
                columns = [self.dt_col] + columns

        if use_dask:
            # Read using dask
            ddf = dask.dataframe.read_parquet(self._parquet_directory())
            if columns is not None:
                ddf = ddf[columns]
            else:
                del ddf[self.geom_col]
            df = ddf.compute().reset_index(drop=True)
        else:
            # Read looping over files sequentially
            # Choose this option for categorical layers with histograms, since columns vary by file
            parquet_files = glob(os.path.join(self._parquet_directory(), '*')).replace('\\', '/')
            df = []
            for file in parquet_files:
                df.append(pandas.read_parquet(file, columns=columns))
            df = pandas.concat(df).reset_index(drop=True)
            if self.geom_col in df.columns:
                del df[self.geom_col]

            # Histogram columns may not be present in all parts
            # Cast back to integer after filling with 0 since these may be float now.
            hist_cols = [c for c in df.columns if c.startswith('count_')]
            if len(hist_cols)>0:
                df[hist_cols] = df[hist_cols].fillna(0).astype(int)

        return df

    
    def to_zarr(
        self,
        chunks={'lat': 256, 'lon': 256, 'time': 256},
        use_dask=True,
        columns=None,
        append=False
    ):
        """Write entire GeoDataFrame to zarr."""
        # Compose DataFrame from the layer's parquet files
        df = self.to_dataframe(use_dask=use_dask, columns=columns)

        # Get latitude and longitude from base4 key
        df_unique = df[[self.key_col]].drop_duplicates().reset_index(drop=True)
        df_unique['lon'], df_unique['lat'] = self.morton.base4_to_center_coords(df_unique[self.key_col])
        df = pandas.merge(df_unique, df, on=self.key_col)
        del df[self.key_col]
        df = df.rename(columns={self.dt_col: 'time'})

        # Convert to xarray dataset
        ds = df.set_index(['time', 'lat', 'lon']).to_xarray()
        ds.attrs = dict(
            dset_id = self.dset_id,
            layer_id = self.layer_id,
            pixel_level = self.pixel_level,
            delta_pixel_hsi = self.delta_pixel_hsi,
            hsi_level = self.hsi_level,
            max_pixel_count = 4**self.delta_pixel_hsi,
        )
        # Rechunk uniformly for zarr
        ds = ds.chunk(chunks)

        filepath = self._zarr_path()
        if append:
            raise NotImplementedError()

        # Write to zarr
        ds.to_zarr(filepath)

    
    def to_csv(self, use_dask=True, columns=None, append=False, epoch=False):
        """Write entire GeoDataFrame to csv."""
        # Compose DataFrame from the layer's parquet files
        df = self.to_dataframe(use_dask=use_dask, columns=columns)

        if epoch:
            # Remember the column order
            columns = df.columns
            # Translate to epoch time
            df_unique = df[[self.dt_col]].drop_duplicates().reset_index(drop=True)
            df_unique[self.dt_col+'_tmp'] = df_unique[self.dt_col].apply(
                lambda x: int(x.timestamp())
            )
            df = pandas.merge(df, df_unique, on=self.dt_col)
            del df[self.dt_col]
            df = df.rename(columns={self.dt_col+'_tmp' : self.dt_col})
            df = df[columns]

        filepath = self._csv_path()

        if append:
            raise NotImplementedError()

        # Write to csv
        df.to_csv(filepath, index=False)

    
    def _zarr_path(self):
        zarr_directory = self._parquet_directory()
        # We don's store the zarr file(s) within the parquet directory, so go one level back.
        zarr_directory = os.path.split(zarr_directory)[0]
        return os.path.join(zarr_directory, 'zarr_level' + str(self.hsi_level)).replace('\\', '/')

    
    def _csv_path(self):
        csv_directory = self._parquet_directory()
        # We don's store the csv file within the parquet directory, so go one level back.
        csv_directory = os.path.split(csv_directory)[0]
        csv_name = self.layer_id + '_level' + str(self.hsi_level) + '.csv'
        return os.path.join(csv_directory, csv_name).replace('\\', '/')

    
    def _json_path(self):
        json_directory = self._parquet_directory()
        # We don's store the json file within the parquet directory, so go one level back.
        json_directory = os.path.split(json_directory)[0]
        json_name = 'level' + str(self.hsi_level) + '.json'
        return os.path.join(json_directory, json_name).replace('\\', '/')

    
    def to_json(self):
        """Dump attributes to a json file."""
        json_dict = {}
        json_dict['pixel_level'] = self.pixel_level
        json_dict['delta_pixel_hsi'] = self.delta_pixel_hsi
        json_dict['hsi_level'] = self.hsi_level
        json_dict['hsi_directory'] = self.hsi_directory
        json_dict['dset_id'] = self.dset_id
        json_dict['layer_id'] = self.layer_id
        json_dict['dimension_values'] = self.dimension_values
        json_dict['dataservice_type'] = self.dataservice_type
        json_dict['dt_col'] = self.dt_col
        json_dict['geom_col'] = self.geom_col
        json_dict['temporal_levels'] = self.temporal_levels
        json_dict['temporal_partitions'] = self.temporal_partitions
        json_dict['spatial_partition_level'] = self.spatial_partition_level
        json_dict['spatial_partitions'] = self.spatial_partitions
        json_dict['stats'] = self.stats
        json_dict['hsi_keys'] = self.hsi_keys
        json_dict['numeric_or_categorical'] = self.numeric_or_categorical
        # The following objects require special attention to serialize and read back
        json_dict['valid_range'] = shapely.to_geojson(self.valid_range)
        json_dict['grid'] = self.grid.__repr__()

        json_path = self._json_path()
        with open(json_path, 'w') as file:
            json.dump(json_dict, file, cls=NpEncoder)


    def from_json(self):
        """Load attributes from a json file."""
        json_path = self._json_path()
        with open(json_path) as file:
            json_dict = json.load(file)
        for k in json_dict:
            if k=='valid_range':
                json_dict[k] = shapely.from_geojson(json_dict[k])
            elif k=='grid':
                # Recreate a grid object from __repr__()
                json_dict[k] = eval("nestedgrid." + json_dict[k])
            setattr(self, k, json_dict[k])


class NpEncoder(json.JSONEncoder):
    # To avoid json type errors
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
