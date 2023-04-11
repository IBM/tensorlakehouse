"""Generate raster overviews.

    Classes

        Rasteroverview    Generating raster overviews.
"""
import os
import sys
from glob import glob
import json
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool
import numpy
import pandas
import geopandas
import pytz
import xarray
import dask

sys.path.insert(0, '/data/mfreitag/software/dataservice_sdk-datacube/src/')
import dataservice.query

import morton
import qtree

os.environ['USE_PYGEOS'] = '0'

class Rasteroverview():
    """Generating raster overviews.

    Writing overviews to spatially and/or temporally partitioned parquet files.

    Attributes:

        dset_id                 Dataset ID
        layer_id                Layer ID
        pixel_level             Pixel level
        overviewstore_directory  Base directory where dataset overviews are stored.
        delta_pixel_overview    Difference between pixel level and overview level.
        dt_col                  Name of timestamp column in DataFrame.
        geom_col                Name of geometry column in DataFrame.
        key_col                 Name of spatial key column in DataFrame.
        numeric_or_categorical  Flag to indicate numeric or categorical type of statistics.
        stats                   Statistics (metadata) concerning available timestamps,
                                overview keys, and spatio-temporal partitions.
        overview_keys           List of the unique overview keys
        temporal_levels         List of the timestamp hierarchy levels (year, month, ...)
        temporal_partitions     List of the temporal partitions
        spatial_partitions      List of the spatial partitions
        spatial_level           Level of the spatial partitions
        timestamps              available timestamps in datetime format
        
    Methods

        available_timestamps    Query the dataservice for global timestamps.
        qtree_spatial_overview  Determine overview cells using qtree algorithm.
        partitions              Let raster object know about the partitions.
        xarray_stats_numeric    Area-weighted statistics for the spatial dimensions of an xarray.
        xarray_stats_categorical  Unweighted categorical statistics for the spatial dimensions 
                                of an xarray.
        overview_statistics     Calculate overviews and append or write to file.
        to_parquet              Write GeoDataFrame partition to parquet.
        to_dataframe            Compose DataFrame from the layer's parquet files.
        to_zarr                 Write entire GeoDataFrame to zarr.
        to_csv                  Write entire GeoDataFrame to csv.
        to_json                 Dump attributes to a json file.
        from_json               Load attributes from a json file.

    """
    # Default values for class attributes
    OVERVIEWSTORE_DIRECTORY    = '/data/raster/overviews/'
    DELTA_PIXEL_OVERVIEW       = 5
    MAX_QUERY_PIXELS           = 5e8
    DT_COL                     = 'timestamp'
    # Geopandas relies on the geometry column being named 'geometry', so enforce this
    GEOM_COL                   = 'geometry'
    KEY_COL                    = 'q_key'
    NUMERIC_OR_CATEGORICAL     = 'numeric'

    QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    ISO_8601 = '%Y-%m-%dT%H:%M:%SZ'

    def __init__(
        self,
        dset_id,
        layer_id,
        pixel_level,
        overviewstore_directory = None,
        delta_pixel_overview = None,
        valid_range = None,
        dt_col = None,
        geom_col = None,
        key_col = None,
        numeric_or_categorical = None,
        verbose = False,
    ):

        # layer attributes
        self.dset_id          = dset_id
        self.layer_id         = layer_id
        self.pixel_level      = pixel_level

        # Overview level calculated relative to pixel level
        if delta_pixel_overview is None:
            self.delta_pixel_overview = self.DELTA_PIXEL_OVERVIEW
        else:
            self.delta_pixel_overview = delta_pixel_overview
        self.overview_level   = self.pixel_level - self.delta_pixel_overview

        # overviewstore_directory base directory
        if overviewstore_directory is None:
            self.overviewstore_directory = self.OVERVIEWSTORE_DIRECTORY
        else:
            self.overviewstore_directory = overviewstore_directory

        # Table specific column information
        self.dt_col = self.DT_COL if dt_col is None else dt_col
        self.key_col = self.KEY_COL if key_col is None else key_col
        if (geom_col is not None) and (geom_col!=self.GEOM_COL):
            raise ValueError("Geopandas dependencies require geom_col to be named geometry." )
        self.geom_col = self.GEOM_COL

        # Opportunity to limit the valid range here
        self.valid_range = morton.valid_range if valid_range is None else valid_range

        if numeric_or_categorical is None:
            self.numeric_or_categorical = self.NUMERIC_OR_CATEGORICAL
        else:
            self.numeric_or_categorical = numeric_or_categorical

        # Dict of statistics about timestamps, overview keys, and spatio-temporal partitions
        self.stats = {}

        self.verbose = verbose

        # For later use
        self.timestamps = None
        self.temporal_levels = None
        self.temporal_partitions = None
        self.temporal_level_encodings = None
        self.overview_keys = None
        self.spatial_partitions = None
        self.spatial_level = None

    def available_timestamps(
        self,
        dt_start = datetime(1970, 1, 1, tzinfo=pytz.utc),
        dt_end = datetime(2100, 12, 31, tzinfo=pytz.utc)-timedelta(seconds=1),
    ):
        """Query the dataservice for global timestamps.

        :param dt_start: starttime
        :param dt_end:   endtime
        """
        epochtimes = dataservice.query.get_global_timestamps(self.layer_id, dt_start, dt_end)

        # The dataservice has a limit on the number of epochtimes returned: 100,000
        # So in case we get that many epochtimes, query by year
        if len(epochtimes)>=100000:
            epochtimes = []
            years = numpy.arange(dt_start.year, dt_end.year+1)
            for year in years:
                dt0 = datetime(year, 1, 1, tzinfo=pytz.utc)
                dt1 = datetime(year, 12, 31, tzinfo=pytz.utc)-timedelta(seconds=1)
                epochtimes.extend(dataservice.query.get_global_timestamps(self.layer_id, dt0, dt1))

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

    def qtree_spatial_overview(self, key_col=None, level_col=None, geom_col=None):
        """Determine overview cells using qtree algorithm."""
        # Get the gridded geometries (boxes) at the overview level
        qt = qtree.QTree(self.valid_range, self.overview_level) # initialize
        qt.quadtree_dfs() # build the quadtree (depth first search)
        gdf_grid = qt.gridded_to_geodataframe(
            key_col=key_col, level_col=level_col, hash_col='q_key', geom_col=geom_col
        ) # grid

        # Set of overview keys
        self.overview_keys = gdf_grid[self.key_col].to_list()
        self.stats['number_overview_keys'] = len(self.overview_keys)
        if self.verbose:
            print('number_overview_keys            ', self.stats['number_overview_keys'])

        # # Building separate (very narrow) trees for each overview cell
        # gdf_grid['qt'] = gdf_grid[self.geom_col].apply(
        #     lambda x: qtree.QTree(x, self.overview_level)
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
            self.temporal_level_encodings = t_part.temporal_level_encodings
            self.stats = self.stats | t_part.stats
        else:
            self.temporal_levels = []
            self.temporal_partitions = []

        # if self.overview_keys is None:
        #     self.qtree_spatial_overview()

        if s_part is not None:
            self.spatial_partitions = s_part.spatial_partitions
            self.spatial_level = s_part.spatial_level
            self.stats = self.stats | s_part.stats
        else:
            self.spatial_partitions = ['0q']
            self.spatial_level = 0

        if (t_part is not None) and (s_part is not None):
            # Calculate additional spatio-temporal stats
            self._partition_stats()

    def _partition_stats(self):
        """Collect spatio-temporal partition statistics (counts per partition)."""
        # Add spatio-temporal stats
        self.stats['max_rows_per_temp_part'] = self.stats[
            'number_overview_keys'] * self.stats['max_ts_per_temp_part']
        self.stats['max_total_records'] = (self.stats['number_global_timestamps']
                                           * self.stats['number_overview_keys'])
        self.stats['max_rows_per_spatial_part'] = (self.stats['number_global_timestamps']
                                                   * self.stats['max_ovw_keys_per_spatial_part'])
        self.stats['max_rows_per_part'] = (self.stats['max_ts_per_temp_part']
                                           * self.stats['max_ovw_keys_per_spatial_part'])
        self.stats['total_number_partitions'] = (self.stats['number_temporal_partitions']
                                                 * self.stats['number_spatial_partitions'])

        if self.verbose:
            print('number_global_timestamps        ', self.stats['number_global_timestamps'])
            print('number_overview_keys            ', self.stats['number_overview_keys'])
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

    def xarray_stats_numeric(self, arr):
        """Area-weighted statistics for the spatial dimensions of an xarray.
        
        :param arr:  xarray of lateral dimensions equaling exactly one overview cell.
                     multiple timestamps may be included 
        """
        lats = numpy.array(arr.lat)
        lons = numpy.array(arr.lon)

        count = arr.count(("lon", "lat"))
        count.name = 'count'
        first = arr.sel(lon=min(lons), lat=min(lats))
        first.name = 'first'
        minimum = arr.min(("lon", "lat"))
        minimum.name = 'min'
        maximum = arr.max(("lon", "lat"))
        maximum.name = 'max'

        # Mean, std, and quantiles are weighted by area
        weights = morton.grid.area_weights(arr.lon, arr.lat)
        #weights = numpy.cos(numpy.deg2rad(arr.lat))
        weights.name = "weights"
        arr_weighted = arr.weighted(weights)

        weighted_mean = arr_weighted.mean(("lon", "lat")).astype(numpy.float32, casting='same_kind')
        weighted_mean.name = 'mean'
        weighted_std = arr_weighted.std(("lon", "lat")).astype(numpy.float32, casting='same_kind')
        weighted_std.name = 'std'
        weighted_quantiles = arr_weighted.quantile(
            self.QUANTILES, dim=("lon", "lat")
        ).astype(numpy.float32, casting='same_kind')
        weighted_quantiles.name = 'quantiles'

        stats = xarray.merge([count, first, minimum, maximum, weighted_mean, weighted_std])
        ds_wq = weighted_quantiles.to_dataset(dim='quantile')
        ds_wq = ds_wq.rename(
            dict(zip(self.QUANTILES, [str(int(q*100))+'%' for q in self.QUANTILES]))
        )
        stats = xarray.merge([stats, ds_wq])

        stats = stats.drop(['lat', 'lon'])
        return stats.to_pandas()

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
        df = df.sort_values(by=['time', 'ovw_x', 'ovw_y', 'lon', 'lat'])
        grp = df[['ovw_x', 'ovw_y', 'time', 'value']].groupby(['ovw_x', 'ovw_y', 'time'])

        # Get count, unique, top, freq statistics in one shot
        stats = grp.describe()['value']

        # Get "first" statistic (bottom left corner value)
        first = grp.first()
        first = first.rename(columns={'value': 'first'})

        stats = pandas.concat([stats, first], axis=1)

        if histogram:
            # Get the full histogram of value counts
            hist = df.set_index(['lon', 'lat']).value_counts()
            hist.name='value_counts'
            hist = hist.reset_index()

            # The values will be column identifiers, so cast to integer then to string
            hist['value'] = hist['value'].astype(int).astype(str)

            # Unstack and fill the counts of missing values with zero.
            hist = hist.sort_values(by=['time', 'value']).set_index(
                ['ovw_x', 'ovw_y', 'time', 'value']
            ).unstack()
            hist = hist.fillna(0).astype(int)

            # Histogram columns may be identified by a prefix: 'count_'.
            hist = hist['value_counts']
            hist.columns.name = None
            hist.columns = ['count_' + h for h in hist.columns]

            # Histogram columns will be merged with the other statistics.
            stats = pandas.concat([stats, hist], axis=1)

        return stats.reset_index().set_index('time')

    def overview_statistics(
        self, temporal_partition=None, spatial_partition=None, n_workers=8, chunk_n=None,
        probe_local_timestamps=False, skip_existing=True
    ):
        """Calculate overviews and append or write to file.
        
        temporal_partition calculating query timestamps based on temporal_partition
        spatial_partition  calculating query box based on spatial_partition
        n_workers          number of workers for the statistics calculation 
                           (the dataservice call uses one worker only)
        chunk_n            limit number of timestamps queried at a time
        """
        if temporal_partition is None:
            # Query all timestamps
            query_epochtimes = [int(t.timestamp()) for t in sorted(self.available_timestamps())]
        else:
            # Query partition timestamps
            if self.timestamps is None:
                query_epochtimes = self._get_query_timestamps(temporal_partition)
            else:
                query_epochtimes = self._get_query_timestamps(temporal_partition, self.timestamps)

        if spatial_partition is None:
            # Query global
            spatial_partition =  '0q'
            query_key = '0q'
            aggregation_keys = self.overview_keys
        else:
            # Query spatial partition
            query_key        = spatial_partition
            aggregation_keys = [o for o in self.overview_keys if o.startswith(query_key)]

            if probe_local_timestamps:
                # probe the query_key area in the four corners as well as the center.
                query_epochtimes = self._probe_query_timestamps(query_key, query_epochtimes)

        # check for existing timestamps and coverage in parquet files
        if skip_existing:
            filepath = self._filepath(spatial_partition, temporal_partition)
            try:
                df1 = pandas.read_parquet(filepath, columns=[self.dt_col, self.key_col])
            except IOError:
                # Likely nothing there yet
                pass
            else:
                df1 = df1.groupby(self.dt_col)[self.key_col].apply(set).reset_index()
                df1 = df1[df1[self.key_col]==set(aggregation_keys)]
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

            gdf_overview = self._overview_statistics(
                query_epochtimes, query_key, aggregation_keys, n_workers=n_workers, chunk_n=chunk_n
            )

            # Append (if existing rows skipped) or overwrite (e.g. to reflect deleted data)
            self.to_parquet(
                gdf_overview, temporal_partition, spatial_partition, append=skip_existing
            )

    def _probe_query_timestamps(self, query_key, query_epochtimes):
        """Given a query key (partition), what timestamps are likely available?

        :param query_key:        spatial key on the partition level
        :param query_epochtimes: "global" timestamps (for current temporal partition)
        """
        probe_keys = []
        probe_keys.append(query_key + '0' * (self.overview_level-self.spatial_level)) #sw
        probe_keys.append(query_key + '1' * (self.overview_level-self.spatial_level)) #se
        probe_keys.append(query_key + '2' * (self.overview_level-self.spatial_level)) #nw
        probe_keys.append(query_key + '3' * (self.overview_level-self.spatial_level)) #ne
        if self.overview_level-self.spatial_level>1:
            # near center
            probe_keys.append(
                query_key + '0' + '3' * (self.overview_level-self.spatial_level-1)
            )

        dt_lst = []
        for probe_key in probe_keys:
            x_coord, y_coord = morton.base4_to_center_coords(probe_key)
            # Get the data from the dataservice
            arr = dataservice.query.to_xarray(
                layer_id=self.layer_id,
                level=self.pixel_level,
                latmin=y_coord,
                lonmin=x_coord,
                latmax=y_coord,
                lonmax=x_coord,
                timestamps=query_epochtimes,
            )
            count = arr.count(dim=['lat', 'lon'])
            dt_lst += list(count[count>0]['time'].to_pandas())
        dt_lst = sorted(set(dt_lst))
        return [int(t.timestamp()) for t in dt_lst]

    def _get_query_timestamps(self, temporal_partition, timestamps=None):
        """Given a temporal partition, what timestamps need to be queried?

        :param temporal_partition: temporal partition
        """
        tmp = temporal_partition
        values = []
        for temporal_level in reversed(self.temporal_levels):
            # parsing from the back of the string (reversed)
            tmp, value = tmp.split(self.temporal_level_encodings[temporal_level])
            value = int(value)
            values.append(value)
        values = list(reversed(values))
        assert len(values)==len(self.temporal_levels)

        if timestamps is None:
            timestamps = self.available_timestamps()

        # get the timestamp attributes (year, month) specified in temporal_levels
        df_timestamps = pandas.DataFrame(timestamps).rename(columns={0: self.dt_col})
        for t_level in self.temporal_levels:
            df_timestamps[t_level] = df_timestamps[self.dt_col].apply(
                lambda x: getattr(x, t_level)
            )

        # Filter by partiton values
        for key, value in zip(self.temporal_levels, values):
            df_timestamps = df_timestamps[df_timestamps[key]==value]

        # Sort and translate back to epoch time
        query_epochtimes = [int(t.timestamp()) for t in sorted(df_timestamps[self.dt_col])]
        return query_epochtimes

    def _overview_statistics(
        self, query_epochtimes, query_key, aggregation_keys, n_workers=8, chunk_n=None
    ):
        """
        query_epochtimes list of timestamps
        query_key        quaternary key to define the query area. (E.g., one spatial partition)
                         Usually called as: query_key = spatial_partition
        aggregation_keys list of q_keys to collect overview stats for. 
                         (E.g., belonging to the spatial partition).
        n_workers        number of workers for the statistics calculation
                         (the dataservice call uses one worker only)
        chunk_n          limit number of timestamps queried at a time
        """
        HISTOGRAM = True

        # query box
        lonmin, latmin, lonmax, latmax = self.valid_range.bounds
        west, south, east, north = morton.base4_to_box(query_key).bounds
        res_x, res_y = morton.resolution(self.pixel_level)
        latmin = max(latmin, south + res_y/2)
        latmax = min(latmax, north - res_y/2)
        lonmin = max(lonmin, west + res_x/2)
        lonmax = min(lonmax, east - res_x/2)

        # Number of timestamps queried at a time
        if chunk_n is None:
            max_query_pixels_per_ts = (
                self.stats['max_ovw_keys_per_spatial_part']
                * 4**self.delta_pixel_overview
            )
            chunk_n = int(self.MAX_QUERY_PIXELS // max_query_pixels_per_ts)

        if chunk_n==0:
            if self.stats['number_global_timestamps']>0:
                s = 'Too many query pixels per timestamp. Define smaller spatial partitions'
                raise ValueError(s)
            else:
                raise RuntimeError('Nothing found.')

        df_stats = []
        for i, chunk in enumerate(self._chunks(query_epochtimes, chunk_n)):
            print('chunk', i)

            # Get the data from the dataservice
            arr = dataservice.query.to_xarray(
                layer_id=self.layer_id,
                level=self.pixel_level,
                latmin=latmin,
                lonmin=lonmin,
                latmax=latmax,
                lonmax=lonmax,
                timestamps=chunk,
            )

            # Multiindex in order to select all the lat/lon values belonging to overview cells
            ovw_x =[l//2**self.delta_pixel_overview for l in range(len(arr.lon))]
            ovw_y =list(reversed([l//2**self.delta_pixel_overview for l in range(len(arr.lat))]))

            midx_x = pandas.MultiIndex.from_arrays(
                [numpy.array(arr.lon), ovw_x], names=("lon", "ovw_x")
            )
            midx_y = pandas.MultiIndex.from_arrays(
                [numpy.array(arr.lat), ovw_y], names=("lat", "ovw_y")
            )
            arr = xarray.DataArray(
                numpy.array(arr),
                dims = ['time', 'y', 'x'],
                coords={'time':arr.time, 'y': midx_y, 'x': midx_x}
            )

            # To do: crop array to valid range
            # (may only be necessary in special cases where we experiment with limited ranges)

            # Cut up the array into overview-cell sized cubes (many timestamps but one overview key)
            arr_small_lst = []
            for x in sorted(set(ovw_x)):
                for y in sorted(set(ovw_y)):
                    arr_small = arr.sel(ovw_x=x, ovw_y=y)
                    arr_small_lst.append(arr_small)

            # Apply the array_stats method either sequentially or in parallel
            if n_workers==1:
                # Work on each spatial cell sequentially
                for arr_small in arr_small_lst:
                    if self.numeric_or_categorical == 'numeric':
                        df_stats.append(self.xarray_stats_numeric(arr_small))
                    elif self.numeric_or_categorical == 'categorical':
                        df_stats.append(
                            self.xarray_stats_categorical(arr_small, histogram=HISTOGRAM)
                        )
            else:
                # from multiprocessing import Pool
                if self.numeric_or_categorical == 'numeric':
                    with Pool(n_workers) as pool:
                        df_stats.extend(pool.map(self.xarray_stats_numeric, arr_small_lst))
                elif self.numeric_or_categorical == 'categorical':
                    partial_stats_categorical = partial(
                        self.xarray_stats_categorical,
                        histogram=HISTOGRAM,
                    )
                    with Pool(n_workers) as pool:
                        df_stats.extend(
                            pool.map(partial_stats_categorical, arr_small_lst)
                        )

        df_stats = pandas.concat(df_stats)
        if len(df_stats)==0:
            print('Nothing Found')
            return pandas.DataFrame()

        if self.numeric_or_categorical == 'categorical' and HISTOGRAM:
            # Histogram columns may not be present in all parts
            hist_cols = [c for c in df_stats.columns if c.startswith('count_')]
            df_stats[hist_cols] = df_stats[hist_cols].fillna(0).astype(int)

        # Get the quaternary key from the ovw_x and ovw_y positions within the xarray
        #x_min, y_min = morton.base4_to_xy_indices(min(aggregation_keys))
        x_min, y_min = morton.base4_to_xy_indices(
            query_key + '0' * (self.overview_level-self.spatial_level)
        )
        gdf_unique = df_stats[['ovw_y', 'ovw_x']].drop_duplicates().reset_index(drop=True)
        xs = morton.x_idx_to_center_coord(gdf_unique['ovw_x'] + x_min, self.overview_level)
        ys = morton.y_idx_to_center_coord(gdf_unique['ovw_y'] + y_min, self.overview_level)
        keys = morton.get_key(numpy.array(ys), numpy.array(xs), [self.overview_level]*len(xs))
        q_keys = morton.encode(keys, [self.overview_level]*len(xs))
        gdf_unique[self.key_col] = q_keys

        # Convert to geopandas GeoDataFrame
        gdf_unique[self.geom_col] = gdf_unique[self.key_col].apply(morton.base4_to_box)
        gdf_unique = geopandas.GeoDataFrame(gdf_unique, geometry=self.geom_col)
        gdf_unique[self.geom_col] = gdf_unique.intersection(self.valid_range)

        # Catch the case when the valid range intersects with the query partition
        gdf_unique = gdf_unique[~gdf_unique[self.geom_col].is_empty].reset_index(drop=True)
        assert set(gdf_unique[self.key_col]).issubset(set(aggregation_keys))

        # Merge keys and stats
        gdf_overview = pandas.merge(
            gdf_unique,
            df_stats.reset_index().rename(columns={'time': self.dt_col}),
            on=['ovw_x', 'ovw_y']
        )
        del gdf_overview['ovw_x']
        del gdf_overview['ovw_y']

        return gdf_overview

    def _chunks(self, lst, chunk_n):
        """
        Yield successive n-sized chunks from lst.
        """
        for i in range(0, len(lst), chunk_n):
            yield lst[i:i + chunk_n]

    def _parquet_directory(self):
        """Composing the parquet directory."""
        parquet_directory = os.path.join(
            self.overviewstore_directory,
            self.dset_id,
            self.layer_id,
            'level' + str(self.overview_level)
        )
        if not os.path.exists(parquet_directory):
            os.makedirs(parquet_directory)
        return parquet_directory

    def _filepath(self, spatial_partition, temporal_partition=None):
        """Composing the filepath from metadata."""
        filepath = self._parquet_directory()
        filename = self.layer_id
        if temporal_partition is not None:
            filepath = os.path.join(filepath, temporal_partition)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filename = '_'.join([filename, temporal_partition])

        filename = '_'.join([filename, spatial_partition])
        filename = '.'.join([filename, 'parquet'])

        return os.path.join(filepath, filename)

    def _decode_filepath(self, filepath):
        dirname, filename = os.path.split(filepath)
        filename = os.path.splitext(filename)[0]

        # Decompose the filename into layer_id, temporal partition, spatial_partition
        parts = filename.split('_')
        assert len(parts)==2 or len(parts)==3
        layer_id = parts[0]
        spatial_partition = parts[-1]
        if len(parts)==3:
            # temporal partition is present
            temporal_partition = parts[1]
            # split off the temporal partition folder from the dirname
            dirname, tail = os.path.split(dirname)
            assert tail==temporal_partition
        else:
            temporal_partition = None

        # split off the overview_level folder from the dirname
        dirname, overview_level = os.path.split(dirname)
        overview_level = int(overview_level.lstrip('level'))

        # split off the layer_id folder from the dirname
        dirname, tail = os.path.split(dirname)
        assert tail==layer_id

        # split off the dset_id folder from the dirname
        overviewstore_directory, dset_id = os.path.split(dirname)
        assert overviewstore_directory.rstrip('/')==self.overviewstore_directory.rstrip('/')
        assert dset_id==self.dset_id
        assert layer_id==self.layer_id
        assert overview_level==self.overview_level

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
                    subset=[self.dt_col, self.key_col], keep='last'
                )

        if len(gdf)>0:
            # Debug: to do: Dimensions need to be considered in the subset above and sorting below
            gdf = gdf.sort_values(by=[self.dt_col, self.key_col]).reset_index(drop=True)
            # Debug: Speedtest to do: try different sorting order [self.key_col, self.dt_col]

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
            parquet_files = glob(os.path.join(self._parquet_directory(), '*'))
            df = []
            for file in parquet_files:
                df.append(pandas.read_parquet(file, columns=columns))
            df = pandas.concat(df).reset_index(drop=True)
            if self.geom_col in df.columns:
                del df[self.geom_col]

            # Histogram columns may not be present in all parts
            hist_cols = [c for c in df.columns if c.startswith('count_')]
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
        df_unique['lon'], df_unique['lat'] = morton.base4_to_center_coords(df_unique[self.key_col])
        df = pandas.merge(df_unique, df, on=self.key_col)
        del df[self.key_col]
        df = df.rename(columns={self.dt_col: 'time'})

        # Convert to xarray dataset
        ds = df.set_index(['time', 'lat', 'lon']).to_xarray()
        ds.attrs = dict(
            dset_id = self.dset_id,
            layer_id = self.layer_id,
            pixel_level = self.pixel_level,
            delta_pixel_overview = self.delta_pixel_overview,
            overview_level = self.overview_level,
            max_pixel_count = 4**self.delta_pixel_overview,
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
        return os.path.join(zarr_directory, 'zarr_level' + str(self.overview_level))

    def _csv_path(self):
        csv_directory = self._parquet_directory()
        # We don's store the csv file within the parquet directory, so go one level back.
        csv_directory = os.path.split(csv_directory)[0]
        csv_name = self.layer_id + '_level' + str(self.overview_level) + '.csv'
        return os.path.join(csv_directory, csv_name)

    def _json_path(self):
        json_directory = self._parquet_directory()
        # We don's store the json file within the parquet directory, so go one level back.
        json_directory = os.path.split(json_directory)[0]
        json_name = self.layer_id + '_level' + str(self.overview_level) + '.json'
        return os.path.join(json_directory, json_name)

    def to_json(self):
        """Dump attributes to a json file."""
        json_dict = {}
        json_dict['dset_id'] = self.dset_id
        json_dict['layer_id'] = self.layer_id
        json_dict['pixel_level'] = self.pixel_level
        json_dict['delta_pixel_overview'] = self.delta_pixel_overview
        json_dict['overview_level'] = self.overview_level
        json_dict['overviewstore_directory'] = self.overviewstore_directory
        json_dict['dt_col'] = self.dt_col
        json_dict['geom_col'] = self.geom_col
        json_dict['temporal_levels'] = self.temporal_levels
        json_dict['temporal_level_encodings'] = self.temporal_level_encodings
        json_dict['temporal_partitions'] = self.temporal_partitions
        json_dict['spatial_partitions'] = self.spatial_partitions
        json_dict['spatial_level'] = self.spatial_level
        json_dict['stats'] = self.stats
        json_dict['overview_keys'] = self.overview_keys

        json_path = self._json_path()
        with open(json_path, 'w') as file:
            json.dump(json_dict, file, cls=NpEncoder)

    def from_json(self):
        """Load attributes from a json file."""
        json_path = self._json_path()
        with open(json_path) as file:
            json_dict = json.load(file)
        for key in json_dict:
            setattr(self, key, json_dict[key])


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
