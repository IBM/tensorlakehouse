import os
os.environ['USE_PYGEOS'] = '0'
import sys
from glob import glob
import shutil

sys.path.insert(1, os.path.abspath(".."))
from qtree_index import qtree, partition
from . import rasteroverview

import pandas
from datetime import datetime, timedelta
import pytz
import geopandas
import dask_geopandas
import shapely
import json
import itertools
import uuid
import pystac_client


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class HSI():
    """Hierarchical Spatial Index - STAC integration.

    Orchestrates the generation of Hirarchical Spatial Indices for GeoDN data registered in STAC.

    Attributes:
        bands                       Bands to be indexed
        pixel_level                 Pixel level of the raw data
        delta_pixel_hsi             Difference between HSI level and raw data level
        spatial_partition_level     Spatial partition level
        hsi_directory               Local HSI directory where parquet files are assembled
        tmp_directory               Temporary directory
        statistics_type             Statistics type (numeric or categorical)
        histogram                   Flag to indicate if histograms are requested for categorical data
        grid                        Nested grid defined for the raw data
        
        stac_url                    STAC URL
        collection_id               STAC collection ID of the raw data
        hsi_collection_id           STAC collection ID of the HSI
        json_folder                 Local json folder for STAC items
        certificate                 STAC certificate

        dataservice_type            Dataservice type (raw data)
        data_bucket                 COS bucket of the raw data
        remote_fs                   Remote filesystem handle for the raw data
        # data_access_key_id          Raw data access key ID
        # data_secret_access_key      Raw data sectre access key ID
        # data_endpoint_url           Raw data endpoint URL

        hsi_dataservice_type        Dataservice type (HSI)
        hsi_bucket                  COS bucket of the HSI data
        hsi_remote_fs               Remote filesystem handle for the HSI data
        hsi_access_key_id           HSI access key ID
        hsi_secret_access_key       HSI sectret access key ID
        hsi_endpoint_url            HSI endpoint URL

        chunk_n                     Number of timestamps to process concurrently (best to keep chunk_n=1)
        skip_existing               Flag to indicate if existing timestamp/dimension combinations should be skipped

        temporal_levels             Temporal partition levels (e.g. ['year', 'month'])
        year                        Temporal partition value for the year (if any)
        month                       Temporal partition value for the month (if any)
        day                         Temporal partition value for the day (if any)
        verbose                     Detailed output 

        # Debug: Remove this flag once we decided how to generalize dimensions
        tile_from_filepath          Get the tile information from the filepath
        # Debug: Not needed anymore in the new HLS collection which has corret epsg information in STAC
        filter_epsg_using_tile_zone Filter the UTM EPSG using the zone info from the zone

        
    Methods

        stac_search_available       Search for available raw data in STAC.
        search_items_local_metadata Create metadataframe from STAC search_items.
        setup_rasteroverview        Setup the Rasteroverviw object with appropriate partitions for HSI creation.
        hsi_worker                  Calculate HSI for one partition 
                                        (uses one CPU and memory dependent on size of partitions.
        register_hsi_items_stac     Register HSI items in STAC.
        
    """
    # Constants
    ISO_8601                   = '%Y-%m-%dT%H:%M:%SZ'
    REQUIRED_STATS_CATEGORICAL = {'count', 'top', 'freq', 'unique', 'first'}
    REQUIRED_STATS_NUMERIC     = {'count', 'min', 'max', 'mean', 'std', 'first'}
    OPTIONAL_STATS_CATEGORICAL_STARTSWITH = 'count_' # Value counts (histogram columns)
    OPTIONAL_STATS_NUMERIC_ENDSWITH = '%' # Quantiles

    def __init__(
        self,
        bands,
        pixel_level,
        delta_pixel_hsi,
        spatial_partition_level,
        hsi_directory,
        tmp_directory,
        statistics_type,
        histogram,
        grid,
        
        stac_url,
        collection_id,
        hsi_collection_id,
        json_folder,
        certificate,

        dataservice_type,
        data_bucket,
        remote_fs,
        # data_access_key_id,
        # data_secret_access_key,
        # data_endpoint_url,

        hsi_dataservice_type,
        hsi_bucket,
        hsi_remote_fs,
        hsi_access_key_id,
        hsi_secret_access_key,
        hsi_endpoint_url,

        chunk_n,
        skip_existing,

        temporal_levels,
        year = None,
        month = None,
        day = None,
        verbose = False,

        tile_from_filepath=None,
        filter_epsg_using_tile_zone=None,
    ):
        self.verbose = verbose

        # Debug: currently we require these parameters since STAC items are not set up to provide all information about dimensions
        self.tile_from_filepath=tile_from_filepath
        self.filter_epsg_using_tile_zone=filter_epsg_using_tile_zone
        
        # Rasteroverview parameters
        self.bands = bands
        self.pixel_level = pixel_level
        self.delta_pixel_hsi = delta_pixel_hsi
        self.spatial_partition_level = spatial_partition_level
        self.hsi_directory = hsi_directory
        self.tmp_directory = tmp_directory
        self.dset_id = None
        self.layer_id = None
        self.valid_range = None
        self.statistics_type = statistics_type
        self.histogram = histogram
                
        # Nested grid
        self.grid = grid

        # STAC access
        self.stac_url = stac_url
        self.collection_id = collection_id
        self.hsi_collection_id = hsi_collection_id
        self.json_folder = json_folder
        self.certificate = certificate

        # COS raw data access
        self.dataservice_type = dataservice_type
        self.data_bucket = data_bucket
        self.remote_fs = remote_fs
        # self.data_access_key_id = data_access_key_id
        # self.data_secret_access_key = data_secret_access_key
        # self.data_endpoint_url = data_endpoint_url

        # COS HSI access
        self.hsi_dataservice_type = hsi_dataservice_type
        self.hsi_bucket = hsi_bucket
        self.hsi_remote_fs = hsi_remote_fs
        self.hsi_access_key_id = hsi_access_key_id
        self.hsi_secret_access_key = hsi_secret_access_key
        self.hsi_endpoint_url = hsi_endpoint_url

        # HSI worker settings
        self.chunk_n = chunk_n
        self.skip_existing = skip_existing

        # Temporal partition levels
        self.temporal_levels = temporal_levels

        # Choose a specific temporal partition if temporal_levels is not an empty list.
        self.year = year
        self.month = month
        self.day = day

        # Note that the order of temporal_partitions keys needs to be 'year', 'month', 'day'
        self.temporal_partitions = {}
        assert set(self.temporal_levels)<={'year', 'month', 'day'}
        if self.year is not None:
            assert set(self.temporal_levels)>={'year'}
            self.temporal_partitions['year'] = [year]
        if self.month is not None:
            assert set(self.temporal_levels)>={'year', 'month'}
            self.temporal_partitions['month'] = [month]
        if self.day is not None:
            assert set(self.temporal_levels)>={'year', 'month', 'day'}
            self.temporal_partitions['day'] = [day]

        # # Temporal partitions
        # t_part = partition.TemporalPartition(
        #     raster_or_vector = 'raster',
        #     timestamps = [],
        # )
        # t_part.get_temporal_partition_levels(temporal_levels=temporal_levels)
        # t_part.get_temporal_partitions()

        # Temporal partitions
        # Debug: getting rid of the temporal partitions dependency on the partition module
        self.t_part = {}
        self.t_part['temporal_levels'] = self.temporal_levels
        self.t_part['temporal_partitions'] = self.temporal_partitions
        self.t_part = dotdict(self.t_part)
    
        if self.verbose:
            print('temporal_levels         ', self.t_part.temporal_levels)
            print('len(temporal_partitions)', len(self.t_part.temporal_partitions))
            print('temporal_partitions     ', self.t_part.temporal_partitions)

        # search_aoi from grid valid bounds
        self.search_aoi = json.loads(shapely.to_geojson(shapely.box(*self.grid.valid_bounds_wgs84)))

        # Filter epsg using STAC
        self.filter_epsg = self.grid.epsg

        # Search the available raw data in STAC.
        self.stac_search_available(
            self.search_aoi,
            filter_epsg=self.filter_epsg
        )

        # Create the metadata table for the Rasteroverview
        self.search_items_local_metadata()

        # Now since we know the local metadata, define the dimensions
        self.dimension_values = {
            'band':self.bands,
            'tile':sorted(self.gdf_local_meta['tile'].unique()),
        }

        # Setup the Rasteroverviw object with appropriate partitions for HSI creation.
        self.setup_rasteroverview()


    def stac_search_available(
        self,
        search_aoi,
        fields = None,
        filter_epsg = None,
    ):
        """Search for available raw data in STAC."""
        LIMIT = 10000
        FIELDS = {
            "include": [
                "id",
                "bbox",
                "datetime",
                "properties.tile",
                "properties.cube:variables",
                "properties.cube:dimensions",
                "properties.cloud_coverage",
            ],
            "exclude": [
            ],
        }
        
        # Stac fields to query
        if fields is None:
            fields = FIELDS
        
        # Temporal search parameters
        if self.day is None:
            if self.month is None:
                if self.year is None:
                    dt_start = None
                else:
                    dt_start = datetime(self.year, 1, 1, tzinfo=pytz.utc)
                    dt_end = (datetime(self.year+1, 1, 1, tzinfo=pytz.utc)-timedelta(seconds=1))
            else:
                if self.year is None:
                    raise ValueError
                else:
                    dt_start = datetime(self.year, self.month, 1, tzinfo=pytz.utc)
                    if self.month<12:
                        dt_end = (datetime(self.year, self.month+1, 1, tzinfo=pytz.utc)-timedelta(seconds=1))
                    elif self.month==12:
                        dt_end = (datetime(self.year+1, 1, 1, tzinfo=pytz.utc)-timedelta(seconds=1))
                    else:
                        raise ValueError
        else:
            if self.year is None:
                raise ValueError
            elif self.month is None:
                raise ValueError
            else:
                dt_start = datetime(self.year, self.month, self.day, tzinfo=pytz.utc)
                dt_end = datetime(self.year, self.month, self.day, tzinfo=pytz.utc)+timedelta(seconds=24*3600-1)
    
        # Translate datetime to format stac understands
        dt_string = None
        if dt_start is not None:
            dt_string = dt_start.strftime(self.ISO_8601)
            if dt_end is not None:
                dt_string = dt_string + '/' + dt_end.strftime(self.ISO_8601)
        if self.verbose:
            print('dt_string               ', dt_string)
            print('search_aoi              ', search_aoi)
            
        # pystac_client search
        stac = pystac_client.Client.open(self.stac_url)
        stac_search_result = stac.search(
            limit = LIMIT,
            collections = [self.collection_id],
            intersects = search_aoi,
            datetime = dt_string,
            fields = fields,
        )
        
        self.search_items = None
        i = 0
        for next_items in stac_search_result.pages():
            if self.verbose:
                print('batch', i, '; raw     ', len(next_items))
    
            # Filter epsg
            if filter_epsg is not None:
                next_items = [item for item in next_items if (
                    item.properties['cube:dimensions']['x']['reference_system']==filter_epsg
                )]
                
            if self.verbose:
                print('batch', i, '; filtered', len(next_items))
    
            if i==0:
                self.search_items = next_items
            else:
                self.search_items = self.search_items + next_items
            i+=1
    
        if self.verbose:
            if self.search_items is not None:
                print('Found', len(self.search_items), 'search items')
            else:
                print('Found no search items')


    def search_items_local_metadata(self):
        """Create metadataframe from STAC search_items."""
        if 'data' in self.search_items[0].assets:
            # (A) Single-asset items:
            self.gdf_local_meta = geopandas.GeoDataFrame([{
                'id': item.id,
                'time': item.datetime,
                #'cloud_coverage': item.properties['cloud_coverage'],
                #'tile': item.properties['tile'],
                'band': list(item.properties['cube:variables'].keys())[0],
                'geometry': shapely.box(*item.bbox),
                'remote_path': item.assets['data'].href.split('s3://')[-1],
            } for item in self.search_items])
        else:
            # (B) Multi-asset items:
            self.gdf_local_meta = geopandas.GeoDataFrame([{
                'id': item.id,
                'time': item.datetime,
                #'cloud_coverage': item.properties['cloud_coverage'],
                #'tile': item.properties['tile'],
                'band': band,
                'geometry': shapely.box(*item.bbox),
                'remote_path': item.assets[band].href.split('s3://')[-1],
            } for item in self.search_items for band in item.assets if band in self.bands])
            
        # Local path from remote path
        if self.dataservice_type=='remote_filesystem':
            # Either (A) download file from remote folder
            self.gdf_local_meta['filepath'] = self.gdf_local_meta['remote_path'].apply(lambda x: 's3://' + x)
        elif self.dataservice_type=='local_filesystem':
            # Or (B) use mounted s3fs
            raise NotImplementedError()
            # self.gdf_local_meta['filepath'] = self.gdf_local_meta['remote_path'].apply(
            #     lambda x: os.path.join('/path/to/mount/point', x.split(self.data_bucket+'/')[-1]).replace('\\', '/')
            # )
        else:
            raise NotImplementedError()
    
        if self.tile_from_filepath:
            def _tile_from_filepath(filepath):
                """Hack the tilename since we don't have this from the item property directly."""
                return filepath.split('.')[-6][1:]
            self.gdf_local_meta['tile'] = self.gdf_local_meta['filepath'].apply(_tile_from_filepath)
    
        # Truncate the time at least to seconds since we use epochtime internally
        self.gdf_local_meta['time_original'] = self.gdf_local_meta['time']
        self.gdf_local_meta['time'] = self.gdf_local_meta['time'].apply(lambda x: datetime(
            x.year, x.month, x.day, x.hour, x.minute, x.second, tzinfo=pytz.utc
        ))
        # debug: Truncating to minute, hour, or day may allow merging multiple timestamps
        #self.gdf_local_meta['time'] = self.gdf_local_meta['time'].apply(lambda x: datetime(x.year, x.month, x.day, tzinfo=pytz.utc))
    
        self.gdf_local_meta['epoch'] = self.gdf_local_meta['time'].apply(lambda x: int(x.timestamp()))
    
        if self.filter_epsg_using_tile_zone:
            # Using the zone_number from the tile designation to deduce the UTM zone 
            # (Note currently reference_system in STAC items cube:dimensions extension is unreliable)
            self.gdf_local_meta['zone_number'] = self.gdf_local_meta['tile'].str[1:3].astype(int)
    
            # Filter UTM zone
            self.gdf_local_meta = self.gdf_local_meta[self.gdf_local_meta['zone_number']==self.grid.zone_number]
    
        # Since we got the geometry in WGS84 coordinates, transform to native grid
        self.gdf_local_meta = self.gdf_local_meta.set_crs('4326').to_crs(self.grid.crs)


    def setup_rasteroverview(self):
        """Setup the Rasteroverview object with appropriate partitions for HSI creation."""
        # Available timestamps
        self.timestamps = sorted(set(self.gdf_local_meta['time']))
        # Debug: getting rid of the temporal partitions dependency in the partition module
        self.t_part.timestamps = self.timestamps
    
        # Total bounds of all the data
        self.total_bounds = shapely.ops.unary_union(self.gdf_local_meta['geometry'].drop_duplicates())
        self.gdf_total_bounds = geopandas.GeoDataFrame([{'geometry': self.total_bounds}])

        if self.verbose:
            print('epsg                    ', self.grid.epsg)
            print('hsi_directory           ', self.hsi_directory)
            print('tmp_directory           ', self.tmp_directory)
            print('dataservice_type        ', self.dataservice_type)
            print('dimension_values        ', self.dimension_values)
            print('delta_pixel_hsi         ', self.delta_pixel_hsi)
            print('hsi level               ', self.pixel_level - self.delta_pixel_hsi)
            print('statistics_type         ', self.statistics_type)
            print('histogram               ', self.histogram)
            print('timestamps              ', len(self.timestamps))
            print('total_bounds            ', self.total_bounds.bounds)
    
        # Initialize the Rasteroverview
        self.raster = rasteroverview.Rasteroverview(
            pixel_level=self.pixel_level,
            delta_pixel_hsi=self.delta_pixel_hsi,
            hsi_directory=self.hsi_directory,
            tmp_directory=self.tmp_directory,
            dset_id=self.dset_id,
            layer_id=self.layer_id,
            dimension_values=self.dimension_values,
            dataservice_type=self.dataservice_type,
            valid_range=self.valid_range,
            statistics_type=self.statistics_type,
            histogram=self.histogram,
            grid=self.grid,
            verbose=self.verbose,
        )
    
        # Spatial partitions: Group hsi cells in appropriate spatial partitions
        self.s_part = partition.UniformSpatialPartition(
            raster_or_vector = 'raster',
            spatial_level = self.spatial_partition_level,
        )
    
        # Let the raster object know about the partitions
        self.raster.partitions(t_part=self.t_part, s_part=self.s_part)
    
        # Cash attributes in json file
        self.raster.to_json()
        #self.raster.from_json()
        
        # Figure out the spatial partitions
        qt = qtree.QTree(self.total_bounds, self.spatial_partition_level, grid=self.grid)
        self.gdf_grid_partitions = qt.gridded_to_geodataframe(key_col=None, level_col=None, crop_valid_range=False)
        
        # Sort largest partitions first, so there won't be any long straggler when workers loop through.
        self.gdf_grid_partitions['area'] = self.gdf_grid_partitions.intersection(self.gdf_total_bounds.loc[0, 'geometry']).area
        # Remove empty partitions that can happen when cells touch but not overlap
        self.gdf_grid_partitions = self.gdf_grid_partitions[self.gdf_grid_partitions['area']>0]
        self.gdf_grid_partitions = self.gdf_grid_partitions.sort_values(by='area', ascending=False).reset_index(drop=True)
        del self.gdf_grid_partitions['area']
            
        if self.verbose:
            try:
                # Plot spatial partitions covering total bounds
                ax = self.gdf_grid_partitions.plot(color='none')
                self.gdf_total_bounds.plot(ax=ax, alpha=0.2)
            except:
                print('Skipping spatial partitions plot.')

        # Get the elements_lst of timestamp/dimension combinations to loop through
        self._elements_lst()

    
    def _elements_lst(self):
        """Get the elements_lst of spatial and temporal partition combinations to loop through"""
        self.elements_lst = []
        if len(self.raster.temporal_partitions) == 0:
            # Only spatial partitions present
            self.elements_lst = list(self.gdf_grid_partitions['q_key'])
        else:
            # Nested Cartesian product of spatial and temporal partitions enables multiprocessing
            for elements in itertools.product(
                    self.gdf_grid_partitions['q_key'],
                    itertools.product(*self.raster.temporal_partitions.values())
            ):
                self.elements_lst.append(elements)
        if self.verbose:
            print('elements_lst            ', self.elements_lst)


    def hsi_worker(
        self,
        elements,
        **kwargs,
    ):
        """Calculate HSI for one partition (uses one CPU and memory dependent on size of partitions."""
        if len(self.raster.temporal_partitions)==0:
            # Only spatial partitions present
            self.spatial_partition = elements
            self.temporal_partition = {}
        else:
            # Both spatial and temporal partitions exist
            self.spatial_partition, temporal_elements = elements
            
            # Looping through the cartesian product of temporal partitions
            self.temporal_partition = {}
            for i, temporal_level in enumerate(self.raster.temporal_levels):
                self.temporal_partition[temporal_level] = temporal_elements[i]
    
        # Path to HSI
        hsi_local_path = self.raster._filepath(self.spatial_partition, self.temporal_partition)
        if self.hsi_dataservice_type == 'local_filesystem': # (mounted drive)
            prefix = os.path.join('data', 'hsi', self.collection_id).replace('\\', '/')
        elif self.hsi_dataservice_type == 'remote_filesystem':
            prefix = os.path.join(self.hsi_bucket, 'hsi', self.collection_id).replace('\\', '/')
        elif self.hsi_dataservice_type == 'hbase':
            raise NotImplementedError('hbase not supported for hsi upload.')
        else:
            raise ValueError(f'"{self.hsi_dataservice_type}" hsi_dataservice_type not understood.')
        self.hsi_remote_path = os.path.join(prefix, hsi_local_path.split(self.raster.hsi_directory)[-1].lstrip('\/')).replace('\\', '/')
    
        if self.skip_existing and self.hsi_dataservice_type == 'remote_filesystem':
            # Download existing HSI for this partition to skip existing timestamp/dimension combinations.
            try:
                self.hsi_remote_fs.download(self.hsi_remote_path, hsi_local_path)
            except FileNotFoundError:
                print('Did not find any existing HSI parquet file in cloud for this partition.')
        
        found_something = self.raster.hsi_statistics(
            self.temporal_partition, 
            self.spatial_partition, 
            probe_local_timestamps=False, 
            chunk_n=self.chunk_n,
            skip_existing=self.skip_existing,
            gdf_local_meta=self.gdf_local_meta,
            remote_fs=self.remote_fs,
            **kwargs,
        )
    
        if found_something:
            # Upload HSI
            if self.hsi_dataservice_type == 'remote_filesystem':
                print(f'hsi_local_path : {hsi_local_path}')
                print(f'hsi_remote_path: {self.hsi_remote_path}')
                self.hsi_remote_fs.upload(hsi_local_path, self.hsi_remote_path)

            # Registering HSI items in STAC
            self.register_hsi_items_stac()
            print(f'Registered hsi for partition {elements}')


    def register_hsi_items_stac(self):
        """Register HSI items in STAC."""
        if self.dataservice_type=='local_filesystem':
            # Accessing in local (or mounted) drive
            self.storage_urls = glob(os.path.join(self.hsi_directory, '**/*.parquet').replace('\\', '/'))
        elif self.dataservice_type=='remote_filesystem':
            # Using s3fs to access
            directory = os.path.split(self.hsi_remote_path)[0].replace('\\', '/')
            self.storage_urls = self.hsi_remote_fs.glob(os.path.join(directory, '**/*.parquet').replace('\\', '/'))
            if len(self.storage_urls)==0:
                # Maybe there are zero subdirectories to glob
                self.storage_urls = self.hsi_remote_fs.glob(os.path.join(directory, '*.parquet').replace('\\', '/'))
    
        if self.verbose:
            print('storage_urls            ', len(self.storage_urls))
            #print(*self.storage_urls, sep='\n')
        self.json_folder_submitted = os.path.join(self.json_folder, 'submitted') 
        os.makedirs(self.json_folder_submitted, exist_ok=True)
    
        for i, storage_url_part in enumerate(self.storage_urls):
            json_filepath = os.path.join(self.json_folder, f'item{i}.json')
    
            if self.dataservice_type=='local_filesystem':
                href = os.path.join(
                    's3://' + self.hsi_bucket,
                    'hsi',
                    storage_url_part.split('/hsi/')[1]
                ).replace('\\', '/')
                try:
                    # If available and installed properly, dask_geopandas can read the metadata faster
                    gdf1 = dask_geopandas.read_parquet(storage_url_part)
                    # Spatial extent in native coordinates based on partition (fast):
                    poly_native = gdf1.spatial_partitions.unary_union
                    gdf1 = gdf1.set_crs(self.grid.crs).compute()
                except:
                    # Read with geopandas
                    gdf1 = geopandas.read_parquet(storage_url_part)
                    # Spatial extent in native coordinates based on hsi cells (slow)
                    poly_native = gdf1.buffer(self.grid.epsilon).unary_union
                    gdf1 = gdf1.set_crs(self.grid.crs)
                    
            elif self.dataservice_type=='remote_filesystem':
                href = 's3://' + storage_url_part
                try:
                    # If available and installed properly, dask_geopandas can read the metadata faster
                    gdf1 = dask_geopandas.read_parquet(
                        's3://'+storage_url_part,
                        storage_options={
                            'key' : self.hsi_access_key_id,
                            'secret' : self.hsi_secret_access_key,
                            'client_kwargs' : {'endpoint_url': self.hsi_endpoint_url},
                        },
                    )
                    # Spatial extent in native coordinates based on partition (fast):
                    poly_native = gdf1.spatial_partitions.unary_union
                    gdf1 = gdf1.set_crs(self.grid.crs).compute()
                except:
                    # Read with geopandas
                    gdf1 = geopandas.read_parquet(
                        's3://'+storage_url_part,
                        storage_options={
                            'key' : self.hsi_access_key_id,
                            'secret' : self.hsi_secret_access_key,
                            'client_kwargs' : {'endpoint_url': self.hsi_endpoint_url},
                        },
                    )
                    # Spatial extent in native coordinates based on HSI cells (slow)
                    poly_native = gdf1.buffer(self.grid.epsilon).unary_union
                    gdf1 = gdf1.set_crs(self.grid.crs)
            
            poly_wgs84 = geopandas.GeoDataFrame([
                {'geometry': poly_native}
            ]).set_crs(self.grid.crs).to_crs(4326).loc[0, 'geometry']
            geometry_native = json.loads(shapely.to_geojson(poly_native))
            geometry_wgs84 = json.loads(shapely.to_geojson(poly_wgs84))
            total_bounds_native = list(poly_native.bounds)
            total_bounds_wgs84 = list(poly_wgs84.bounds)
    
            table_columns = []
            for col in gdf1.columns:
                if col in self.REQUIRED_STATS_CATEGORICAL | self.REQUIRED_STATS_NUMERIC:
                    description = 'statistic: ' + col
                elif col.startswith(self.OPTIONAL_STATS_CATEGORICAL_STARTSWITH):
                    description = 'value_count for category ' + col[len(self.OPTIONAL_STATS_CATEGORICAL_STARTSWITH):]
                elif col.endswith(self.OPTIONAL_STATS_NUMERIC_ENDSWITH):
                    description = 'quantile: ' + col
                elif col=='q_key':
                    description = 'base4 key of the observation location'
                elif col=='geometry':
                    description = 'location of the observation'
                elif col=='crs':
                    description = 'coordinate reference system'
                elif col=='time':
                    description = 'time of the observation'
                elif col=='dset_id':
                    description = 'PAIRS dataset ID'
                elif col=='layer_id':
                    description = 'PAIRS layer ID'
                elif col=='hsi_level':
                    description = 'spatial level of the HSI'
                elif col=='spatial_partition':
                    description = 'base4 key of the spatial partition'
                elif col in ('year', 'month', 'day'):
                    description = 'temporal partition: ' + col
                elif col=='dimension_band':
                    description = 'band'
                elif col=='dimension_tile':
                    description = 'tile'
                else:
                    print('Gdf columns', gdf1.columns)
                    raise ValueError(f'"{col}" column name not understood.')
                    description = None
    
                table_columns.append(
                    {
                        "name": col,
                        "description": description,
                        "type": repr(gdf1[col].dtype),
                    }
                )
    
            datetime_lst = [t.strftime(self.ISO_8601) for t in sorted(gdf1['time'].unique())]
            hsi_level = int([
                d for d in storage_url_part.split('/') if "hsi_level" in d
            ][0].split('=')[1])
    
            cube_dimensions = {
                "q_key": {
                    "axis": "q_key",
                    "extent": None,
                    "description": 'base4 key of the Morton curve',
                    "step": None,
                    "type": "spatial",
                    "reference_system": None
                },
                "time": {
                    "extent": [
                        datetime_lst[0],
                        datetime_lst[-1],
                    ],
                    "description": None,
                    "step": None,
                    "type": "temporal"
                }
            }
    
            for col in gdf1.columns:
                if col.startswith('dimension'):
                    cube_dimensions[col] = {
                        "axis": col,
                        "extent": sorted(gdf1[col].unique()),
                        "description": None,
                        "step": None,
                        "type": "other",
                        "reference_system": None
                        }
    
            cube_variables = {}
            for col in gdf1.columns:
                if (
                    (col in self.REQUIRED_STATS_CATEGORICAL) |
                    (col in self.REQUIRED_STATS_NUMERIC) | 
                    col.startswith(self.OPTIONAL_STATS_CATEGORICAL_STARTSWITH) | 
                    col.endswith(self.OPTIONAL_STATS_NUMERIC_ENDSWITH)
                ):
                    cube_variables[col] = {
                        "dimensions": [
                              "q_key",
                              "time",
                        ] + [c for c in gdf1.columns if c.startswith('dimension')],
                        "type": "data",
                        "description": "",
                        "unit": "",
                    }
    
            stac_item_dict = {
                "type": "Feature",
                "stac_version": "1.0.0",
                "stac_extensions": [
                    "https://stac-extensions.github.io/projection/v1.1.0/schema.json",
                    "https://stac-extensions.github.io/table/v1.2.0/schema.json",
                ],
                "id": uuid.uuid4().hex,
                "collection": self.hsi_collection_id,
                "bbox": total_bounds_wgs84,
                "geometry": geometry_wgs84,
    
                "properties": {
                    # debug: may need to indicate that datetime can be a list?
                    "datetime": datetime_lst[0],
                    "start_datetime": datetime_lst[0],
                    "end_datetime": datetime_lst[-1],
    
                    # Projection Extension (https://github.com/stac-extensions/projection)
                    #debug: need to set the projection in the geoparquet so that we can import here
                    "proj:epsg": self.grid.epsg,
                    "proj:bbox": total_bounds_native,
                    "proj:geometry": geometry_native,
    
                    "table:columns": table_columns,
                    "table:primary_geometry": "geometry",
                    "table:row_count": len(gdf1),
    
                    #debug: this is a hack to make the item openEO compatible
                    "cube:dimensions": cube_dimensions,
                    #debug: this is a hack to make the item openEO compatible
                    "cube:variables": cube_variables,
                },
                "links": [
                    {
                        "href": "./collection.json",
                        "rel": "collection",
                    },
                ],
                "assets": {
                    "data": {
                        "href": href,
                        "type": "table/parquet; application=geoparquet; profile=cloud-optimized",
                        "title": self.hsi_collection_id,
                        "roles": [
                            "hierarchical spatial index"
                        ],
                        "description": ""
                    },
                }
            }
    
            with open(json_filepath, 'w') as outfile:
                json.dump(stac_item_dict, outfile, indent=4, sort_keys=False)
    
        # Upload files to STAC
        stac_collection_url = os.path.join(
            self.stac_url, 'collections', self.hsi_collection_id.replace(" ", "%20"), 'items'
        ).replace('\\', '/')
        for file in glob(os.path.join(self.json_folder, '*.json')):
            file = file.replace('\\', '/')
            os.system(f'curl -H "Content-Type: application/json" -X POST {stac_collection_url} -kL {self.certificate} -d "@{file}"')
            os.system(f'mv {file} {self.json_folder_submitted}')


    def register_hsi_collection_stac(
        self,
        bbox,
        dt_start,
        dt_end,
        title,
        description,
        outpath,
    ):
        """
        Register HSI collection in STAC.

        Debug: NEED TO CHECK IF THIS METHOD IS STILL WORKING
        """
        # # Debug
        # stac = pystac_client.Client.open(self.stac_url)
        # stac_collections = list(stac.get_all_collections())
        # collection = stac_collections[[c.id for c in stac_collections].index(self.hsi_collection_id)]
    
        stac_collection_dict = {
            "id": self.hsi_collection_id,
            "type": "Collection",
            "stac_version": "1.0.0",
            "title": title,
            "description": description,
            "extent": {
                "spatial": {
                    "bbox": bbox,
                },
                "temporal": {
                    "interval": [
                        [dt_start.strftime(self.ISO_8601), dt_end.strftime(self.ISO_8601)]
                    ]
                }
            },
            "license": "Unknown",
            "links": [
                {
                    "href": "",
                    "rel": "self",
                },
                {
                    "href": "",
                    "rel": "item",
                },
            ],
        }
    
        #json.dumps(stac_collection_dict)
        with open(outpath, "w") as outfile:
            json.dump(stac_collection_dict, outfile, indent=4, sort_keys=False)
    
        return


def search_hsi(
    stac,
    access_key_id, 
    secret_access_key,
    endpoint_url,
    hsi_collection_id,
    search_aoi = None,
    dt_start = None,
    dt_end = None,
    fields = None,
    required_cols = [],
    filters = [],
    required_crs = None,
    columns = None,
    limit = None,
    verbose = False,
):
    LIMIT = 200
        
    #FIELDS = {"include": [], "exclude": []}
    FIELDS = {
        "include": [
            "id",
            "bbox",
            "datetime",
            "properties.table:columns",
            "properties.proj:epsg",
            #"properties.tile",
            #"properties.cube:variables",
            #"properties.cube:dimensions",
            #"properties.cloud_coverage",
        ],
        "exclude": [
        ],
    }

    if limit is None:
        limit = LIMIT
    
    if fields is None:
        fields = FIELDS
    
    if verbose:
        print('limit             ', limit)

    # Get the collection from its ID
    hsi_collection = stac.get_collection(hsi_collection_id)
    
    if verbose:
        print('hsi_collection_id ', hsi_collection_id)
    
    ISO_8601 = '%Y-%m-%dT%H:%M:%SZ'
    dt_string = None
    if dt_start is not None:
        dt_string = dt_start.strftime(ISO_8601)
        
        if dt_end is not None:
            dt_string = dt_string + '/' + dt_end.strftime(ISO_8601)
            
    if verbose:
        print('dt_string         ', dt_string)

    # pystac_client search 
    stac_search_result = stac.search(
        limit = limit,
        collections = [hsi_collection_id],
        intersects = search_aoi,
        datetime = dt_string,
        fields = fields,
    )
    search_items = next(stac_search_result.pages())
    
    if verbose:
        print('Found', len(search_items), 'search items')
        
    # Filter search items for presence of specific column
    for col in required_cols:
        search_items = [s for s in search_items if any([(col in c['name']) for c in s.properties['table:columns']])]
        
    # Filter by the STAC-informed reference_system
    if required_crs is not None:
        search_items = [
            item for item in search_items if item.properties['proj:epsg']==required_crs
        ]

    if verbose:
        print('Found', len(search_items), 'search items')

    filters = filters + [
            ('time', '>=', dt_start),
            ('time', '<=', dt_end),
        ]
    
    if verbose:
        print('filters           [', *filters, sep=',\n    ')
        print(']')

    if verbose:
        print('columns           ', columns)

    gdf_concat = []
    for search_item in search_items:
        storage_url = search_item.assets['data'].href
        epsg = search_item.properties['proj:epsg']
        try:
            gdf_hsi = geopandas.read_parquet(
                storage_url,
                storage_options={
                    'key' : access_key_id,
                    'secret' : secret_access_key,
                    'client_kwargs' : {'endpoint_url': endpoint_url}
                },
                columns=columns,
                filters=filters,
            ).set_crs(epsg)
        except ValueError as e:
            pass
        else:
            # Populate the bounds into columns
            #gdf_hsi = pandas.concat([gdf_hsi, gdf_hsi.bounds], axis=1)

            # DO DO: MAKE THIS WORK FOR NATIVE GRID
            # Spatial filtering (sub file-level) 
            #poly_filter = shapely.Polygon(search_aoi['coordinates'][0])
            #gdf_hsi = gdf_hsi[gdf_hsi.intersects(poly_filter)].reset_index(drop=True)
            
            gdf_concat.append(gdf_hsi)

    return pandas.concat(gdf_concat).reset_index(drop=True)


# def _epsg_from_file(filepath):
#     arr = xarray.open_dataarray(
#         filepath,
#         masked=True, 
#     )
#     epsg = arr.rio.crs.to_epsg()
    
#     if epsg is None:
#         # Parse the wkt string instead
#         crs = arr.rio.crs
#         assert crs.data['proj']=='utm'
#         zone_number = crs.data['zone']
#         if 'Northern Hemisphere' in pyproj.Proj(crs).crs.name:
#             north_or_south = 'north'
#             epsg = int(f'326{zone_number:02}')
#         elif 'Southern Hemisphere' in pyproj.Proj(crs).crs.name:
#             north_or_south = 'south'
#             epsg = int(f'327{zone_number:02}')
#         else:
#             raise ValueError('projection not understood')
        
#     return epsg



# def upload_hsi_cos(
#     src,
#     dst,
#     dataservice_type,
#     verbose = False,
#     **kwargs,
# ):
#     """
#     Upload all the parquet files in the src directory to dst.
#     If dataservice_type=='remote_filesystem' please provide a remote_fs with write credentials in the kwargs.
#     """
#     for src_path in glob(os.path.join(src, '**/*.parquet'), recursive=True):
#         dst_path = dst + src_path.split(src)[-1]
        
#         if dataservice_type=='local_filesystem':
#             # Using local (or mounted) drive to upload to
#             if not os.path.exists(dst_path):
#                 dst_dir = os.path.dirname(dst_path)
#                 os.makedirs(dst_dir, exist_ok=True)
#                 shutil.copy(src_path, dst_path)
#             else:
#                 if verbose:
#                     print('WARNING: file exists', dst_path)
#         elif dataservice_type=='remote_filesystem':
#             # Using s3fs to upload data to COS
#             remote_fs = kwargs.get('remote_fs')
#             if not remote_fs.exists(dst_path):
#                 remote_fs.upload(src_path, dst_path)
#             else:
#                 if verbose:
#                     print('WARNING: file exists', dst_path)
#         elif dataservice_type=='hbase':
#             raise NotImplementedError('hbase not supported for hsi upload.')
#         else:
#             raise ValueError(f'"{dataservice_type}" dataservice_type not understood.')
#     return
