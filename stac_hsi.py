import os
os.environ['USE_PYGEOS'] = '0'
from glob import glob
import shutil

# import nestedgrid
# import mortoncurve
import qtree
import partition
import rasteroverview

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

def stac_search_available(
    stac_url,
    collection_id,
    search_aoi,
    year = None,
    month = None,
    day = None,
    fields = None,
    filter_band = None,
    filter_crs = None, #DEBUG, DOES NOT WORK YET SINCE RAW DATA HAS WRONG CRS
    verbose = False,
):
    """Search for available raw data."""
    LIMIT = 10000
    ISO_8601 = '%Y-%m-%dT%H:%M:%SZ'
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
    if day is None:
        if month is None:
            if year is None:
                dt_start = None
            else:
                dt_start = datetime(year, 1, 1)
                dt_end = (datetime(year+1, 1, 1)-timedelta(seconds=1))
        else:
            if year is None:
                raise ValueError
            else:
                dt_start = datetime(year, month, 1)
                if month<12:
                    dt_end = (datetime(year, month+1, 1)-timedelta(seconds=1))
                elif month==12:
                    dt_end = (datetime(year+1, 1, 1)-timedelta(seconds=1))
                else:
                    raise ValueError
    else:
        if year is None:
            raise ValueError
        elif month is None:
            raise ValueError
        else:
            dt_start = datetime(year, month, day)
            dt_end = datetime(year, month, day)+timedelta(seconds=24*3600-1)

    # Translate datetime to format stac understands
    dt_string = None
    if dt_start is not None:
        dt_string = dt_start.strftime(ISO_8601)
        if dt_end is not None:
            dt_string = dt_string + '/' + dt_end.strftime(ISO_8601)
    if verbose:
        print('dt_string', dt_string)
        print('search_aoi', search_aoi)
        
    # pystac_client search
    stac = pystac_client.Client.open(stac_url)
    stac_search_result = stac.search(
        limit = LIMIT,
        collections = [collection_id],
        intersects = search_aoi,
        datetime = dt_string,
        fields = fields,
    )
    
    search_items = None
    i = 0
    for next_items in stac_search_result.pages():
        if verbose:
            print('batch', i, '; raw     ', len(next_items))

        # Filter band
        if filter_band is not None:
            next_items = [
                item for item in next_items if (filter_band in item.properties['cube:variables'])
            ]

        # Filter crs
        if filter_crs is not None:
            next_items = [item for item in next_items if (
                f'EPSG:'+str(item.properties['cube:dimensions']['x']['reference_system'])==filter_crs
            )]
            
        if verbose:
            print('batch', i, '; filtered', len(next_items))

        if i==0:
            search_items = next_items
        else:
            search_items = search_items + next_items
        i+=1

    if verbose:
        if search_items is not None:
            print('Found', len(search_items), 'search items')
        else:
            print('Found no search items')
    
    return search_items


# If necessary, get additional data from the filepath or file metadata
def _dt_from_filepath(filepath):
    dt_string = filepath.split('.')[-5]
    return datetime.strptime(dt_string, '%Y%jT%H%M%S').replace(tzinfo=pytz.utc)

def _tile_from_filepath(filepath):
    return filepath.split('.')[-6]

def _band_from_filepath(filepath):
    return filepath.split('.')[-2]

def _product_from_filepath(filepath):
    return filepath.split('.')[-7]

def _bounds_from_file(filepath):
    # Finding the bounds of the data and the available timestamps
    arr = xarray.open_dataarray(
        filepath,
        masked=True, 
    )
    #dx, dy = morton.resolution(pixel_level)
    dx = abs(arr.x[1].item()-arr.x[0].item())
    dy = abs(arr.y[1].item()-arr.y[0].item())
    return shapely.box(
        arr.x.min().item()-dx/2,
        arr.y.min().item()-dy/2,
        arr.x.max().item()+dx/2,
        arr.y.max().item()+dy/2,
    )

def _crs_from_file(filepath):
    arr = xarray.open_dataarray(
        filepath,
        masked=True, 
    )
    crs = arr.rio.crs.to_epsg()
    if crs is None:
        # Parse the wkt string instead
        crs = arr.rio.crs
        assert crs.data['proj']=='utm'
        zone_number = crs.data['zone']

    if 'Northern Hemisphere' in pyproj.Proj(crs).crs.name:
        north_or_south = 'north'
        crs = int(f'326{zone_number:02}')
    elif 'Southern Hemisphere' in pyproj.Proj(crs).crs.name:
        north_or_south = 'south'
        crs = int(f'327{zone_number:02}')
    else:
        raise ValueError('projection not understood')
        
    return crs

def stac_search_items_to_raster_local_metadata(
    search_items,
    cos_bucket,
    grid,
    dataservice_type, # 'local_filesystem', 'remote_filesystem'
):
    # Create metadataframe from stac search_items
    gdf_local_meta = geopandas.GeoDataFrame([{
        #'item': item,
        'id': item.id,
        #'time': item.datetime,
        'cloud_coverage': item.properties['cloud_coverage'],
        'tile': item.properties['tile'],
        'band': list(item.properties['cube:variables'].keys())[0],
        'geometry': shapely.box(*item.bbox),
        'remote_path': item.assets['data'].href.split('s3://')[-1],
    } for item in search_items])

    # Local path from remote path
    if dataservice_type=='remote_filesystem':
        # Either (A) download file from remote folder
        gdf_local_meta['filepath'] = gdf_local_meta['remote_path'].apply(lambda x: 's3://' + x)
    elif dataservice_type=='local_filesystem':
        # Or (B) use mounted s3fs
        raise NotImplementedError()
        # gdf_local_meta['filepath'] = gdf_local_meta['remote_path'].apply(
        #     lambda x: os.path.join('/path/to/mount/point', x.split(cos_bucket+'/')[-1])
        # )
    else:
        raise NotImplementedError()
    
    # Gather additional properties from filepath
    gdf_local_meta['time'] = gdf_local_meta['filepath'].apply(_dt_from_filepath)
    gdf_local_meta['epoch'] = gdf_local_meta['time'].apply(lambda x: int(x.timestamp()))
    # gdf_local_meta['tile'] = gdf_local_meta['filepath'].apply(_tile_from_filepath)
    # gdf_local_meta['band'] = gdf_local_meta['filepath'].apply(_band_from_filepath)
    gdf_local_meta['product'] = gdf_local_meta['filepath'].apply(_product_from_filepath)
    #gdf_local_meta['geometry_utm'] = gdf_local_meta['filepath'].apply(_bounds_from_filepath)
    #gdf_local_meta['crs'] = gdf_local_meta['filepath'].apply(_crs_from_file)

    # Instead of relying on the crs directly, we can use the tile information to deduce the UTM zone
    gdf_local_meta['zone_number'] = gdf_local_meta['tile'].str[1:3].astype(int)

    # Filter UTM zone
    gdf_local_meta = gdf_local_meta[gdf_local_meta['zone_number']==grid.zone_number]

    # Since we got the geometry in WGS84 coordinates, transform to native grid
    gdf_local_meta = gdf_local_meta.set_crs('4326').to_crs(grid.crs)

    # Sort
    gdf_local_meta = gdf_local_meta.sort_values(by=['product', 'band', 'tile', 'time']).reset_index(drop=True)
    
    return gdf_local_meta


def setup_hsi(
    overviewstore_directory,
    tmp_directory,
    grid,
    gdf_local_meta,
    bands,
    numeric_or_categorical, 
    dataservice_type, # 'local_filesystem', 'remote_filesystem'
    verbose = False,
):
    """Create the Hierarchical Spatial Index from Data in COS."""
    PIXEL_LEVEL = 20 # 30m UTM30m grid
    DELTA_PIXEL_OVERVIEW = 5
    SPATIAL_PARTITION_LEVEL = 7

    # Projection from grid
    crs = int(grid.crs.split('EPSG:')[-1])

    dimension_values = {
        'product':[
            'L30', # Landsat
            'S30', # Sentinel
        ],
        'band':bands,
        'tile':sorted(gdf_local_meta['tile'].unique()),
    }

    # Available timestamps
    timestamps = []
    for filepath in gdf_local_meta['filepath']:
        # Timestamps
        timestamps.append(_dt_from_filepath(filepath))
    timestamps = sorted(set(timestamps))

    # # Valid range
    # valid_bounds = morton.valid_range.bounds
    # valid_range = shapely.box(*valid_bounds)
    # print('valid_range            ', valid_range.bounds)

    # Total bounds of all the data
    total_bounds = shapely.ops.unary_union(gdf_local_meta['geometry'])

    if verbose:
        print('crs                     ', crs)
        print('overviewstore_directory ', overviewstore_directory)
        print('tmp_directory           ', tmp_directory)
        print('dataservice_type        ', dataservice_type)
        print('dimension_values        ', dimension_values)
        print('delta_pixel_overview    ', DELTA_PIXEL_OVERVIEW)
        print('overview level          ', PIXEL_LEVEL - DELTA_PIXEL_OVERVIEW)
        print('numeric_or_categorical  ', numeric_or_categorical)
        print('timestamps              ', len(timestamps))
        print('total_bounds            ', total_bounds.bounds)

    # For plotting purposes
    gdf_total_bounds = geopandas.GeoDataFrame([{'geometry': total_bounds}])

    # Initialize
    raster = rasteroverview.Rasteroverview(
        pixel_level=PIXEL_LEVEL,
        delta_pixel_overview=DELTA_PIXEL_OVERVIEW,
        overviewstore_directory=overviewstore_directory,
        tmp_directory=tmp_directory,
        dset_id=None,
        layer_id=None,
        dimension_values=dimension_values,
        dataservice_type=dataservice_type,
        valid_range=None,
        numeric_or_categorical=numeric_or_categorical,
        grid=grid,
        verbose=False,
    )

    # Determine overview cells using qtree algorithm
    gdf_grid = raster.qtree_spatial_overview()

    # Temporal partitions
    t_part = partition.TemporalPartition(
        raster_or_vector = 'raster',
        timestamps = timestamps,
    )
    t_part.get_temporal_partition_levels(['year', 'month'])
    #t_part.get_temporal_partition_levels()
    t_part.get_temporal_partitions()
    if verbose:
        print('temporal_levels         ', t_part.temporal_levels)
        print('len(temporal_partitions)', len(t_part.temporal_partitions))
        print('temporal_partitions     ', t_part.temporal_partitions)

    # Spatial partitions: Group overview cells in appropriate spatial partitions

    # Eiter Split the geometric area until the number of overview keys per partition is below a threshold level
    # split_n = 100000 #256 #1024 #200 #500 
    # s_part = partition.SpatialPartition(
    #     raster_or_vector = 'raster',
    #     max_spatial_level = raster.overview_level,
    #     split_n = split_n,
    # )

    # Or use uniform SPATIAL_PARTITION_LEVEL

    s_part = partition.UniformSpatialPartition(
        raster_or_vector = 'raster',
        spatial_level = SPATIAL_PARTITION_LEVEL,
    )

    if verbose:
        print('spatial_partition_level ', SPATIAL_PARTITION_LEVEL)

    df_partition_count = s_part.planned_partitions(gdf_grid)

    # Let the raster object know about the partitions
    raster.partitions(t_part=t_part, s_part=s_part)

    # Cash attributes in json file
    raster.to_json()
    raster.from_json()
    
    # Figure out the spatial partitions
    qt = qtree.QTree(total_bounds, raster.spatial_partition_level, grid=grid)
    gdf_grid_partitions = qt.gridded_to_geodataframe(key_col=None, level_col=None, crop_valid_range=False)
    
    # Sort largest partitions first, so there won't be any long straggler when workers loop through.
    gdf_grid_partitions['area'] = gdf_grid_partitions.intersection(gdf_total_bounds.loc[0, 'geometry']).area
    # Remove empty partitions that can happen when cells touch but not overlap
    gdf_grid_partitions = gdf_grid_partitions[gdf_grid_partitions['area']>0]
    gdf_grid_partitions = gdf_grid_partitions.sort_values(by='area', ascending=False).reset_index(drop=True)
    del gdf_grid_partitions['area']
        
    if verbose:
        try:
            # Plot spatial partitions covering total bounds
            ax = gdf_grid_partitions.plot(color='none')
            gdf_total_bounds.plot(ax=ax, alpha=0.2)
        except:
            print('Skipping spatial partitions plot.')
            pass
        
    return raster, gdf_grid, gdf_grid_partitions, gdf_total_bounds


def hsi_worker(
    elements,
    raster,
    CHUNK_N, #1
    SKIP_EXISTING, #False
    **kwargs,
):
    print('elements ', elements)
    
    if len(raster.temporal_partitions)==0:
        # Only spatial partitions present
        spatial_partition = elements
        temporal_partition = {}
    else:
        # Both spatial and temporal partitions exist
        spatial_partition, temporal_elements = elements
        
        # Looping through the cartesian product of temporal partitions
        temporal_partition = {}
        for i, temporal_level in enumerate(raster.temporal_levels):
            temporal_partition[temporal_level] = temporal_elements[i]

    raster.overview_statistics(
        temporal_partition, 
        spatial_partition, 
        probe_local_timestamps=False, 
        skip_existing=SKIP_EXISTING,
        n_workers = 1,
        chunk_n = CHUNK_N,
        **kwargs,
    )
    return spatial_partition, temporal_partition


def upload_hsi_cos(
    src,
    dst,
    dataservice_type, # 'local_filesystem', 'remote_filesystem'
    verbose = False,
    **kwargs,
):
    """
    Upload all the parquet files in the src directory to dst.
    If dataservice_type=='remote_filesystem' please provide a remote_fs with write credentials in the kwargs.
    """
    for src_path in glob(os.path.join(src, '**/*.parquet'), recursive=True):
        dst_path = dst + src_path.split(src)[-1]
        
        if dataservice_type=='local_filesystem':
            # Using local (or mounted) drive to upload to
            if not os.path.exists(dst_path):
                dst_dir = os.path.dirname(dst_path)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src_path, dst_path)
            else:
                if verbose:
                    print('WARNING: file exists', dst_path)
        elif dataservice_type=='remote_filesystem':
            # Using s3fs to upload data to COS
            remote_fs = kwargs.get('remote_fs')
            if not remote_fs.exists(dst_path):
                remote_fs.upload(src_path, dst_path)
            else:
                if verbose:
                    print('WARNING: file exists', dst_path)
        elif dataservice_type=='hbase':
            raise NotImplementedError('hbase not supported for hsi upload.')
        else:
            raise ValueError(f'"{dataservice_type}" dataservice_type not understood.')
    return


def register_hsi_collection_stac(
    stac_url,
    hsi_collection_id,
    bbox,
    dt_start,
    dt_end,
    title,
    description,
    outpath,
):
    ISO_8601 = '%Y-%m-%dT%H:%M:%SZ'
    # # Debug
    # stac = pystac_client.Client.open(stac_url)
    # stac_collections = list(stac.get_all_collections())
    # collection = stac_collections[[c.id for c in stac_collections].index(hsi_collection_id)]

    stac_collection_dict = {
        "id": hsi_collection_id,
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
                    [dt_start.strftime(ISO_8601), dt_end.strftime(ISO_8601)]
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


def register_hsi_items_stac(
    cos_bucket,
    hsi_collection_id,
    hsi_directory,
    grid,
    json_folder,
    dataservice_type,
    load_hsi_stac_filepath, #'/path/to/shellscript/load_hsi_stac.sh',
    stac_url,
    **kwargs,
):
    """
    Register HSI items in STAC.
    If dataservice_type=='remote_filesystem' please provide
    remote_fs, access_key_id, secret_access_key, and endpoint_url in the kwargs.
    """
    CERTIFICATE = "ca.cert.txt"
    if dataservice_type=='remote_filesystem':
        remote_fs = kwargs.get('remote_fs')
        access_key_id = kwargs.get('access_key_id')
        secret_access_key = kwargs.get('secret_access_key')
        endpoint_url = kwargs.get('endpoint_url')
        
    ISO_8601 = '%Y-%m-%dT%H:%M:%SZ'
    REQUIRED_STATS_CATEGORICAL = {'count', 'top', 'freq', 'unique', 'first'}
    REQUIRED_STATS_NUMERIC = {'count', 'min', 'max', 'mean', 'std', 'first'}
    OPTIONAL_STATS_CATEGORICAL_STARTSWITH = 'count_' # Value counts (histogram columns)
    OPTIONAL_STATS_NUMERIC_ENDSWITH = '%' # Quantiles

    if dataservice_type=='local_filesystem':
        # Accessing in local (or mounted) drive
        storage_urls = glob(os.path.join(hsi_directory, '*/*/*/*.parquet'))
    elif dataservice_type=='remote_filesystem':
        # Using s3fs to access
        storage_urls = remote_fs.glob(os.path.join(hsi_directory, '**/*.parquet').replace('\\', '/'))
        if len(storage_urls)==0:
            # Maybe there are zero subdirectories to glob
            storage_urls = remote_fs.glob(os.path.join(hsi_directory, '*.parquet').replace('\\', '/'))

    print('storage_urls', len(storage_urls))
    #print(*storage_urls, sep='\n')

    for i, storage_url_part in enumerate(storage_urls):
        print('debug storage_url_part', storage_url_part)
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
        json_filepath = f'{json_folder}/item{i}.json'

        crs = int(grid.crs.split('EPSG:')[-1])
        if dataservice_type=='local_filesystem':
            href = os.path.join(
                's3://' + cos_bucket,
                'hsi',
                storage_url_part.split('/hsi/')[1]
            )
            try:
                # If available and installed properly, dask_geopandas can read the metadata faster
                gdf1 = dask_geopandas.read_parquet(storage_url_part)
                # Spatial extent in native coordinates based on partition (fast):
                poly_native = gdf1.spatial_partitions.unary_union
                gdf1 = gdf1.set_crs(crs).compute()
            except:
                # Read with geopandas
                gdf1 = geopandas.read_parquet(storage_url_part)
                # Spatial extent in native coordinates based on overview cells (slow)
                poly_native = gdf1.buffer(grid.epsilon).unary_union
                gdf1 = gdf1.set_crs(crs)
                
        elif dataservice_type=='remote_filesystem':
            href = 's3://' + storage_url_part
            try:
                # If available and installed properly, dask_geopandas can read the metadata faster
                gdf1 = dask_geopandas.read_parquet(
                    's3://'+storage_url_part,
                    storage_options={
                        'key' : access_key_id,
                        'secret' : secret_access_key,
                        'client_kwargs' : {'endpoint_url': endpoint_url},
                    },
                )
                # Spatial extent in native coordinates based on partition (fast):
                poly_native = gdf1.spatial_partitions.unary_union
                gdf1 = gdf1.set_crs(crs).compute()
            except:
                # Read with geopandas
                gdf1 = geopandas.read_parquet(
                    's3://'+storage_url_part,
                    storage_options={
                        'key' : access_key_id,
                        'secret' : secret_access_key,
                        'client_kwargs' : {'endpoint_url': endpoint_url},
                    },
                )
                # Spatial extent in native coordinates based on overview cells (slow)
                poly_native = gdf1.buffer(grid.epsilon).unary_union
                gdf1 = gdf1.set_crs(crs)
        
        poly_wgs84 = geopandas.GeoDataFrame([
            {'geometry': poly_native}
        ]).set_crs(crs).to_crs(4326).loc[0, 'geometry']
        geometry_native = json.loads(shapely.to_geojson(poly_native))
        geometry_wgs84 = json.loads(shapely.to_geojson(poly_wgs84))
        total_bounds_native = list(poly_native.bounds)
        total_bounds_wgs84 = list(poly_wgs84.bounds)

        table_columns = []
        for col in gdf1.columns:
            if col in REQUIRED_STATS_CATEGORICAL | REQUIRED_STATS_NUMERIC:
                description = 'statistic: ' + col
            elif col.startswith(OPTIONAL_STATS_CATEGORICAL_STARTSWITH):
                description = 'value_count for category ' + col.lstrip(OPTIONAL_STATS_CATEGORICAL_STARTSWITH)
            elif col.endswith(OPTIONAL_STATS_NUMERIC_ENDSWITH):
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
                description = 'dataset ID'
            elif col=='layer_id':
                description = 'layer ID'
            elif col=='overview_level':
                description = 'spatial level of the overview'
            elif col=='spatial_partition':
                description = 'base4 key of the spatial partition'
            elif col in ('year', 'month', 'day'):
                description = 'temporal partition: ' + col
            elif col=='dimension_product':
                description = 'product'
            elif col=='dimension_band':
                description = 'band'
            elif col=='dimension_tile':
                description = 'tile'
            else:
                raise ValueError(f'"{col}" column name not understood.')
                description = None

            table_columns.append(
                {
                    "name": col,
                    "description": description,
                    #"type": str(gdf1[col].dtype),
                    "type": repr(gdf1[col].dtype),
                }
            )

        datetime_lst = [t.strftime(ISO_8601) for t in sorted(gdf1['time'].unique())]
        overview_level = int([
            d for d in storage_url_part.split('/') if "overview_level" in d
        ][0].split('=')[1])
        #resolution = raster.morton.resolution(overview_level)[0]

        cube_dimensions = {
            "q_key": {
                "axis": "q_key",
                "extent": None, #sorted(gdf1['q_key'].unique()),
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
                (col in REQUIRED_STATS_CATEGORICAL) |
                (col in REQUIRED_STATS_NUMERIC) | 
                col.startswith(OPTIONAL_STATS_CATEGORICAL_STARTSWITH) | 
                col.endswith(OPTIONAL_STATS_NUMERIC_ENDSWITH)
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
            #"id": storage_url_part.lstrip('s3://').rsplit('/',1)[0],
            "id": uuid.uuid4().hex,
            "collection": hsi_collection_id,
            "bbox": total_bounds_wgs84,
            "geometry": geometry_wgs84,

            "properties": {
                # debug: may need to indicate that datetime can be a list?
                "datetime": datetime_lst[0],
                "start_datetime": datetime_lst[0],
                "end_datetime": datetime_lst[-1],

                # Projection Extension (https://github.com/stac-extensions/projection)
                #debug: need to set the projection in the geoparquet so that we can import here
                "proj:epsg": crs,
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
                    "title": hsi_collection_id,
                    "roles": [
                        "hierarchical spatial index"
                    ],
                    "description": ""
                },
            }
        }

        with open(json_filepath, 'w') as outfile:
            json.dump(stac_item_dict, outfile, indent=4, sort_keys=False)
            
    # See what json files are there
    # glob(os.path.join(json_folder, '*.json'))
    
    # Debug: catch the case where the collection_id contains space characters
    arg_collection_id = hsi_collection_id.replace(" ", "%20")
    
    # Upload using shellscript
    #os.system(f'{load_hsi_stac_filepath} >/dev/null 2>&1')
    #print(os.system(f'{load_hsi_stac_filepath}'))
    os.system(f'{load_hsi_stac_filepath} --collection_id={arg_collection_id} --json_folder={json_folder} --stac_url={stac_url} --CERTIFICATE={CERTIFICATE}')
    
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
        crs = search_item.properties['proj:epsg']
        
        # if (required_crs is not None) and (crs!=required_crs):
        #     continue

        # if verbose:
        #     print('storage_url       ', storage_url)

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
            ).set_crs(crs)
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