import requests
import numpy
import pandas
import json
import logging
from typing import Tuple, Dict

import pairs_quadtree

logger = logging.getLogger(__name__)

DATASERVICEENDPOINT = 'http://pairs-interactive01.pok.ibm.com:9084/pairsdataservice'
#DATASERVICEENDPOINT = 'http://wattsun5.pok.ibm.com:9082/pairsdataservice'
PERCENTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

def getMortonInv(z):
    """
    Inverse function of getMorton(y,x), i.e. `y, x == getMortonInv(getMorton(y,x))`
    yields `True`.
    :z long:    z-ordered key
    :returns:   latitude and longitude integer index of PAIRS raster array
                (lower left corner is root, i.e. [0,0,])
    :rtype:     (int, int)
    """
    x = z               & 0x5555555555555555
    x = (x | x >> 1)    & 0x3333333333333333
    x = (x | x >> 2)    & 0x0f0f0f0f0f0f0f0f
    x = (x | x >> 4)    & 0x00ff00ff00ff00ff
    x = (x | x >> 8)    & 0x0000ffff0000ffff
    x = (x | x >> 16)   & 0x00000000ffffffff
    y = (z >> 1)        & 0x5555555555555555
    y = (y | y >> 1)    & 0x3333333333333333
    y = (y | y >> 2)    & 0x0f0f0f0f0f0f0f0f
    y = (y | y >> 4)    & 0x00ff00ff00ff00ff
    y = (y | y >> 8)    & 0x0000ffff0000ffff
    y = (y | y >> 16)   & 0x00000000ffffffff
    return y, x

def getMorton(y: numpy.array, x: numpy.array) -> numpy.array:
    """
    Calculating the morton number (z-ordered index of x-y plane).
    :param y:   y-coordinate
    :param x:   x-coordinate
    :returns:   Morton number
    """
    y = y.astype(numpy.int64)
    y = (y | (y << 16)) & 0x0000FFFF0000FFFF #0b0000000000000000111111111111111100000000000000001111111111111111
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF  #0b0000000011111111000000001111111100000000111111110000000011111111
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F  #0b0000111100001111000011110000111100001111000011110000111100001111
    y = (y | (y << 2)) & 0x3333333333333333  #0b0011001100110011001100110011001100110011001100110011001100110011
    y = (y | (y << 1)) & 0x5555555555555555  #0b0101010101010101010101010101010101010101010101010101010101010101
    y = y<<1
    
    x = x.astype(numpy.int64)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF  
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF  
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F  
    x = (x | (x << 2)) & 0x3333333333333333 
    x = (x | (x << 1)) & 0x5555555555555555 

    return x | y  #return the interleaved values (morton number)

def getParentKey(key: numpy.array, levelsUp: int) -> numpy.array:
    """
    Calculating parent key 'levelsUp' levels above the current level.
    :param Key:      PAIRS key
    :param levelsUp: Difference in levels between two resolution layers
    :returns:        The key at levelsUp coarser resolution
    """
    return key>>(2*levelsUp)

def quadtree(poly, level):
    # Get the quadtree representation
    polyQuadTree, _ = pairs_quadtree.QuadTreePAIRS(poly, max_level=level)
    # Get all the keys on the same resolution level
    polyCells = pairs_quadtree.QuadTreeCellsPAIRS(polyQuadTree, level)
    return polyQuadTree, polyCells

def pairs_x_to_lon(x, level):
    return x * pairs_quadtree.getResolution(level) - 180

def pairs_y_to_lat(y, level):
    return y * pairs_quadtree.getResolution(level) - 90

def pairs_x_to_center_lon(x, level):
    return pairs_x_to_lon(x+.5, level)

def pairs_y_to_center_lat(y, level):
    return pairs_y_to_lat(y+.5, level)

def get_global_timestamps(layerid, starttime, endtime):
    result = requests.get(f'{DATASERVICEENDPOINT}/v2/dataquery/layer/{layerid}/timestamp/global',
        params = {
            'count' : 100000,
            'start' : starttime,
            'end' : endtime
        }
    )
    timestamps = sorted(result.json()['timestamps'])
    return timestamps

def _query_data_service(layer_id: str, level: int, latmin: float, lonmin: float, latmax: float, lonmax: float, timestamp: int, dimensions: Dict={}) -> requests.models.Response:
    layer = {
        'id' : layer_id,
        'temporal' : 
        {
            'intervals' : [
                {
                    'snapshot' : timestamp,
                }
            ]
        },
        'dimensions' : None,
    }
    
    api_response = requests.get(f'{DATASERVICEENDPOINT}/v2/dataquery/layer/raster',
        params = {
            'level' : level,
            'bbox' : f'{lonmin},{latmin},{lonmax},{latmax}',
            'ibmpairslayer' : json.dumps(layer),
        }
    )
    
    return api_response

def query_data_service(layer_id: str, level: int, latmin: float, lonmin: float, latmax: float, lonmax: float, timestamp: int, dimensions: Dict={}) -> Tuple[Dict, numpy.array]:
    '''
    Args:
    
    Returns:
        Tuple (pairs_headers, data). pairs_headers is a dictionary with keys height, width, boundingBox.
        data is a 2-dimensional numpy array. The data[0, 0] is the northwesternmost pixel.
    '''
    api_response = _query_data_service(layer_id, level, latmin, lonmin, latmax, lonmax, timestamp, dimensions)
    
    pairs_headers = json.loads(api_response.headers['ibmpairs'])
    data_height = pairs_headers['height']
    data_width = pairs_headers['width']

    data = numpy.frombuffer(
        api_response.content,
        dtype=numpy.dtype(numpy.float32).newbyteorder('big')
    ).reshape(data_height, data_width)
    
    data = numpy.where(data!=-9999., data, numpy.nan)
    
    return pairs_headers, data

def query_key_to_frame(
    layer_id: str, 
    level: int, 
    query_key: int, 
    query_level:int, 
    timestamp: int, 
    dimensions: Dict={}, 
    overview=False, 
    delta_pixel_overview=5,
    delta_pixel_cell=5,
    xy=False
) -> pandas.DataFrame:
    delta_pixel_query = level-query_level
    y_query, x_query = getMortonInv(query_key)
    y_pixel = y_query*2**delta_pixel_query+numpy.arange(2**delta_pixel_query)
    x_pixel = x_query*2**delta_pixel_query+numpy.arange(2**delta_pixel_query)
    
    latmin = pairs_y_to_center_lat(y_pixel.min(), level)
    latmax = pairs_y_to_center_lat(y_pixel.max(), level)
    lonmin = pairs_x_to_center_lon(x_pixel.min(), level)
    lonmax = pairs_x_to_center_lon(x_pixel.max(), level)
    
    """
    print('Old:', latmin, latmax, lonmin, lonmax)
    res = pairs_quadtree.getResolution(level-delta_pixel_cell) # Resolution on the pairs cell level
    query_res = pairs_quadtree.getResolution(query_level) # Resolution on the query key level
    
    # Corner coordinates
    latmin, lonmin = pairs_quadtree.getLatLon(query_key, query_level)
    latmax = latmin + query_res
    lonmax = lonmin + query_res
    
    # Center cell coordinates
    latmin += res/2-1e-13
    latmax -= res/2+1e-13
    lonmin += res/2-1e-13
    lonmax -= res/2+1e-13
    print('New:', latmin, latmax, lonmin, lonmax)
    """
    
    pairs_headers, data = query_data_service(layer_id, level, latmin, lonmin, latmax, lonmax, timestamp)
    data_height = pairs_headers['height']
    data_width = pairs_headers['width']
    
    clip_height = min(data_height, 2**delta_pixel_query) 
    clip_width = min(data_width, 2**delta_pixel_query)
    if (clip_height!=data_height) or (clip_width!=data_width):
        print('WARNING: QUERY ENDPOINT RETURNED TOO MUCH DATA. CLIPPING HERE')
        print('data.shape', data.shape)
        print('data_height, data_width', data_height, data_width)
        print('clip_height, clip_width', clip_height, clip_width)
    data = data[-clip_height:, :clip_width]
    x_pixel = x_pixel[:clip_width]
    y_pixel = y_pixel[:clip_height]
    
    x_pixel_grid, y_pixel_grid = numpy.meshgrid(x_pixel, y_pixel[::-1], indexing='xy')
    key_pixel = getMorton(y_pixel_grid, x_pixel_grid)
    
    data_frame = pandas.DataFrame(
        {
            'value' : data.reshape(-1),
            'timestamp' : timestamp
        },
        index=pandas.Index(key_pixel.reshape(-1), name='key')
    )
    if xy:
        data_frame['x'] = x_pixel_grid.reshape(-1)
        data_frame['y'] = y_pixel_grid.reshape(-1)
    data_frame = data_frame.dropna(how='any')
    data_frame = data_frame.sort_index()
    
    if overview:
        overview_keys = getParentKey(data_frame.index.values, delta_pixel_overview)
        grouped = data_frame.groupby(overview_keys, as_index=True)
        overview_timestamps = grouped['timestamp'].first()
        overview_base_stats = grouped['value'].aggregate(['first'])
        overview_advanced_stats = grouped['value'].describe(percentiles=PERCENTILES)
        overview_frame = overview_timestamps.to_frame().join(overview_base_stats).join(overview_advanced_stats)
        overview_frame.index.name = 'key'
        return overview_frame
    
    else:
        return data_frame