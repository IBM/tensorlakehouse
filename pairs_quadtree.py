"""
    Navigating the PAIRS grid:
        - getMorton:              Calculating the morton number (z-ordered index of x-y plane)
        - getMortonInv:           Inverse function of getMorton(y,x), i.e. `y, x==getMortonInv(getMorton(y,x))` yields `True`.
        - getResolution:          Calculating pixel resolution based on level
        - lon_to_x:               Longitude to x-coordinate
        - lat_to_y:               Latitude to y-coordinate
        - x_to_lon:               x-coordinate to longitude
        - y_to_lat:               y-coordinate to latitude
        - x_to_center_lon:        x-coordinate to center longitude
        - y_to_center_lat:        y-coordinate to center latitude
            - _latlon_to_pairs_cell_xy: Pre-calculate the coordinates expected from a query to PAIRS with full PAIRS cell extent
        - getKey:                 Calculating a key based on lat/lon and level (encode)
        - getLatLon:              Calculating bottom-left lat/lon pixel corner from a key and level (decode)
        - getCenterLatLon:        Calculating center lat/lon from a key and level
        - getParentKey:           Calculating parent key 'levelsUp' levels above the current level
        - getChildrenKeys:        Calculating children keys

    Quadtree index (recursive, depth first search): 
        - quadTreePAIRS_dfs:      Walking the PAIRS z-order Quadtree recursively to retrieve leaf nodes that lie within a Polygon
        - quadTreeCellsPAIRS_dfs: Replacing internal nodes with a block of leaf nodes.
        - quadtree_dfs:           Get the keys at fixed level in one shot.
        - quadtree:               Legacy wrapper to get the keys at fixed level in one shot.

    Quadtree index (numpy vectorized, breadth first search): 
            - _step_quadTreePAIRS_bfs: Descending one level into the PAIRS z-order Quadtree (breadth first) 
            - _key_to_box:        Translate keys to shapely boxes (e.g. for performing intersections or containment operations)
            - _qt_to_index:       Composing "pyramid" index columns from the quadtree information. 
            - _quadTreePAIRS_bfs: Walking the PAIRS z-order Quadtree breadth first to retrieve nodes that lie within a Polygon
        - quadTreePAIRS_bfs:      Walking the PAIRS z-order Quadtree breadth first to retrieve nodes that lie within a Polygon
        - qt_to_boxes:            Translate entire quadtree to shapely boxes (e.g. for performing intersections or containment operations)
        - gridCellsPAIRS_bfs:     Replacing internal nodes with a block of leaf nodes. Returns all leaf nodes at the specified level.
        - quadtree_bfs:           Get the keys at fixed level in one shot.
"""

from typing import Tuple, Iterable, Dict

import math
import numpy
import shapely
import pandas

def getMorton(y: numpy.array, x: numpy.array) -> numpy.array:
    """
    Calculating the morton number (z-ordered index of x-y plane).
    :param y:   y-coordinate
    :param x:   x-coordinate
    :returns:   Morton number
    """
    y = y.astype(numpy.int64)
    y = (y | (y << 16)) & 0x0000FFFF0000FFFF
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y << 2)) & 0x3333333333333333
    y = (y | (y << 1)) & 0x5555555555555555
    y = y<<1
    
    x = x.astype(numpy.int64)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF  
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF  
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F  
    x = (x | (x << 2)) & 0x3333333333333333 
    x = (x | (x << 1)) & 0x5555555555555555 

    return x | y  #return the interleaved values (morton number)

def getMortonInv(z: numpy.array) -> Tuple[numpy.array, numpy.array]:
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

def getResolution(level: int) -> numpy.float:
    """
    Calculating pixel resolution based on level.
    :param level:   resolution level
    :returns:       resolution in degrees
    """
    return (2**(29-level))/1000000.

def getResolution_np(level: numpy.array) -> numpy.array:
    """
    Calculating pixel resolution based on level.
    :param level:   resolution level
    :returns:       resolution in degrees
    """
    return (2**(29-level.astype(float)))/1000000.

# Cashing 30 resolutions (0 to 29) as a numpy array. Recall using RESOLUTIONS[level]
RESOLUTION = getResolution_np(numpy.arange(30))

def lon_to_x(lon: numpy.array, level: int) -> numpy.array:
    return (lon + 180.) / getResolution(level)

def lat_to_y(lat: numpy.array, level: int) -> numpy.array:
    return (lat + 90.) / getResolution(level)

def x_to_lon(x: numpy.array, level: int) -> numpy.array:
    return x * getResolution(level) - 180

def y_to_lat(y: numpy.array, level: int) -> numpy.array:
    return y * getResolution(level) - 90

def x_to_center_lon(x: numpy.array, level: int) -> numpy.array:
    return (x+.5) * getResolution(level) - 180

def y_to_center_lat(y: numpy.array, level: int) -> numpy.array:
    return (y+.5) * getResolution(level) - 90

def _latlon_to_pairs_cell_xy(level: int, latmin: int, lonmin: int, latmax: int, lonmax: int, delta_pixel_cell=5) -> Tuple[Iterable, Iterable]:
    """
    Pre-calculate the coordinates expected from a query to PAIRS with full PAIRS cell extent
    """
    cellminx = math.floor(lon_to_x(numpy.array([lonmin]), level-delta_pixel_cell))
    cellminy = math.floor(lat_to_y(numpy.array([latmin]), level-delta_pixel_cell))
    cellminx = max(0, cellminx)
    cellminy = max(0, cellminy)
    cellmaxx = math.ceil(lon_to_x(numpy.array([lonmax]), level-delta_pixel_cell))
    cellmaxy = math.ceil(lat_to_y(numpy.array([latmax]), level-delta_pixel_cell))
    cellrx = range(cellminx, cellmaxx)
    cellry = range(cellminy, cellmaxy)
    return cellrx, cellry

def getKey(lat: numpy.array, lon: numpy.array, level: int) -> numpy.array:
    """
    Calculating a key based on lat/lon and level (encode)
    :param lat:   latitude
    :param lon:   longitude
    :param level: PAIRS resolution layer level
    :returns:     PAIRS spatial key
    """
    resolution = getResolution(level)
    x = (lon + 180.) / resolution
    y = (lat + 90.) / resolution
    return getMorton(y, x) 

def getLatLon(key: numpy.array, level: int) -> Tuple[numpy.array, numpy.array]:
    """
    Calculating bottom-left lat/lon pixel corner from a key and level (decode)
    :param key:     z-ordered key (type long)
    :param level:   resolution level
    :returns:       lat-long tuple derived from the key

    This returns the bottom-left corner of the pixel.
    note: beware of rounding errors if lat, lon is used again in getKey(lat, lon, level)
    """
    y = 0
    x = 0
    for i in range(level):
        mask = 1<<2*i+1
        y+= (key&mask)>>(i+1)
        mask = 1<<2*i
        x+= (key&mask)>>i
    resolution = getResolution(level)
    lat = y * resolution - 90
    lon = x * resolution - 180
    return lat, lon

def getCenterLatLon(key: numpy.array, level: int) -> Tuple[numpy.array, numpy.array]:
    """
    Calculating center lat/lon from a key and level.
    :param key:     PAIRS key
    :param level:   resolution level
    :returns:       lat-long tuple derived from the key

    This returns the center of the pixel.
    note: this is safer than getLatLon if the lat, lon is used again in getKey(lat, lon, level)
    """
    y = 0
    x = 0
    for i in range(level):
        mask = 1<<2*i+1
        y+= (key&mask)>>(i+1)
        mask = 1<<2*i
        x+= (key&mask)>>i
    resolution = getResolution(level)
    lat = (y+.5) * resolution - 90 
    lon = (x+.5) * resolution - 180
    return lat, lon

def getParentKey(key: numpy.array, levelsUp: int) -> numpy.array:
    """
    Calculating parent key 'levelsUp' levels above the current level.
    :param Key:      PAIRS key
    :param levelsUp: Difference in levels between two resolution layers
    :returns:        The key at levelsUp coarser resolution
    """
    return key.astype(numpy.int64)>>(2*levelsUp)

def getChildrenKeys(key: numpy.array, levelsDown: int) -> numpy.array:
    """
    Calculating children keys.
    :param key:      PAIRS key
    :returns:        A 2d array of the 4**LevelsDown descendent for each key
    """
    key = key.astype(numpy.int64)
    key = key<<(2*levelsDown)
    key_prime = numpy.arange(4**levelsDown)
    key = numpy.tile(numpy.array([key]).T, (1, key_prime.shape[0]))
    key_prime = numpy.tile(key_prime, (key.shape[0], 1))
    return key + key_prime

def quadTreePAIRS_dfs(poly: shapely.Polygon, max_level: int, level=0, key=0):
    """
    Walking the PAIRS z-order Quadtree recursively (depth first search) to retrieve leaf nodes that lie within a Polygon
    Stop descending whenever the node lies completely within the polygon
    :param poly:       Polygon defined by lat/lon values in degrees
    :param max_level:  maximum depth of traversing the tree
    :param level:      PAIRS level (The root level is level = 1, setting level=0 means that the check for division is alway done 
                       the first time this quadTreePAIRS_dfs is called)
    :param key:        PAIRS key (The PAIRS root is key = 0)
    """
    if level >= max_level: 
        # return this node even though it may not be a leaf node
        return [(level, key)], True
    else:
        # check if the node needs to be devided further (polygon does not yet fully occupy all quadrants) 
        south, west = getLatLon(key, level)
        #res = getResolution(level)
        res = RESOLUTION[level]
        north = south + res
        east = west + res
        box_full = shapely.box(west, south, east, north)
        if poly.contains(box_full): 
            return [(level, key)], True
        else:  
            # This node may be an internal node
            center_y = south + res/2
            center_x = west + res/2
            quadrants = shapely.box(
                numpy.array([west, center_x, west, center_x]),
                numpy.array([south, south, center_y, center_y]),
                numpy.array([center_x, east, center_x, east]),
                numpy.array([center_y, center_y, north, north]),
            )
            intersects = shapely.intersects(poly, quadrants)
            quad_tree = []
            count = 0
            if intersects[0]:
                # It is possible to make a nested list here by using append exclusively instead of extend
                branchData, quadrantIsLeaf = quadTreePAIRS_dfs(poly, max_level, level+1, key<<2) #south-west
                if quadrantIsLeaf:
                    count+=1
                quad_tree.extend(branchData)
            if intersects[1]:
                branchData, quadrantIsLeaf = quadTreePAIRS_dfs(poly, max_level, level+1, (key<<2)+1) #south-east 
                if quadrantIsLeaf:
                    count+=1
                quad_tree.extend(branchData)
            if intersects[2]:
                branchData, quadrantIsLeaf = quadTreePAIRS_dfs(poly, max_level, level+1, (key<<2)+2) #north-west 
                if quadrantIsLeaf:
                    count+=1
                quad_tree.extend(branchData)
            if intersects[3]:
                branchData, quadrantIsLeaf = quadTreePAIRS_dfs(poly, max_level, level+1, (key<<2)+3) ##north-east 
                if quadrantIsLeaf:
                    count+=1
                quad_tree.extend(branchData)
            nodeIsLeaf = count == 4 
            #nodeIsLeaf = count in [4]   #[4] typical, [3,4] if fewer keys but slightly larger squares are preferred.
            if nodeIsLeaf:     #overwrite the higher-level data if enough quadrants (3 or 4) are present          
                quad_tree = [(level, key)] 
            return quad_tree, nodeIsLeaf

def quadTreeCellsPAIRS_dfs(quad_tree, level):
    """
    Replacing internal nodes with a block of leaf nodes.
    Returns all leaf nodes at the specified level.
    :param quad_tree: Polygon information in quadtree list form
    :param level:     PAIRS level
    :returns:         leaf nodes at this level
    """
    keys = []
    for node in quad_tree:
        if node[0] == level:
            keys.append(node[1])
        else:
            #replace the square key with the children keys on level "level"
            index = numpy.arange(2**(2*(level - node[0])))
            keys.extend(node[1] * 4 ** (level - node[0]) + index)
    return keys

def quadtree_dfs(poly, level):
    """
    Get the polyKeys in one shot. Using the (slower) recursive depth-first-search algorithm
    """
    # Get the quadtree representation
    quad_tree, _ = quadTreePAIRS_dfs(poly, max_level=level)
    # Get all the keys on the same resolution level
    keys = quadTreeCellsPAIRS_dfs(quad_tree, level)
    return keys

def quadtree(poly, level):
    """
    Get the keys at fixed level in one shot. Legacy
    """
    return quadtree_dfs(poly, level)

def _step_quadTreePAIRS_bfs(poly: shapely.Polygon, parent_keys: numpy.array, parent_level: int) -> Tuple[numpy.array, numpy.array]:
    """
    Descending one level into the PAIRS z-order Quadtree (breadth first) to retrieve nodes that lie within a Polygon
    :param poly:         Polygon defined by lat/lon values in degrees
    :param parent_keys:  Numpy Array of parent keys
    :param parent_level: Parent level
    """
    level = parent_level + 1

    # All children keys belonging to level "level" 
    keys = getChildrenKeys(parent_keys, 1).flatten()
    
    # Corresponding bounding boxes
    boxes = _key_to_box(keys, level)
    
    # Check for full containment. These keys are recorded as leaf keys and don't have to be devided further
    full_containments = shapely.contains(poly, boxes)
    leaf_keys = keys[full_containments]#.flatten()
    # The others need to be investigated further
    keys = keys[~full_containments]#.flatten()
    boxes = boxes[~full_containments]#.flatten()
    
    # Check for intersection with polygon
    intersects = shapely.intersects(poly, boxes)
    keys = keys[intersects]
        
    return leaf_keys, keys

def _key_to_box(key: numpy.array, level: int) -> shapely.Polygon:
    """
    Translate keys to shapely boxes (e.g. for performing intersections or containment operations)
    :key:   numpy array of keys
    :level: level
    """
    south, west = getLatLon(key, level)
    #res = getResolution(level)
    res = RESOLUTION[level]
    north = south + res
    east = west + res
    return shapely.box(west, south, east, north)

def _qt_to_index(qt: Dict, max_level: int) -> pandas.DataFrame:
    """
    Composing "pyramid" index columns from the quadtree information. 
    These can be used on levels up to the leaf nodes for partitioning/filtering 
    :param qt:        qt quadtree dictionary
    :param max_level: Maximum level to descend into the tree. Nodes at max_level will be returned as leaf nodes
    """
    df = pandas.DataFrame()
    for leaf in range(1, max_level+1):
        df1 = pandas.DataFrame(qt[leaf]).rename(columns={0: leaf})
        for l in reversed(range(1, leaf)):
            df1[l] = getParentKey(numpy.array(df1[l+1]), 1)
        if len(df)>0:
            df = pandas.merge(df, df1, on=list(numpy.arange(0 ,leaf-1)+1), how='outer')
        else:
            df = df1
    df = df.sort_values(by=list(range(1, max_level+1))).reset_index(drop=True)
    df = df[list(range(1, max_level+1))]
    return df

def _quadTreePAIRS_bfs(poly: shapely.Polygon, max_level: int) -> Dict:
    """
    Walking the PAIRS z-order Quadtree breadth first to retrieve nodes that lie within a Polygon
    This funstion returns a quadtree that may still need pruning (4 children -> parent)
    :param poly:         Polygon defined by lat/lon values in degrees
    :param max_level:    Maximum level to descend into the tree. Nodes at max_level will be returned as leaf nodes
    :returns qt:         qt quadtree dictionary indexed by level
    """
    qt = {}
    for parent_level in range(0, max_level):
        level = parent_level + 1
        if parent_level==0:
            parent_keys = numpy.array([0])
        else: 
            parent_keys = keys
        leaf_keys, keys = _step_quadTreePAIRS_bfs(poly, parent_keys, parent_level)
        
        # Remember the leaf keys at parent level
        qt[level] = leaf_keys
    
    # Remember the keys at max_level even though they may not be true leafs
    qt[parent_level+1] = numpy.hstack([qt[parent_level+1], keys])
    
    return qt
    
def quadTreePAIRS_bfs(poly: shapely.Polygon, max_level: int) -> Dict:
    """
    Walking the PAIRS z-order Quadtree breadth first to retrieve nodes that lie within a Polygon
    :param poly:         Polygon defined by lat/lon values in degrees
    :param max_level:    Maximum level to descend into the tree. Nodes at max_level will be returned as leaf nodes
    :returns qt:         qt quadtree dictionary indexed by level
    :returns df_qt:      quadtree (only the leaf nodes) as a pandas DataFrame
    :returns df_index:   quadtree index as a pandas DataFrame
    """
    # Get the quadtree (that may still need pruning)
    _qt = _quadTreePAIRS_bfs(poly, max_level)
    
    # Convert into tabular form
    df_index = _qt_to_index(_qt, max_level)
    
    # We may still have to combine quadrants:
    # Whenever four quadrants with the same parent key are intersecting the polygon, we need to replace them by the parent
    for l in reversed(range(1, max_level)):
        # Checking if there are 4 of the same keys at level l present
        combine = (df_index.groupby(l).transform('count')[l+1]==4)
        # Checking if expected sucessive values on level l+1 are all present (no duplicates)
        combine = combine & (df_index.groupby(l)[l+1].transform('nunique')==4)
        # Checking if those sucessive values are leaf nodes
        if l+2<=max_level:
            combine = combine & (df_index.groupby(l)[l+2].transform('nunique')==0)
        
        df_index.loc[combine, l+1] = numpy.nan
        df_index = df_index.drop_duplicates().reset_index(drop=True)
    df_index = df_index.sort_values(by=list(range(1, max_level+1))).reset_index(drop=True)
    df_index = df_index[list(range(1, max_level+1))]
    
    # Calculate the correct quadtree (with combined nodes) using the index
    qt = {}
    df_qt = df_index.copy()
    for l in reversed(df_qt.columns):
        # Grab a single level starting with the deepest
        qt[l] = numpy.array(df_qt[l].dropna()).astype(int)
        # If we found a key, remove all lower-level keys in this row
        df_qt.loc[~df_qt[l].isnull(), numpy.arange(1, l)] = numpy.nan
    
    return qt, df_qt, df_index #, _qt

def qt_to_boxes(qt):
    """
    translate entire quadtree to shapely boxes (e.g. for performing intersections or containment operations)
    """
    boxes = []
    for l in qt.keys():
        boxes.append(numpy.array(_key_to_box(qt[l], l)))
    return numpy.hstack(boxes)

def gridCellsPAIRS_bfs(qt, level):
    """
    Replacing internal nodes with a block of leaf nodes.
    Returns all leaf nodes at the specified level.
    Usually used with level==max_level of the qt.
    Note that when level>max_level of the qt, cells may be returned that are outside the original polygon
    :param qt:     Quadtree dictionary
    :param level:  PAIRS level
    :returns:      all nodes at this level
    """
    keys = []
    for l in qt.keys():
        if len(qt[l])>0:
            if l==level:
                # Take the keys as they are
                keys.append(qt[l])
            elif l<level:
                # Replace the square key withs the children keys on level "level"
                index = numpy.arange(2**(2*(level - l)))
                qt_level = qt[l] * 4 ** (level - l)
                keys.extend(numpy.add.outer(qt_level, index))
            elif l>level:
                # Combine higher-level keys 
                p = getParentKey(numpy.array([qt[l]]), l-level)
                p = numpy.array([numpy.unique(p)])
                keys.extend(p)
    return numpy.sort(numpy.unique(numpy.concatenate(keys)))

def quadtree_bfs(poly, level):
    """
    Get the keys at fixed level in one shot.
    """
    # Get the quadtree representation
    qt, df_qt, df_index = quadTreePAIRS_bfs(poly, max_level=level)
    # Get all the keys on the same resolution level
    keys = gridCellsPAIRS_bfs(qt, level)
    return keys

