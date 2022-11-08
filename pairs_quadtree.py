"""
    Helper Functions:
        - get_aoiSquare:         Translation from Shapely polygon to PAIRS aoiSquare
        - get_polygon:           Translation from PAIRS aoiSquare to Shapely polygon
        - getMorton:             Calculating the morton number (z-ordered index of x-y plane)
        - getResolution:         Calculating pixel resolution based on level
        - getKey:                Calculating a key based on lat/lon and level (encode)
        - getLatLon:             Calculating bottom-left lat/lon pixel corner from a key and level (decode)
        - getCenterLatLon:       Calculating center lat/lon from a key and level
        - getParentKey:          Calculating parent key 'levelsUp' levels above the current level
        - getChildrenKeys:       Calculating children keys
        - merge_on_pKey_others:  Merge two dataframes on latitude and longitude columns by transforming to pKey at pLevel
        - mode:                  Fast Majority Vote function for pandas groupby objects
        - get_counts:            Count the number of rows per category in a Pandas DataFrame
        - getPixelArea:          Get the approximate pixel area in m2

    Quadtree functions: 
        - QuadTreePAIRS:       Walking the PAIRS z-order Quadtree recursively to retrieve leaf nodes that lie within a Polygon
        - getQuadTreeSquares:  Function to conveniently get poly_squares for plotting with matplotlib, shapely, and descartes.
        - QuadTreeCellsPAIRS:  Replacing internal nodes with a block of cell nodes

"""

import numpy
import pandas
from shapely.geometry import Point, box, Polygon

def get_aoiSquare(my_polygon):
    """
    Translation from Shapely polygon to PAIRS aoiSquare
    :my_polygon:    Shapely polygon
    :returns:       PAIRS aoiSquare
    """
    return [my_polygon.bounds[1],     # min. latitude
            my_polygon.bounds[0],     # min. longitude
            my_polygon.bounds[3],     # max. latitude
            my_polygon.bounds[2]]     # max. longitude

def get_polygon(aoiSquare):
    """
    Translation from PAIRS aoiSquare to Shapely polygon
    :aoiSquare:     PAIRS aoiSquare
    :returns:       Shapely polygon
    """
    return box(aoiSquare[1], aoiSquare[0], aoiSquare[3], aoiSquare[2])

def getMorton(y, x):
    """
    Calculating the morton number (z-ordered index of x-y plane).
    :param y:   y-coordinate
    :param x:   x-coordinate
    :returns:   Morton number
    """
    y = int(y)
    y = (y | (y << 16)) & 0x0000FFFF0000FFFF #0b0000000000000000111111111111111100000000000000001111111111111111
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF  #0b0000000011111111000000001111111100000000111111110000000011111111
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F  #0b0000111100001111000011110000111100001111000011110000111100001111
    y = (y | (y << 2)) & 0x3333333333333333  #0b0011001100110011001100110011001100110011001100110011001100110011
    y = (y | (y << 1)) & 0x5555555555555555  #0b0101010101010101010101010101010101010101010101010101010101010101
    y = y<<1
    
    x = int(x)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF  
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF  
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F  
    x = (x | (x << 2)) & 0x3333333333333333 
    x = (x | (x << 1)) & 0x5555555555555555 

    return x | y  #return the interleaved values (morton number)

def getResolution(level):
    """
    Calculating pixel resolution based on level.
    EVENTUALLY NEED TO REPLACE THIS FUNCTION WITH A PRE_INITIALIZED DICT
    :param level:   resolution level
    :returns:       resolution in degrees
    """
    return (2**(29-level))/1000000.0

getResolution_vect = numpy.vectorize(getResolution)

def getKey(lat, lon, level):
    """
    Calculating a key based on lat/lon and level (encode)
    :param lat:   latitude
    :param lon:   longitude
    :param level: PAIRS resolution layer level
    :returns:     PAIRS spatial key
    """
    resolution = getResolution(level)
    x = (lon + 180) / resolution
    y = (lat + 90) / resolution
    return getMorton(y, x) 

getKey_vect = numpy.vectorize(getKey)

def getLatLon(key, level): 
    """
    Calculating bottom-left lat/lon pixel corner from a key and level (decode)
    BEWARE THAT ROUNDING ERRORS CAN BECOME PROBLEMATIC
    :param key:     z-ordered key (type long)
    :param level:   resolution level
    :returns:       lat-long tuple derived from the key

    This returns the bottom-left corner of the pixel.
    note: BEWARE THAT ROUNDING ERRORS CAN BECOME PROBLEMATIC
    """
    lat = 0
    lon = 0
    for i in range(level):
        mask = 1<<2*i+1
        lat+= (key&mask)>>(i+1)
        mask = 1<<2*i
        lon+= (key&mask)>>i
    resolution = getResolution(level)
    lat = lat*resolution-90
    lon = lon*resolution-180
    return lat, lon

getLatLon_vect = numpy.vectorize(getLatLon)

def getCenterLatLon(key, level):  
    """
    Calculating center lat/lon from a key and level.
    :param key:     PAIRS key
    :param level:   resolution level
    :returns:       lat-long tuple derived from the key

    This returns the center of the pixel.
    note: this is safer than getLatLon if the lat-lon is used again in getKey(lat, lon, level)
    """
    lat = 0
    lon = 0
    for i in range(level):
        mask = 1<<2*i+1
        lat+= (key&mask)>>(i+1)
        mask = 1<<2*i
        lon+= (key&mask)>>i
    resolution = getResolution(level)
    lat = lat*resolution-90+resolution/2  #+resolution/2 for center of pixel insetead of bottom-left of pixel
    lon = lon*resolution-180+resolution/2  #+resolution/2 for center of pixel insetead of bottom-left of pixel
    return lat, lon

getCenterLatLon_vect = numpy.vectorize(getCenterLatLon)

def getParentKey(key, levelsUp):
    """
    Calculating parent key 'levelsUp' levels above the current level.
    :param Key:      PAIRS key
    :param levelsUp: Difference in levels between two resolution layers
    :returns:        The key at levelsUp coarser resolution
    """
    return key>>(2*levelsUp)

getParentKey_vect = numpy.vectorize(getParentKey)

def getChildrenKeys(key, levelsDown=1):
    """
    Calculating children keys.
    :param key:      PAIRS key
    :returns:        A tuple of the 4 children keys (or 4**LevelsDown descendent keys)
    """
    """
    key = key<<2
    return key, key+1, key+2, key+3
    """
    key = key<<(2*levelsDown)
    return tuple(key + numpy.arange(4**levelsDown))

getChildrenKeys_vect = numpy.vectorize(getChildrenKeys)

def merge_on_pKey_others(df_left, df_right, how='inner', other_merge_columns=[], pLevel=18):
    """
    Merge two dataframes on latitude and longitude columns by transforming to pKey at pLevel
    :df_left:              Left DataFrame
    :df_right:             Right DataFrame
    :how:                  Merge method
    :other_merge_columns:  List of merge columns other than pKey
    :pLevel:               Pixel level
    :returns:              merged DataFrames
    """
    #Make sure we are not altering the original dataframes
    df_left = df_left.copy()
    df_right = df_right.copy()
    
    # Make sure the other columns are provided as a list even if it's just one element
    if type(other_merge_columns) is list:
        pass
    elif type(other_merge_columns) is tuple:
        pass
    else:
        other_merge_columns = [other_merge_columns]
    merge_columns = ['pKey'] + other_merge_columns

    if 'pKey' not in df_left.columns:
        df_left['pKey'] = getKey_vect(df_left['latitude'], df_left['longitude'], pLevel)
    if 'pKey' not in df_right.columns:
        df_right['pKey'] = getKey_vect(df_right['latitude'], df_right['longitude'], pLevel)
    del df_right['latitude']
    del df_right['longitude']

    df_merged = pandas.merge(df_left, df_right, on=merge_columns, how=how)
    return df_merged

def change_resolution(df, pLevel_in, pLevel_out, how='mean', other_key_columns=[]):
    """
    Currently only how='mean supported for downsampling
    Currently only 'near' supported for upsampling
    """
    # Make sure we are not altering the original dataframes
    df = df.copy()
    
    # Index needs to be reset for code further down
    df = df.reset_index(drop=True)
    
    # Make sure the other columns are provided as a list even if it's just one element
    if type(other_key_columns) is list:
        pass
    elif type(other_key_columns) is tuple:
        pass
    else:
        other_key_columns = [other_key_columns]
    key_columns = ['pKey'] + other_key_columns

    if 'pKey' not in df.columns:
        df['pKey'] = getKey_vect(df['latitude'], df['longitude'], pLevel_in)
    del df['latitude']
    del df['longitude']
    
    if pLevel_out>pLevel_in:
        # Upscaling (or interpolating)
        #getChildrenKeys_vect(df['pKey'].values, pLevel_out - pLevel_in)
        df['pKey'] = df['pKey'].apply(lambda x: getChildrenKeys(x, pLevel_out - pLevel_in))
        id_vars = [c for c in df.columns if c!='pKey']
        df = pandas.concat([df[id_vars], pandas.DataFrame(df['pKey'].values.tolist())], axis=1)
        df = df.melt(id_vars=id_vars, value_name='pKey')
        del df['variable']
        
    elif pLevel_out<pLevel_in:
        # Downsampling (or averaging)
        df['pKey'] = df['pKey'].apply(lambda x: getParentKey(x, pLevel_in - pLevel_out))
        #df['pKey'] = getParentKey_vect(df['pKey'], pLevel_in - pLevel_out)
        df = df.groupby(key_columns).mean()
        df = df.reset_index()
        
    #df['latitude'], df['longitude'] = getCenterLatLon_vect(df['pKey'], pLevel_out)
    df = df.reset_index(drop=True)
    return df

def mode(df, key_cols, value_col, count_col):
    """
    Fast Majority Vote function for pandas groupby objects
    :df:        Pandas DataFrame
    :key_cols:  Groupby columns
    :value_col: Column for which we want to get the mode (majority vote per groupby group).
    :count_col: Column name for a column in the returned DataFrame indicting how many times the mode appeared in its group
    :returns:   DataFrame with one record per group
    
    Pandas does not provide a `mode` aggregation function for its `GroupBy` objects. 
    This function is meant to fill that gap, though the semantics are not exactly the same.

    The input is a DataFrame with the columns `key_cols` that you would like to group on, 
    and the column `value_col` for which you would like to obtain the mode.

    The output is a DataFrame with a record per group that has at least one mode
    (null values are not counted). The `key_cols` are included as columns, `value_col
    contains a mode (ties are broken arbitrarily and deterministically) for each
    group, and `count_col` indicates how many times each mode appeared in its group.
    """
    return df.groupby(key_cols + [value_col]).size() \
             .to_frame(count_col).reset_index() \
             .sort_values(count_col, ascending=False) \
             .drop_duplicates(subset=key_cols)

def get_counts(df, cat_column_name, count_column_name):
    """
    Count the number of rows per category in a Pandas DataFrame
    :df:                 DataFrame
    :cat_column_name:    Column name of the category column (must be present in the df)
    :count_column_name:  Column name of the new count column (will be added by the function)
    :returns:            
    
    Example use: (Reduce the number of rotation classes by combining classes with too few members)
    rotation_counts = get_counts(df_field, 'rotation string', 'rotation count')
    df_reduced = pandas.merge(df_field, rotation_counts, on='rotation string', how='outer')
    """
    rotation_counts = pandas.DataFrame.from_dict(Counter(df[cat_column_name]), orient='index')
    rotation_counts.reset_index(inplace=True, drop=False)
    rotation_counts.columns=[cat_column_name, count_column_name]
    return rotation_counts


def getPixelArea(lat, lon, res):
    """
    Get the approximate pixel area in m2
    lat:    latitude
    lon:    longitude
    res:    pixel resolution
    return: area
    """
    # approximate radius of earth in km
    R = 6373000.0

    # dy does not change with latitude or longitude for a spherical earth
    dy = res * 111320

    # Convert to radians
    lat = numpy.radians(lat)
    dlon = res * numpy.pi / 180.0

    a = numpy.cos(lat)**2 * numpy.sin(dlon / 2)**2
    dx = R * 2 * numpy.arctan2(numpy.sqrt(a), numpy.sqrt(1 - a))

    return dx * dy


def QuadTreePAIRS(poly, max_level, level=0, key=0):
    """
    Walking the PAIRS z-order Quadtree recursively to retrieve leaf nodes that lie within a Polygon
    This version of QuadTreePAIRS combines four (or three) leafs into one leaf at lower resolution level.
    :param poly:      Polygon defined by lat/lon values in degrees
    :type poly:       shapely.geometry.Polygon
    :param max_level: maximum depth of traversing the tree
    :param level:     PAIRS level (The root level is level = 1, setting level=0 means that the check for division is alway done 
                      the first time this QuadTreePairs is called)
    :param key:       PAIRS key (The PAIRS root is key = 0)
    :returns:         Quadtree
    """
    if level >= max_level: #RETURN THIS NODE EVEN THOUGH MAY Not BE A LEAF NODE
        return [(level, key)], True
    else:
        #check if the node needs to be devided further (polygon does not yet fully occupy all quadrants) 
        bottom, left = getLatLon(key, level)
        center_y, center_x = getCenterLatLon(key, level)
        right = 2 * center_x - left
        top = 2 * center_y - bottom
        box_full = box(left, bottom, right, top)
        if poly.contains(box_full): 
            return [(level, key)], True
        else:  #THIS NODE MAY BE AN INTERNAL NODE
            box0 = box(left, bottom, center_x, center_y)
            box1 = box(center_x, bottom, right, center_y)
            box2 = box(left, center_y, center_x, top)
            box3 = box(center_x, center_y, right, top)
            intersecting = (not box0.intersection(poly).is_empty) +\
                            (not box1.intersection(poly).is_empty) +\
                            (not box2.intersection(poly).is_empty) +\
                            (not box3.intersection(poly).is_empty)
            data = []
            count = 0
            if not box0.intersection(poly).is_empty:
                #It is possible to make a nested list here by using append exclusively instead of extend
                branchData, quadrantIsLeaf = QuadTreePAIRS(poly, max_level, level+1, key<<2) #south-west
                if quadrantIsLeaf:
                    count+=1
                data.extend(branchData)
            if not box1.intersection(poly).is_empty:
                branchData, quadrantIsLeaf = QuadTreePAIRS(poly, max_level, level+1, (key<<2)+1) #south-east 
                if quadrantIsLeaf:
                    count+=1
                data.extend(branchData)
            if not box2.intersection(poly).is_empty:
                branchData, quadrantIsLeaf = QuadTreePAIRS(poly, max_level, level+1, (key<<2)+2) #north-west 
                if quadrantIsLeaf:
                    count+=1
                data.extend(branchData)
            if not box3.intersection(poly).is_empty:
                branchData, quadrantIsLeaf = QuadTreePAIRS(poly, max_level, level+1, (key<<2)+3) ##north-east 
                if quadrantIsLeaf:
                    count+=1
                data.extend(branchData)
            nodeIsLeaf = count == 4 
            #nodeIsLeaf = count in [4]   #[4] typical, [3,4] if fewer keys but slightly larger squares are preferred.
            if nodeIsLeaf:     #overwrite the higher-level data if enough quadrants (3 or 4) are present          
                data = [(level, key)] 
            return data, nodeIsLeaf 

        
def getQuadTreeSquares(polyQuadTree):
    """
    Function to conveniently get poly_squares for plotting with matplotlib, shapely, and descartes (PolygonPatch).
    :param polyQuadTree: Polygon information in quadtree list form
    :param cell_level:   cell level
    :returns:            Cell-level leaf nodes
    """
    poly_squares=[]
    for square in polyQuadTree:
        south, west = getLatLon(square[1], square[0])
        res = getResolution(square[0])
        north = south + res
        east = west + res
        poly_squares.append(box(west, south, east, north))
        
    #Plotting recipe:
    """
    import matplotlib
    matplotlib.use('Agg')
    %matplotlib inline
    import matplotlib.pyplot as plt
    from shapely.geometry import box
    from descartes import PolygonPatch
    
    fig = plt.figure(num=1,figsize=(10,10))
    ax = fig.gca() 
    #ax.add_patch(PolygonPatch(my_polygon, fc='#6699cc', alpha=0.5, zorder=2)) #ec='#6699cc', 
    for poly_square in poly_squares:
        ax.add_patch(PolygonPatch(poly_square, fc='#6699cc', alpha=0.5, zorder=2)) #ec='#6699cc', 
    ax.axis('scaled')
    plt.show()
    """
    return poly_squares

    
def QuadTreeCellsPAIRS(polyQuadTree, cell_level):
    """
    Replacing internal nodes with a block of cell nodes
    This version of QuadTreePAIRS returns only cell-level leaf nodes.
    :param polyQuadTree: Polygon information in quadtree list form
    :param cell_level:   cell level
    :returns:            Cell-level leaf nodes
    """
    
    cKeys = []
    for node in polyQuadTree:
        if node[0] == cell_level:
            cKeys.append(node[1])
        else:
            #replace the square key with the cell key children
            index = numpy.arange(2**(2*(cell_level - node[0])))
            cKeys.extend(node[1] * 4 ** (cell_level - node[0]) + index)
    """
    # ALTERNATIVE CODE REDUCING THE NUMBER OF LOOP ITERATIONS BUT NOT CURRENTLY FASTER 
    pqt = numpy.array(polyQuadTree)
    
    # First extract values that are already on the cell level
    mask = pqt[:,0]==cell_level
    cKeys = list(pqt[mask,1])
    pqt = pqt[~mask,:]
    
    # Now deal with the other levels
    for level in sorted(set(pqt[:,0])):
        #replace the square keys with the cell key children
        mask = pqt[:,0]==level
        pqt_l = pqt[mask,1]
        index = numpy.arange(2**(2*(cell_level - level)))
        val = numpy.tile(pqt_l, (len(index), 1))
        index = numpy.tile(numpy.array([index]).transpose(), (1, len(pqt_l)))
        cKeys.extend((val * 4 ** (cell_level - level) + index).ravel())
    """
    return cKeys