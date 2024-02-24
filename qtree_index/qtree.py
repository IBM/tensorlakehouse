"""Traverse the nested grid using a quadtree.

Classes:

    Node      Tree Datastructure.
    QTree     Quadtree operations based on the Node Tree class.

Functions:

    Quaternary hash encodes quadtree key and level together in a compact string.
        - encode_qt_dict:         Encode an entire qt dictionary into quaternary hashes.
        - to_qt_dict:             Convert quaternary string back to a qt dictionary.
"""

import numpy
import shapely
import pandas
import geopandas

from . import mortoncurve


class Node():
    """Implementation of a Quadtree datastructure (Node with up to four children).

    Attributes:

        level                   Resolution level.
        key                     Z-order key.
        is_leaf                 Flag indicating Node does not have any children (is_leaf=True).
        is_early_stopping_leaf  Flag indicating we don't know if Node has any children 
                                due to reaching maximum level of search.
        is_qualified_leaf       Flag indicating Node does not have any recorded children.
        children                Dictionary of children nodes
        count_children          Total number of children
        count_children_leafs    Total number of children that are 
        res                     Size of Node 
        south                   Coordinate south
        west                    Coordinate west
        north                   Coordinate north
        east                    Coordinate east
        
    Methods:

        encode                  Quaternary representation of node.
        add_child               Add a child node.
    """
    def __init__(self, morton, level=0, key=0):
        self.level = level
        self.key = key

        # Nodes encompass half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        #self.epsilon = morton.grid.epsilon

        # is_leaf is true when node does not have any children (genuine leaf node).
        # Polygon completely contains the node. The set of "is_leaf" can be used to find
        # the qtree keys that don't have to be divided any further
        self.is_leaf = False

        # is_early_stopping_leaf is true when node does not have any children due to early stopping.
        # the set of "is_early_stopping_leaf" is usually located at the boundary of the polygon.
        # Intersection operations need to consult the original geometries
        self.is_early_stopping_leaf = False

        # is_qualified_leaf is true when is_leaf or is_early_stopping_leaf are true
        # the set of "is_qualified_leafs" can be used to define an entire qtree that
        # fully contains the polygon
        self.is_qualified_leaf = False

        # Using a dictionary of children instead of a list so we can distinguish between
        # 0:sw, 1:se, 2:nw, 3:ne
        self.children = {}
        self.count_children = 0
        self.count_children_leafs = 0

        # Bounding box coordinates.
        # Nodes encompass half-open intervals [south,north) and [west,east), so that points
        # (and some lines) are assigned to exactly one box on each resolution level.
        self.res_x, self.res_y = morton.resolution(self.level)
        self.west, self.south = morton.coordinates(key, level)
        self.east = self.west + self.res_x - morton.grid.epsilon
        self.north = self.south + self.res_y - morton.grid.epsilon

    def __repr__(self):
        """Adding the quaternay hash string to the representation."""
        cls = type(self)
        return f"<{cls.__module__}.{cls.__name__} {self.encode()} at {hex(id(self))}>"

    def encode(self):
        """Quaternary representation of node.

        z-order key and level encoded in base-4 string with prepended zeros dependent on level.
        """
        if self.level==0:
            return "0q" # 0q in analogy to 0b for binary and 0x for hex

        base_4 = numpy.base_repr(self.key, 4)
        return f"0q{base_4:0>{self.level}}"

    def add_child(self, quadrant, obj):
        """Add a child node.

        Valid quadrants according to z-order are:
        "0": south-west
        "1": south-east
        "2": north-west
        "3": south-east
        Note these are quaternary (base-4) strings.
        """
        assert(quadrant in ["0", "1", "2", "3"])
        self.children[quadrant] = obj


class QTree():
    """Quadtree operations based on the Node Tree class.
    
    QTree and Node attributes contain information about intersections and containments.

        self.head                   smallest node that contains polygon
        node.is_leaf                polygon contains node
        node.is_early_stopping_leaf polygon intersects node AND polygon does not contain node
        node.is_qualified_leaf      polygon intersects node

    Methods:
        quadtree_dfs                Walking the z-order Quadtree recursively (depth first search).
        gridded                     Get the grid keys as a numpy array.
        gridded_to_geodataframe     Get the polyKeys as a geopandas GeoDataFrame.
        seeker                      Descend the QTree from root to a specific node.
        walker                      Walk the QTree breadth-first.
        to_geodataframe             Convert the quadtree into a geopandas GeoDataFrame.
    """

    def __init__(self, poly, max_level, max_depth=None, head_only=False, simplify_n=None, grid=None):
        """
        :param poly:          Polygon on a coordinate reference system as defined in morton.grid.crs
        :param max_level:     Maximum level when traversing the tree
        :param max_depth:     Maximum depth of tree (measured from level before first split (head))
        :param head_only:     If true, don't descend into the tree. 
                              Just return the head node (highest level containing full polygon)
        :param simplify_n:    Simplify the quadtree by merging n=4 or n=3+ quadrants
        :param grid:          Nestedgrid
        """
        self.poly = poly
        self.max_level = max_level
        self.max_depth = max_depth
        self.head_only = head_only
        self.simplify_n = simplify_n

        # Nested grid to overwrite the default PAIRS grid
        self.grid = grid

        # Definition of the morton curve on top of the nested grid
        self.morton = mortoncurve.Morton(self.grid)

        # Nodes encompass half-open intervals [south,north) and [west,east), so that points
        # (and some lines) are assigned to exactly one box on each resolution level.
        self.epsilon = self.morton.grid.epsilon

        # Reserved for gridding the qtree at fixed level
        self._keys = None

        # Root Node
        self.root = Node(self.morton) # Level 0,  Key 0

        # Remember the node at which the entire polygon fits into a qtree cell
        # Allows snapping to head of qtree (just before the first split), bypassing lower levels.
        # Allows limiting search to level = max_depth + head.level
        # May be consulted for smart vectorstore partitioning decisions
        self.head = None
        self._head_node() # Find the head

    def _quadtree_dfs(self, node):
        """Walking the z-order Quadtree recursively (depth first search).

        Retrieving leaf nodes that lie within or intersect a Polygon.
        Stop descending whenever the node lies completely within the polygon
        :param node:          Tree object
        """
        if node.level>=self.max_level:
            # Call this a leaf node even though it may only be due to early stopping
            node.is_leaf = False
            node.is_early_stopping_leaf = True
            node.is_qualified_leaf = True
            return

        if ((self.head is not None) and (self.max_depth is not None) and
            (node.level > self.head.level + self.max_depth)
        ):
            node.is_leaf = False
            node.is_early_stopping_leaf = True
            node.is_qualified_leaf = True
            return

        # Check if the polygon fully contains the node -> node is leaf
        if self.poly.contains(shapely.box(node.west, node.south, node.east, node.north)):
            # This is a genuine leaf node
            node.is_leaf = True
            node.is_early_stopping_leaf = False
            node.is_qualified_leaf = True
            return

        # This node may be an internal node
        center_y = node.south + node.res_y/2
        center_x = node.west + node.res_x/2
        quadrants = shapely.box(
            numpy.array([node.west, center_x, node.west, center_x]),
            numpy.array([node.south, node.south, center_y, center_y]),
            numpy.array([center_x-self.epsilon, node.east, center_x-self.epsilon, node.east]),
            numpy.array([center_y-self.epsilon, center_y-self.epsilon, node.north, node.north]),
        )
        intersects = shapely.intersects(self.poly, quadrants)

        if (self.head is None) and sum(intersects)>1:
            # Found the head of the qtree
            self.head = node

        if self.head_only and sum(intersects)>1:
            # The head defines the pixel containing the entire polygon
            node.is_leaf = False
            node.is_early_stopping_leaf = False
            node.is_qualified_leaf = False
            return

        count_children_leafs = 0
        if intersects[0]:
            node_sw = Node(self.morton, node.level+1, (node.key<<2)) #south-west
            node.add_child('0', node_sw)
            self._quadtree_dfs(node_sw)
            if node_sw.is_qualified_leaf:
                count_children_leafs+=1
        if intersects[1]:
            node_se = Node(self.morton, node.level+1, (node.key<<2)+1) #south-east
            node.add_child('1', node_se)
            self._quadtree_dfs(node_se)
            if node_se.is_qualified_leaf:
                count_children_leafs+=1
        if intersects[2]:
            node_nw = Node(self.morton, node.level+1, (node.key<<2)+2) #north-west
            node.add_child('2', node_nw)
            self._quadtree_dfs(node_nw)
            if node_nw.is_qualified_leaf:
                count_children_leafs+=1
        if intersects[3]:
            node_ne = Node(self.morton, node.level+1, (node.key<<2)+3) #north-east
            node.add_child('3', node_ne)
            self._quadtree_dfs(node_ne)
            if node_ne.is_qualified_leaf:
                count_children_leafs+=1

        node.count_children = len(node.children)
        node.count_children_leafs = count_children_leafs

        if self.simplify_n is not None:
            # Combine [4] typical, [3,4] if fewer keys but slightly larger squares are preferred.
            if node.count_children_leafs in list(range(self.simplify_n, 4+1)):  # E.g. [4] #[3,4]
                #Prune the branch
                node.children = {}
                node.is_early_stopping_leaf = True

        node.is_leaf = False
        node.is_qualified_leaf = node.is_early_stopping_leaf #is_leaf or is_early_stopping_leaf

    def _head_node(self):
        """Find the head node."""
        self.head_only=True
        self._quadtree_dfs(self.root)
        self.head = self.root
        while len(self.head.children.keys())==1:
            self.head = list(self.head.children.values())[0]

    def quadtree_dfs(self):
        """Walking the z-order Quadtree recursively (depth first search).

        Retrieving leaf nodes that lie within a Polygon.
        Stop descending whenever the node lies completely within the polygon.
        """
        self.head_only=False
        self._quadtree_dfs(self.head)

    def _gridded_dfs(self, node, level):
        """Recursive depth-first search, replacing internal leaf nodes with a block of keys.

        Returns all leaf nodes at the specified level.
        :param node:      tree
        :param level:     level
        :returns:         leaf nodes at this level
        """
        if node.is_qualified_leaf:
            if node.level==level:
                self._keys.append(node.key)
            elif node.level<level:
                #replace the square key with the children keys on level "level"
                index = numpy.arange(4**(level - node.level))
                self._keys.extend(node.key * 4**(level - node.level) + index)
            else:
                pass

        for child in node.children.values():
            self._gridded_dfs(child, level)

    def gridded(self, level=None):
        """Get the grid keys in one shot.
        
        Usually used with level==self.max_level. 
        When level<max_level, only leaf nodes inside the polygon are returned
        When level>max_level, cells may be returned that are outside the original polygon
        """
        if level is None:
            level = self.max_level

        if self.head_only:
            # Full qtree has not been calculated yet
            self.quadtree_dfs()

        # Collect the grid keys in a flat list
        self._keys = []

        # Recursively, get all the keys on the same resolution level
        self._gridded_dfs(self.root, level)
        return numpy.array(self._keys)

    def gridded_to_geodataframe(
        self,
        key_col='key',
        level_col='level',
        hash_col='q_key',
        geom_col='geometry',
        level=None,
        crop_valid_range=True,
    ):
        """Get the polyKeys in one shot.
        
        Usually used with level==self.max_level. 
        When level<max_level, only leaf nodes inside the polygon are returned
        When level>max_level, cells may be returned that are outside the original polygon
        """
        if level is None:
            level = self.max_level

        keys = self.gridded(level=level)
        gdf_grid = {}
        if key_col is not None:
            gdf_grid[key_col] = keys
        if level_col is not None:
            gdf_grid[level_col] = [level]*len(keys)
        if hash_col is not None:
            gdf_grid[hash_col] = self.morton.encode(keys, [level]*len(keys))
        if geom_col is not None:
            gdf_grid[geom_col] = self.morton.key_to_box(keys, level)

        gdf_grid = pandas.DataFrame(gdf_grid)

        if hash_col is not None:
            gdf_grid = gdf_grid.sort_values(by=hash_col).reset_index(drop=True)
        if geom_col is not None:
            gdf_grid = geopandas.GeoDataFrame(gdf_grid, geometry=geom_col).set_crs(self.morton.grid.crs)
            if crop_valid_range:
                gdf_grid[geom_col] = self.morton.valid_range.intersection(gdf_grid[geom_col])

        return gdf_grid

    def seeker(self, q_key):
        """Descend the QTree (starting at root) to the node defined by quaternary hash."""
        node = self.root
        try:
            for letter in q_key[2:]:
                # descending sucessivele levels
                node = node.children[letter]
            return node
        except KeyError:
            return None

    def walker(self, max_level=None, max_depth=None):
        """Walk the QTree breadth-first.
        
        Returning all nodes in a dictionary indexed by level.
        """
        def tree_bfs(root):
            level = root.level
            current = [root]
            while current:
                yield current # Yield the current level nodes

                level+=1
                if (max_level is not None) and level>max_level:
                    return
                if (max_depth is not None) and (level - self.head.level)>max_depth:
                    return

                # Next time we will yield all the next level nodes
                current = [c for n in current for c in n.children.values()]
            # No more levels to iterate over. Generator will be considered empty

        qt_dict = {i + self.root.level: lst for i, lst in enumerate(tree_bfs(self.root))}

        return qt_dict

    def to_geodataframe(
        self, key_col='key', level_col='level', hash_col='q_key', geom_col='geometry',
        max_level=None, max_depth=None, filter_attr=None,
    ) -> geopandas.GeoDataFrame:
        """Convert the quadtree into a geopandas GeoDataFrame.
        
        Option to get a pandas Dataframe when geom_col==None
        """
        qt_dict = self.walker(max_level=max_level, max_depth=max_depth)
        gdf_qt = {}
        if key_col is not None:
            gdf_qt[key_col] = []
        if level_col is not None:
            gdf_qt[level_col] = []
        if hash_col is not None:
            gdf_qt[hash_col] = []
        if geom_col is not None:
            gdf_qt[geom_col] = []
        if filter_attr is not None:
            if not filter_attr in ['is_qualified_leaf', 'is_leaf', 'is_early_stopping_leaf']:
                raise ValueError('filter_attr not understood.')

        for level in sorted(qt_dict.keys()):
            if len(qt_dict[level])>0:
                if key_col is not None:
                    if filter_attr is None:
                        keys = [n.key for n in qt_dict[level]]
                    else:
                        keys = [n.key for n in qt_dict[level] if getattr(n, filter_attr)]
                    gdf_qt[key_col].extend(keys)
                if level_col is not None:
                    if filter_attr is None:
                        levels = [level]*len(qt_dict[level])
                    else:
                        levels = [n.level for n in qt_dict[level] if getattr(n, filter_attr)]
                    gdf_qt[level_col].extend(levels)
                if hash_col is not None:
                    if filter_attr is None:
                        q_key = [n.encode() for n in qt_dict[level]]
                    else:
                        q_key = [n.encode() for n in qt_dict[level] if getattr(n, filter_attr)]
                    gdf_qt[hash_col].extend(q_key)
                if geom_col is not None:
                    if filter_attr is None:
                        boxes = [
                            shapely.box(n.west, n.south, n.east, n.north) for n in qt_dict[level]
                        ]
                    else:
                        boxes = [
                            shapely.box(n.west, n.south, n.east, n.north) for n in qt_dict[level]
                            if getattr(n, filter_attr)
                        ]
                    gdf_qt[geom_col].extend(boxes)
        gdf_qt = pandas.DataFrame(gdf_qt)
        if hash_col is not None:
            gdf_qt = gdf_qt.sort_values(by=hash_col).reset_index(drop=True)
        if geom_col is not None:
            gdf_qt = geopandas.GeoDataFrame(gdf_qt, geometry=geom_col).set_crs(self.morton.grid.crs)
        gdf_qt[geom_col] = self.morton.valid_range.intersection(gdf_qt[geom_col])

        return gdf_qt


def encode_qt_dict(qt_dict):
    """Encode an entire qt dictionary into quaternary hashes."""
    q_key = []
    for level in sorted(qt_dict.keys()):
        if len(qt_dict[level])>0:
            q_key.extend(self.morton.encode(qt_dict[level], level))
    return q_key

def to_qt_dict(q_key: str):
    """Convert the quaternary string representation back to a qt_dict dictionary (decode)."""
    key, level = self.morton.decode(q_key)
    df_tmp = pandas.DataFrame({'level': level, 'key': key})
    return dict(df_tmp.groupby('level')['key'].apply(numpy.array))

def _filter_qualified_leafs(qt_dict):
    qt_dict_filtered = {}
    for level in sorted(qt_dict.keys()):
        lst = [n for n in qt_dict[level] if n.is_qualified_leaf]
        if len(lst)>0:
            qt_dict_filtered[level] = lst
    return qt_dict_filtered

def _filter_leafs(qt_dict):
    qt_dict_filtered = {}
    for level in sorted(qt_dict.keys()):
        lst = [n for n in qt_dict[level] if n.is_leaf]
        if len(lst)>0:
            qt_dict_filtered[level] = lst
    return qt_dict_filtered

def _filter_early_stopping_leafs(qt_dict):
    qt_dict_filtered = {}
    for level in sorted(qt_dict.keys()):
        lst = [n for n in qt_dict[level] if n.is_early_stopping_leaf]
        if len(lst)>0:
            qt_dict_filtered[level] = lst
    return qt_dict_filtered

def _select_attributes(qt_dict, attributes):
    qt_dict_filtered = {}
    for level in sorted(qt_dict.keys()):
        lst = [tuple([getattr(n, attr) for attr in attributes]) for n in qt_dict[level]]
        if len(lst)>0:
            qt_dict_filtered[level] = lst
    return qt_dict_filtered

def _flatten_dict(qt_dict):
    qt_lst = [node for lst in qt_dict.values() for node in lst]
    return qt_lst










# # def qt_to_geodataframe(qt_dict, key_col='key', level_col='level', hash_col='q_key', geom_col='geometry') -> geopandas.GeoDataFrame:
# #     """
# #     Convert the quadtree into geopandas GeoDataFrame (or pandas Dataframe when geometry==False)
# #     """
# #     a = numpy.array([(
# #         n.key, n.level, n.encode(), n.west, n.south, n.east, n.north
# #     ) for v in qt_dict.values() for n in v])

# #     gdf_qt = pandas.DataFrame({
# #         key_col: a[:, 0],
# #         level_col: a[:, 1],
# #         hash_col: a[:, 2],
# #         geom_col: shapely.box(a[:, 3].astype(float), a[:, 4].astype(float), a[:, 5].astype(float), a[:, 6].astype(float))
# #     })
# #     gdf_qt = gdf_qt.sort_values(by=hash_col).reset_index(drop=True)
# #     gdf_qt = geopandas.GeoDataFrame(gdf_qt, geometry=geom_col).set_crs(self.morton.grid.crs)
# #     gdf_qt['geom_col'] = self.morton.valid_range.intersection(gdf_qt['geom_col'])
# #     return gdf_qt

# def qt_to_geodataframe(qt_dict, key_col='key', level_col='level', hash_col='q_key', geom_col='geometry') -> geopandas.GeoDataFrame:
#     """
#     Convert the quadtree into geopandas GeoDataFrame (or pandas Dataframe when geometry==False)
#     """
#     gdf_qt = {}
#     if key_col is not None:
#         gdf_qt[key_col] = []
#     if level_col is not None:
#         gdf_qt[level_col] = []
#     if hash_col is not None:
#         gdf_qt[hash_col] = []
#     if geom_col is not None:
#         gdf_qt[geom_col] = []

#     for l in sorted(qt_dict.keys()):
#         if len(qt_dict[l])>0:
#             if key_col is not None:
#                 keys = [n.key for n in qt_dict[l]]
#                 gdf_qt[key_col].extend(keys)
#             if level_col is not None:
#                 levels = [l]*len(qt_dict[l])
#                 gdf_qt[level_col].extend(levels)
#             if hash_col is not None:
#                 q_key = [n.encode() for n in qt_dict[l]]
#                 gdf_qt[hash_col].extend(q_key)
#             if geom_col is not None:
#                 gdf_qt[geom_col].extend([shapely.box(n.west, n.south, n.east, n.north) for n in qt_dict[l]])
#                 #gdf_qt[geom_col].extend(numpy.array(self.morton.key_to_box(qt_dict[l], l)))
#     gdf_qt = pandas.DataFrame(gdf_qt)
#     if hash_col is not None:
#         gdf_qt = gdf_qt.sort_values(by=hash_col).reset_index(drop=True)
#     if geom_col is not None:
#         gdf_qt = geopandas.GeoDataFrame(gdf_qt, geometry=geom_col).set_crs(self.morton.grid.crs)
#     gdf_qt[geom_col] = self.morton.valid_range.intersection(gdf_qt[geom_col])

#     return gdf_qt












# # def walker(self, attribute=None, max_level=None, max_depth=None):
# #     """
# #     Walk the QTree breadth-first, returning all nodes in a dictionary indexed by level
# #     """
# #     def tree_bfs(root):
# #         level = root.level
# #         current = [root]
# #         while current:
# #             yield current # Yield the current level nodes

# #             level+=1
# #             if (max_level is not None) and level>max_level:
# #                 return
# #             if (max_depth is not None) and (level - self.head.level)>max_depth:
# #                 return

# #             # Next time we will yield all the next level nodes
# #             current = [c for n in current for c in n.children.values()]
# #         # No more levels to iterate over. Generator will be considered empty

# #     if attribute is None:
# #         qt_dict = {i + self.root.level: lst for i, lst in enumerate(tree_bfs(self.root))}
# #     else:
# #         # Return only the attribute requested
# #         qt_dict = {i + self.root.level: [getattr(n, attribute) for n in lst] for i, lst in enumerate(tree_bfs(self.root))}

# #     return qt_dict


# # Still needs to be implemented as methods:

# def _step_quadtreePAIRS_bfs(poly: shapely.Polygon, parent_keys: numpy.array, parent_level: int) -> Tuple[numpy.array, numpy.array]:
#     """
#     Descending one level into the PAIRS z-order Quadtree (breadth first) to retrieve nodes that lie within a Polygon
#     :param poly:         Polygon defined by lat/lon values in degrees
#     :param parent_keys:  Numpy Array of parent keys
#     :param parent_level: Parent level
#     """
#     level = parent_level + 1

#     # All children keys belonging to level "level"
#     keys = self.morton.children_keys(parent_keys, 1).flatten()

#     # Corresponding bounding boxes
#     boxes = self.morton.key_to_box(keys, level)

#     # Check for full containment. These keys are recorded as leaf keys and don't have to be devided further
#     full_containments = shapely.contains(poly, boxes)
#     leaf_keys = keys[full_containments]#.flatten()
#     # The others need to be investigated further
#     keys = keys[~full_containments]#.flatten()
#     boxes = boxes[~full_containments]#.flatten()

#     # Check for intersection with polygon
#     intersects = shapely.intersects(poly, boxes)
#     keys = keys[intersects]

#     return leaf_keys, keys

# def _qt_dict_to_index(qt_dict: Dict, max_level: int) -> pandas.DataFrame:
#     """
#     Composing "pyramid" index columns from the quadtree information.
#     These can be used on levels up to the leaf nodes for partitioning/filtering
#     :param qt_dict: quadtree dictionary
#     :param max_level: Maximum level to descend into the tree. Nodes at max_level will be returned as leaf nodes
#     """
#     df = pandas.DataFrame()
#     for leaf in range(1, max_level+1):
#         df1 = pandas.DataFrame(qt_dict[leaf]).rename(columns={0: leaf})
#         for l in reversed(range(1, leaf)):
#             df1[l] = parent_key(numpy.array(df1[l+1]), 1)
#         if len(df)>0:
#             df = pandas.merge(df, df1, on=list(numpy.arange(0 ,leaf-1)+1), how='outer')
#         else:
#             df = df1
#     df = df.sort_values(by=list(range(1, max_level+1))).reset_index(drop=True)
#     df = df[list(range(1, max_level+1))]
#     return df

# def _quadtreePAIRS_bfs(poly: shapely.Polygon, max_level: int) -> Dict:
#     """
#     Walking the PAIRS z-order Quadtree breadth first to retrieve nodes that lie within a Polygon
#     This funstion returns a quadtree that may still need pruning (4 children -> parent)
#     :param poly:         Polygon defined by lat/lon values in degrees
#     :param max_level:    Maximum level to descend into the tree. Nodes at max_level will be returned as leaf nodes
#     :returns qt_dict:    quadtree dictionary indexed by level
#     """
#     qt_dict = {}
#     for parent_level in range(0, max_level):
#         level = parent_level + 1
#         if parent_level==0:
#             parent_keys = numpy.array([0])
#         else:
#             parent_keys = keys
#         leaf_keys, keys = _step_quadtreePAIRS_bfs(poly, parent_keys, parent_level)

#         # Remember the leaf keys at parent level
#         qt_dict[level] = leaf_keys

#     # Remember the keys at max_level even though they may not be true leafs
#     qt_dict[parent_level+1] = numpy.hstack([qt_dict[parent_level+1], keys])

#     return qt_dict

# def quadTreePAIRS_bfs(poly: shapely.Polygon, max_level: int) -> Dict:
#     """
#     Walking the PAIRS z-order Quadtree breadth first to retrieve nodes that lie within a Polygon
#     :param poly:         Polygon defined by lat/lon values in degrees
#     :param max_level:    Maximum level to descend into the tree. Nodes at max_level will be returned as leaf nodes
#     :returns qt_dict:    quadtree dictionary indexed by level
#     :returns df_qt:      quadtree (only the leaf nodes) as a pandas DataFrame
#     :returns df_index:   quadtree index as a pandas DataFrame
#     """
#     # Get the quadtree (that may still need pruning)
#     _qt_dict = _quadtreePAIRS_bfs(poly, max_level)

#     # Convert into tabular form
#     df_index = _qt_dict_to_index(_qt_dict, max_level)

#     # We may still have to combine quadrants:
#     # Whenever four quadrants with the same parent key are intersecting the polygon, we need to replace them by the parent
#     for l in reversed(range(1, max_level)):
#         # Checking if there are 4 of the same keys at level l present
#         combine = (df_index.groupby(l).transform('count')[l+1]==4)
#         # Checking if expected sucessive values on level l+1 are all present (no duplicates)
#         combine = combine & (df_index.groupby(l)[l+1].transform('nunique')==4)
#         # Checking if those sucessive values are leaf nodes
#         if l+2<=max_level:
#             combine = combine & (df_index.groupby(l)[l+2].transform('nunique')==0)

#         df_index.loc[combine, l+1] = numpy.nan
#         df_index = df_index.drop_duplicates().reset_index(drop=True)
#     df_index = df_index.sort_values(by=list(range(1, max_level+1))).reset_index(drop=True)
#     df_index = df_index[list(range(1, max_level+1))]

#     # Calculate the correct quadtree (with combined nodes) using the index
#     qt_dict = {}
#     df_qt = df_index.copy()
#     for l in reversed(df_qt.columns):
#         # Grab a single level starting with the deepest
#         qt_dict[l] = numpy.array(df_qt[l].dropna()).astype(int)

#         # If we found a key, remove all lower-level keys in this row
#         df_qt.loc[~df_qt[l].isnull(), numpy.arange(1, l)] = numpy.nan

#         # Remove empty arrays
#         if len(qt_dict[l])==0:
#             del qt_dict[l]

#     return qt_dict, df_qt, df_index #, _qt_dict

# def qt_to_boxes(qt_dict):
#     """
#     translate entire quadtree to shapely boxes (e.g. for performing intersections or containment operations)
#     """
#     boxes = []
#     for l in sorted(qt_dict.keys()):
#         boxes.append(numpy.array(self.morton.key_to_box(qt_dict[l], l)))
#     return numpy.hstack(boxes)

# def gridCellsPAIRS_bfs(qt_dict, level):
#     """
#     Replacing internal nodes with a block of leaf nodes.
#     Returns all leaf nodes at the specified level.
#     Usually used with level==max_level of the quadtree.
#     Note that when level>max_level of the qt_dict, cells may be returned that are outside the original polygon
#     :param qt_dict: Quadtree dictionary
#     :param level:   PAIRS level
#     :returns:       all nodes at this level
#     """
#     keys = []
#     for l in qt_dict.keys():
#         if len(qt_dict[l])>0:
#             if l==level:
#                 # Take the keys as they are
#                 keys.append(qt_dict[l])
#             elif l<level:
#                 # Replace the square key withs the children keys on level "level"
#                 index = numpy.arange(4**(level - l))
#                 qt_level = qt_dict[l] * 4**(level - l)
#                 keys.extend(numpy.add.outer(qt_level, index))
#             elif l>level:
#                 # Combine higher-level keys
#                 p = parent_key(numpy.array([qt_dict[l]]), l-level)
#                 p = numpy.array([numpy.unique(p)])
#                 keys.extend(p)
#     return numpy.sort(numpy.unique(numpy.concatenate(keys)))

# def gridCells_bfs(poly, level):
#     """
#     Get the keys at fixed level in one shot.
#     """
#     # Get the quadtree representation
#     qt_dict, _, _ = quadTreePAIRS_bfs(poly, max_level=level)
#     # Get all the keys on the same resolution level
#     keys = gridCellsPAIRS_bfs(qt_dict, level)
#     return keys

# def dfs_to_bfs(qt_lst: list[tuple[int, int]]) -> dict:
#     """
#     Translate betweeen the two quadtree representations (depth first search -> breadth first search)
#     """
#     qt_dict = dict(pandas.DataFrame(qt_lst).groupby(0)[1].apply(numpy.sort))
#     return qt_dict

# def bfs_to_dfs(qt_dict: dict) -> list[tuple[int, int]]:
#     """
#     Translate betweeen the two quadtree representations (breadth first search -> depth first search)
#     """
#     qt_lst = [(l, key) for l in qt_dict.keys() for key in qt_dict[l]]
#     return qt_lst
