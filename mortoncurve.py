from typing import Tuple, Iterable
import os
import math
import numpy
import shapely

import nestedgrid

# The PyGEOS package was merged with Shapely in 2021 and will be released as part of Shapely 2.0
# No further development will take place for the PyGEOS package.
# Here we use a beta-version of Shapely 2.0 and explicitly disable PyGEOS
os.environ['USE_PYGEOS'] = '0'

class Morton():
    """Navigating the GeoDN grid using the Morton (z-order) index.

    Attributes:

        grid                  Nested grid the Morton curve is built on.
        valid_range           valid range geometry
        epsilon

    Methods:

        morton                Morton number (z-ordered index of x-y plane).
        morton_inv            Inverse of morton.
        resolution            Pixel resolution based on level.
        resolution_x          Resolution in x dimension at specified level.
        resolution_y          Resolution in y dimension at specified level.
        x_coord_to_idx        Translate x-coordinate (e.g., longitude) to x-index at level.
        y_coord_to_idx        Translate y-coordinate (e.g., latitude) to y-index at level.
        x_idx_to_coord        Translate x-index at level to x-coordinate (e.g., longitude).
        y_idx_to_coord        Translate y-index at level to y-coordinate (e.g., latitude).
        x_idx_to_center_coord Translate x-index at level to center x-coordinate.
        y_idx_to_center_coord Translate y-index at level to center y-coordinate.
        coordinates_to_xy_cell_indices: Calculate the coordinates expected from a full-cell query.
        get_key               Key based on coordinates and level (encode).
        coordinates           Bottom-left corner of pixel associated with key/level (decode).
        center_coordinates    Translate key/level to center coordinates of associated pixel.
        parent_key            Parent key 'levels_up' levels above the current level.
        children_keys         Get children keys 'levels_down' levels below current level.
        key_to_box            Translate keys to shapely boxes.

    Quaternary hash encodes quadtree key and level together in a compact string. 
    Can be used as an index in tables.

        encode                Encode key, level in a quaternary hash.
        decode                Convert the quaternary string representation back to key/level.
        base4_to_xy_indices   Convert the quaternary string to y and x coordinate indices.

    Navigating the GeoDN grid using base4 strings.

        base4_to_coords        Translate base4 quaternary number to xy coordinate pair.
        base4_to_center_coords Translate base4 quaternary number to xy center coordinate pair.
        base4_to_box           Translate keys to shapely boxes.
    """
    # Default values for class attributes
    GRID = nestedgrid.PAIRS()

    def __init__(self, grid=None):
        # Basic grid definition containing level0 extent, valid range, etc.
        self.grid = self.GRID if grid is None else grid

        # Valid range geometry
        self.valid_range = shapely.geometry.box(*self.grid.valid_bounds)

        # Nodes encompass half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        # We thus subtract epsilon to calculate node attributes north and east
        self.epsilon = self.grid.epsilon

        # Some convenience functions vectorized
        self.base_repr = numpy.vectorize(numpy.base_repr)
        self.len_np = numpy.vectorize(len)
        self.int_np = numpy.vectorize(int)

        # Cashing valid resolutions as numpy arrays.
        self.RESOLUTION_X, self.RESOLUTION_Y = self._resolution_np(numpy.arange(self.grid.max_levels + 1))

    def morton(self, x_idx: numpy.array, y_idx: numpy.array) -> numpy.array:
        """Morton number (z-ordered index of x-y plane).
        
        :param x_idx:  x-index
        :param y_idx:  y-index
        :returns:      Morton number
        """
        # Bit-wise operations using
        # 0x0000FFFF0000FFFF: 0b0000000000000000111111111111111100000000000000001111111111111111
        # 0x00FF00FF00FF00FF: 0b0000000011111111000000001111111100000000111111110000000011111111
        # 0x0F0F0F0F0F0F0F0F: 0b0000111100001111000011110000111100001111000011110000111100001111
        # 0x3333333333333333: 0b0011001100110011001100110011001100110011001100110011001100110011
        # 0x5555555555555555: 0b0101010101010101010101010101010101010101010101010101010101010101
    
        x_idx = x_idx.astype(numpy.int64)
        x_idx = (x_idx | (x_idx << 16)) & 0x0000FFFF0000FFFF
        x_idx = (x_idx | (x_idx << 8))  & 0x00FF00FF00FF00FF
        x_idx = (x_idx | (x_idx << 4))  & 0x0F0F0F0F0F0F0F0F
        x_idx = (x_idx | (x_idx << 2))  & 0x3333333333333333
        x_idx = (x_idx | (x_idx << 1))  & 0x5555555555555555
    
        y_idx = y_idx.astype(numpy.int64)
        y_idx = (y_idx | (y_idx << 16)) & 0x0000FFFF0000FFFF
        y_idx = (y_idx | (y_idx << 8))  & 0x00FF00FF00FF00FF
        y_idx = (y_idx | (y_idx << 4))  & 0x0F0F0F0F0F0F0F0F
        y_idx = (y_idx | (y_idx << 2))  & 0x3333333333333333
        y_idx = (y_idx | (y_idx << 1))  & 0x5555555555555555
        y_idx = y_idx << 1
    
        return x_idx | y_idx  #return the interleaved values (morton number)
    
    def morton_inv(self, z_idx: numpy.array) -> Tuple[numpy.array, numpy.array]:
        """Inverse of morton.
        
        Note: "x_idx, y_idx==morton_inv(morton(x_idx, y_idx))" yields "True".
        :z_idx long:  z-ordered key
        :returns:     y_idx, x_idx integer indices of PAIRS raster array
                      (lower left corner is root, i.e. [0,0,])
        :rtype:       (int, int)
        """
        x_idx = z_idx                 & 0x5555555555555555
        x_idx = (x_idx | x_idx >> 1)  & 0x3333333333333333
        x_idx = (x_idx | x_idx >> 2)  & 0x0f0f0f0f0f0f0f0f
        x_idx = (x_idx | x_idx >> 4)  & 0x00ff00ff00ff00ff
        x_idx = (x_idx | x_idx >> 8)  & 0x0000ffff0000ffff
        x_idx = (x_idx | x_idx >> 16) & 0x00000000ffffffff
    
        y_idx = (z_idx >> 1)          & 0x5555555555555555
        y_idx = (y_idx | y_idx >> 1)  & 0x3333333333333333
        y_idx = (y_idx | y_idx >> 2)  & 0x0f0f0f0f0f0f0f0f
        y_idx = (y_idx | y_idx >> 4)  & 0x00ff00ff00ff00ff
        y_idx = (y_idx | y_idx >> 8)  & 0x0000ffff0000ffff
        y_idx = (y_idx | y_idx >> 16) & 0x00000000ffffffff
    
        return x_idx, y_idx
    
    def _resolution(self, level: int) -> Tuple[float, float]:
        """Pixel resolution based on level.
        
        :param level:   resolution level
        :returns:       resolution in degrees
        """
        factor = 2**(-float(level))
        return self.grid.res0.x * factor, self.grid.res0.y * factor
    
    def _resolution_np(self, level: numpy.array) -> Tuple[numpy.array, numpy.array]:
        """Pixel resolution based on level.
        
        :param level:   resolution level
        :returns:       resolution in degrees
        """
        factor = 2**(-level.astype(float))
        return self.grid.res0.x * factor, self.grid.res0.y * factor
    
    def resolution(self, level: int) -> Tuple:
        """Pixel resolution based on level."""
        try:
            # Get the resolution from a pre-computed dictionary
            return self.RESOLUTION_X[level], self.RESOLUTION_Y[level]
        except KeyError:
            try:
                return self._resolution_np(level)
            except AttributeError:
                return self._resolution(level)
    
    def resolution_x(self, level: int):
        """Resolution in x dimension at specified level."""
        try:
            # Get the resolution from a pre-computed dictionary
            return self.RESOLUTION_X[level]
        except KeyError:
            try:
                return self._resolution_np(level)[0]
            except AttributeError:
                return self._resolution(level)[0]
    
    def resolution_y(self, level: int):
        """Resolution in y dimension at specified level."""
        try:
            # Get the resolution from a pre-computed dictionary
            return self.RESOLUTION_Y[level]
        except KeyError:
            try:
                return self._resolution_np(level)[1]
            except AttributeError:
                return self._resolution(level)[1]
    
    def x_coord_to_idx(self, x_coord: numpy.array, level: int) -> numpy.array:
        """Translate x-coordinate (e.g., longitude) to x-index at level."""
        return (x_coord - self.grid.origin.x) / self.resolution_x(level)
    
    def y_coord_to_idx(self, y_coord: numpy.array, level: int) -> numpy.array:
        """Translate y-coordinate (e.g., latitude) to y-index at level."""
        return (y_coord - self.grid.origin.y) / self.resolution_y(level)
    
    def x_idx_to_coord(self, x_idx: numpy.array, level: int) -> numpy.array:
        """Translate x-index at level to x-coordinate (e.g., longitude)."""
        return x_idx * self.resolution_x(level) + self.grid.origin.x
    
    def y_idx_to_coord(self, y_idx: numpy.array, level: int) -> numpy.array:
        """Translate y-index at level to y-coordinate (e.g., latitude)."""
        return y_idx * self.resolution_y(level) + self.grid.origin.y
    
    def x_idx_to_center_coord(self, x_idx: numpy.array, level: int) -> numpy.array:
        """Translate x-index at level to center x-coordinate."""
        return (x_idx + .5) * self.resolution_x(level) + self.grid.origin.x
    
    def y_idx_to_center_coord(self, y_idx: numpy.array, level: int) -> numpy.array:
        """Translate y-index at level to center y-coordinate."""
        return (y_idx + .5) * self.resolution_y(level) + self.grid.origin.y
    
    def coordinates_to_xy_cell_indices(
        self, bounds: Tuple, level: int, delta_pixel_cell: int
    ) -> Tuple[Iterable, Iterable]:
        """Calculate the coordinates expected from a full-cell query.
    
        :param bounds:           tuple of geometry bounds.
        :param level:            pixel level.
        :param delta_pixel_cell: difference betw. pixel and cell level (5 in PAIRS).
        """
        south = bounds[0]
        west = bounds[1]
        north = bounds[2]
        east = bounds[3]
        cellminx = math.floor(self.x_coord_to_idx(numpy.array([west]), level-delta_pixel_cell))
        cellminy = math.floor(self.y_coord_to_idx(numpy.array([south]), level-delta_pixel_cell))
        cellminx = max(0, cellminx)
        cellminy = max(0, cellminy)
        cellmaxx = math.ceil(self.x_coord_to_idx(numpy.array([east]), level-delta_pixel_cell))
        cellmaxy = math.ceil(self.y_coord_to_idx(numpy.array([north]), level-delta_pixel_cell))
        cellrx = range(cellminx, cellmaxx)
        cellry = range(cellminy, cellmaxy)
        return cellrx, cellry
    
    def get_key(self, y_coord: numpy.array, x_coord: numpy.array, level: int) -> numpy.array:
        """Key based on coordinates and level (encode).
        
        :param y_coord:  y-coordinate (e.g., latitude)
        :param x_coord:  x-coordinate (e.g., longitude)
        :param level:    resolution level
        :returns:        z-order spatial key
        """
        res_x, res_y = self.resolution(level)
        x_idx = (x_coord - self.grid.origin.x) / res_x
        y_idx = (y_coord - self.grid.origin.y) / res_y
        return self.morton(x_idx, y_idx)
    
    def coordinates(self, key: numpy.array, level: int) -> Tuple[numpy.array, numpy.array]:
        """Bottom-left corner of pixel associated with key/level (decode).
        
        :param key:   z-ordered key
        :param level: resolution level
        :returns:     coordinate tuple derived from the key
    
        Returns the bottom-left corner of the pixel.
        Note: beware of rounding errors if coordinates are re-used in get_key(y_coord, x_coord, level)
        """
        x_idx = 0
        y_idx = 0
        for i in range(level):
            mask = 1<<2*i
            x_idx+= (key&mask)>>i
            mask = 1<<2*i+1
            y_idx+= (key&mask)>>(i+1)
        res_x, res_y = self.resolution(level)
        x_coord = x_idx * res_x + self.grid.origin.x
        y_coord = y_idx * res_y + self.grid.origin.y
        return x_coord, y_coord
    
    def center_coordinates(self, key: numpy.array, level: int) -> Tuple[numpy.array, numpy.array]:
        """Translate key/level to center coordinates of associated pixel.
        
        :param key:   z-ordered key
        :param level: resolution level
        :returns:     coordinate tuple derived from the key
    
        Returns the center of the pixel.
        Note: safer than get_coordinates if coordinates are re-used in get_key(y_coord, x_coord, level)
        """
        x_idx = 0
        y_idx = 0
        for i in range(level):
            mask = 1<<2*i
            x_idx+= (key&mask)>>i
            mask = 1<<2*i+1
            y_idx+= (key&mask)>>(i+1)
        res_x, res_y = self.resolution(level)
        x_coord = (x_idx + .5) * res_x + self.grid.origin.x
        y_coord = (y_idx + .5) * res_y + self.grid.origin.y
        return x_coord, y_coord
    
    def parent_key(self, key: numpy.array, levels_up: int) -> numpy.array:
        """Parent key 'levels_up' levels above the current level.
        
        :param Key:       PAIRS key
        :param levels_up: Difference in levels between two resolution layers
        :returns:         The key at levels_up coarser resolution
        """
        return key.astype(numpy.int64)>>(2*levels_up)
    
    def children_keys(self, key: numpy.array, levels_down: int) -> numpy.array:
        """Get children keys 'levels_down' levels below current level.
        
        :param key:      PAIRS key
        :returns:        A 2d array of the 4**levels_down descendent for each key
        """
        key = key.astype(numpy.int64)
        key = key<<(2*levels_down)
        key_prime = numpy.arange(4**levels_down)
        key = numpy.tile(numpy.array([key]).T, (1, key_prime.shape[0]))
        key_prime = numpy.tile(key_prime, (key.shape[0], 1))
        return key + key_prime
    
    def key_to_box(self, key: numpy.array, level: int) -> shapely.Polygon:
        """Translate keys to shapely boxes.
    
        (e.g. for performing intersections or containment operations)
        :key:   numpy array of keys
        :level: level
        """
        # Nodes encompass half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        west, south = self.coordinates(key, level)
        res_x, res_y = self.resolution(level)
        east = west + res_x - self.epsilon
        north = south + res_y - self.epsilon
        return shapely.box(west, south, east, north)
    
    def encode(self, key: numpy.array, level: numpy.array) -> numpy.array:
        """Encode key, level in a quaternary hash.
    
        The level will be encoded in the length of the string
        """
        # Just making sure several data types are working (int, lists, arrays)
        level = numpy.array(level)
        if len(level.shape)==0:
            try:
                len(key)
            except:
                level = numpy.array([level])
            else:
                level = numpy.array([level]*len(key))
        key = numpy.array(key)
        key = numpy.mod(key, 4**level) # Truncate if necessary
        assert key.shape==level.shape
    
        # Base 4 string representation
        n_base4 = self.base_repr(key, 4)
        # Note: 0q in analogy to 0b for binary and 0x for hex
        q_key = [f"0q{k:0>{l}}" if l>0 else '0q' for k, l in zip(n_base4, list(level))]
        return q_key
    
    def decode(self, q_key: numpy.array(str)) -> Tuple[numpy.array, numpy.array]:
        """Convert the quaternary string representation back to key/level."""
        # Catch cases where q_key is a numpy array of objects or q_key is a sting
        q_key = numpy.array(q_key).astype(str)
    
        level = self.len_np(q_key)-2  # -2 to remove the leading 0q
        try:
            # Try vectorized first for speed
            key = self.int_np(numpy.char.replace(q_key, '0q', ''), 4)
        except ValueError:
            # Catch the case where level==0
            try:
                # Assuming we are dealing with a numpy array
                q_key2 = numpy.char.replace(q_key, '0q', '')
                key = [int(q, 4) if len(q)>0 else 0 for q in q_key2]
            except TypeError:
                if q_key=='0q':
                    key = 0
    
        return key, numpy.array(level)
    
    def base4_to_xy_indices(self, q_key: numpy.array(str)) -> Tuple[numpy.array, numpy.array]:
        """Convert the quaternary string to y and x coordinate indices."""
        key, level = self.decode(q_key)
        max_level = level.max()
        y_idx = 0
        x_idx = 0
        for i in range(max_level):
            stop_mask = i<=level
            mask = 1<<2*i+1
            y_idx += ((key&mask)>>(i+1)) * stop_mask
            mask = 1<<2*i
            x_idx += ((key&mask)>>i) * stop_mask
        return x_idx, y_idx
    
    # Adding these base4 methods here because they depend on the grid
    # (nestedgrid module imported here)
    def base4_to_coords(self, q_string: numpy.array(str)) -> Tuple[numpy.array, numpy.array]:
        """Translate base4 quaternary number to xy coordinate pair."""
        key, level = self.decode(q_string)
        max_level = level.max()
        y_idx = 0
        x_idx = 0
        for i in range(max_level):
            stop_mask = i<=level
            mask = 1<<2*i+1
            y_idx += ((key&mask)>>(i+1)) * stop_mask
            mask = 1<<2*i
            x_idx += ((key&mask)>>i) * stop_mask
        res_x, res_y = self.resolution(level)
        x_coord = x_idx * res_x + self.grid.origin.x
        y_coord = y_idx * res_y + self.grid.origin.y
        return x_coord, y_coord
    
    def base4_to_center_coords(self, q_string: numpy.array(str)) -> Tuple[numpy.array, numpy.array]:
        """Translate base4 quaternary number to xy center coordinate pair."""
        key, level = self.decode(q_string)
        max_level = level.max()
        y_idx = 0
        x_idx = 0
        for i in range(max_level):
            stop_mask = i<=level
            mask = 1<<2*i+1
            y_idx += ((key&mask)>>(i+1)) * stop_mask
            mask = 1<<2*i
            x_idx += ((key&mask)>>i) * stop_mask
        res_x, res_y = self.resolution(level)
        x_coord = (x_idx + .5) * res_x + self.grid.origin.x
        y_coord = (y_idx + .5) * res_y + self.grid.origin.y
        return x_coord, y_coord
    
    def base4_to_box(self, q_string: numpy.array(str)) -> numpy.array(shapely.Polygon):
        """Translate keys to shapely boxes.
        
        (e.g. for performing intersections or containment operations).
        """
        key, level = self.decode(q_string)
        max_level = level.max()
        y_idx = 0
        x_idx = 0
        for i in range(max_level):
            stop_mask = i<=level
            mask = 1<<2*i+1
            y_idx += ((key&mask)>>(i+1)) * stop_mask
            mask = 1<<2*i
            x_idx += ((key&mask)>>i) * stop_mask
    
        # Boxes encompass half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        res_x, res_y = self.resolution(level)
        south = y_idx * res_y + self.grid.origin.y
        west = x_idx * res_x + self.grid.origin.x
        east = west + res_x - self.epsilon
        north = south + res_y - self.epsilon
        return shapely.box(west, south, east, north)
