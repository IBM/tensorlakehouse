"""Define the specifics of the nested grid here.

    Classes:
        PAIRS              PAIRS grid based on WGS 84 and defined by resolution on level 29.
        World              Commensurate WGS 84 grid with square size of 360x360 degrees.
        WorldFitting       World fitting WGS 84 grid filling the 360x180 lon x lat space exactly.
                           Pixels are rectangular rather than square.
        WorldMercator      World Mercator (EPSG:2295).
        UTM30m_North       WGS 84 / UTM zone n North (EPSG:32601 - 32660).
                           Southern coordinates are represented with negative y values.

    Helper Classes:
        Res0               Helper Dataclass
        Origin             Helper Dataclass
"""

from dataclasses import dataclass
from rasterio import CRS
import numpy

@dataclass
class Res0:
    """Pixel resolution at level 0."""
    x: float
    y: float


@dataclass
class Origin:
    """Origin of the grid (lower-left corner)."""
    x: float
    y: float


def parse(epsg):
    """In case a string is passed, convert to integer epsg code."""
    try: 
        epsg = int(epsg)
    except:             
        try: 
            epsg = int(epsg.lstrip("EPSG:"))
        except:
            pass
    assert isinstance(epsg, int)
    return epsg


@dataclass
class PAIRS():
    """PAIRS grid based on WGS 84 and defined by resolution on level 29 equal to 1e-6 degrees."""
    # Resolution at pixel level 29
    RES_29 = 1e-6
    def __init__(self, epsg=4326):
        self.epsg = parse(epsg)

        # PAIRS grid requires WGS84
        assert self.epsg==4326

        # get the coordinate reference system from rasterio
        self.crs = CRS.from_epsg(self.epsg)

        # Define the origin of the grid (lower-left corner)
        self.origin = Origin(
            x = -180.,
            y = -90.,
        )

        # Define x and y extent (corresponds to pixel resolution at level 0)
        self.res0 = Res0(
            x = self.RES_29*2**29,
            y = self.RES_29*2**29,
        )

        # Extent of the level0 box (typically a square so pixels will be square as well)
        self.level0_bounds = [
            self.origin.x,
            self.origin.y,
            self.origin.x + self.res0.x,
            self.origin.y + self.res0.y,
        ] #west, south, east, north

        self.max_levels = 29

        # Define an epsilon here, smaller than the resolution of the highest level
        # These are used to approximate half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        self.epsilon = 1e-13

        # Define the valid range (west, south, east, north)
        self.valid_bounds = [-180, -90, 180, 90]

        self.valid_bounds_wgs84 = [-180, -90, 180, 90]

    def __repr__(self):
        """Adding the epsg to the class so that we can use eval to instantiate."""
        cls = type(self)
        return f"{cls.__name__}({self.epsg.__repr__()})"

    def area_weights(self, x_coord, y_coord):
        """Weights for area normalizations (return 1 if equal area grid)."""
        return numpy.cos(numpy.deg2rad(y_coord))

    def _y_degrees_to_meters(self, y_coord):
        """Calculate the length of a degree of longitude and latitude in meters.

        Adapted from the accepted answer from whuber
        https://gis.stackexchange.com/questions/75528/
            understanding-terms-in-length-of-degree-formula/75535#75535
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        """
        # y-coordinate in radians
        phi = numpy.deg2rad(y_coord)
        # The principal radius of the WGS84 spheroid is
        a = 6378137.
        # meters and its inverse flattening is
        f = 298.257223563
        # , whence the squared eccentricity is 0.0066943799901413165
        e2 = (2 - 1/f)/f
        # The meridional radius of curvature at latitude phi is
        M = a * (1 - e2) / (1 - e2 * numpy.sin(phi)**2)**(3/2)
        # and the radius of curvature along the prime vertical is
        N = a / (1 - e2 * numpy.sin(phi)**2)**(1/2)
        # Furthermore, the radius of the parallel is
        r = N * numpy.cos(phi)
        # Finally then length (in meters) per degree in x and y direction are
        x_length = r * numpy.pi / 180
        y_length = M * numpy.pi / 180
        return x_length, y_length

    def pixel_area(self, x_coord, y_coord, res_x, res_y):
        """Pixel area [square meters].

        Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
        :param x_coord:  x coordinate of pixel (e.g. longitude)
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        :param res_x:    width of pixel
        :param res_y:    height of pixel
        """
        x_length, y_length = self._y_degrees_to_meters(y_coord)
        return res_x * x_length * res_y * y_length


@dataclass
class World():
    """Commensurate WGS 84 grid with square size of 360x360 degrees."""
    def __init__(self, epsg=4326):
        self.epsg = parse(epsg)

        # World grid requires WGS84
        assert self.epsg==4326

        # get the coordinate reference system from rasterio
        self.crs = CRS.from_epsg(self.epsg)

        # Define the origin of the grid (lower-left corner)
        self.origin = Origin(
            x = -180.,
            y = -90.,
        )

        # Define x and y extent (corresponds to pixel resolution at level 0)
        self.res0 = Res0(
            x = 360.,
            y = 360.,
        )

        # Extent of the level0 box (typically a square so pixels will be square as well)
        self.level0_bounds = [
            self.origin.x,
            self.origin.y,
            self.origin.x + self.res0.x,
            self.origin.y + self.res0.y,
        ] #west, south, east, north

        self.max_levels = 29

        # Define an epsilon here, smaller than the resolution of the highest level
        # These are used to approximate half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        self.epsilon = 1e-13

        # Define the valid range (west, south, east, north)
        self.valid_bounds = [-180, -90, 180 - self.epsilon, 90 - self.epsilon]

        self.valid_bounds_wgs84 = [-180, -90, 180 - self.epsilon, 90 - self.epsilon]

    def __repr__(self):
        """Adding the epsg to the class so that we can use eval to instantiate."""
        cls = type(self)
        return f"{cls.__name__}({self.epsg.__repr__()})"
        
    def area_weights(self, x_coord, y_coord):
        """Weights for area normalizations (return 1 if equal area grid)."""
        return numpy.cos(numpy.deg2rad(y_coord))

    def _y_degrees_to_meters(self, y_coord):
        """Calculate the length of a degree of longitude and latitude in meters.

        Adapted from the accepted answer from whuber
        https://gis.stackexchange.com/questions/75528/
            understanding-terms-in-length-of-degree-formula/75535#75535
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        """
        # y-coordinate in radians
        phi = numpy.deg2rad(y_coord)
        # The principal radius of the WGS84 spheroid is
        a = 6378137.
        # meters and its inverse flattening is
        f = 298.257223563
        # , whence the squared eccentricity is 0.0066943799901413165
        e2 = (2 - 1/f)/f
        # The meridional radius of curvature at latitude phi is
        M = a * (1 - e2) / (1 - e2 * numpy.sin(phi)**2)**(3/2)
        # and the radius of curvature along the prime vertical is
        N = a / (1 - e2 * numpy.sin(phi)**2)**(1/2)
        # Furthermore, the radius of the parallel is
        r = N * numpy.cos(phi)
        # Finally then length (in meters) per degree in x and y direction are
        x_length = r * numpy.pi / 180
        y_length = M * numpy.pi / 180
        return x_length, y_length

    def pixel_area(self, x_coord, y_coord, res_x, res_y):
        """Pixel area [square meters].

        Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
        :param x_coord:  x coordinate of pixel (e.g. longitude)
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        :param res_x:    width of pixel
        :param res_y:    height of pixel
        """
        x_length, y_length = self._y_degrees_to_meters(y_coord)
        return res_x * x_length * res_y * y_length


@dataclass
class WorldFitting():
    """World fitting WGS 84 grid filling the 360x180 lon x lat space exactly.

    Pixels are rectangular rather than square since the level0_bounds are not square either.
    """
    def __init__(self, epsg=4326):
        self.epsg = parse(epsg)

        # WorldFitting grid requires WGS84
        assert self.epsg==4326

        # get the coordinate reference system from rasterio
        self.crs = CRS.from_epsg(self.epsg)

        # Define the origin of the grid (lower-left corner)
        self.origin = Origin(
            x = -180.,
            y = -90.,
        )

        # Define x and y extent (corresponds to pixel resolution at level 0)
        self.res0 = Res0(
            x = 360.,
            y = 180.,
        )

        # Extent of the level0 box (typically a square so pixels will be square as well)
        self.level0_bounds = [
            self.origin.x,
            self.origin.y,
            self.origin.x + self.res0.x,
            self.origin.y + self.res0.y,
        ] #west, south, east, north

        self.max_levels = 29

        # Define an epsilon here, smaller than the resolution of the highest level
        # These are used to approximate half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        self.epsilon = 1e-13

        # Define the valid range (west, south, east, north)
        self.valid_bounds = [-180, -90, 180 - self.epsilon, 90 - self.epsilon]

        self.valid_bounds_wgs84 = [-180, -90, 180 - self.epsilon, 90 - self.epsilon]

    def __repr__(self):
        """Adding the crs to the class name."""
        cls = type(self)
        return f"{cls.__name__}({self.crs.lstrip('EPSG:')})"
        
    def area_weights(self, x_coord, y_coord):
        """Weights for area normalizations (return 1 if equal area grid)."""
        return numpy.cos(numpy.deg2rad(y_coord))

    def _y_degrees_to_meters(self, y_coord):
        """Calculate the length of a degree of longitude and latitude in meters.

        Adapted from the accepted answer from whuber
        https://gis.stackexchange.com/questions/75528/
            understanding-terms-in-length-of-degree-formula/75535#75535
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        """
        # y-coordinate in radians
        phi = numpy.deg2rad(y_coord)
        # The principal radius of the WGS84 spheroid is
        a = 6378137.
        # meters and its inverse flattening is
        f = 298.257223563
        # , whence the squared eccentricity is 0.0066943799901413165
        e2 = (2 - 1/f)/f
        # The meridional radius of curvature at latitude phi is
        M = a * (1 - e2) / (1 - e2 * numpy.sin(phi)**2)**(3/2)
        # and the radius of curvature along the prime vertical is
        N = a / (1 - e2 * numpy.sin(phi)**2)**(1/2)
        # Furthermore, the radius of the parallel is
        r = N * numpy.cos(phi)
        # Finally then length (in meters) per degree in x and y direction are
        x_length = r * numpy.pi / 180
        y_length = M * numpy.pi / 180
        return x_length, y_length

    def pixel_area(self, x_coord, y_coord, res_x, res_y):
        """Pixel area [square meters].

        Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
        :param x_coord:  x coordinate of pixel (e.g. longitude)
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        :param res_x:    width of pixel
        :param res_y:    height of pixel
        """
        x_length, y_length = self._y_degrees_to_meters(y_coord)
        return res_x * x_length * res_y * y_length


@dataclass
class WorldMercator():
    """World Mercator (EPSG:3395).

    Cylindrical projection preserving angles, distorting areas especially those close to the poles.
    """
    def __init__(self, epsg=3395):
        self.epsg = parse(epsg)

        # WorldMercator grid requires specific World Mercator grid
        assert self.epsg==3395

        # get the coordinate reference system from rasterio
        self.crs = CRS.from_epsg(self.epsg)

        # Define the origin of the grid (lower-left corner)
        self.origin = Origin(
            x = -20037508.34,
            y = -20037508.34,
        )

        # Define x and y extent (corresponds to pixel resolution at level 0)
        self.res0 = Res0(
            x = 40075016.68,
            y = 40075016.68,
        )

        # Extent of the level0 box (typically a square so pixels will be square as well)
        self.level0_bounds = [
            self.origin.x,
            self.origin.y,
            self.origin.x + self.res0.x,
            self.origin.y + self.res0.y,
        ] #west, south, east, north

        self.max_levels = 29

        # Define an epsilon here, smaller than the resolution of the highest level
        # These are used to approximate half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        self.epsilon = 1e-7

        # Define the valid range (west, south, east, north)
        self.valid_bounds = [
            -20037508.34,
            -20037508.34,              # -15496570.74,
            20037508.34 - self.epsilon,
            20037508.34 - self.epsilon,  # 18764656.23-self.epsilon,
        ]

        self.valid_bounds_wgs84 = [-180, -90, 180, 90]

    def __repr__(self):
        """Adding the epsg to the class so that we can use eval to instantiate."""
        cls = type(self)
        return f"{cls.__name__}({self.epsg.__repr__()})"
        
    def area_weights(self, x_coord, y_coord):
        """Weights for area normalizations (return 1 if equal area grid)."""
        return numpy.cos(y_coord / 20037508.34 * numpy.pi / 2)

    def _y_coord_to_meters(self, y_coord):
        """Calculate the length of a degree of longitude and latitude in meters.

        Adapted from the accepted answer from whuber
        https://gis.stackexchange.com/questions/75528/
            understanding-terms-in-length-of-degree-formula/75535#75535
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        """
        # y-coordinate in radians
        phi = y_coord / 20037508.34 * numpy.pi / 2
        # The principal radius of the WGS84 spheroid is
        a = 6378137.
        # meters and its inverse flattening is
        f = 298.257223563
        # , whence the squared eccentricity is 0.0066943799901413165
        e2 = (2 - 1/f)/f
        # The meridional radius of curvature at latitude phi is
        M = a * (1 - e2) / (1 - e2 * numpy.sin(phi)**2)**(3/2)
        # and the radius of curvature along the prime vertical is
        N = a / (1 - e2 * numpy.sin(phi)**2)**(1/2)
        # Furthermore, the radius of the parallel is
        r = N * numpy.cos(phi)
        # Finally then length (in meters) per degree in x and y direction are
        x_length = r * numpy.pi / 180
        y_length = M * numpy.pi / 180
        return x_length, y_length

    def pixel_area(self, x_coord, y_coord, res_x, res_y):
        """Pixel area [square meters].

        Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
        :param x_coord:  x coordinate of pixel (e.g. longitude)
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        :param res_x:    width of pixel
        :param res_y:    height of pixel
        """
        x_length, y_length = self._y_coord_to_meters(y_coord)
        return res_x * x_length * res_y * y_length


@dataclass
class UTM30m_North():
    """WGS 84 / UTM zone North (EPSG:32601 - 32660).
    
    Southern coordinates are represented with negative y values.
    Commensurate with the Harmonized Landsat and Sentinel-2 (HLS).
    Transverse Mercator.
    """
    def __init__(self, epsg: int):
        """Instantiate the object.

        epsg: EPSG number of UTM coordinate reference system.
            valid (32601, 32602, .., 32660) 
        """
        self.epsg = parse(epsg)

        # get the coordinate reference system from rasterio
        self.crs = CRS.from_epsg(self.epsg)

        # Southern areas are modeled using the corresponding northern UTM grid
        southern_crs = [CRS.from_epsg(i) for i in numpy.arange(32701, 32761)]
        if self.crs in southern_crs:
            raise ValueError('For Southern latitudes use Northern UTM zones with negative y values')
        
        # Valid UTM North grids are ('EPSG:32601', 'EPSG:32602', .., 'EPSG:32660')
        valid_crs = [CRS.from_epsg(i) for i in numpy.arange(32601, 32661)]
        assert self.crs in valid_crs
        
        self.zone_number = self.epsg - 32600
        
        # Define the origin of the grid (lower-left corner)
        self.origin = Origin(
            x = 0.0,  # Origin set so that we get a commensurate square at level 0
            y = -30*(2**19),  # Origin 1/2 Res0 (-15728640)
        )

        # Define x and y extent (corresponds to pixel resolution at level 0)
        self.res0 = Res0(
            x = 30*(2**20),
            y = 30*(2**20),  # (31457280) commensurate with 30m and encompassing 84deg North
        )

        # Extent of the level0 box (typically a square so pixels will be square as well)
        self.level0_bounds = [
            self.origin.x,
            self.origin.y,
            self.origin.x + self.res0.x,
            self.origin.y + self.res0.y,
        ] #west, south, east, north

        self.max_levels = 29

        # Define an epsilon here, smaller than the resolution of the highest level
        # These are used to approximate half-open intervals [south,north) and [west,east),
        # so that points (and some lines) are assigned to exactly one box on each resolution level.
        self.epsilon = 1e-7

        west = numpy.arange(-180, 180, 6)[self.zone_number-1]
        east = numpy.arange(-180, 180, 6)[self.zone_number]
        south = -80
        north = 84
        
        # Define the valid range (west, south, east, north)
        self.valid_bounds_wgs84 = [west, south, east, north]
        
        self.valid_bounds = [
            166021.44,   # 102 deg west
            -8883084.96,  # 80 deg south
            833978.56,   # 96 deg west
            9329005.18,  # 84 deg north
        ]
        
    def __repr__(self):
        """Adding the epsg to the class so that we can use eval to instantiate."""
        cls = type(self)
        return f"{cls.__name__}({self.epsg.__repr__()})"

    def area_weights(self, x_coord, y_coord):
        """Weights for area normalizations (return 1 if equal area grid)."""
        return y_coord*0.+1.

    def _y_coord_to_meters(self, y_coord):
        """Calculate the length of a degree of longitude and latitude in meters.

        :param y_coord:  y coordinate of pixel
        """
        return 0.9996, 0.9996

    def pixel_area(self, x_coord, y_coord, res_x, res_y):
        """Pixel area [square meters].

        Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
        :param x_coord:  x coordinate of pixel (e.g. longitude)
        :param y_coord:  y coordinate of pixel (e.g. latitude)
        :param res_x:    width of pixel
        :param res_y:    height of pixel
        """
        return 0.9996 * res_x * 0.9996 * res_y


# @dataclass
# class UTM30m_North:
#     """WGS 84 / UTM zone n North (EPSG:32601 - 32660).
    
#     Southern coordinates are represented with negative y values.
#     This grid is commensurate with the system HLS uses for their rasters
#     Transverse Mercator.
#     """
#     def __init__(self, zone):
#         # parse the zone string
#         if zone.endswith('N'):
#             self.north_or_south = 'north'
#             self.zone_number = int(zone.rsplit('N')[0])
#         elif zone.endswith('S'):
#             self.north_or_south = 'south'
#             self.zone_number = int(zone.rsplit('S')[0])
#             raise ValueError('For Southern latitudes use Northern UTM zones with negative y values')
#         else:
#             raise ValueError('UTM zone string not understood')
            
#         assert (self.zone_number>=1) and (self.zone_number<=60)
        
#         # Coordinate reference system
#         self.crs = f'EPSG:326{self.zone_number:02}'

#         # Define the origin of the grid (lower-left corner)
#         self.origin = Origin(
#             x = 0.0,  # Origin set so that we get a commensurate square at level 0
#             y = -30*(2**19),  # Origin 1/2 Res0 (15728640)
#         )

#         # Define x and y extent (corresponds to pixel resolution at level 0)
#         self.res0 = Res0(
#             x = 30*(2**20),
#             y = 30*(2**20),  # (31457280) commensurate with 30m and encompassing 84deg North
#         )

#         # Extent of the level0 box (typically a square so pixels will be square as well)
#         self.level0_bounds = [
#             self.origin.x,
#             self.origin.y,
#             self.origin.x + self.res0.x,
#             self.origin.y + self.res0.y,
#         ] #west, south, east, north

#         self.max_levels = 29

#         # Define an epsilon here, smaller than the resolution of the highest level
#         # These are used to approximate half-open intervals [south,north) and [west,east),
#         # so that points (and some lines) are assigned to exactly one box on each resolution level.
#         self.epsilon = 1e-7

#         west = numpy.arange(-180, 180, 6)[self.zone_number-1]
#         east = numpy.arange(-180, 180, 6)[self.zone_number]
#         south = -80
#         north = 84
        
#         # Define the valid range (west, south, east, north)
#         self.valid_bounds_wgs84 = [west, south, east, north]
        
#         self.valid_bounds = [
#             166021.44,   # 102 deg west
#             -8883084.96,  # 80 deg south
#             833978.56,   # 96 deg west
#             9329005.18,  # 84 deg north
#         ]
        
#     def __repr__(self):
#         """Adding the epsg to the class so that we can use eval to instantiate."""
#         cls = type(self)
#         return f"{cls.__name__}({self.epsg.__repr__()})"
        
#     def area_weights(self, x_coord, y_coord):
#         """Weights for area normalizations (return 1 if equal area grid)."""
#         return y_coord*0.+1.

#     def _y_coord_to_meters(self, y_coord):
#         """Calculate the length of a degree of longitude and latitude in meters.

#         :param y_coord:  y coordinate of pixel
#         """
#         return 0.9996, 0.9996

#     def pixel_area(self, x_coord, y_coord, res_x, res_y):
#         """Pixel area [square meters].

#         Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
#         :param x_coord:  x coordinate of pixel (e.g. longitude)
#         :param y_coord:  y coordinate of pixel (e.g. latitude)
#         :param res_x:    width of pixel
#         :param res_y:    height of pixel
#         """
#         return 0.9996 * res_x * 0.9996 * res_y


# @dataclass
# class UTM13N:
#     """WGS 84 / UTM zone 13N (EPSG:32613).

#     Transverse Mercator.
#     """
#     def __init__(self):
#         # Coordinate reference system
#         self.crs = 'EPSG:32613'

#         # Define the origin of the grid (lower-left corner)
#         self.origin = Origin(
#             x = 0.0,  # Origin set so that we get a commensurate square at level 0
#             y = 0.0,  # Origin at equator
#         )

#         # Define x and y extent (corresponds to pixel resolution at level 0)
#         self.res0 = Res0(
#             x = 10*(2**20),
#             y = 10*(2**20),  # (10485760) commensurate with 10m and encompassing 84deg North
#         )

#         # Extent of the level0 box (typically a square so pixels will be square as well)
#         self.level0_bounds = [
#             self.origin.x,
#             self.origin.y,
#             self.origin.x + self.res0.x,
#             self.origin.y + self.res0.y,
#         ] #west, south, east, north

#         self.max_levels = 29

#         # Define an epsilon here, smaller than the resolution of the highest level
#         # These are used to approximate half-open intervals [south,north) and [west,east),
#         # so that points (and some lines) are assigned to exactly one box on each resolution level.
#         self.epsilon = 1e-7

#         # Define the valid range (west, south, east, north)
#         self.valid_bounds = [
#             166021.44,   # 108 deg west
#             0.0,         # 0 deg north (equator)
#             833978.56,   # 102 deg west
#             9329005.18,  # 84 deg north
#         ]
        
#         self.valid_bounds_wgs84 = [
#             -108,   # 108 deg west
#             0.0,         # 0 deg north (equator)
#             -102,   # 102 deg west
#             84,  # 84 deg north
#         ]
        
#     def area_weights(self, x_coord, y_coord):
#         """Weights for area normalizations (return 1 if equal area grid)."""
#         return y_coord*0.+1.

#     def _y_coord_to_meters(self, y_coord):
#         """Calculate the length of a degree of longitude and latitude in meters.

#         :param y_coord:  y coordinate of pixel
#         """
#         return 0.9996, 0.9996

#     def pixel_area(self, x_coord, y_coord, res_x, res_y):
#         """Pixel area [square meters].

#         Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
#         :param x_coord:  x coordinate of pixel (e.g. longitude)
#         :param y_coord:  y coordinate of pixel (e.g. latitude)
#         :param res_x:    width of pixel
#         :param res_y:    height of pixel
#         """
#         return 0.9996 * res_x * 0.9996 * res_y


# @dataclass
# class UTM14N:
#     """WGS 84 / UTM zone 14N (EPSG:32614).

#     Transverse Mercator.
#     """
#     def __init__(self):
#         # Coordinate reference system
#         self.crs = 'EPSG:32614'

#         # Define the origin of the grid (lower-left corner)
#         self.origin = Origin(
#             x = 0.0,  # Origin set so that we get a commensurate square at level 0
#             y = 0.0,  # Origin at equator
#         )

#         # Define x and y extent (corresponds to pixel resolution at level 0)
#         self.res0 = Res0(
#             x = 10*(2**20),
#             y = 10*(2**20),  # (10485760) commensurate with 10m and encompassing 84deg North
#         )

#         # Extent of the level0 box (typically a square so pixels will be square as well)
#         self.level0_bounds = [
#             self.origin.x,
#             self.origin.y,
#             self.origin.x + self.res0.x,
#             self.origin.y + self.res0.y,
#         ] #west, south, east, north

#         self.max_levels = 29

#         # Define an epsilon here, smaller than the resolution of the highest level
#         # These are used to approximate half-open intervals [south,north) and [west,east),
#         # so that points (and some lines) are assigned to exactly one box on each resolution level.
#         self.epsilon = 1e-7

#         # Define the valid range (west, south, east, north)
#         self.valid_bounds = [
#             166021.44,   # 102 deg w
#             0.0,         # 0 deg north (equator)
#             833978.56,   # 96 deg west
#             9329005.18,  # 84 deg north
#         ]
        
#         self.valid_bounds_wgs84 = [
#             -102,   # 102 deg west
#             0.0,         # 0 deg north (equator)
#             -96,   # 96 deg west
#             84,  # 84 deg north
#         ]
        
#     def area_weights(self, x_coord, y_coord):
#         """Weights for area normalizations (return 1 if equal area grid)."""
#         return y_coord*0.+1.

#     def _y_coord_to_meters(self, y_coord):
#         """Calculate the length of a degree of longitude and latitude in meters.

#         :param y_coord:  y coordinate of pixel
#         """
#         return 0.9996, 0.9996

#     def pixel_area(self, x_coord, y_coord, res_x, res_y):
#         """Pixel area [square meters].

#         Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
#         :param x_coord:  x coordinate of pixel (e.g. longitude)
#         :param y_coord:  y coordinate of pixel (e.g. latitude)
#         :param res_x:    width of pixel
#         :param res_y:    height of pixel
#         """
#         return 0.9996 * res_x * 0.9996 * res_y


# @dataclass
# class UTM30m:
#     """WGS 84 / UTM zone n(N/S) (EPSG:32601 - 32660, 32701 - 32760).

#     Transverse Mercator.
#     """
#     def __init__(self, zone):
#         # parse the zone string
#         if zone.endswith('N'):
#             self.north_or_south = 'north'
#             self.zone_number = int(zone.rsplit('N')[0])
#         elif zone.endswith('S'):
#             self.north_or_south = 'south'
#             self.zone_number = int(zone.rsplit('S')[0])
#         else:
#             raise ValueError('UTM zone string not understood')
            
#         assert (self.zone_number>=1) and (self.zone_number<=60)
        
#         # Coordinate reference system
#         if self.north_or_south=='north':
#             self.crs = f'EPSG:326{self.zone_number:02}'
#         else:
#             # southern hemisphere
#             self.crs = f'EPSG:327{self.zone_number:02}'

#         if self.north_or_south=='north':
#             # Define the origin of the grid (lower-left corner)
#             self.origin = Origin(
#                 x = 0.0,  # Origin set so that we get a commensurate square at level 0
#                 y = 0.0,  # Origin at equator --> We may need to change this
#             )

#             # Define x and y extent (corresponds to pixel resolution at level 0)
#             self.res0 = Res0(
#                 x = 30*(2**20),
#                 y = 30*(2**20),  # (31457280) commensurate with 30m and encompassing 84deg North
#             )
#         else:
#             raise NotImplementedError("To be implemented")

#         # Extent of the level0 box (typically a square so pixels will be square as well)
#         self.level0_bounds = [
#             self.origin.x,
#             self.origin.y,
#             self.origin.x + self.res0.x,
#             self.origin.y + self.res0.y,
#         ] #west, south, east, north

#         self.max_levels = 29

#         # Define an epsilon here, smaller than the resolution of the highest level
#         # These are used to approximate half-open intervals [south,north) and [west,east),
#         # so that points (and some lines) are assigned to exactly one box on each resolution level.
#         self.epsilon = 1e-7

#         if self.north_or_south=='north':
#             west = numpy.arange(-180, 180, 6)[self.zone_number-1]
#             east = numpy.arange(-180, 180, 6)[self.zone_number]
#             south = 0
#             north = 84
#         else:
#             raise NotImplementedError("To be implemented")
        
#         # Define the valid range (west, south, east, north)
#         self.valid_bounds_wgs84 = [west, south, east, north]
        
#         self.valid_bounds = [
#             166021.44,   # 102 deg west
#             0.0,         # 0 deg north (equator)
#             833978.56,   # 96 deg west
#             9329005.18,  # 84 deg north
#         ]
        
#     def area_weights(self, x_coord, y_coord):
#         """Weights for area normalizations (return 1 if equal area grid)."""
#         return y_coord*0.+1.

#     def _y_coord_to_meters(self, y_coord):
#         """Calculate the length of a degree of longitude and latitude in meters.

#         :param y_coord:  y coordinate of pixel
#         """
#         return 0.9996, 0.9996

#     def pixel_area(self, x_coord, y_coord, res_x, res_y):
#         """Pixel area [square meters].

#         Pixel with lateral dimensions res_x, res_y at position (x_coord, y_coord).
#         :param x_coord:  x coordinate of pixel (e.g. longitude)
#         :param y_coord:  y coordinate of pixel (e.g. latitude)
#         :param res_x:    width of pixel
#         :param res_y:    height of pixel
#         """
#         return 0.9996 * res_x * 0.9996 * res_y