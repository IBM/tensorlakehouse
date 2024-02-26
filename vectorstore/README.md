# Vectorstore and Vector-Raster Fusion

Provides rapid access to large vector data persisted in GeoParquet and fuses vector data with GeoDN raster data.

## Vectorstore

- Persists spatio-temporal vector data (e.g. Points, Linestrings, or Polygons) in partitioned GeoParquet files in COS.
- Provides a spatial index based on quadtree (z-order defined on top of a nested grid).
    - Base4 key indexes the root of the geometry (smallest quadtree cell that fully encompases the vector geometry).
- GoeParquet files partitioned based on spatial index (and optionally on temporal, and dimension keys).
- Exposes each GeoParquet partition through spatio-temporal asset catalogue (STAC).
- Provide methods that accelerate queries, reprojection, containment, and intersection operations.
- Accelerates spatial joins (Overlays) with another similarly indexed Vectorstore.

## Querying the Vectorstore

- SQL-like queries (including the ST_geometry extension) are supported.

## Vector-Raster Fusion

Facilitates rapid spatial joins between vector and raster data. 

- Extension of the quadtree spatial index that indexes not only the root (quadtree cell that fully encompases the vector geometry) but also the interior of the geometry
- As soon as a quadtree cell is fully contained in the geometry, it isn't split any further.

## Combined Vector-Raster Queries

- Query through openEO with Vectorcubes as return values.

## Vector HSI

Summarizes vector data on a grid, commensurate with raster data.

- Temporal aggregation statistics of vector attributes at specific resolution levels, similar to what we do for raster data in the Hierarchical Spatial Index (HSI).
- Statistics for numeric attributes: min, max, first, mean, std, count, quantiles: (e.g. 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
- Statistics for categorical attributes: top, freq, first, unique, count, value_counts.
