# Parquet Vectorstore and Overview Statistics

## Facilitating rapid spatial joins between vector and raster data

Data discovery and filter queries are improved if dedicated overview statistics are available for both raster and vector layers. 
Here we implement such overviews using the PAIRS key, the parquet file format, and eventually the cloud for storage. 
We also add a new Vectorstore for big geospatial tables that makes use of parquet storage and processing provided by geopandas/shapely2.0.

## Vectorstore
Vector data is first loaded into a PAIRS-commensurate Vectorstore with the following features:
-	Parquet store (local or S3) based on Geopandas/PyArrow for an entire dataset with multiple layer columns, one timestamp column, and one geometry column. The geometry can be anything supported by Shapely, such as Point, Linestring, Polygon.
-	One overview level *L* (PAIRS cell level) needs to be specified for every dataset. The geometries will be distributed in separate parquet files, specified by a PAIRS key at level *L*. 
-	Geometries that intersect multiple PAIRS keys at overview level *L* will either be cut into smaller geometries fully within a PAIRS cell or duplicated and stored in multiple files, based on user specification (intersection_policy flag).
-	Partitioning option using temporal and/or spatial partitions facilitates efficient search in large tables. Partitions for dimensions to be implemented...

## Querying the Vectorstore
-	To be implemented…

## Vector overviews (at level *L*)
-	Aggregate statistics at the overview level *L* are calculated for user-specified layers
-	Numerical layers: 
    - count, mean, std, min, max
    - user-specified quantiles (e.g. 0%, 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, 100%)
-	Categorical layers:
    - full histogram
    - count, unique, top, frequency

## Vector pyramids (at levels 0, 1, 2, …, *L*)
-	Selection of aggregate statistics that can be calculated precisely at lower-resolution levels using the overview statistics as the basis.
-	Geometries that intersect multiple overview cells are accounted for.
-	Numerical layers: 
    - count, mean, min, max
    - no quantiles
-	Categorical layers:
    - full histogram of values
    - count, unique, top, frequency


## Raster overviews

The schema for the raster overviews is currently as follows

| key | timestamp | first | sum | count | mean | std | min | 1% | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | max |
|-----|-----------|-------|-----|-------|------|-----|-----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|

Where overviews are stored at cell level (pixel level - 5). The filename convention is `layer{layer_id}_level{level-5}`. **Currently these are sorted**. The order is effectively `timestamp, key`.
