# Parquet Vectorstore and Overview Statistics

## Facilitating rapid spatial joins between vector and raster data

Data discovery and filter queries are improved if dedicated overview statistics are available for both raster and vector layers. 
Here we implement such overviews using the PAIRS key, the parquet file format, and eventually the cloud for storage. 
We also add a new Vectorstore for big geospatial tables that makes use of parquet storage and processing provided by geopandas/shapely2.0.


## Raster overviews

The schema for the raster overviews is currently as follows

| key | timestamp | first | sum | count | mean | std | min | 1% | 5% | 10% | 25% | 50% | 75% | 90% | 95% | 99% | max |
|-----|-----------|-------|-----|-------|------|-----|-----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|

Where overviews are stored at cell level (pixel level - 5). The filename convention is `layer{layer_id}_level{level-5}`. **Currently these are sorted**. The order is effectively `timestamp, key`.
