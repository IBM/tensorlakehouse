# Parquet Vectorstore and Overview Statistics

## Facilitating rapid spatial joins between vector and raster data

Data discovery and filter queries are improved if dedicated overview statistics are available for both raster and vector layers. 
Here we implement such overviews using the PAIRS key, the parquet file format, and eventually the cloud for storage. 
We also add a new Vectorstore for big geospatial tables that makes use of parquet storage and processing provided by geopandas/shapely2.0.
