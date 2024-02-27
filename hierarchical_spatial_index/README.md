# HSI: Hierarchical Spatial Index

HSI summarizes statistics of geospatial raster data at one or multiple lower-resolution levels in geoparquet files in S3. 

## HSI statistics

Dependent on the type of data (numeric or categorical), the statistics available are:

### Numeric data

- min, max, mean, std, first (bottom-left corner pixel), count (number of non-nan pixels)
- quantiles: (e.g. 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

Note that mean, std and the quantiles are weighted by area.

### Categorical data

- top: top category
- freq: frequency of the top category
- first: first non-nan category, starting from bottom-left corner pixel
- unique: number of unique categories
- count: number of non-nan pixels
- value_counts: Full histogram of the number of occurences of each value (category)

## Typical use-cases for the HSI

- Speed up raster filter queries
    - Query the HSI using min/max statistics or the value_counts to reduce the raw-data area/timestamps needed to be queried.

- Facilitate rapid spatial joins between vector and raster data
    - Data discovery and filter queries are improved if dedicated HSI statistics are available for both raster and vector (see vectorstore) layers.

- Cloud filtering
    - Cloud flags, e.g. fmask in Harmonized Landsat and Sentinel (HLS) datasets, are available at a high-resolution pixel level or as a percentage on the tile level. The HSI provides fast access to sub-tile level fmask information for medium-sized patches as needed for machine-learning.

- Find exact timestamps for irregular data
    - e.g. satellite data swaths, on a sub-tile level
    - Cheaply detect gaps in the data to facilitate (re)-ingestion

- Machine Learning
    - The "first" statistic (bottom-left corner) may be useful for global pre-training of time-series models, since it contains the exact value of a singel pixel.

- Speed up spatial aggregation over random polygons

- Search for specific values in categorical layers
    - e.g. calculate total area of soy grown in every US county

- Become aware of available data
    - "Just give me the low-resolution data at scale"
    - Improves user experience over STAC search
