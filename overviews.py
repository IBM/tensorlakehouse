import os
import time
import numpy
import pandas
from datetime import datetime, timedelta
import pytz
import geopandas
from functools import partial

import pairs_quadtree
import vectorstore


class Overviews(vectorstore.Vectorstore):
    """
    Overviews Class to calculate vectordata overviews for numerical or categorical layers (columns)
    and save them in an overviewstore
    """
    
    # Default values for variables
    VECTORSTORE_DIRECTORY      = 'data/vector/vectorstore/'
    OVERVIEWSTORE_DIRECTORY    = 'data/vector/overviews/'

    # Overview statistics
    NUMERIC_LAYERS             = []
    TIMESTAMP_LAYERS           = []
    CATEGORICAL_LAYERS         = []
    QUANTILES                  = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    FIRST                      = True # First statistics for vector overviews
    # Set to True for timeseries data, where we have too many timestamps to calculate individual statistics (e.g. >100)
    TIMESTAMP_AGGREGATION      = True 
    
    # Pyramids
    PYRAMID_LEVELS             = []  # Initialize as empty list. Will be populated when pyramids are generated
    
    def __init__(self,
                 dataset,
                 vectorstore_directory = None,
                 overviewstore_directory = None,
                 numeric_layers = None,
                 timestamp_layers = None,
                 categorical_layers = None,
                 quantiles = None,
                 first = None,
                 timestamp_aggregation=None,
                ):
        # Dataset name
        self.dataset                      = dataset
        
        # Vectorstore base directory
        self.vectorstore_directory        = self.VECTORSTORE_DIRECTORY if vectorstore_directory is None else vectorstore_directory
        
        super().__init__(
            dataset=dataset,
            vectorstore_directory=vectorstore_directory,
        )
        
        # If the vectorstore already exists, read the settings and metadata
        try:
            self.read_vectorstore_settings()
            self.metadata_from_parquet()
        except:
            print('WARNING: error reading vectorstore settings or metadata')

        # Overview base directory
        self.overviewstore_directory = self.OVERVIEWSTORE_DIRECTORY if overviewstore_directory is None else overviewstore_directory
        self.overview_directory = os.path.join(self.overviewstore_directory, self.dataset)
        if not os.path.exists(self.overview_directory): os.makedirs(self.overview_directory)
        """
        # Overview statistics variables
        self.numeric_layers = self.NUMERIC_LAYERS if numeric_layers is None else numeric_layers
        self.timestamp_layers = self.TIMESTAMP_LAYERS if timestamp_layers is None else timestamp_layers
        self.categorical_layers = self.CATEGORICAL_LAYERS if categorical_layers is None else categorical_layers
        self.quantiles = self.QUANTILES if quantiles is None else quantiles
        self.first = self.FIRST if first is None else first
        self.timestamp_aggregation = self.TIMESTAMP_AGGREGATION if timestamp_aggregation is None else timestamp_aggregation

        # Pyramid variables
        self.pyramid_levels = self.PYRAMID_LEVELS  # Will be set when pyramids are generated
        """

    def _calc_write_intersected_geometry(self):
        # Determine the geometries that intersect overview cells
        df_intersected = []
        for i, row in self.gdf_meta[[self.composite_key_col, 'filepath']].iterrows():
            df1 = pandas.read_parquet(row['filepath'], columns=[self.composite_key_col, self.id_col, self.intersection_flag_col])
            df1 = df1.drop_duplicates().reset_index(drop=True)
            df_intersected.append(df1[df1[self.intersection_flag_col]])
        df_intersected = pandas.concat(df_intersected).reset_index(drop=True)
        
        # Generate a two column table (id_col and lists of composite_key_col) 
        df_intersected = df_intersected.groupby(self.id_col, group_keys=False)[self.composite_key_col].apply(list).reset_index()

        filepath = os.path.join(self.overview_directory, 'intersected_geometry.parquet')
        df_intersected = df_intersected.to_parquet(path=filepath, engine='pyarrow', compression='snappy')

    def read_intersected_geometry(self):
        filepath = os.path.join(self.overview_directory, 'intersected_geometry.parquet')
        try:
            df = pandas.read_parquet(filepath)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
            
    def _calc_write_overview_statistics_numeric(self):
        if self.timestamp_aggregation:
            timestamp_agg_cols = ['timestamp_start', 'timestamp_end']
        else:
            timestamp_agg_cols = []
        df_cell_statistics = []
        for i, row in self.gdf_meta[[self.composite_key_col, 'filepath']].iterrows():
            # Read one-by-one so we won't have any memory issues
            df_parquet = pandas.read_parquet(row['filepath'], columns=self.timestamp_layers+self.numeric_layers)
            if self.timestamp_aggregation:
                # Groupby spatial only
                df_parquet['dummy_key'] = 1
                grp = df_parquet.groupby('dummy_key')
            else:
                # Groupby spatio-temporal
                grp = df_parquet.groupby(self.timestamp_layers)

            # Most of the descriptive statistics calculated here
            df1 = grp.describe(percentiles=self.quantiles).reset_index()
            if self.first:
                # "First" statistics requested
                df2 = grp.first().reset_index()
                # df1 has a multi-index, so we need one for df2 too
                df2.columns = pandas.MultiIndex.from_product([df2.columns, ['first']])
                df1 = df1.join(df2)
            if self.timestamp_aggregation:
                # Add start and end timestamp columns
                df1['timestamp_start'] = df_parquet[self.timestamp_layers[0]].min()
                df1['timestamp_end'] = df_parquet[self.timestamp_layers[0]].max()
            df1[self.composite_key_col] = row[self.composite_key_col]
            df_cell_statistics.append(df1)

        df_cell_statistics = pandas.concat(df_cell_statistics, axis=0).reset_index(drop=True)
        self._decompose_composite_keys(df_cell_statistics)

        for layer in self.numeric_layers: 
            # Pick relevant overviews by layer
            df_cell_statistics_layer = pandas.concat([df_cell_statistics[layer], df_cell_statistics[
                timestamp_agg_cols + self.temporal_keys + self.dimension_keys + [
                    self.spatial_key_col, self.spatial_level_col, self.composite_key_col
                ]].droplevel(1, axis=1)
            ], axis=1)

            # Add geometry column
            gdf_cell_statistics_layer = pandas.merge(
                self._polyCells2geodataframe(df_cell_statistics_layer[self.spatial_key_col].drop_duplicates()),
                df_cell_statistics_layer, 
                on=[self.spatial_key_col, self.spatial_level_col],
                how='right',
            )

            # Save to geo-parquet
            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
            gdf_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
 
    def _sanitize_dt_column(self, dt64):
        dt64 = dt64.apply(lambda x: x.tz_localize(None))
        ts = numpy.round((dt64 - numpy.datetime64('1970-01-01T00:00:00')) / numpy.timedelta64(1, 'us')) /1e6
        ts = ts.apply(lambda x: datetime.utcfromtimestamp(x).replace(tzinfo=pytz.utc))
        return ts

    def _calc_write_overview_statistics_categorical(self):
        """
        Categorical layers (count, unique, top, frequency)
        For object data (e.g. strings), the result’s index will include count, unique, top, and freq. 
        The top is the most common value. The freq is the most common value’s frequency. 
        """
        if self.timestamp_aggregation:
            timestamp_agg_cols = ['timestamp_start', 'timestamp_end']
        else:
            timestamp_agg_cols = []
        df_cell_statistics = []
        for i, row in self.gdf_meta[[self.composite_key_col, 'filepath']].iterrows():
            # Read one-by-one so we won't have any memory issues
            df_parquet = pandas.read_parquet(row['filepath'], columns=self.timestamp_layers+self.categorical_layers)
            df_parquet = df_parquet.astype(object) # Make sure the columns will be interpreted as categorical
            if self.timestamp_aggregation:
                # Groupby spatial only
                df_parquet['dummy_key'] = 1
                grp = df_parquet.groupby('dummy_key')
            else:
                # Groupby spatio-temporal
                grp = df_parquet.groupby(self.timestamp_layers)

            # Most of the descriptive statistics calculated here
            df1 = grp.describe().reset_index()
            if self.first:
                # "First" statistics requested
                df2 = grp.first().reset_index()
                # df1 has a multi-index, so we need one for df2 too
                df2.columns = pandas.MultiIndex.from_product([df2.columns, ['first']])
                df1 = df1.join(df2)

            if self.timestamp_aggregation:
                # Add start and end timestamp columns
                df1['timestamp_start'] = df_parquet[self.timestamp_layers[0]].min()
                df1['timestamp_end'] = df_parquet[self.timestamp_layers[0]].max()
            df1[self.composite_key_col] = row[self.composite_key_col]
            df_cell_statistics.append(df1)

        df_cell_statistics = pandas.concat(df_cell_statistics, axis=0).reset_index(drop=True)
        self._decompose_composite_keys(df_cell_statistics)

        for layer in self.categorical_layers: 
            # Pick relevant overviews by layer
            df_cell_statistics_layer = pandas.concat([df_cell_statistics[layer], df_cell_statistics[
                timestamp_agg_cols + self.temporal_keys + self.dimension_keys + [
                    self.spatial_key_col, self.spatial_level_col, self.composite_key_col
                ]].droplevel(1, axis=1)
            ], axis=1)

            # Add geometry column
            gdf_cell_statistics_layer = pandas.merge(
                self._polyCells2geodataframe(df_cell_statistics_layer[self.spatial_key_col].drop_duplicates()),
                df_cell_statistics_layer, 
                on=[self.spatial_key_col, self.spatial_level_col],
                how='right',
            )

            # Save to geo-parquet
            filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
            gdf_cell_statistics_layer.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
    
    def _calc_write_overview_histogram(self, histogram_layer):
        # Categorical and timestamp layers only
        assert(histogram_layer in (self.categorical_layers + self.timestamp_layers))

        df_cell_hist = []
        for i, row in self.gdf_meta[[self.composite_key_col, 'filepath']].iterrows():
            df1 = pandas.read_parquet(row['filepath'], columns=[histogram_layer])
            df2 = df1[histogram_layer].value_counts()
            df2.name = row[self.composite_key_col]
            df_cell_hist.append(df2)
        df_cell_hist = pandas.concat(df_cell_hist, axis=1)
        df_cell_hist = df_cell_hist.sort_index().fillna(0).astype(int)

        filepath = os.path.join(self.overview_directory, 'overview_histogram_' + histogram_layer + '.parquet')
        df_cell_hist.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
        
    def calc_overview_statistics(
        self, 
        numeric_layers=None, 
        timestamp_layers=None, 
        categorical_layers=None, 
        quantiles=None, 
        first=None,
        timestamp_aggregation=None,
    ):
        """
        Caluculate the overview layer statistics and write them out to parquet
        """
        if numeric_layers is not None:
            self.numeric_layers = numeric_layers 
        if timestamp_layers is not None:
            self.timestamp_layers = timestamp_layers 
        if categorical_layers is not None:
            self.categorical_layers = categorical_layers 
        if quantiles is not None:
            self.quantiles = quantiles 
        if first is not None:
            self.first = first 
        if timestamp_aggregation is not None:
            self.timestamp_aggregation = timestamp_aggregation 
        
        # Dump the settings to a json file (existing will be overwritten)
        #self.write_overviewstore_settings()
        
        # Assert that we don't have layers in multiple categories at once
        all_layers = self.numeric_layers + self.timestamp_layers + self.categorical_layers
        assert(len(all_layers)==len(list(set(all_layers))))
        
        # Make sure gdf_meta is present
        try:
            self.gdf_meta
        except:
            self.metadata_from_parquet()
        
        if len(self.numeric_layers)>0:
            self._calc_write_overview_statistics_numeric()
        if len(self.categorical_layers)>0:
            self._calc_write_overview_statistics_categorical()
            for categorical_layer in self.categorical_layers:
                self._calc_write_overview_histogram(categorical_layer)
        if len(self.timestamp_layers)>0:
            for timestamp_layer in self.timestamp_layers:
                self._calc_write_overview_histogram(timestamp_layer)
                    
        # Summary of intersected geometries (at overview cell boundaries)
        self._calc_write_intersected_geometry()
            
    def read_overview_statistics(self, layer, **kwargs):
        filepath = os.path.join(self.overview_directory, 'overview_statistics_' + layer + '.parquet')
        try:
            # If needed, transpose after reading the parquet instead of in the parquet file itself, 
            # because parquet has limitations in storing mixed datatypes in columns.
            #df = pandas.read_parquet(filepath, **kwargs)
            df = geopandas.read_parquet(filepath, **kwargs) # Note reading using geopandas is slightly slower
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df

    def read_overview_histogram(self, layer, **kwargs):
        filepath = os.path.join(self.overview_directory, 'overview_histogram_' + layer + '.parquet')
        try:
            df = pandas.read_parquet(filepath, **kwargs)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
    
    def _pyramid_groupby_cols(self, pyramid_level):
        cols = ['pyramid_level' + str(pyramid_level)] + self.temporal_keys + self.dimension_keys
        return cols
    
    def _numeric_pyramids(self, numeric_layer, verbose=False):
        if verbose:
            print('numeric_layer', numeric_layer)
            
        if self.timestamp_aggregation:
            timestamp_keys = ['timestamp_start', 'timestamp_end']
        else:
            timestamp_keys = self.temporal_keys
            
        # Read the detailed overview statistics at the overview level
        df_cell_statistics = self.read_overview_statistics(numeric_layer)

        # Keep a limited number of statistics and add the pyramid keys
        df = df_cell_statistics[['count', 'mean', 'min', 'max'] + timestamp_keys] # + self.dimension_keys
        df = df.join(self.df_pyramid_keys)

        # Calculate the sum from the count and mean
        df['sum'] = df['count'] * df['mean']

        # Correct the count and sum statistics for polygons intersecting multiple overview cells 
        # (otherwise they would be counted multiple times)
        df_multiple = []
        for i, row in self.df_intersected_T.iterrows():
            #composite_key = row[self.composite_key_col]
            filepath = row['filepath']
            lst_filter_id = row[self.id_col]
            df_tmp = pandas.read_parquet(
                filepath, 
                columns=[self.composite_key_col, self.id_col, numeric_layer] + timestamp_keys # + self.dimension_keys
            )  
            df_tmp = df_tmp[df_tmp[self.id_col].isin(lst_filter_id)]
            df_multiple.append(df_tmp)

        df_multiple = pandas.concat(df_multiple).reset_index(drop=True)
        df_multiple = df_multiple.set_index(self.composite_key_col)

        for pyramid_level in self.pyramid_levels:
            cols = self._pyramid_groupby_cols(pyramid_level)

            # Naively aggregate count, sum, min, max
            df_naive = []
            if verbose:
                print('pyramid grouping columns', cols)
            df_naive.append(df[cols + ['count']].groupby(cols).sum())
            df_naive.append(df[cols + ['sum']].groupby(cols).sum())
            df_naive.append(df[cols + ['min']].groupby(cols).min())
            df_naive.append(df[cols + ['max']].groupby(cols).max())
            df_naive = pandas.concat(df_naive, axis=1).reset_index()

            # First group the intersecting geometries by overview cell level 
            grp = df_multiple[[self.id_col, numeric_layer]].reset_index().groupby([self.id_col, self.composite_key_col])
            df1_count = grp.count().rename(columns={numeric_layer: 'count'}).reset_index()
            df1_sum = grp.sum().rename(columns={numeric_layer: 'sum'}).reset_index()
            df1 = pandas.merge(df1_sum, df1_count, on=['geom_id', self.composite_key_col])
            df1 = df1.set_index(self.composite_key_col).join(self.df_pyramid_keys)

            # Groupby pyramid cols (level and partition keys) and individual geometries
            grp = df1[[self.id_col, 'sum'] + cols].groupby([self.id_col] + cols)
            df2_count = grp.count().rename(columns={'sum': 'count'}).reset_index()
            df2_sum = grp.sum().reset_index()
            df2 = pandas.merge(df2_count, df2_sum, on=[self.id_col] + cols)
            df2['overcount'] = df2['count'] - 1
            df2['oversum'] = df2['overcount'] * df2['sum'] / df2['count']
            df2 = df2.groupby(cols)[['overcount', 'oversum']].sum().reset_index()
            df_corrected = pandas.merge(df_naive, df2, on=cols, how='left').fillna(0)
            df_corrected['count'] = df_corrected['count'] - df_corrected['overcount']
            df_corrected['sum'] = df_corrected['sum'] - df_corrected['oversum']
            df_corrected['mean'] = df_corrected['sum'] / df_corrected['count']
            #del df_corrected['sum']
            df_corrected['count'] = df_corrected['count'].astype(int)
            df_corrected = df_corrected[cols + ['count', 'mean', 'min', 'max']].sort_values(by=cols).set_index(cols)

            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + numeric_layer + '.parquet')
            df_corrected.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
            
    def _histogram_pyramids(self, categorical_layer, verbose=False):
        if verbose:
            print('histogram_layer', categorical_layer)
            
        # Get the histogram at the overview level
        df = self.read_overview_histogram(categorical_layer)
        df = df.T
        histogram_categories = df.columns

        # Get the temporal (and dimension) keys from the overview key
        df.index.name = self.composite_key_col
        df = df.reset_index()
        self._decompose_composite_keys(df)
        #del df[self.spatial_level_col]
        #del df[self.spatial_key_col]
        df = df.set_index(self.composite_key_col)

        df = df.join(self.df_pyramid_keys)

        for pyramid_level in self.pyramid_levels:
            # Naively aggregate the counts for each historgram category
            df1 = []
            cols = self._pyramid_groupby_cols(pyramid_level)
            if verbose:
                print('pyramid grouping columns', cols)
            for category in histogram_categories:
                df1.append(df[cols + [category]].groupby(cols).sum())
            df1 = pandas.concat(df1, axis=1)

            # Correct the statistics for polygons intersecting multiple overview cells 
            # (otherwise they would be counted multiple times)
            df3 = []
            for i, row in self.df_intersected_T.iterrows():
                #composite_key = row[self.composite_key_col]
                filepath = row['filepath']
                lst_filter_id = row[self.id_col]
                df2 = pandas.read_parquet(
                    filepath, 
                    columns=[self.composite_key_col, self.id_col, categorical_layer] + self.temporal_keys + self.dimension_keys
                )
                df2 = df2[df2[self.id_col].isin(lst_filter_id)]
                df3.append(df2)

            if len(df3)>0:
                df3 = pandas.concat(df3).reset_index(drop=True)
                df3 = df3.set_index(self.composite_key_col)
                df3 = df3.join(self.df_pyramid_keys)

                # Groupby id and pyramid level together to find relevant overcounts
                df4 = []
                df4.append(df3[cols + [self.id_col, categorical_layer]].groupby([self.id_col] + cols).count().rename(
                    columns={categorical_layer: 'count'}
                ))
                df4.append(df3[cols + [self.id_col, categorical_layer]].groupby([self.id_col] + cols).first().rename(
                    columns={categorical_layer: 'category'}
                ))

                df4 = pandas.concat(df4, axis=1)

                # Prepare for summing up by pyramid key
                df4 = df4[df4['count']>1]
                df4 = df4.reset_index()
                df4['overcount'] = df4['count']-1

                # Sum up the overcount and oversum by pyramid level key
                df5 = df4[cols + ['category', 'overcount']].groupby(cols + ['category']).sum()

                # Join with the original statistics and correct them
                df1.columns.name = 'category'
                df1_stacked = pandas.DataFrame(df1.stack().rename('count'))
                df6 = df1_stacked.join(df5)
                df6['count'] = df6['count'] - df6['overcount'].fillna(0)
                df6 = df6[['count']]
                df6 = df6.unstack()['count'].astype(int)
                df6.columns.name = None
                #df6 = df6.reset_index()
            else:
                df6=df1

            # Integer or float value column names are not allowed in parquet
            df6.columns = [str(c) for c in df6.columns]
            
            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_histogram_' + categorical_layer + '.parquet')
            df6.to_parquet(path=filepath, engine='pyarrow', compression='snappy')            
            
    def _categorical_pyramids(self, categorical_layer, verbose=False):
        if verbose:
            print('categorical_layer', categorical_layer)
        for pyramid_level in self.pyramid_levels:
            cols = self._pyramid_groupby_cols(pyramid_level)
            if verbose:
                print('pyramid grouping columns', cols)
            filepath_histogram = os.path.join(
                self.overview_directory, 
                'pyramid_level' + str(pyramid_level) + '_histogram_' + categorical_layer + '.parquet'
            )
            try:
                df = pandas.read_parquet(filepath_histogram)
            except:
                # Generate histogram pyramid first
                print('WARNING: (Re-)generating histogram needed for categorical pyramid')
                self._histogram_pyramids(categorical_layer, verbose=verbose)
                df = pandas.read_parquet(filepath_histogram)

            df.columns.name = 'categories'
            df1 = df[[]].copy()
            df1['count'] = df.sum(axis=1)
            df_stacked = df[df!=0].stack().reset_index()
            df1['unique'] = df_stacked.groupby(cols).count()['categories']
            df1['top'] = df.idxmax(axis=1)
            df1['freq'] = df_stacked.groupby(cols).max()[0].astype(int)
            df1.columns.name = None

            # Save to parquet file
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + categorical_layer + '.parquet')
            df1.to_parquet(path=filepath, engine='pyarrow', compression='snappy')
                
    def _create_pyramid_columns(self, gdf):
        self.pyramid_levels = list(range(self.overview_level + 1))
        for pyramid_level in self.pyramid_levels:
            levelsUp = self.overview_level - pyramid_level
            getParentKey_part = partial(pairs_quadtree.getParentKey, levelsUp=levelsUp)
            gdf['pyramid_level' + str(pyramid_level)] = gdf[self.spatial_key_col].apply(getParentKey_part)
            
    def pyramid_keys(self):
        self.df_pyramid_keys = self.gdf_meta.copy()
        self._create_pyramid_columns(self.df_pyramid_keys)
        self.df_pyramid_keys = self.df_pyramid_keys[
            [self.composite_key_col] + ['pyramid_level' + str(l) for l in self.pyramid_levels]
        ].set_index(self.composite_key_col)
                
    def calc_pyramids(self, verbose=False):
        if verbose:
            print('Calculating pyramids')
            
        # Generate the pyramid keys
        self.pyramid_keys()
        
        # Find the geometries intersecting overview cells
        self.df_intersected = self.read_intersected_geometry()

        # Transpose df_intersected so we can make quick filter queries using the id_col
        self.df_intersected_T = self.df_intersected.set_index(self.id_col)[self.composite_key_col].explode().reset_index()
        self.df_intersected_T = self.df_intersected_T.groupby(self.composite_key_col, group_keys=False)[self.id_col].apply(list).reset_index()
        self.df_intersected_T = pandas.merge(self.df_intersected_T, self.gdf_meta[[self.composite_key_col, 'filepath']], on=self.composite_key_col)
        
        for numeric_layer in self.numeric_layers:
            self._numeric_pyramids(numeric_layer, verbose=verbose)
            
        for categorical_layer in self.categorical_layers:
            self._histogram_pyramids(categorical_layer, verbose=verbose)
            self._categorical_pyramids(categorical_layer, verbose=verbose)
            
        # Update the vectorstore settings
        #self.write_overviewstore_settings()
        
    def read_pyramid(self, pyramid_level, layer, histogram=False, **kwargs):
        if histogram:
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_histogram_' + layer + '.parquet')
        else:
            filepath = os.path.join(self.overview_directory, 'pyramid_level' + str(pyramid_level) + '_layer_' + layer + '.parquet')
        try:
            # If needed, transpose after reading the parquet instead of in the parquet file itself, 
            # because parquet has limitations in storing mixed datatypes in columns.
            df = pandas.read_parquet(filepath, **kwargs)
        except Exception as e:
            print(e)
            df = pandas.DataFrame()
        return df
    
