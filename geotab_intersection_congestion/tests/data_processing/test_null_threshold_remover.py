import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark_test import assert_pyspark_df_equal
from geotab_intersection_congestion.data_processing.transformers import \
    NullThresholdRemover


@pytest.fixture(scope='module')
def spark():
    return SparkSession.builder.appName('geotab-testing').getOrCreate()


def test_null_threshold_remover(spark):

    mock_df_schema = StructType([StructField("Latitude", FloatType(), True),
                                 StructField("Longitude", FloatType(),
                                             True),
                                 StructField('EntryHeading', StringType(),
                                             True),
                                 StructField('Path', StringType(), True),
                                 StructField('Month', IntegerType(), True)])
    mock_df_data = [[33.75094, None, None, None, 6],
                    [33.75094, None, None, 'Peachtree Street', 6],
                    [33.75094, -84.393032, None, 'Peachtree Street', None],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', None],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, None, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', None]]
    mock_df = spark.createDataFrame(mock_df_data, schema=mock_df_schema)
    expected_df_schema = StructType([StructField("Latitude", FloatType(),
                                                 True),
                                 StructField('Path', StringType(), True)])
    expected_df_data = [[33.75094, None],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street'],
                        [33.75094, 'Peachtree Street']]
    expected_df = spark.createDataFrame(expected_df_data,
                                        schema=expected_df_schema)
    null_remove = NullThresholdRemover(inputCols=mock_df.columns,
                                       threshold=0.2)
    transformed_df = null_remove.transform(mock_df)
    assert_pyspark_df_equal(expected_df, transformed_df)

def test_no_input_cols(spark):
    mock_df_schema = StructType([StructField("Latitude", FloatType(), True),
                                 StructField("Longitude", FloatType(),
                                             True),
                                 StructField('EntryHeading', StringType(),
                                             True),
                                 StructField('Path', StringType(), True),
                                 StructField('Month', IntegerType(), True)])
    mock_df_data = [[33.75094, None, None, None, 6],
                    [33.75094, None, None, 'Peachtree Street', 6],
                    [33.75094, -84.393032, None, 'Peachtree Street', None],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', None],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, None, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', None]]
    mock_df = spark.createDataFrame(mock_df_data, schema=mock_df_schema)
    null_remove = NullThresholdRemover(inputCols=[],
                                       threshold=0.2)
    transformed_df = null_remove.transform(mock_df)
    assert_pyspark_df_equal(mock_df, transformed_df)

def test_invalid_threshold(spark):
    mock_df_schema = StructType([StructField("Latitude", FloatType(), True),
                                 StructField("Longitude", FloatType(),
                                             True),
                                 StructField('EntryHeading', StringType(),
                                             True),
                                 StructField('Path', StringType(), True),
                                 StructField('Month', IntegerType(), True)])
    mock_df_data = [[33.75094, None, None, None, 6],
                    [33.75094, None, None, 'Peachtree Street', 6],
                    [33.75094, -84.393032, None, 'Peachtree Street', None],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', None],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, None, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', 6],
                    [33.75094, -84.393032, 'NE', 'Peachtree Street', None]]
    mock_df = spark.createDataFrame(mock_df_data, schema=mock_df_schema)

    with pytest.raises(ValueError):
        null_remove = NullThresholdRemover(inputCols=mock_df.columns,
                                           threshold=1.2)
        transformed_df = null_remove.transform(mock_df)