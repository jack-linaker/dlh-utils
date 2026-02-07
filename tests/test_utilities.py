import datetime as dt
import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.testing import assertDataFrameEqual

from dlh_utils.utilities import (
    chunk_list,
    describe_metrics,
    pandas_to_spark,
    regex_match,
    search_files,
    value_counts,
)


class TestChunkList:
    def test_expected(self) -> None:
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = chunk_list(data, 4)
        assert result == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]


class TestDescribeMetrics:
    def test_expected(self, spark: SparkSession) -> None:
        input_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "colA": ["A", "A", "B", None, "C", "C", "C", None],
                    "colB": [None, 1, 2, 3, 4, 5, 6, 7],
                }
            )
        )
        expected = pd.DataFrame(
            {
                "variable": ["colA", "colB"],
                "type": ["string", "double"],
                "count": [8, 8],
                "distinct": [3, 8],
                "percent_distinct": [37.5, 100],
                "null": [2, 1],
                "percent_null": [25, 12.5],
                "not_null": [6, 7],
                "percent_not_null": [75, 87.5],
            }
        )
        actual = describe_metrics(spark, input_df, output_mode="pandas")
        assert_frame_equal(actual, expected)


class TestPandasToSpark:
    def test_expected(self, spark: SparkSession) -> None:
        pandas_df = pd.DataFrame(
            {
                "colDate": ["19000101"],
                "colInt": [1],
                "colBigInt": [1],
                "colFloat": [1.0],
                "colBigFloat": [1.0],
                "colString": ["hello"],
            }
        )
        pandas_df["colDate"] = pandas_df["colDate"].astype("datetime64[ns]")
        pandas_df["colInt"] = pandas_df["colInt"].astype("int32")
        pandas_df["colBigInt"] = pandas_df["colInt"].astype("int64")
        pandas_df["colFloat"] = pandas_df["colFloat"].astype("float32")
        pandas_df["colBigFloat"] = pandas_df["colBigFloat"].astype("float64")
        expected_schema = StructType(
            [
                StructField("colDate", TimestampType()),
                StructField("colInt", LongType()),
                StructField("colBigInt", LongType()),
                StructField("colFloat", DoubleType()),
                StructField("colBigFloat", DoubleType()),
                StructField("colString", StringType()),
            ]
        )
        date_val = dt.datetime(1900, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
        expected_data: list[list[dt.datetime | float | str]] = [
            [date_val, 1, 1, 1.0, 1.0, "hello"],
        ]
        expected = spark.createDataFrame(expected_data, expected_schema)
        actual = pandas_to_spark(spark, pandas_df)
        assertDataFrameEqual(actual, expected)


class TestRegexMatch:
    def test_expected(self, spark: SparkSession) -> None:
        df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "colA": ["abc123_hello", "", None],
                    "colB": [" abc123_hello", "", None],
                    "colC": ["123_hello", "", None],
                    "colD": ["abc_hello", "", None],
                    "colE": ["abc123hello", "", None],
                    "colF": ["abc123", "", None],
                    "colG": ["abc123_hello", "abc123_hello", "abc123_hello"],
                }
            )
        )
        regex = "^[a-z]*[0-9]+_"
        result = regex_match(df, regex, limit=10000, cut_off=0.0)
        assert result == ["colA", "colC", "colG"]
        result = regex_match(df, regex, limit=10000, cut_off=0.6)
        assert result == ["colG"]


class TestSearchFiles:
    def test_expected(self) -> None:
        path = os.path.dirname(os.path.realpath(__file__))
        result = search_files(path, "import")
        assert sorted(result.keys()) == sorted(
            [
                "test_formatting.py",
                "test_linkage.py",
                "test_profiling.py",
                "test_standardisation.py",
                "test_dataframes.py",
                "test_flags.py",
                "test_utilities.py",
                "conftest.py",
            ]
        )


class TestValueCounts:
    def test_expected(self, spark: SparkSession) -> None:
        df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "colA": ["A", "A", "B", None, "C", "C", "C", None],
                    "colB": [None, 1, 2, 3, 3, 5, 5, 6],
                }
            )
        )
        result_df = value_counts(spark, df, limit=6, output_mode="pandas")
        intended_df = pd.DataFrame(
            {
                "colA": ["C", None, "A", "B", "", ""],
                "colA_count": [3, 2, 2, 1, 0, 0],
                "colB": [3.0, 5.0, 1.0, 2.0, 6.0, np.nan],
                "colB_count": [2, 2, 1, 1, 1, 1],
            }
        )
        assert_frame_equal(result_df, intended_df)
