"""Import required libraries."""

import tempfile
from collections.abc import Iterator

import pytest
from pyspark.sql import SparkSession

GRAPHFRAMES_PACKAGE = "io.graphframes:graphframes-spark4_2.13:0.10.0"


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Set up the Spark session by using a fixture decorator.

    This will be passed to all the tests by having spark as an input.
    Being able to define the Spark session in this way is one advantage
    of Pytest over unittest.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        spark = (
            SparkSession.builder.master("local")
            .config("spark.checkpoint.dir", tmp_dir)
            .config("spark.jars.packages", GRAPHFRAMES_PACKAGE)
            .getOrCreate()
        )
    yield spark
    spark.stop()
