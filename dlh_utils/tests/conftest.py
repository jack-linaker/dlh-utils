"""Import required libraries."""

import tempfile
from collections.abc import Iterator

import pyspark
import pytest
from pyspark.sql import SparkSession

GRAPHFRAMES_3 = "io.graphframes:graphframes-spark3_2.12:0.10.0"
GRAPHFRAMES_4 = "io.graphframes:graphframes-spark4_2.13:0.10.0"
PYSPARK_VERSION = pyspark.__version__


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
            .config(
                "spark.jars.packages",
                GRAPHFRAMES_3 if PYSPARK_VERSION.startswith("3.5") else GRAPHFRAMES_4,
            )
            .getOrCreate()
        )
    yield spark
    spark.stop()
