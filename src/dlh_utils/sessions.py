"""Functions used to create and start different sized spark sessions."""

import os
from typing import Literal

import graphframes_jars as graphframes
from IPython.display import HTML, display
from pyspark.sql import SparkSession


def get_or_create_spark_session(
    app_name: str = "DE_DL",
    size: Literal["small", "medium", "large", "extra_large", "custom"] = "large",
    show_console_progress: Literal["false", "true"] = "false",
    shuffle_partitions: int = 200,
    default_parallelism: int = 200,
    memory: str = "10g",
    memory_overhead: str = "1g",
    cores: int = 5,
    max_executors: int = 5,
) -> SparkSession:
    """Start a Spark session (size/custom) and show Spark UI link.

    Starts spark session dependent on size category specified or starts
    custom session on specified parameters. Also generates Spark UI link
    in console for monitoring session progress/resource use.

    Parameters
    ----------
    app_name : str, optional
        The name of the spark session. Defaults to "DE_DL".
    size : {"small", "medium", "large", "extra_large", "custom"}, optional
        The size category of session to be started. Defaults to "large".
    show_console_progress : {"false", "true"}, optional
        Option to display UI metrics in console. Defaults to "false".
    shuffle_partitions : int, optional
        The default number of partitions to be used in repartitioning.
        Defaults to 200.
    default_parallelism : int, optional
        Default number of partitions in resilient distributed datasets
        (RDDs) returned by transformations like join, reduceByKey, and
        parallelize when no shufflePartition number is set. Defaults to
        200.
    memory : str, optional
        Executor memory allocation. Defaults to "10g".
    memory_overhead : str, optional
        The amount of off-heap memory to be allocated per driver in
        cluster mode. Defaults to "1g".
    cores : int, optional
        The number of cores to use on each executor. Defaults to 5.
    max_executors : int, optional
        Upper bound for the number of executors. Defaults to 5.

    Returns
    -------
    pyspark.sql.SparkSession
        Also displays a Spark UI web link in the workbench console.
    """
    # Obtain spark UI URL parameters.
    url = (
        "spark-"
        + str(os.environ["CDSW_ENGINE_ID"])
        + "."
        + str(os.environ["CDSW_DOMAIN"])
    )
    display(HTML(f"<a href=http://{url}s>Spark UI</a>"))

    try:
        # Get graphframes jar path to configure session with.
        graphframes_path = graphframes.__file__
        graphframes_path = graphframes_path.rsplit("/", 1)[0]

        for file in os.listdir(graphframes_path):
            if file.endswith(".jar"):
                # Get the latest jar file.
                jar_path = os.path.join(graphframes_path, file)

    except FileNotFoundError:
        print(
            "graphframes wrapper package not found. Please install this to use the "
            "`cluster_number` function."
        )
        jar_path = None

    if size == "small":
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.executor.memory", "1g")
            .config("spark.executor.cores", 1)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 3)
            .config("spark.sql.shuffle.partitions", 12)
            .config("spark.jars", jar_path)
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.ui.showConsoleProgress", show_console_progress)
            .enableHiveSupport()
            .getOrCreate()
        )
    elif size == "medium":
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.executor.memory", "6g")
            .config("spark.executor.cores", 3)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 3)
            .config("spark.sql.shuffle.partitions", 18)
            .config("spark.jars", jar_path)
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.ui.showConsoleProgress", show_console_progress)
            .enableHiveSupport()
            .getOrCreate()
        )
    elif size == "large":
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.executor.memory", "10g")
            .config("spark.yarn.executor.memoryOverhead", "1g")
            .config("spark.executor.cores", 5)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 5)
            .config("spark.jars", jar_path)
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.ui.showConsoleProgress", show_console_progress)
            .enableHiveSupport()
            .getOrCreate()
        )
    elif size == "extra_large":
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.executor.memory", "20g")
            .config("spark.yarn.executor.memoryOverhead", "2g")
            .config("spark.executor.cores", 5)
            .config("spark.dynamicAllocation.enabled", "true")
            .config("spark.dynamicAllocation.maxExecutors", 12)
            .config("spark.jars", jar_path)
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.ui.showConsoleProgress", show_console_progress)
            .enableHiveSupport()
            .getOrCreate()
        )
    elif size == "custom":
        spark = (
            SparkSession.builder.appName(app_name)
            .config("spark.executor.memory", memory)
            .config("spark.executor.memoryOverhead", memory_overhead)
            .config("spark.executor.cores", cores)
            .config("spark.dynamicAllocation.maxExecutors", max_executors)
            .config("spark.sql.shuffle.partitions", shuffle_partitions)
            .config("spark.default.parallelism", default_parallelism)
            .config("spark.ui.showConsoleProgress", show_console_progress)
            .config("spark.jars", jar_path)
            .config("spark.sql.repl.eagerEval.enabled", "true")
            .config("spark.shuffle.service.enabled", "true")
            .config("spark.dynamicAllocation.enabled", "true")
            .getOrCreate()
        )

    return spark
