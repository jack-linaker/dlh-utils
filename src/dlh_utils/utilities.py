"""Utility functions used to ease difficulty in querying databases.

Also used to produce descriptive metrics about a DataFrame.
"""

import os
import re
import subprocess
import warnings
from typing import Any, Literal

import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession
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

from dlh_utils import dataframes


def chunk_list(_list: list[Any], _num: int) -> list[Any]:
    """Split a list into a specified number of chunks.

    Parameters
    ----------
    _list : list[Any]
        A list.
    _num : int
        The number of chunks.

    Returns
    -------
    list[Any]
    """
    return [
        _list[i * _num : (i + 1) * _num] for i in range((len(_list) + _num - 1) // _num)
    ]


def clone_hive_table(
    database: str, table_name: str, new_table: str, suffix: str = ""
) -> None:
    """Duplicate Hive table.

    Parameters
    ----------
    database : str
        Name of database.
    table_name : str
        Name of table being cloned.
    new_table : str
        Name of cloned table.
    suffix : str, optional
        String appended to table name. Defaults to "".
    """
    spark = SparkSession.builder.getOrCreate()

    spark.sql(
        f"CREATE TABLE {database}.{new_table}{suffix} AS SELECT * FROM "
        f"{database}.{table_name}"
    )


def create_hive_table(df: DataFrame, database: str, table_name: str) -> None:
    """Create Hive table from dataframe.

    Saves all information within a dataframe into a Hive table.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe being saved as a Hive table.
    database : str
        Name of database Hive table is being saved to.
    table_name : str
        Name of table df is being named to.
    """
    spark = SparkSession.builder.getOrCreate()

    df.createOrReplaceTempView("tempTable")
    spark.sql(f"CREATE TABLE {database}.{table_name} AS SELECT * FROM tempTable")


def describe_metrics(
    spark: SparkSession,
    df: DataFrame,
    output_mode: Literal["pandas", "spark"] = "pandas",
) -> pd.DataFrame | DataFrame:
    """Summarise variable metrics.

    Used to describe information about variables within a dataframe,
    including:

    - type
    - count
    - distinct value count
    - percentage of distinct values
    - null count
    - percentage of null values
    - non-null value count
    - percentage of non-null values

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to produce descriptive metrics about.
    output_mode : {"pandas", "spark"}, optional
        The type of DataFrame to return. Defaults to "pandas".

    Returns
    -------
    pandas.DataFrame | pyspark.sql.DataFrame
        A DataFrame with columns detailing descriptive metrics on each
        variable.

    Examples
    --------
    >>> describe_metrics(df, output_mode="spark").show()
    +-----------+------+-----+--------+----------------+----+------------+--------+----------------+
    |   variable|  type|count|distinct|percent_distinct|null|percent_null|not_null|percent_not_null|
    +-----------+------+-----+--------+----------------+----+------------+--------+----------------+
    |         ID|string|    6|       5| 83.333333333334|   0|         0.0|       6|           100.0|
    |   Forename|string|    6|       5| 83.333333333334|   0|         0.0|       6|           100.0|
    |Middle_name|string|    6|       4| 66.666666666666|   1|16.666666664|       5| 83.333333333334|
    |    Surname|string|    6|       1|16.6666666666664|   0|         0.0|       6|           100.0|
    |        DoB|string|    6|       5| 83.333333333334|   0|         0.0|       6|           100.0|
    |        Sex|string|    6|       2| 33.333333333333|   0|         0.0|       6|           100.0|
    |   Postcode|string|    6|       1|16.6666666666664|   0|         0.0|       6|           100.0|
    +-----------+------+-----+--------+----------------+----+------------+--------+----------------+
    """
    distinct_df = df.agg(
        *(sf.countDistinct(sf.col(c)).alias(c) for c in df.columns)
    ).withColumn("summary", sf.lit("distinct"))

    exprs = []
    for col in df.schema:
        col_name = col.name
        c = sf.col(col_name)
        if isinstance(col.dataType, (FloatType, DoubleType)):
            is_missing = sf.isnan(c) | c.isNull()
        else:
            is_missing = c.isNull()
        exprs.append(
            sf.sum(sf.when(is_missing, 1).otherwise(0)).cast("long").alias(col_name)
        )
    null_df = df.agg(*exprs).withColumn("summary", sf.lit("null"))

    describe_df = dataframes.union_all(distinct_df, null_df).persist()

    count = df.count()

    types = df.dtypes
    types = dict(zip([x[0] for x in types], [x[1] for x in types], strict=False))

    describe_df = describe_df.toPandas()
    describe_df = describe_df.transpose().reset_index()
    describe_df.columns = ["variable"] + list(
        describe_df[describe_df["index"] == "summary"]
        .reset_index(drop=True)
        .transpose()[0]
    )[1:]
    describe_df = describe_df[describe_df["variable"] != "summary"]
    describe_df["count"] = count
    describe_df["not_null"] = describe_df["count"] - describe_df["null"]
    for variable in ["distinct", "null", "not_null"]:
        describe_df["percent_" + variable] = (
            describe_df[variable] / describe_df["count"]
        ) * 100
    describe_df["type"] = [types[x] for x in describe_df["variable"]]

    describe_df = describe_df[
        [
            "variable",
            "type",
            "count",
            "distinct",
            "percent_distinct",
            "null",
            "percent_null",
            "not_null",
            "percent_not_null",
        ]
    ]

    describe_df["count"] = describe_df["count"].astype(int)
    describe_df["distinct"] = describe_df["distinct"].astype(int)
    describe_df["null"] = describe_df["null"].astype(int)
    describe_df["not_null"] = describe_df["not_null"].astype(int)
    describe_df["percent_null"] = describe_df["percent_null"].astype(float)
    describe_df["percent_not_null"] = describe_df["percent_not_null"].astype(float)
    describe_df["percent_distinct"] = describe_df["percent_distinct"].astype(float)

    if output_mode == "spark":
        describe_df = spark.createDataFrame(describe_df)

    return describe_df


def drop_hive_table(database: str, table_name: str) -> None:
    """Delete Hive table from Hive if it exists.

    Parameters
    ----------
    database : str
      Name of database.
    table_name : str
      Name of table.
    """
    spark = SparkSession.builder.getOrCreate()

    spark.sql(f"DROP TABLE IF EXISTS {database}.{table_name}")


def list_checkpoints(checkpoint: str) -> list[str]:
    """List checkpoints in HDFS directory.

    Parameters
    ----------
    checkpoint : str
        String path of checkpoint directory.

    Returns
    -------
    list[str]
        List of files in checkpoint directory.

    Examples
    --------
    >>> list_checkpoints(checkpoint="/user/edwara5/checkpoints")
    ['hdfs://prod1/user/checkpoints/0299d46e-96ad-4d3a-9908-c99b9c6a7509/connected-components-985ca288']
    """
    return list_files(list_files(checkpoint, walk=False)[0])


def list_files(
    file_path: str,
    regex: str | None = None,
    *,
    walk: bool = False,
    full_path: bool = True,
) -> list[str]:
    """List files in HDFS directory.

    Lists files in a given HDFS directory, and/or all of that
    directory's subfolders if specified

    Parameters
    ----------
    file_path : str
        String path of directory.
    regex : str, optional
        Use regex to find certain words within the listed files.
        Defaults to None.
    walk : bool, optional
        When False, lists files only in immediate directory specified.
        When True, lists all files in immediate directory and all
        subfolders. Defaults to False.
    full_path : bool, optional
        When True, show full file path. When False, show just files.
        Defaults to True.

    Returns
    -------
    list[str]
        List of file names.
    """
    list_of_filenames = []

    if walk is True:
        process = subprocess.Popen(
            ["hadoop", "fs", "-ls", "-R", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        process = subprocess.Popen(
            ["hadoop", "fs", "-ls", "-C", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    std_out, _std_error = process.communicate()

    try:
        std_out = str(std_out).split("\\n")[:-1]
        std_out[0] = std_out[0].strip("b'")

        if full_path is True:
            for i in std_out:
                file_name = str(i).split(" ")[-1]
                list_of_filenames.append(file_name)

        elif full_path is False:
            for i in std_out:
                file_name = str(i).split("/")[-1]
                list_of_filenames.append(file_name)

        if regex is not None:
            list_of_filenames = list(
                filter(re.compile(regex).search, list_of_filenames)
            )
    except:
        print("no files in this directory")
        list_of_filenames = []

    return list_of_filenames


def list_tables(database: str) -> list[str]:
    """Return list of tables in a Hive database.

    Returns the tables in a database from hive, it takes an argument of
    the database name as a string. It then returns a dataframe listing
    the tables within the database given.

    Parameters
    ----------
    database : str
        String name of database.

    Returns
    -------
    list[str]
        List of tables in database.

    Examples
    --------
    >>> list_tables("baby_names")
    ['baby_names_boy_raw', 'baby_names_boy_std', 'baby_names_girl_raw', 'baby_names_girl_std', 'bv_girl_names_raw', 'bv_girl_names_std']
    """
    spark = SparkSession.builder.getOrCreate()

    df = spark.sql(f"SHOW TABLES IN {database}")
    return list((df.select("tableName").toPandas())["tableName"])


def most_recent(
    path: str, filetype: Literal["csv", "parquet", "hive"], regex: str | None = None
) -> tuple[str, Literal["csv", "parquet", "hive"]]:
    """Return the most recently edited `filetype` in `path`.

    Returns most recently edited Hive table or directory containing most
    recently edited csv/parquet file(s) in location or database.

    Parameters
    ----------
    path : str
        The path or database which will be searched.
    filetype : {"csv", "parquet", "hive"}
        The format of data that is to be searched for.
    regex : str, optional
        A regular expression to filter the search results by, eg "^VOA".
        Defaults to None.

    Returns
    -------
    tuple[str, {"csv", "parquet", "hive"}]
        The filepath or table reference for most recent data, and the
        format of the data for which a filepath has been returned.

    Raises
    ------
    FileNotFoundError if search query does not exist in HDFS.

    Examples
    --------
    >>> most_recent(path="baby_names", filetype="hive", regex=None)
    ('baby_names.bv_girl_names_raw', 'hive')

    >>> most_recent(path="baby_names", filetype="hive", regex="std$")
    ('baby_names.bv_girl_names_std', 'hive')
    """
    # Pass spark context to function.
    spark = SparkSession.builder.getOrCreate()

    if regex is None:
        if filetype == "hive":
            try:
                # List all tables in directory.
                tables = spark.sql(f"SHOW TABLES IN {path}").select("tableName")

                # Create full filepath from directory & table name.
                filepaths = tables.withColumn(
                    "path", sf.concat(sf.lit(path), sf.lit("."), sf.col("tableName"))
                )

                # Convert to list.
                filepaths = list(filepaths.select("path").toPandas()["path"])

                # Initialise empty dictionary.
                filepath_dict = {}

                # Loop through paths, appending path and time to
                # dictionary.
                for filepath in filepaths:
                    time = spark.sql(
                        f"SHOW tblproperties {filepath} ('transient_lastDdlTime')"
                    ).collect()[0][0]

                    filepath_dict.update({filepath: time})

                # Sort by max time since epoch and return corresponding
                # path.
                most_recent_filepath = max(filepath_dict, key=filepath_dict.get)

            except Exception as exc:
                raise FileNotFoundError(
                    filetype + " file not found in this directory: " + path
                ) from exc

        # If filetype != hive.
        else:
            # Return all files in dir recursively, sorted by
            # modification date (ascending), decode from bytes-like to
            # str.
            files = subprocess.check_output(
                ["hdfs", "dfs", "-ls", "-R", "-t", "-C", path]
            ).decode()

            # Split by newline to return list of old -> new files.
            files = files.split("\n")

            if filetype == "csv":
                try:
                    # Filter for .csv ext and take last element of list.
                    result = [f for f in files if f.endswith("csv")][-1]

                    # Return path up until last '/'.
                    most_recent_filepath = re.search(r".*\/", result).group(0)

                except Exception as exc:
                    raise FileNotFoundError(
                        filetype + " file not found in this directory: " + path
                    ) from exc

            elif filetype == "parquet":
                try:
                    # Filter for .csv ext and take last element of list.
                    result = [f for f in files if f.endswith("parquet")][-1]

                    # Return path up until last '/'.
                    most_recent_filepath = re.search(r".*\/", result).group(0)

                except Exception as exc:
                    raise FileNotFoundError(
                        filetype + " file not found in this directory: " + path
                    ) from exc

    # If regex argument specified.
    else:
        if filetype == "hive":
            try:
                # List all tables in directory.
                tables = spark.sql(f"SHOW TABLES IN {path}").select("tableName")

                # Create full filepath from directory & table name.
                filepaths = tables.withColumn(
                    "path", sf.concat(sf.lit(path), sf.lit("."), sf.col("tableName"))
                )

                # Filter filepaths based on regex.
                filtered_filepaths = filepaths.filter(filepaths["path"].rlike(regex))

                # Convert to list.
                filtered_filepaths = list(
                    filtered_filepaths.select("path").toPandas()["path"]
                )

                # Initialise empty dictionary.
                filepath_dict = {}

                # Loop through paths, appending path and time to dict.
                for filepath in filtered_filepaths:
                    time = spark.sql(
                        f"SHOW tblproperties {filepath} ('transient_lastDdlTime')"
                    ).collect()[0][0]

                    filepath_dict.update({filepath: time})

                # Sort by max time since epoch and return corresponding
                # path.
                most_recent_filepath = max(filepath_dict, key=filepath_dict.get)

            except Exception as exc:
                raise FileNotFoundError(
                    filetype
                    + " file, matching this regular expression: "
                    + regex
                    + " not found in this directory: "
                    + path
                ) from exc

        # If filetype != hive.
        else:
            # Return all files in dir recursively, sorted by
            # modification date (ascending), decode from bytes-like to
            # str.
            files = subprocess.check_output(
                ["hdfs", "dfs", "-ls", "-R", "-t", "-C", path]
            ).decode()

            # Split by newline to return list of old -> new files.
            files = files.split("\n")

            r = re.compile(regex)

            # APply regex filter.
            filtered_files = list(filter(r.match, files))

            if filetype == "csv":
                try:
                    # Filter for .csv ext and take last element of list.
                    result = [f for f in filtered_files if f.endswith("csv")][-1]

                    # Return path up until last '/'.
                    most_recent_filepath = re.search(R".*\/", result).group(0)

                except Exception as exc:
                    raise FileNotFoundError(
                        filetype
                        + " file, matching this regular expression: "
                        + regex
                        + " not found in this directory: "
                        + path
                    ) from exc

            elif filetype == "parquet":
                try:
                    # Filter for .csv ext and take last element of list.
                    result = [f for f in filtered_files if f.endswith("parquet")][-1]

                    # Return path up until last '/'.
                    most_recent_filepath = re.search(R".*\/", result).group(0)

                except Exception as exc:
                    raise FileNotFoundError(
                        filetype
                        + " file, matching this regular expression: "
                        + regex
                        + " not found in this directory: "
                        + path
                    ) from exc

    return most_recent_filepath, filetype


def pandas_to_spark(spark: SparkSession, pandas_df: pd.DataFrame) -> DataFrame:
    """Create a Spark DataFrame from a given pandas DataFrame.

    .. deprecated:: 0.4.2
       Use `pyspark.sql.SparkSession.createDataFrame` instead.

    Parameters
    ----------
    df : pandas.DataFrame
        Pandas DataFrame being converted.

    Returns
    -------
    pyspark.sql.DataFrame
        A Spark DataFrame.
    """
    warnings.warn(
        "`pandas_to_spark` is deprecated. Use "
        "`pyspark.sql.SparkSession.createDataFrame` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return spark.createDataFrame(pandas_df)


def read_format(
    read: Literal["csv", "parquet", "hive"],
    path: str | None = None,
    file_name: str | None = None,
    sep: str = ",",
    header: Literal["true", "false"] = "true",
    infer_schema: Literal["True", "False"] = "True",
) -> DataFrame:
    """Read DataFrame from specified format.

    Can read from HDFS in csv or parquet format and from database hive
    table format.

    Parameters
    ----------
    read : {"csv", "parquet", "hive"}
        The format from which data is to be read.
    path : str, optional
        The path or database from which DataFrame is to be read.
        Defaults to None.
    file_name : str, optional
        The file or table name from which DataFrame is to be read. Note
        that if None, function. will read from HDFS path specified in
        case of csv or parquet. Defaults to None.
    sep : str, optional
        Specified separator for data in csv format. Defaults to ",".
    header : {"true", "false"}, optional
        Boolean indicating whether or not data will be read to include a
        header. Defaults to "true".
    infer_schema : {"True", "False"}, optional
        Boolean indicating whether data should be read with inferred
        data types and schema. If false, all data will read as string
        format. Defaults to "True".

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame of data read from specified path and format.

    Examples
    --------
    >>> df = read_format(
    ...     read="parquet",
    ...     path="/user/edwara5/simpsons.parquet",
    ...     file_name=None,
    ...     header="true",
    ...     infer_schema="True",
    ... )
    >>> df.show()
    +---+--------+-----------+-------+----------+---+--------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+--------+-----------+-------+----------+---+--------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ZZ99 9SZ|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ZZ99 5GB|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ZZ99 2SP|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ZZ99 2FA|
    +---+--------+-----------+-------+----------+---+--------+
    """
    spark = SparkSession.builder.getOrCreate()
    if file_name is None:
        if read == "csv":
            df = (
                spark.read.format("csv")
                .option("sep", sep)
                .option("header", header)
                .option("inferSchema", infer_schema)
                .load(f"{path}")
            )
        if read == "parquet":
            df = (
                spark.read.format("parquet")
                .option("header", header)
                .option("inferSchema", infer_schema)
                .load(f"{path}")
            )
        if read == "hive":
            df = spark.sql(f"SELECT * FROM {path}")

    else:
        if read == "csv":
            df = (
                spark.read.format("csv")
                .option("sep", sep)
                .option("header", header)
                .option("inferSchema", infer_schema)
                .load(f"{path}/{file_name}")
            )
        if read == "parquet":
            df = (
                spark.read.format("parquet")
                .option("header", header)
                .option("inferSchema", infer_schema)
                .load(f"{path}/{file_name}")
            )
        if read == "hive":
            df = spark.sql(f"SELECT * FROM {path}.{file_name}")

    return df


def regex_match(
    df: DataFrame, regex: str, limit: int = 10000, cut_off: float = 0.75
) -> list[str]:
    r"""Find columns matching regex pattern across DataFrame rows.

    Returns a list of columns, for an input DataFrame, that match a
    specified regex pattern.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame being searched for a text pattern.
    regex : str
        Regex pattern to match against.
    limit : int, optional
        Number of rows from DataFrame to search for a text pattern.
        Defaults to 10000.
    cut_off : float, optional
        The minimum rate of matching values in a column for it to be
        considered a regex match. Defaults to 0.75.

    Returns
    -------
    list[str]
        A list of all columns matching specified regex pattern.

    Examples
    --------
    >>> df.show()
    +---+--------+-----------+-------+----------+---+--------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+--------+-----------+-------+----------+---+--------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+-----------+-------+----------+---+--------+
    >>> regex_match(df, regex="([A-Z])\w+", limit=5, cut_off=0.75)
    ['Forename', 'Middle_name', 'Surname', 'Postcode']
    """
    sample_df = (df.limit(limit)).persist()

    row_count = sample_df.count()

    counts_df = sample_df.groupBy().agg(
        *[
            sf.sum(sf.when(sf.col(col).rlike(regex), 1)).alias(col)
            for col in sample_df.columns
        ]
    )

    counts_df = (
        counts_df.toPandas()
        .transpose()
        .dropna()
        .reset_index()
        .rename(
            columns={
                "index": "variable",
                0: "count",
            }
        )
    )

    counts_df["match_rate"] = counts_df["count"] / row_count

    counts_df = counts_df[counts_df["match_rate"] >= cut_off].reset_index(drop=True)

    sample_df.unpersist()

    return list(counts_df["variable"])


def rename_hive_table(database: str, table_name: str, new_name: str) -> None:
    """Rename Hive table.

    Parameters
    ----------
    database : str
        Name of database.
    table_name : str
        Name of table being renamed.
    new_name : str
        Name of new table.
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"ALTER TABLE {database}.{table_name} RENAME TO {database}.{new_name}")


def search_files(path: str, string: str) -> dict[str, list[int]]:
    """Find occurrences of a string in files and return file: lines.

    Finds file and line number(s) of specified string within a specified
    file path.

    Parameters
    ----------
    path : str
        Path directory for which the search function is applied to.
    string : str
        String value that is searched within the files of the directory
        given.

    Returns
    -------
    dict[str, list[int]]
        Dictionary with keys of file names containing the string and
        values of line numbers indicating where there is a match on the
        string.

    Examples
    --------
    >>> search_files(path="/home/cdsw/random_stuff", string="Homer")
    >>> {"simpsons.csv": [2]}
    """
    files_in_dir = os.listdir(path)
    diction = {}  # try empty dictionary

    for file in files_in_dir:
        count = 0
        count_list = []

        try:
            with open(f"{path}/{file}") as f:
                datafile = f.readlines()

            for line in datafile:
                count = count + 1

                if string in line:
                    count_list.append(count)

            if len(count_list) != 0:
                diction[file] = count_list
        except IsADirectoryError:
            continue
        except UnicodeDecodeError:
            continue

    return diction


def value_counts(
    spark: SparkSession,
    df: DataFrame,
    limit: int = 20,
    output_mode: Literal["pandas", "spark"] = "pandas",
) -> DataFrame:
    """Count the most common values in all columns of a DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to produce summary counts from.
    limit : int, optional
        The top n values to search for. Defaults to 20.
    output_mode : {"pandas", "spark"}, optional
        The type of DataFrame to return. Defaults to "pandas".

    Returns
    -------
    pyspark.sql.DataFrame
        A DataFrame with original DataFrame columns and a count of their
        most common values.

    Examples
    --------
    >>> value_counts(df, limit=5, output_mode="spark").show()
    +---+--------+--------+--------------+-----------+-----------------+-------+-------------+
    | ID|ID_count|Forename|Forename_count|Middle_name|Middle_name_count|Surname|Surname_count|
    +---+--------+--------+--------------+-----------+-----------------+-------+-------------+
    |  3|       2|    Bart|             2|      Jo-Jo|                2|Simpson|            6|
    |  5|       1|   Homer|             1|       NULL|                1|       |            0|
    |  1|       1|   Marge|             1|     Juliet|                1|       |            0|
    |  4|       1|  Maggie|             1|      Marie|                1|       |            0|
    |  2|       1|    Lisa|             1|        Jay|                1|       |            0|
    +---+--------+--------+--------------+-----------+-----------------+-------+-------------+
    """

    def value_count(df: DataFrame, col: str, limit: int) -> pd.DataFrame:
        return (
            df.groupBy(col)
            .count()
            .sort(["count", col], ascending=[False, True])
            .limit(limit)
            .withColumnRenamed("count", col + "_count")
            .toPandas()
        )

    dfs = [value_count(df, col, limit) for col in df.columns]

    def make_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
        count = df.shape[0]

        if count < limit:
            dif = limit - count

            dif_df = pd.DataFrame({0: [""] * dif, 1: [0] * dif})[[0, 1]]

            dif_df.columns = list(df)

            df = pd.concat([df, dif_df]).reset_index(drop=True)

        return df

    dfs = [make_limit(df, limit) for df in dfs]

    df = pd.concat(dfs, axis=1)

    if output_mode == "spark":
        df = spark.createDataFrame(df)

    return df


def write_format(
    df: DataFrame,
    write: Literal["csv", "parquet", "hive"],
    path: str,
    file_name: str | None = None,
    sep: str = ",",
    header: Literal["true", "false"] = "true",
    mode: Literal["overwrite", "append"] = "overwrite",
) -> None:
    """Write DataFrame in specified format.

    Can write data to HDFS in csv or parquet format and to database in
    hive table format.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe to be written.
    write : {"csv", "parquet", "hive"}
        The format in which data is to be written.
    path : str
        The path or database to which dataframe is to be written.
    file_name : str, optional
        The file or table name under which dataframe is to be saved.
        Note that if None, function will write to the HDFS path
        specified in case of csv or parquet. Defaults to None.
    sep : str, optional
        Specified separator for data in csv format. Defaults to ","
    header : {"true", "false"}, optional
        Boolean indicating whether or not data will include a header.
        Defaults to "true".
    mode : {"overwrite", "append"}, optional
        Choice to overwrite existing file or table or to append new data
        into it. Defaults to "overwrite".

    Returns
    -------
    file or table
        Written version of DataFrame in specified format.

    Examples
    --------
    >>> write_format(
    ...     df, write="parquet", path="user/edwara5/simpsons.parquet", mode="overwrite"
    ... )
    """
    if file_name is None:
        if write == "csv":
            df.write.format("csv").option("header", header).mode(mode).option(
                "sep", sep
            ).save(f"{path}")
        if write == "parquet":
            df.write.parquet(path=f"{path}", mode=mode)
        if write == "hive":
            df.write.mode("overwrite").saveAsTable(f"{path}")

    else:
        if write == "csv":
            df.write.format("csv").option("header", header).mode(mode).option(
                "sep", sep
            ).save(f"{path}/{file_name}")
        if write == "parquet":
            df.write.parquet(path=f"{path}/{file_name}", mode=mode)
        if write == "hive":
            df.write.mode("overwrite").saveAsTable(f"{path}.{file_name}")
