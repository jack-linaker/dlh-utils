"""Profiling functions used to produce summaries of data."""

import re
from typing import Literal

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import IntegerType, StringType

from dlh_utils import dataframes as da
from dlh_utils import utilities as ut


def create_table_statements(
    database: str,
    regex: str | None = None,
    output_mode: Literal["pandas", "spark"] = "spark",
) -> DataFrame | pd.DataFrame:
    """Summarise Hive CREATE TABLE statement into a DataFrame.

    Returns a dataframe summarising the SQL CREATE TABLE statement that
    creates a named table within a Hive database. Can use regular
    expressions to filter for given tables.

    This gives detailed information on table creation parameters,
    including: - All variables in the table. - Their corresponding data
    types. - The time and date of the table's creation.

    Parameters
    ----------
    database : str
        Database to be queried.
    regex : str, optional
        Regex pattern to match Hive tables against. Defaults to None.
    output_mode : typing.Literal["pandas", "spark"], optional
        Whether to output a Pandas or Spark DataFrame. Defaults to
        "spark".

    Returns
    -------
    pyspark.sql.DataFrame | pandas.DataFrame
        DataFrame summarising the CREATE TABLE statement.

    """
    spark = SparkSession.builder.getOrCreate()

    tables = ut.list_tables(database)

    if regex is not None:
        tables = list(filter(re.compile(regex).match, tables))

    if output_mode == "pandas":
        out = pd.concat(
            [
                (
                    spark.sql(f"SHOW CREATE TABLE {database}.{table}")
                    .withColumn("table", F.lit(table))
                    .toPandas()
                )
                for table in tables
            ]
        )[["table", "createtab_stmt"]]

    if output_mode == "spark":
        out = spark.createDataFrame(out)

    return out


def df_describe(
    df: DataFrame,
    output_mode: Literal["pandas", "spark"] = "pandas",
    approx_distinct: bool = False,
    rsd: float = 0.05,
) -> DataFrame | pd.DataFrame:
    """Profile DataFrame variables: type, length, distinctness, nulls.

    Produces a DataFrame containing descriptive metrics on each variable
    within a specified DataFrame, including: - The variable type. - The
    maximum and minimum value length. - The maximum and minimum lengths
    before/after decimal places for float variables. - The number and
    percentage of distinct values. - The number and percentage of null
    and non-null values.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Data to be summarised.
    output_mode : typing.Literal["pandas", "spark"], optional
        The output DataFrame format. Defaults to "pandas".
    approx_distinct : bool, optional
        Whether to return approximate distinct counts of values in the
        data. Used to improve performance of the function. Defaults to
        False.
    rsd : float, optional
        Maximum relative standard deviation allowed for
        approxcountdistinct. Note: for rsd < 0.01, it is more efficient
        to set approx_distinct to False. Defaults to 0.05.

    Returns
    -------
    pyspark.sql.DataFrame | pandas.DataFrame
        Pandas or Spark DataFrame summarising the data being queried.

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

    >>> df_describe(df, output_mode="spark", approx_distinct=False, rsd=0.05).show()
    +-----------+------+---------+--------+------------------+----+
    |   variable|  type|row_count|distinct|  percent_distinct|null|
    +-----------+------+---------+--------+------------------+----+
    |         ID|string|        6|       5| 83.33333333333334|   0|
    |   Forename|string|        6|       5| 83.33333333333334|   0|
    |Middle_name|string|        6|       4| 66.66666666666666|   1|
    |    Surname|string|        6|       1|16.666666666666664|   0|
    |        DoB|string|        6|       5| 83.33333333333334|   0|
    |        Sex|string|        6|       2| 33.33333333333333|   0|
    |   Postcode|string|        6|       1|16.666666666666664|   0|
    +-----------+------+---------+--------+------------------+----+ ... output truncated
    """
    spark = SparkSession.builder.getOrCreate()

    types = df.dtypes
    types = dict(zip([x[0] for x in types], [x[1] for x in types]))

    for col in [k for k, v in types.items() if v in ["timestamp", "date", "boolean"]]:
        df = df.withColumn(col, F.col(col).cast(StringType()))

    count = df.count()

    if approx_distinct:
        distinct_df = df.agg(
            *(F.approxCountDistinct(F.col(c), rsd).alias(c) for c in df.columns)
        ).withColumn("summary", F.lit("distinct"))

    else:
        distinct_df = df.agg(
            *(F.countDistinct(F.col(c)).alias(c) for c in df.columns)
        ).withColumn("summary", F.lit("distinct"))

    empty_df = df.agg(
        *(F.count(F.when(F.col(c).rlike("^\s*$"), c)).alias(c) for c in df.columns)
    ).withColumn("summary", F.lit("empty"))

    max_l_df = df.agg(
        *(F.max(F.length(F.col(c))).alias(c) for c in df.columns)
    ).withColumn("summary", F.lit("max_l"))

    min_l_df = df.agg(
        *(F.min(F.length(F.col(c))).alias(c) for c in df.columns)
    ).withColumn("summary", F.lit("min_l"))

    point_variables = [
        x for x in df.columns if types[x] == "double" or types[x].startswith("decimal")
    ]

    if len(point_variables) != 0:
        max_l_bp_df = df.agg(
            *(
                F.max(F.length(F.col(c).cast(IntegerType()))).alias(c)
                for c in point_variables
            )
        ).withColumn("summary", F.lit("max_l_before_point"))

        min_l_bp_df = df.agg(
            *(
                F.min(F.length(F.col(c).cast(IntegerType()))).alias(c)
                for c in point_variables
            )
        ).withColumn("summary", F.lit("min_l_before_point"))

        max_l_ap_df = df.agg(
            *(
                F.max(F.length(F.reverse(F.col(c)).cast(IntegerType()))).alias(c)
                for c in point_variables
            )
        ).withColumn("summary", F.lit("max_l_after_point"))

        min_l_ap_df = df.agg(
            *(
                F.min(F.length(F.reverse(F.col(c)).cast(IntegerType()))).alias(c)
                for c in point_variables
            )
        ).withColumn("summary", F.lit("min_l_after_point"))

    else:
        max_l_bp_df = spark.createDataFrame(
            (pd.DataFrame({"summary": ["max_l_before_point"]}))
        )

        min_l_bp_df = spark.createDataFrame(
            (pd.DataFrame({"summary": ["min_l_before_point"]}))
        )

        max_l_ap_df = spark.createDataFrame(
            (pd.DataFrame({"summary": ["max_l_after_point"]}))
        )

        min_l_ap_df = spark.createDataFrame(
            (pd.DataFrame({"summary": ["min_l_after_point"]}))
        )

    describe_df = da.union_all(
        df.describe(),
        distinct_df,
        # null_df,
        empty_df,
        max_l_df,
        min_l_df,
        max_l_bp_df,
        min_l_bp_df,
        max_l_ap_df,
        min_l_ap_df,
    ).toPandas()

    describe_df = describe_df.transpose().reset_index()
    describe_df.columns = ["variable"] + list(
        describe_df[describe_df["index"] == "summary"]
        .reset_index(drop=True)
        .transpose()[0]
    )[1:]
    describe_df = describe_df[describe_df["variable"] != "summary"]

    describe_df = describe_df.rename(columns={"count": "not_null"})

    describe_df["row_count"] = count
    describe_df["null"] = describe_df["row_count"] - describe_df["not_null"].astype(int)
    for variable in [
        "distinct",
        "null",
        "not_null",
        "empty",
    ]:
        describe_df["percent_" + variable] = (
            describe_df[variable].astype(int) / describe_df["row_count"]
        ) * 100
    describe_df["type"] = [types[x] for x in describe_df["variable"]]

    describe_df = describe_df.reset_index(drop=True)
    describe_df["min"] = [
        y if describe_df["type"][x] != "string" else None
        for x, y in enumerate(describe_df["min"])
    ]

    describe_df = describe_df.reset_index(drop=True)
    describe_df["max"] = [
        y if describe_df["type"][x] != "string" else None
        for x, y in enumerate(describe_df["max"])
    ]

    describe_df = describe_df[
        [
            "variable",
            "type",
            "row_count",
            "distinct",
            "percent_distinct",
            "null",
            "percent_null",
            "not_null",
            "percent_not_null",
            "empty",
            "percent_empty",
            "min",
            "max",
            "min_l",
            "max_l",
            "max_l_before_point",
            "min_l_before_point",
            "max_l_after_point",
            "min_l_after_point",
        ]
    ]

    if output_mode == "spark":
        describe_df = ut.pandas_to_spark(describe_df)

    return describe_df


def value_counts(
    df: DataFrame, limit: int = 20, output_mode: Literal["pandas", "spark"] = "pandas"
) -> tuple[DataFrame, DataFrame]:
    """Summarise top and bottom distinct value counts per DataFrame.

    Produces DataFrames summarising the top and bottom distinct value
    counts within a supplied spark DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Data to be summarised.
    limit : int, optional
        The top n values to be summarised. Defaults to 20.
    output_mode : typing.Literal["pandas", "spark"], optional
        The output DataFrame format. Defaults to "pandas".

    Returns
    -------
    tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]
        A DataFrame summarising the top n distinct values within data
        and a DataFrame summarising the bottom n distinct values within
        data.

    Examples
    --------
    >>> high_values = value_counts(df, limit=20, output_mode="pandas")[0]
    >>> high_values
    Year_Of_Birth  Year_Of_Birth_count  Cluster_Number  Cluster_Number_count
    -9             16                   248             5
    1944           14                   205             4
    1947           14                   292             4
    1954           12                   373             4

    >>> low_values = value_counts(df, limit=20, output_mode="pandas")[1]
    >>> low_values
    Year_Of_Birth  Year_Of_Birth_count  Cluster_Number  Cluster_Number_count
    2008           1                    262             2
    2001           1                    353             2
    1986           2                    325             2
    """
    spark = SparkSession.builder.getOrCreate()

    def value_count(df, col, limit):
        grouped = (
            (
                df.select(col)
                .dropna()
                .withColumn("count", F.count(col).over(Window.partitionBy(col)))
            )
            .dropDuplicates()
            .persist()
        )

        high = (
            grouped.sort("count", ascending=False)
            .limit(limit)
            .withColumnRenamed("count", col + "_count")
            .toPandas()
        )

        low = (
            grouped.sort("count", ascending=True)
            .limit(limit)
            .withColumnRenamed("count", col + "_count")
            .toPandas()
        )

        grouped.unpersist()

        return (high, low)

    dfs = [value_count(df, col, limit) for col in df.columns]
    high = [x[0] for x in dfs]
    low = [x[1] for x in dfs]

    def make_limit(df, limit):
        count = df.shape[0]

        if count < limit:
            dif = limit - count

            dif_df = pd.DataFrame({0: [""] * dif, 1: [0] * dif})[[0, 1]]

            dif_df.columns = list(df)

            df = df.append(dif_df).reset_index(drop=True)

        return df

    high = [make_limit(df, limit) for df in high]
    high = pd.concat(high, axis=1)

    low = [make_limit(df, limit) for df in low]
    low = pd.concat(low, axis=1)

    if output_mode == "spark":
        high = ut.pandas_to_spark(high)

        low = ut.pandas_to_spark(low)

    return high, low


def hive_dtypes(database: str, table: str) -> list:
    """Return list of variables and data types for a Hive table.

    Returns a list of variables and their corresponding data types
    within a hive table.

    Parameters
    ----------
    df : str
        Database to query for a given table.
    table : str
        The name of the Hive table to be summarised.

    Returns
    -------
    list
        A list of tuples containing each variable and its data type.
    """
    spark = SparkSession.builder.getOrCreate()

    df = spark.sql(f"DESCRIBE {database}.{table}").toPandas()
    df = df[["col_name", "data_type"]]
    return list(df.to_records(index=False))


def hive_variable_matrix(
    database: str,
    regex: str | None = None,
    output_mode: Literal["pandas", "spark"] = "spark",
) -> DataFrame | pd.DataFrame:
    """List variables and their tables in a Hive database as DataFrame.

    Returns a DataFrame detailing all variables and their corresponding
    tables within a queried Hive database.

    Parameters
    ----------
    database : str
        Database to query.
    regex : str, optional
        Regex pattern to match Hive tables against. Defaults to None.
    output_mode : typing.Literal["pandas", "spark"], optional
        Type of DataFrame to return the variable matrix in. Defaults to
        "spark".

    Returns
    -------
    pyspark.sql.DataFrame | pandas.DataFrame
        A DataFrame containing the variable matrix.
    """
    spark = SparkSession.builder.getOrCreate()

    tables = ut.list_tables(database)

    if regex is not None:
        tables = list(filter(re.compile(regex).match, tables))

    variable_types = [(table, hive_dtypes(database, table)) for table in tables]

    all_variables = list(
        {
            y[0]
            for y in [
                item for sublist in [x[1] for x in variable_types] for item in sublist
            ]
        }
    )

    out = pd.DataFrame({"variable": all_variables})

    for table, dtype in variable_types:
        df = pd.DataFrame(
            {"variable": [x[0] for x in dtype], table: [x[1] for x in dtype]}
        )

        out = (out.merge(df, on="variable", how="left")).reset_index(drop=True)

    out = (out.fillna("").sort_values("variable")).reset_index(drop=True)

    if output_mode == "spark":
        out = ut.pandas_to_spark(out)

    return out
