"""Profiling functions used to produce summaries of data."""

import re
from typing import Any, Literal

import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import StringType

from dlh_utils import utilities


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
    output_mode : {"pandas", "spark"}, optional
        Whether to output a Pandas or Spark DataFrame. Defaults to
        "spark".

    Returns
    -------
    pyspark.sql.DataFrame | pandas.DataFrame
        DataFrame summarising the CREATE TABLE statement.

    """
    spark = SparkSession.builder.getOrCreate()

    tables = utilities.list_tables(database)

    if regex is not None:
        tables = list(filter(re.compile(regex).match, tables))

    if output_mode == "pandas":
        out = pd.concat(
            [
                (
                    spark.sql(f"SHOW CREATE TABLE {database}.{table}")
                    .withColumn("table", sf.lit(table))
                    .toPandas()
                )
                for table in tables
            ]
        )[["table", "createtab_stmt"]]

    if output_mode == "spark":
        out = spark.createDataFrame(out)

    return out


def df_describe(
    spark: SparkSession,
    df: DataFrame,
    *,
    output_mode: Literal["pandas", "spark"] = "pandas",
    rsd: float = 0.05,
    approx_distinct: bool = False,
) -> DataFrame | pd.DataFrame:
    """Profile DataFrame variables: type, length, distinctness, nulls.

    Produces a DataFrame containing descriptive metrics on each variable
    within a specified DataFrame, including:

    - The variable type.
    - The maximum and minimum value length.
    - The maximum and minimum lengths before/after decimal places for
      float variables.
    - The number and percentage of distinct values.
    - The number and percentage of null and non-null values.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Data to be summarised.
    output_mode : {"pandas", "spark"}, optional
        The output DataFrame format. Defaults to "pandas".
    approx_distinct : bool, optional
        Whether to return approximate distinct counts of values in the
        data. Used to improve performance of the function. Defaults to
        False.
    rsd : float, optional
        Maximum relative standard deviation allowed for approx_distinct.
        Note: for rsd < 0.01, it is more efficient to set
        approx_distinct to False. Defaults to 0.05.

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
    |  5|  Maggie|       NULL|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+-----------+-------+----------+---+--------+

    >>> df_describe(df, output_mode="spark").show()
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
    # Map of column -> Spark type name.
    types = dict(df.dtypes)

    # Cast timestamps/dates/booleans to str so length/min/max works
    # consistently.
    for col, t in list(types.items()):
        if t in ("timestamp", "date", "boolean"):
            df = df.withColumn(col, sf.col(col).cast(StringType()))

    row_count = df.count()

    # Build aggregations: we compute one big agg and then pull values
    # out.
    agg_exprs = []
    for c in df.columns:
        # Distinct.
        if approx_distinct:
            agg_exprs.append(
                sf.approxCountDistinct(sf.col(c), rsd).alias(f"{c}_distinct")
            )
        else:
            agg_exprs.append(sf.countDistinct(sf.col(c)).alias(f"{c}_distinct"))

        # Not null (Spark's count treats NaN as non-null).
        agg_exprs.append(sf.count(sf.col(c)).alias(f"{c}_not_null"))

        # Empty: trim and compare to empty string.
        agg_exprs.append(
            sf.count(
                sf.when(sf.trim(sf.col(c).cast(StringType())) == "", sf.col(c))
            ).alias(f"{c}_empty")
        )

        agg_exprs.append(sf.min(sf.col(c)).alias(f"{c}_min"))
        agg_exprs.append(sf.max(sf.col(c)).alias(f"{c}_max"))

        # Lengths computed from string representation (handles NaN by
        # producing 'NaN' length).
        col_str = sf.col(c).cast(StringType())
        agg_exprs.append(sf.max(sf.length(col_str)).alias(f"{c}_max_l"))
        agg_exprs.append(sf.min(sf.length(col_str)).alias(f"{c}_min_l"))

        # For point types (double/decimal) compute lengths before/after
        # decimal point.
        t = types.get(c, "")
        if t == "double" or t.startswith("decimal"):
            col_str = sf.col(c).cast(StringType())
            before = sf.when(
                sf.col(c).isNull() | sf.isnan(sf.col(c)), sf.lit(None)
            ).otherwise(sf.substring_index(col_str, ".", 1))
            after = sf.when(
                sf.col(c).isNull() | sf.isnan(sf.col(c)), sf.lit("")
            ).otherwise(
                sf.when(
                    sf.col(c).contains("."), sf.substring_index(col_str, ".", -1)
                ).otherwise(sf.lit(""))
            )
            agg_exprs.append(sf.max(sf.length(before)).alias(f"{c}_max_l_bp"))
            agg_exprs.append(sf.min(sf.length(before)).alias(f"{c}_min_l_bp"))
            agg_exprs.append(sf.max(sf.length(after)).alias(f"{c}_max_l_ap"))
            agg_exprs.append(sf.min(sf.length(after)).alias(f"{c}_min_l_ap"))
        else:
            # Keep placeholders (they'll be filled as None later).
            agg_exprs.append(sf.lit(None).alias(f"{c}_max_l_bp"))
            agg_exprs.append(sf.lit(None).alias(f"{c}_min_l_bp"))
            agg_exprs.append(sf.lit(None).alias(f"{c}_max_l_ap"))
            agg_exprs.append(sf.lit(None).alias(f"{c}_min_l_ap"))

    # Run the aggregated query and pull results into a dict.
    agg_row = df.agg(*agg_exprs).collect()[0].asDict()

    # Build per-column result rows.
    rows = []
    for c in df.columns:
        distinct_v = agg_row.get(f"{c}_distinct")
        not_null_v = agg_row.get(f"{c}_not_null")
        empty_v = agg_row.get(f"{c}_empty")
        min_v = agg_row.get(f"{c}_min")
        max_v = agg_row.get(f"{c}_max")
        max_l_v = agg_row.get(f"{c}_max_l")
        min_l_v = agg_row.get(f"{c}_min_l")
        max_l_bp_v = agg_row.get(f"{c}_max_l_bp")
        min_l_bp_v = agg_row.get(f"{c}_min_l_bp")
        max_l_ap_v = agg_row.get(f"{c}_max_l_ap")
        min_l_ap_v = agg_row.get(f"{c}_min_l_ap")

        row = {
            "variable": c,
            "type": types.get(c, ""),
            "row_count": int(row_count),
            "distinct": None if distinct_v is None else int(distinct_v),
            "percent_distinct": (int(distinct_v) / row_count) * 100
            if distinct_v is not None
            else None,
            "null": row_count - int(not_null_v) if not_null_v is not None else None,
            "percent_null": ((row_count - int(not_null_v)) / row_count) * 100
            if not_null_v is not None
            else None,
            "not_null": None if not_null_v is None else int(not_null_v),
            "percent_not_null": (int(not_null_v) / row_count) * 100
            if not_null_v is not None
            else None,
            "empty": None if empty_v is None else int(empty_v),
            "percent_empty": (int(empty_v) / row_count) * 100
            if empty_v is not None
            else None,
            "min": None if types.get(c) == "string" else min_v,
            "max": None if types.get(c) == "string" else max_v,
            "min_l": None if min_l_v is None else int(min_l_v),
            "max_l": None if max_l_v is None else int(max_l_v),
            "max_l_before_point": None if max_l_bp_v is None else int(max_l_bp_v),
            "min_l_before_point": None if min_l_bp_v is None else int(min_l_bp_v),
            "max_l_after_point": None if max_l_ap_v is None else int(max_l_ap_v),
            "min_l_after_point": None if min_l_ap_v is None else int(min_l_ap_v),
        }
        rows.append(row)

    describe_df = pd.DataFrame(rows)

    if output_mode == "spark":
        describe_df = spark.createDataFrame(describe_df)

    return describe_df


def value_counts(
    spark: SparkSession,
    df: DataFrame,
    limit: int = 20,
    output_mode: Literal["pandas", "spark"] = "pandas",
) -> tuple[pd.DataFrame | DataFrame, pd.DataFrame | DataFrame]:
    """Summarise top and bottom distinct value counts per DataFrame.

    Produces DataFrames summarising the top and bottom distinct value
    counts within a supplied spark DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Data to be summarised.
    limit : int, optional
        The top n values to be summarised. Defaults to 20.
    output_mode : {"pandas", "spark"}, optional
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
    SparkSession.builder.getOrCreate()

    def value_count(
        df: DataFrame, col: str, limit: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        grouped = (
            (
                df.select(col)
                .dropna()
                .withColumn("count", sf.count(col).over(Window.partitionBy(col)))
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

    def make_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
        count = df.shape[0]

        if count < limit:
            dif = limit - count

            dif_df = pd.DataFrame({0: [""] * dif, 1: [0] * dif})[[0, 1]]

            dif_df.columns = list(df)

            df = pd.concat([df, dif_df]).reset_index(drop=True)

        return df

    high = [make_limit(df, limit) for df in high]
    high = pd.concat(high, axis=1)

    low = [make_limit(df, limit) for df in low]
    low = pd.concat(low, axis=1)

    if output_mode == "spark":
        high = spark.createDataFrame(high)
        low = spark.createDataFrame(low)

    return high, low


def hive_dtypes(database: str, table: str) -> list[Any]:
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
    spark: SparkSession,
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
    output_mode : {"pandas", "spark"}, optional
        Type of DataFrame to return the variable matrix in. Defaults to
        "spark".

    Returns
    -------
    pyspark.sql.DataFrame | pandas.DataFrame
        A DataFrame containing the variable matrix.
    """
    SparkSession.builder.getOrCreate()

    tables = utilities.list_tables(database)

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
        out = spark.createDataFrame(out)

    return out
