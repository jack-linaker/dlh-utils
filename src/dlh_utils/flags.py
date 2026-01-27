"""Flag functions used to highlight anomalous values within data."""

from functools import reduce
from operator import add
from typing import Any, Literal

import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType


def flag(
    df: DataFrame,
    ref_col: str,
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "isNull", "isNotNull", "regex"]
    | None = None,
    condition_value: Any = None,
    condition_col: Any = None,
    alias: str | None = None,
    prefix: str = "FLAG",
    *,
    fill_null: bool | None = None,
) -> DataFrame:
    """Add boolean flag column from conditions for quality checks.

    Adds True or False flags to supplied DataFrame that can then be used
    for quality checks.

    Conditions can be set in comparison to columns or specific values
    (e.g. == column, ==1).  Conditions covered are equals, not equals,
    greater/less than, is/is not null, and regex. Optional TRUE/FALSE
    fill for null outputs of comparison. Designed for use in conjunction
    with flag_summary() and flag_check() functions.

    This function creates a column filled with TRUE or FALSE values
    based on whether a condition has been met.

    NOTE: If an alias is not specified, a Flag column name is
    automatically generated.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame the function is applied to.
    ref_col : str
        The column title that the conditions are performing checks upon.
    condition : typing.Literal["==", "!=", ">", ">=", "<", "<=", "isNull", "isNotNull", "regex"], optional
        Conditional statements used to compare values to the ref_col.
        Defaults to None.
    condition_value : Any, optional
        The value the ``ref_col`` is being compared against. Defaults to
        None.
    condition_col : Any, optional
        Comparison column for flag condition. Defaults to None.
    alias : str, optional
        Alias for flag column. Defaults to None.
    prefix : str, optional
        Default alias flag column prefix. Defaults to "FLAG".
    fill_null : bool, optional
        True or False fill where condition operations return null.
        Defaults to None.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with additional window column.

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

    >>> flag(
            df,
            ref_col="Middle_name",
            condition="isNotNull",
            condition_value=None,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        ).show()
    +---+--------+-----------+-------+----------+---+--------+------------------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|FLAG_Middle_nameisNotNull|
    +---+--------+-----------+-------+----------+---+--------+------------------------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|                    true|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|                    true|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                    true|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                    true|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|                    true|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|                   false|
    +---+--------+-----------+-------+----------+---+--------+------------------------+
    """
    if alias is None and condition_value is not None:
        alias_value = str(condition_value)
        alias = f"{prefix}_{ref_col}{condition}{alias_value}"

    if alias is None and condition_col is not None:
        alias = f"{prefix}_{ref_col}{condition}_{condition_col}"

    if alias is None and condition_col is None and condition_value is None:
        alias = f"{prefix}_{ref_col}{condition}"

    if condition == "==" and condition_col is not None:
        df = df.withColumn(alias, F.col(ref_col) == F.col(condition_col))

    if condition == "==" and condition_col is None:
        df = df.withColumn(alias, F.col(ref_col) == condition_value)

    if condition == ">" and condition_col is not None:
        df = df.withColumn(alias, F.col(ref_col) > F.col(condition_col))

    if condition == ">" and condition_col is None:
        df = df.withColumn(alias, F.col(ref_col) > condition_value)

    if condition == ">=" and condition_col is not None:
        df = df.withColumn(alias, F.col(ref_col) >= F.col(condition_col))

    if condition == ">=" and condition_col is None:
        df = df.withColumn(alias, F.col(ref_col) >= condition_value)

    if condition == "<" and condition_col is not None:
        df = df.withColumn(alias, F.col(ref_col) < F.col(condition_col))

    if condition == "<" and condition_col is None:
        df = df.withColumn(alias, F.col(ref_col) < condition_value)

    if condition == "<=" and condition_col is not None:
        df = df.withColumn(alias, F.col(ref_col) <= F.col(condition_col))

    if condition == "<=" and condition_col is None:
        df = df.withColumn(alias, F.col(ref_col) <= condition_value)

    if condition == "!=" and condition_col is not None:
        df = df.withColumn(alias, F.col(ref_col) != F.col(condition_col))

    if condition == "!=" and condition_col is None:
        df = df.withColumn(alias, F.col(ref_col) != condition_value)

    if condition == "isNull":
        df = df.withColumn(alias, (F.col(ref_col).isNull()) | (F.isnan(F.col(ref_col))))

    if condition == "isNotNull":
        df = df.withColumn(
            alias, (F.col(ref_col).isNotNull()) & ~(F.isnan(F.col(ref_col)))
        )

    if condition == "regex":
        df = df.withColumn(alias, (F.col(ref_col).rlike(condition_value)))

    if fill_null is not None:
        df = df.withColumn(
            alias, F.when(F.col(alias).isNull(), fill_null).otherwise(F.col(alias))
        )

    return df


def flag_summary(
    df: DataFrame, flags: str | list[str] | None = None, *, pandas: bool = False
) -> DataFrame | pd.DataFrame:
    """Produce summary table of boolean flag columns.

    Produces a summary of True/False counts and percentages. Option to
    output as pandas or spark DataFrame (default spark).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame the function is applied to.
    flags : str | list[str], optional
        A boolean flag column title in the format of a string or a list
        of strings of boolean flag column titles. Defaults to None.
    pandas : bool, optional
        Option to output as a pandas DataFrame. Defaults to False.

    Returns
    -------
    pyspark.sql.DataFrame | pandas.DataFrame
        DataFrame with additional window column.

    Examples
    --------
    >>> df.show()
    +---+--------+-----------+-------+----------+---+--------+-------------------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|FLAG_Middle_nameisNotNull|
    +---+--------+-----------+-------+----------+---+--------+-------------------------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|                     true|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|                     true|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                     true|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                     true|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|                     true|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|                    false|
    +---+--------+-----------+-------+----------+---+--------+-------------------------+

    >>> flag_summary(df, flags=None, pandas=False).show()
    +---------------------+----+-----+----+-----------------+------------------+
    |                 flag|true|false|rows|     percent_true|     percent_false|
    +---------------------+----+-----+----+-----------------+------------------+
    |FLAG_Middle_nameis...|   5|    1|   6|83.33333333333334|16.666666666666657|
    +---------------------+----+-----+----+-----------------+------------------+
    """
    spark = SparkSession.builder.getOrCreate()

    if flags is None:
        flags = [column for column in df.columns if column.startswith("FLAG_")]

    if not isinstance(flags, list):
        flags = [flags]

    rows = df.count()

    flags_out = []

    for col in flags:
        flags_out.append((df.select(col).where(F.col(col) == F.lit(True)).count()))

    out = pd.DataFrame(
        {
            "flag": flags,
            "true": flags_out,
            "false": [rows - x for x in flags_out],
            "rows": rows,
            "percent_true": [(x / rows) * 100 for x in flags_out],
            "percent_false": [100 - ((x / rows) * 100) for x in flags_out],
        }
    )

    out = out[["flag", "true", "false", "rows", "percent_true", "percent_false"]]

    if pandas is False:
        out = spark.createDataFrame(out).coalesce(1)

    return out


def flag_check(
    df: DataFrame,
    prefix: str = "FLAG_",
    flags: list[str] | None = None,
    mode: Literal["master", "split", "pass", "fail"] = "master",
    *,
    summary: bool = False,
) -> DataFrame:
    """Read flag columns and counts True/False values.

    Adds flag count column (counting TRUE/Fail values) and overall fail
    column (TRUE/FALSE and flag TRUE/Fail). If any rows in the flag
    count column are greater than 0, the overall fail value for this row
    will be True so this is quickly highlighted to the user.

    Option to produce flag summary stats employing ``flag_summary``.
    Option to return full DataFrame, only passes, only fails, or passes
    and fails(residuals) as two separate dataframes.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame the function is applied to.
    prefix : str, optional
        For dynamic identification of flag columns if prefixed. Defaults
        to "FLAG_".
    flags : list[str], optional
        List of flag manually specified flags to operate the function
        on. If this is kept as default value, all columns in ``df`` that
        start with "FLAG_" are assumed to be flag columns by the
        function. Defaults to None.
    mode : typing.Literal["master", "split", "pass", "fail"], optional
        "master": returns all results (full DataFrame). "pass": only
        returns rows where pass is True. "fail": only returns rows where
        fail is True. "split": returns two separate DataFrames for both
        pass and fail results. Defaults to "master".
    summary : bool, optional
        Optional flag summary employing ``flag_summary`` function.
        Defaults to False.

    Returns
    -------
    pyspark.sql.DataFrame
        Returns DataFrame with results depending on the mode argument.
        If the mode argument is set to split, it will return two
        DataFrames.

    Examples
    --------
    >>> df.show()
    +---+--------+-----------+-------+---+-------------------------+
    | ID|Forename|Middle_name|Surname|Sex|FLAG_Middle_nameisNotNull|
    +---+--------+-----------+-------+---+-------------------------+
    |  1|   Homer|        Jay|Simpson|  M|                     true|
    |  2|   Marge|     Juliet|Simpson|  F|                     true|
    |  3|    Bart|      Jo-Jo|Simpson|  M|                     true|
    |  3|    Bart|      Jo-Jo|Simpson|  M|                     true|
    |  4|    Lisa|      Marie|Simpson|  F|                     true|
    |  5|  Maggie|       null|Simpson|  F|                    false|
    +---+--------+-----------+-------+---+-------------------------+

    >>> flag_check(df, prefix="FLAG_", flags=None, mode="master", summary=False).show()
    +---+--------+-----------+-------+---+-------------------------+----------+-----+
    | ID|Forename|Middle_name|Surname|Sex|FLAG_Middle_nameisNotNull|flag_count| FAIL|
    +---+--------+-----------+-------+---+-------------------------+----------+-----+
    |  1|   Homer|        Jay|Simpson|  M|                     true|         1| true|
    |  2|   Marge|     Juliet|Simpson|  F|                     true|         1| true|
    |  3|    Bart|      Jo-Jo|Simpson|  M|                     true|         1| true|
    |  3|    Bart|      Jo-Jo|Simpson|  M|                     true|         1| true|
    |  4|    Lisa|      Marie|Simpson|  F|                     true|         1| true|
    |  5|  Maggie|       null|Simpson|  F|                    false|         0|false|
    +---+--------+-----------+-------+---+-------------------------+----------+-----+

    See Also
    --------
    ``flag_summary``

    Notes
    -----
    In all instances summary will be the last component returned eg
    master, summary.
    """
    if flags is None:
        flags = [column for column in df.columns if column.startswith(prefix)]

    if len(flags) == 0:
        print(
            "No flag columns found! Please specify which flag column to summarise\
        with the flags = argument, or specify the correct prefix"
        )

    df = df.withColumn("flag_count", F.lit(0))

    for flag in flags:
        df = df.withColumn(
            "flag_count",
            F.when(F.col(flag), F.col("flag_count") + F.lit(1)).otherwise(
                F.col("flag_count")
            ),
        )

    df = df.withColumn(
        "FAIL", reduce(add, [F.col(flag).cast(IntegerType()) for flag in flags])
    )
    df = df.withColumn("FAIL", F.col("FAIL") > 0)

    if summary is True:
        summary_df = flag_summary(df, flags + ["FAIL"], pandas=False)

        if mode == "master":
            return (df, summary_df)

        if mode == "split":
            return (
                (df.where(F.col("FAIL") == F.lit(False))),
                (df.where(F.col("FAIL") == F.lit(True))),
                summary_df,
            )

        if mode == "pass":
            return (df.where(F.col("FAIL") == F.lit(False)), summary_df)

        if mode == "fail":
            return (df.where(F.col("FAIL") == F.lit(True)), summary_df)

    else:
        if mode == "master":
            return df

        if mode == "split":
            return (
                (df.where(F.col("FAIL") == F.lit(False))),
                (df.where(F.col("FAIL") == F.lit(True))),
            )

        if mode == "pass":
            return df.where(F.col("FAIL") == F.lit(False))

        if mode == "fail":
            return df.where(F.col("FAIL") == F.lit(True))
