"""Functions used to modify aspects of a dataframe prior to linkage."""

import re
from typing import Any, Literal

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window

from dlh_utils import standardisation as st


def clone_column(df: DataFrame, target: str, clone: str) -> DataFrame:
    """Duplicate a DataFrame column and add it with a new name.

    Clones a column within a DataFrame and gives it a new column header.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    target : str
        The name of the column to be cloned.
    clone : str
        Name of the new column.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with column cloned.

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

    >>> clone_column(df, target="Sex", clone="Gender").show()
    +---+--------+-----------+-------+----------+---+--------+------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|Gender|
    +---+--------+-----------+-------+----------+---+--------+------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|     M|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|     F|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|     M|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|     M|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|     F|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|     F|
    +---+--------+-----------+-------+----------+---+--------+------+
    """
    df = df.withColumn(clone, F.col(target))

    return df


def coalesced(
    df: DataFrame,
    subset: list[str] | None = None,
    output_col: str = "coalesced_col",
    *,
    drop: bool = False,
) -> DataFrame:
    """Add column with each row's first non-null value.

    Produces a new column from a supplied DataFrame, that contains the
    first non-null value from each row.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    subset : list[str], optional
        Subset of columns being coalesced together into a single column,
        if subset=None then subset=[all columns in DataFrame]. Defaults
        to None.
    output_col : str, optional
        Name of the output column for results of the coalesced columns.
        Defaults to "coalesced_col".
    drop : bool, optional
        If True, the columns that were coalesced will be dropped from
        the DataFrame. Defaults to False.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with coalesced columns results appended to original
        DataFrame in the ``output_col`` column.

    Examples
    --------
    >>> df3.show()
    +----+----+
    |   a|   b|
    +----+----+
    |null|null|
    |   1|null|
    |null|   2|
    +----+----+

    >>> coalesced(df3, subset=None, output_col="coalesced_col").show()
    +----+----+-------------+
    |   a|   b|coalesced_col|
    +----+----+-------------+
    |null|null|        null |
    |   1|null|           1 |
    |null|   2|           2 |
    +----+----+-------------+

    >>> coalesced(df3, subset=None, output_col="coalesced_col", drop=True).show()
    +-------------+
    |coalesced_col|
    +-------------+
    |        null |
    |           1 |
    |           2 |
    +-------------+
    """
    if subset is None:
        subset = df.columns

    df = df.withColumn(output_col, F.coalesce(*[F.col(x) for x in subset]))

    if drop:
        df = drop_columns(df, subset=subset, drop_duplicates=False)

    return df


def concat(
    df: DataFrame, out_col: str, sep: str = " ", columns: list[str] | None = None
) -> DataFrame:
    """Concatenate column strings to new column.

    Concatenates strings from specified columns into a single string and
    stores the new string value in a new column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe to which the function is applied.
    out_col : str
        The name, in string format, of the output column for the new
        concatenated strings to be stored in.
    sep : str, optional
        This is the value used to separate the strings in the different
        columns when combining them into a single string. Defaults to "
        ".
    columns : list[str], optional
        The list of columns being concatenated into one string. Defaults
        to None.

    Returns
    -------
    pyspark.sql.DataFrame
        Returns dataframe with ``out_col`` column containing the
        concatenated string.

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

    >>> concat(
    ...     df,
    ...     out_col="Full Name",
    ...     sep=" ",
    ...     columns=["Forename", "Middle_name", "Surname"],
    ... ).show()
    +---+--------+-----------+-------+----------+---+--------+--------------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|           Full Name|
    +---+--------+-----------+-------+----------+---+--------+--------------------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|   Homer Jay Simpson|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|Marge Juliet Simpson|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|  Bart Jo-Jo Simpson|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|  Bart Jo-Jo Simpson|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|  Lisa Marie Simpson|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|      Maggie Simpson|
    +---+--------+-----------+-------+----------+---+--------+--------------------+
    """
    if columns is None:
        columns = []
    df = df.withColumn(out_col, F.concat_ws(sep, *[F.col(x) for x in columns]))

    if sep != "":
        df = (
            df.withColumn(out_col, F.regexp_replace(F.col(out_col), f"[{sep}]+", sep))
            .withColumn(
                out_col,
                F.regexp_replace(F.col(out_col), f"^[{sep}]|[{sep}]$", ""),
            )
            .withColumn(
                out_col,
                F.when(F.col(out_col).rlike("^$"), None).otherwise(F.col(out_col)),
            )
        )

    return df


def cut_off(
    df: DataFrame,
    threshold_column: str,
    val: int,
    mode: Literal["<", "<=", ">", ">="],
) -> DataFrame:
    """Cut off rows that do not meet certain thresholds.

    Takes a DataFrame column and a cutoff value and returns the
    DataFrame with rows in which the mode and cutoff value condition is
    met. Will also work for date values if the ``threshold_column`` is a
    timestamp.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    threshold_column : str
        Column to which the cutoff values are being applied.
    val : int
        Value against which the mode operation is checking
        ``threshold_column`` values.
    mode : typing.Literal["<", "<=", ">", ">="]
        Operation used to cutoff values that do not meet the operation
        requirement.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with rows removed where they did not meet the cut off
        specification.

    Examples
    --------
    >>> df3.show()
    +---+---+
    |  a|  b|
    +---+---+
    |  1|  2|
    |100|200|
    +---+---+

    >>> cut_off(df3, threshold_column="a", val=5, mode=">").show()
    +---+---+
    |  a|  b|
    +---+---+
    |100|200|
    +---+---+
    """
    if mode == ">=":
        df = df.where(F.col(threshold_column) >= val)
    elif mode == ">":
        df = df.where(F.col(threshold_column) > val)
    elif mode == "<":
        df = df.where(F.col(threshold_column) < val)
    elif mode == "<=":
        df = df.where(F.col(threshold_column) <= val)
    return df


def date_diff(
    df: DataFrame,
    col_name1: str,
    col_name2: str,
    diff: str = "Difference",
    in_date_format: str = "dd-MM-yyyy",
    units: Literal["days", "months", "years"] = "days",
    *,
    absolute: bool = True,
) -> DataFrame:
    """Compute days/months/years difference between two date columns.

    Finds the number of days/months/years between two date columns by
    subtracting the dates in the second column from the dates in the
    first. Note, using just months is currently inaccurate as all months
    are assumed to have 31 days.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    col_name1 : str
        Name of the first column with values representing dates.
    col_name2 : str
        Name of second column with values representing dates.
    diff : str, optional
        Name of the column in which the difference between dates will be
        shown. Defaults to "Difference".
    in_date_format : str, optional
        User must specify the format of how the dates are entered in
        both ``col_name1`` and ``col_name2`` and use this argument to do
        so. Defaults to "dd-MM-yyyy".
    units : typing.Literal["days", "months", "years"], optional
        Units of how the difference in the two date columns will be
        represented in the ``diff`` column. Defaults to "days".
    absolute : bool, optional
        Bool toggle allowing user to display all values as absolute or
        non-absolute values in the ``diff`` column. Defaults to True.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with new column appended showing the time difference
        between ``col_name1`` and ``col_name2`` columns in the units
        specified.

    Examples
    --------
    >>> df.show()
    +---+--------+-------+----------+---+--------+----------+
    | ID|Forename|Surname|       DoB|Sex|Postcode|     Today|
    +---+--------+-------+----------+---+--------+----------+
    |  1|   Homer|Simpson|1983-05-12|  M|ET74 2SP|2022-11-07|
    |  2|   Marge|Simpson|1983-03-19|  F|ET74 2SP|2022-11-07|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|
    |  4|    Lisa|Simpson|2014-05-09|  F|ET74 2SP|2022-11-07|
    |  5|  Maggie|Simpson|2021-01-12|  F|ET74 2SP|2022-11-07|
    +---+--------+-------+----------+---+--------+----------+

    >>> date_diff(
    ...     df,
    ...     "DoB",
    ...     "Today",
    ...     diff="Difference",
    ...     in_date_format="yyyy-MM-dd",
    ...     units="days",
    ...     absolute=True,
    ... ).show()
    +---+--------+-------+----------+---+--------+----------+----------+
    | ID|Forename|Surname|       DoB|Sex|Postcode|     Today|Difference|
    +---+--------+-------+----------+---+--------+----------+----------+
    |  1|   Homer|Simpson|1983-05-12|  M|ET74 2SP|2022-11-07|  14424.04|
    |  2|   Marge|Simpson|1983-03-19|  F|ET74 2SP|2022-11-07|   14478.0|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|   3872.04|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|   3872.04|
    |  4|    Lisa|Simpson|2014-05-09|  F|ET74 2SP|2022-11-07|   3104.04|
    |  5|  Maggie|Simpson|2021-01-12|  F|ET74 2SP|2022-11-07|     664.0|
    +---+--------+-------+----------+---+--------+----------+----------+
    """
    df = df.withColumn(
        diff,
        F.unix_timestamp(F.col(col_name1), in_date_format)
        - F.unix_timestamp(F.col(col_name2), in_date_format),
    )

    if units == "days":
        df = df.withColumn(diff, (F.col(diff) / 86400))
        df = df.withColumn(diff, F.round(diff, 2))
    elif units == "months":
        # "months" value is slightly inaccurate as it assumes every
        # month is a 31-day month.
        df = df.withColumn(diff, F.col(diff) / (31 * 86400))
        df = df.withColumn(diff, F.round(diff, 2))
    elif units == "years":
        df = df.withColumn(diff, F.col(diff) / (86400 * 365))
        df = df.withColumn(diff, F.round(diff, 2))
    if absolute is True:
        df = df.withColumn(diff, F.abs(F.col(diff)))
    return df


def drop_columns(
    df: DataFrame,
    subset: str | list[str] | None = None,
    startswith: str | None = None,
    endswith: str | None = None,
    contains: str | None = None,
    regex: str | None = None,
    *,
    drop_duplicates: bool = True,
) -> DataFrame:
    """Drop columns from ``df``.

    Allows user to specify one or more columns to be dropped from the
    dataframe.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe to which the function is applied.
    subset : str | list[str], optional
      The subset can be entered as a list of column headers that are the
      columns to be dropped. If a single string that is a name of a
      column is entered, it will drop that column. Defaults to None.
    startswith : str, optional
      This parameter takes a string value and drops columns from the
      dataframe if the column title starts with the string value.
      Defaults to None.
    endswith : str, optional
      This parameter takes a string value and drops columns from the
      dataframe if the column title ends with the string value. Defaults
      to None.
    contains : str, optional
      This parameter takes a string value and drops columns from the
      dataframe if the column title contains the string value. Defaults
      to None.
    regex : str, optional
      This parameter takes a string value in regex format and drops
      columns from the dataframe if the column title matches the
      conditions of the regex string. Defaults to None.
    drop_duplicates : bool, optional
      This parameter drops duplicated columns. Defaults to True.

    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe with columns dropped based on parameters.

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

    >>> drop_columns(df, subset=None, startswith="S").show()
    +---+--------+-----------+----------+--------+
    | ID|Forename|Middle_name|       DoB|Postcode|
    +---+--------+-----------+----------+--------+
    |  2|   Marge|     Juliet|1983-03-19|ET74 2SP|
    |  3|    Bart|      Jo-Jo|2012-04-01|ET74 2SP|
    |  4|    Lisa|      Marie|2014-05-09|ET74 2SP|
    |  5|  Maggie|       null|2021-01-12|ET74 2SP|
    |  1|   Homer|        Jay|1983-05-12|ET74 2SP|
    +---+--------+-----------+----------+--------+

    >>> drop_columns(
    ...     df, subset=None, endswith="e", drop_duplicates=False
    ... ).show()
    +---+----------+---+
    | ID|       DoB|Sex|
    +---+----------+---+
    |  1|1983-05-12|  M|
    |  2|1983-03-19|  F|
    |  3|2012-04-01|  M|
    |  3|2012-04-01|  M|
    |  4|2014-05-09|  F|
    |  5|2021-01-12|  F|
    +---+----------+---+

    >>> drop_columns(df, subset=None, contains="name").show()
    +---+----------+---+--------+
    | ID|       DoB|Sex|Postcode|
    +---+----------+---+--------+
    |  4|2014-05-09|  F|ET74 2SP|
    |  2|1983-03-19|  F|ET74 2SP|
    |  3|2012-04-01|  M|ET74 2SP|
    |  5|2021-01-12|  F|ET74 2SP|
    |  1|1983-05-12|  M|ET74 2SP|
    +---+----------+---+--------+

    >>> drop_columns(df, subset=None, regex="^[A-Z]{2}$").show()
    +--------+-----------+-------+----------+---+--------+
    |Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +--------+-----------+-------+----------+---+--------+
    |   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|
    +--------+-----------+-------+----------+---+--------+
    """
    if startswith is not None:
        df = df.drop(*[x for x in df.columns if x.startswith(startswith)])

    if endswith is not None:
        df = df.drop(*[x for x in df.columns if x.endswith(endswith)])

    if contains is not None:
        df = df.drop(*[x for x in df.columns if contains in x])

    if regex is not None:
        df = df.drop(*[x for x in df.columns if re.search(regex, x)])

    if subset is not None:
        if not isinstance(subset, list):
            subset = [subset]
        df = df.drop(*subset)

    if drop_duplicates:
        df = df.dropDuplicates()

    return df


def drop_nulls(
    df: DataFrame, subset: str | list[str] | None = None, val: str | None = None
) -> DataFrame:
    """Drop rows with Null or specified values.

    This drops rows containing nulls in any columns by default.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The DataFrame the function is applied to.
    subset : str | list[str], optional
        A list of columns to drop null values from. Defaults to None.
    val : str, optional
        The specified value for nulls. Defaults to None.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with Null or ``val`` values dropped on the columns
        where the function is applied.

    Examples
    --------
    >>> df.show()
    +---+--------+-----------+-------+----------+---+--------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+--------+-----------+-------+----------+---+--------+
    |  1|   Homer|       null|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|       null|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|       null|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|       null|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+-----------+-------+----------+---+--------+

    >>> drop_nulls(df, subset=None, val=None).show()
    +---+--------+-----------+-------+----------+---+--------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+--------+-----------+-------+----------+---+--------+
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    +---+--------+-----------+-------+----------+---+--------+
    """
    if subset is not None:
        if not isinstance(subset, list):
            subset = [subset]

    if val is not None:
        df = df.replace(val, value=None, subset=subset)

    df = df.dropna(how="any", subset=subset)

    return df


def explode(
    df: DataFrame,
    column: str,
    on: str = " ",
    flag: str | None = None,
    *,
    retain: bool = False,
    drop_duplicates: bool = True,
) -> DataFrame:
    """Split a column on a specified separator.

    Splits a string column on specified separator (default=" ") and
    creates a new row for each element of the split string array
    maintaining values in all other columns.

    Parameters
    ----------
    df : pyspark.sql.DataFrame.
        DataFrame function is being applied to.
    column : str
        Column to be exploded.
    on : str, optional
        This argument takes a string or regex value in string format and
        explodes the values where either the string or regex value
        matches. Defaults to " ".
    flag: str, optional
        Name of flag column, that contains False values for rows that
        are equal. For a flag column to be appended, retain needs to be
        True. Defaults to None.
    retain : bool, optional
        Option to retain original string values. Defaults to False.
    drop_duplicates: bool, optional
        Option to drop duplicate values. Defaults to True.

    Returns
    -------
    pyspark.sql.DataFrame
        dataframe with additional rows accommodating all elements of
        exploded string column.

    Examples
    --------
    >>> df.show(truncate=False)
    +---+--------+-----------+-------+----------+---+--------+----------------------+
    |ID |Forename|Middle_name|Surname|DoB       |Sex|Postcode|Description           |
    +---+--------+-----------+-------+----------+---+--------+----------------------+
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |ET74 2SP|Balding Lazy          |
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |ET74 2SP|Blue-hair Kind-hearted|
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |ET74 2SP|Spiky-hair Rebellious |
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |ET74 2SP|Spiky-hair Rebellious |
    |4  |Lisa    |Marie      |Simpson|2014-05-09|F  |ET74 2SP|Red-dress Smart       |
    |5  |Maggie  |null       |Simpson|2021-01-12|F  |ET74 2SP|Star-hair Mute        |
    +---+--------+-----------+-------+----------+---+--------+----------------------+

    Eg if you wanted to separate the record's appearance from their
    personality description:

    >>> explode(
    ...     df,
    ...     column="Description",
    ...     on=" ",
    ...     retain=False,
    ...     drop_duplicates=True,
    ...     flag=None,
    ... ).show()
    +---+--------+-----------+-------+----------+---+--------+------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode| Description|
    +---+--------+-----------+-------+----------+---+--------+------------+
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP| Spiky-hair |
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|     Balding|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|       Smart|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|   Star-hair|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|   Red-dress|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|  Rebellious|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|   Blue-hair|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|Kind-hearted|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|        Mute|
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|        Lazy|
    +---+--------+-----------+-------+----------+---+--------+------------+

    If you wanted to also keep the original overall description:

    >>> explode(
    ...     df,
    ...     column="Description",
    ...     on=" ",
    ...     retain=True,
    ...     drop_duplicates=True,
    ...     flag=None,
    ... ).show()
    +---+--------+-----------+-------+----------+---+--------+----------------------+
    |ID |Forename|Middle_name|Surname|DoB       |Sex|Postcode|Description           |
    +---+--------+-----------+-------+----------+---+--------+----------------------+
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |ET74 2SP|Spiky-hair            |
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |ET74 2SP|Balding               |
    |4  |Lisa    |Marie      |Simpson|2014-05-09|F  |ET74 2SP|Smart                 |
    |5  |Maggie  |null       |Simpson|2021-01-12|F  |ET74 2SP|Star-hair             |
    |4  |Lisa    |Marie      |Simpson|2014-05-09|F  |ET74 2SP|Red-dress             |
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |ET74 2SP|Rebellious            |
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |ET74 2SP|Blue-hair             |
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |ET74 2SP|Balding Lazy          |
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |ET74 2SP|Kind-hearted          |
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |ET74 2SP|Spiky-hair Rebellious |
    |5  |Maggie  |null       |Simpson|2021-01-12|F  |ET74 2SP|Star-hair Mute        |
    |5  |Maggie  |null       |Simpson|2021-01-12|F  |ET74 2SP|Mute                  |
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |ET74 2SP|Lazy                  |
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |ET74 2SP|Blue-hair Kind-hearted|
    |4  |Lisa    |Marie      |Simpson|2014-05-09|F  |ET74 2SP|Red-dress Smart       |
    +---+--------+-----------+-------+----------+---+--------+----------------------+
    """
    if retain is False:
        df = (
            df.where(F.col(column).rlike(on))
            .select(
                *[x for x in df.columns if x != column],
                F.explode(F.split(F.col(column), on)).alias(column),
            )
            .unionByName(
                df.where(
                    ~(F.col(column).rlike(on)) | (F.col(column).rlike(on).isNull())
                )
            )
        )

    if retain is True:
        if flag is None:
            df = (
                df.where(F.col(column).rlike(on))
                .select(
                    *[x for x in df.columns if x != column],
                    F.explode(F.split(F.col(column), on)).alias(column),
                )
                .unionByName(df)
            )

        else:
            df = (
                df.where(F.col(column).rlike(on))
                .withColumn(flag, F.lit(col=True))
                .select(
                    *[x for x in [*df.columns, flag] if x != column],
                    F.explode(F.split(F.col(column), on)).alias(column),
                )
                .unionByName(df.withColumn(flag, F.lit(col=False)))
            )

    if drop_duplicates is True:
        df = df.dropDuplicates()

    return df


def filter_window(
    df: DataFrame,
    filter_window: str | list[str],
    target: str,
    mode: Literal["count", "countDistinct"],
    value: int | None = None,
    *,
    condition: bool = True,
) -> DataFrame:
    """Perform operation on collection of rows.

    Performs statistical operations such as ``count``,
    ``countDistinct``, ``max``, or ``min`` on a collection of rows and
    returns results for each row individually.

    This function filters the results of the window operation in two
    ways; for ``count`` and ``countDistinct``, it filters the results to
    show the results where the 'value' argument value is matched. For
    ``max`` and ``min`` operations it filters window results to only the
    minimum or maximum values respectively.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    filter_window : str | list[str]
        List of columns defining a window.
    target : str
        Target column for operations.
    mode : typing.Literal["count", "countDistinct"]
        Operation applied to the window.
    value : int, optional
        A value to filter the data by after applying the window
        operation. Defaults to None.
    condition : bool, optional
        Option to include (True) or exclude (False) rows that match the
        filter value. Defaults to True.

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

    >>> filter_window(
    ...     df,
    ...     filter_window="Forename",
    ...     target="ID",
    ...     mode="count",
    ...     value=1,
    ...     condition=True,
    ... ).show()
    +---+--------+-----------+-------+----------+---+--------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+--------+-----------+-------+----------+---+--------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|
    +---+--------+-----------+-------+----------+---+--------+

    >>> filter_window(
    ...     df,
    ...     filter_window="Forename",
    ...     target="ID",
    ...     mode="count",
    ...     value=1,
    ...     condition=False,
    ... ).show()
    +---+--------+-----------+-------+----------+---+--------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+--------+-----------+-------+----------+---+--------+
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    +---+--------+-----------+-------+----------+---+--------+

    >>> df.show()
    +---+---+-----------+-------+----------+---+--------+
    |Age| ID|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+---+-----------+-------+----------+---+--------+
    |  3|  2|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  5|  4|      Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  6|  3|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  1|  5|       null|Simpson|2021-01-12|  F|ET74 2SP|
    |  4|  3|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  2|  1|        Jay|Simpson|1983-05-12|  M|ET74 2SP|
    +---+---+-----------+-------+----------+---+--------+

    >>> filter_window(
    ...     df, filter_window="ID", target="Age", mode="min", value=None, condition=True
    ... ).show()
    +---+---+-----------+-------+----------+---+--------+
    | ID|Age|Middle_name|Surname|       DoB|Sex|Postcode|
    +---+---+-----------+-------+----------+---+--------+
    |  3|  4|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  5|  1|       null|Simpson|2021-01-12|  F|ET74 2SP|
    |  1|  2|        Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  4|  5|      Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  2|  3|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    +---+---+-----------+-------+----------+---+--------+

    The records are grouped by ID, and then the minimum age for each
    record is returned. Therefore, the age '6' for ID '3' is removed.

    See Also
    --------
    ``standardisation.fill_nulls``
    """
    if not isinstance(filter_window, list):
        filter_window = [filter_window]

    w = Window.partitionBy(filter_window)

    if mode in ["count", "countDistinct"]:
        if condition:
            df = (
                window(df, filter_window, target, mode, alias="count")
                .where(F.col("count") == value)
                .drop("count")
            )

        else:
            df = (
                window(df, filter_window, target, mode, alias="count")
                .where(F.col("count") != value)
                .drop("count")
            )

    if mode in ["min", "max"]:
        dt_target = [dtype for name, dtype in df.dtypes if name == target][0]
        df = window(df, filter_window, target, mode, alias="value")
        df = st.fill_nulls(df, fill="<<<>>>", subset=["value"] + [target])

        if condition:
            df = df.where(F.col(target) == F.col("value")).drop("value")
        else:
            df = df.where(F.col(target) != F.col("value")).drop("value")

        df = st.standardise_null(df=df, replace="^<<<>>>$", subset=target)
        df = df.withColumn(target, F.col(target).cast(dt_target))

    return df


def index_select(
    df: DataFrame,
    split_col: str,
    out_col: str,
    index: int | tuple[int, int],
    sep: str = " ",
) -> DataFrame:
    """Select elements from an array column by index into a new column.

    Allows indices to be selected within a column of arrays and casts
    those values to a new column (out_col arg).

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    split_col : str
        Column header of the split array column to be indexed.
    out_col : str
        Column header of the output column of selected index.
    index : int | tuple[int, int]
        Index or indices of required element(s) being selected.
    sep : str, optional
        Separator when using a tuple to select more than one element
        using index selection. Defaults to " ".

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with new variable selected from split array column.

    Examples
    --------
    >>> df3.show()
    +------+----+
    |     a|   b|
    +------+----+
    |[1, 2]|null|
    |[4, 5]|null|
    |[7, 8]|   2|
    +------+----+

    >>> index_select(df3, split_col="a", out_col="a_index_1", index=0, sep=" ").show()
    +------+----+---------+
    |     a|   b|a_index_1|
    +------+----+---------+
    |[1, 2]|null|        1|
    |[4, 5]|null|        4|
    |[7, 8]|   2|        7|
    +------+----+---------+
    """
    if isinstance(index, tuple):
        for i in range(index[1])[index[0] :]:
            df = df.withColumn(f"index_select_{i}", F.col(split_col).getItem(i))

        df = concat(
            df,
            out_col,
            sep,
            [f"index_select_{i}" for i in range(index[1])[index[0] :]],
        )

        df = df.drop(*[f"index_select_{i}" for i in range(index[1])[index[0] :]])

    else:
        if index >= 0:
            df = df.withColumn(out_col, F.col(split_col).getItem(index))

        if index < 0:
            df = df.withColumn(
                out_col, F.reverse(F.col(split_col)).getItem(abs(index) - 1)
            )

    return df


def literal_column(df: DataFrame, col_name: str, literal: Any) -> DataFrame:
    """Add a column containing a specified value.

    Returns the original DataFrame along with a new column added
    containing values specified by the user.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    col_name : str
        New column title.
    literal : typing.Any
        Values populating the ``col_name`` column.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with new literal column.

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

    >>> literal_column(
    ...     df, col_name="Next-door neighbour", literal="Ned Flanders"
    ... ).show()
    +---+--------+-----------+-------+----------+---+--------+-------------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|Postcode|Next-door neighbour|
    +---+--------+-----------+-------+----------+---+--------+-------------------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|       Ned Flanders|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|       Ned Flanders|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|       Ned Flanders|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|       Ned Flanders|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|       Ned Flanders|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|       Ned Flanders|
    +---+--------+-----------+-------+----------+---+--------+-------------------+
    """
    df = df.withColumn(col_name, F.lit(literal))
    return df


def prefix_columns(
    df: DataFrame, prefix: str, exclude: str | list[str] | None = None
) -> DataFrame:
    """Rename columns with specified prefix string.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        A DataFrame containing columns to be renamed.
    prefix : str
        The prefix string that will be appended to column names.
    exclude : str | list[str], optional
        This argument either takes a list of column names or a string
        value that is a column name. These values are excluded from the
        renaming of columns. Defaults to None.

    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe with prefixed column names.

    Examples
    --------
    You want to join the Simpsons df to the Flintstones df. Suffixing or
    prefixing will allow you to identify which dataset the columns
    relate to:

    >>> df.show()
    +---+--------+-----------+-------+----------+---+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|
    +---+--------+-----------+-------+----------+---+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|
    +---+--------+-----------+-------+----------+---+

    >>> prefix_columns(df, prefix="Simpsons_").show()
    +-----------+-----------------+--------------------+----------------+------------+------------+
    |Simpsons_ID|Simpsons_Forename|Simpsons_Middle_name|Simpsons_Surname|Simpsons_DoB|Simpsons_Sex|
    +-----------+-----------------+--------------------+----------------+------------+------------+
    |          1|            Homer|                 Jay|         Simpson|  1983-05-12|           M|
    |          2|            Marge|              Juliet|         Simpson|  1983-03-19|           F|
    |          3|             Bart|               Jo-Jo|         Simpson|  2012-04-01|           M|
    |          3|             Bart|               Jo-Jo|         Simpson|  2012-04-01|           M|
    |          4|             Lisa|               Marie|         Simpson|  2014-05-09|           F|
    |          5|           Maggie|                null|         Simpson|  2021-01-12|           F|
    +-----------+-----------------+--------------------+----------------+------------+------------+

    >>> prefix_columns(df, prefix="Simpsons_", exclude="Surname").show()
    +-----------+-----------------+--------------------+-------+------------+------------+
    |Simpsons_ID|Simpsons_Forename|Simpsons_Middle_name|Surname|Simpsons_DoB|Simpsons_Sex|
    +-----------+-----------------+--------------------+-------+------------+------------+
    |          1|            Homer|                 Jay|Simpson|  1983-05-12|           M|
    |          2|            Marge|              Juliet|Simpson|  1983-03-19|           F|
    |          3|             Bart|               Jo-Jo|Simpson|  2012-04-01|           M|
    |          3|             Bart|               Jo-Jo|Simpson|  2012-04-01|           M|
    |          4|             Lisa|               Marie|Simpson|  2014-05-09|           F|
    |          5|           Maggie|                null|Simpson|  2021-01-12|           F|
    +-----------+-----------------+--------------------+-------+------------+------------+
    """
    if not isinstance(exclude, list):
        exclude = [exclude]

    old = [x for x in df.columns if x not in exclude]
    new = [prefix + x for x in old]

    rename = dict(zip(old, new))

    for old, new in rename.items():
        df = df.withColumnRenamed(old, new)

    return df


def rename_columns(df: DataFrame, rename_dict: dict[str, str]) -> DataFrame:
    """Rename multiple columns via a rename dictionary.

    Allows multiple columns to be renamed in one command from {before:
    after} replacement dictionary.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe to which the function is applied.
    rename_dict : dict[str, str]
        The dictionary to rename columns, with format {before: after}.

    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe with columns renamed

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

    >>> rename_columns(df, rename_dict={"ID": "Number", "DoB": "Birth Date"}).show()
    +------+--------+-----------+-------+----------+---+--------+
    |Number|Forename|Middle_name|Surname|Birth Date|Sex|Postcode|
    +------+--------+-----------+-------+----------+---+--------+
    |     1|   Homer|        Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |     2|   Marge|     Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |     3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |     3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |     4|    Lisa|      Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |     5|  Maggie|       null|Simpson|2021-01-12|  F|ET74 2SP|
    +------+--------+-----------+-------+----------+---+--------+
    """
    for before, after in rename_dict.items():
        df = df.withColumnRenamed(before, after)

    return df


def select(
    df: DataFrame,
    columns: str | list[str] | None = None,
    startswith: str | None = None,
    endswith: str | None = None,
    contains: str | None = None,
    regex: str | None = None,
    *,
    drop_duplicates: bool = True,
) -> DataFrame:
    """Retain only specified list of columns.

    Retains only specified list of columns or columns meeting
    startswith, endswith, contains or regex arguments.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe to which the function is applied.
    columns : str | list[str], optional
        This argument can be entered as a list of column headers that
        are the columns to be selected. If a single string that is a
        name of a column is entered, it will select only that column.
        Defaults to None.
    startswith : str, optional
        This parameter takes a string value and selects columns from the
        dataframe if the column title starts with the string value.
        Defaults to None.
    endswith : str, optional
        This parameter takes a string value and selects columns from the
        dataframe if the column title ends with the string value.
        Defaults to None.
    contains : str, optional
        This parameter takes a string value and selects columns from the
        dataframe if the column title contains the string value.
        Defaults to None.
    regex : str, optional
        This parameter takes a string value in regex format and selects
        columns from the dataframe if the column title matches the
        conditions of the regex string. Defaults to None
    drop_duplicates : bool, optional
        This parameter drops duplicated rows. Defaults to True.

    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe with columns limited to those specified by the
        parameters.

    Examples
    --------
    >>> data = [
    ...     ("1", "Homer", "Jay", "Simpson", "1983-05-12", "M", "ET74 2SP"),
    ...     ("2", "Marge", "Juliet", "Simpson", "1983-03-19", "F", "ET74 2SP"),
    ...     ("3", "Bart", "Jo-Jo", "Simpson", "2012-04-01", "M", "ET74 2SP"),
    ...     ("3", "Bart", "Jo-Jo", "Simpson", "2012-04-01", "M", "ET74 2SP"),
    ...     ("4", "Lisa", "Marie", "Simpson", "2014-05-09", "F", "ET74 2SP"),
    ...     ("5", "Maggie", None, "Simpson", "2021-01-12", "F", "ET74 2SP"),
    ... ]
    >>> df = spark.createDataFrame(
    ...     data=data,
    ...     schema=[
    ...         "ID",
    ...         "Forename",
    ...         "Middle_name",
    ...         "Surname",
    ...         "DoB",
    ...         "Sex",
    ...         "Postcode",
    ...     ],
    ... )

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

    >>> select(df, columns=None, startswith="F").show()
    +--------+
    |Forename|
    +--------+
    |   Homer|
    |   Marge|
    |  Maggie|
    |    Bart|
    |    Lisa|
    +--------+

    >>> select(df, columns=None, endswith="e", drop_duplicates=False).show()
    +--------+-----------+-------+--------+
    |Forename|Middle_name|Surname|Postcode|
    +--------+-----------+-------+--------+
    |   Homer|        Jay|Simpson|ET74 2SP|
    |   Marge|     Juliet|Simpson|ET74 2SP|
    |    Bart|      Jo-Jo|Simpson|ET74 2SP|
    |    Bart|      Jo-Jo|Simpson|ET74 2SP|
    |    Lisa|      Marie|Simpson|ET74 2SP|
    |  Maggie|       null|Simpson|ET74 2SP|
    +--------+-----------+-------+--------+

    >>> select(df, columns=None, contains="name").show()
    +--------+-----------+-------+
    |Forename|Middle_name|Surname|
    +--------+-----------+-------+
    |    Bart|      Jo-Jo|Simpson|
    |   Marge|     Juliet|Simpson|
    |   Homer|        Jay|Simpson|
    |    Lisa|      Marie|Simpson|
    |  Maggie|       null|Simpson|
    +--------+-----------+-------+

    >>> select(df, columns=None, regex="^[A-Z]{2}$").show()
    +---+
    | ID|
    +---+
    |  3|
    |  5|
    |  1|
    |  4|
    |  2|
    +---+
    """
    if columns is not None:
        df = df.select(columns)

    if startswith is not None:
        df = df.select([x for x in df.columns if x.startswith(startswith)])

    if endswith is not None:
        df = df.select([x for x in df.columns if x.endswith(endswith)])

    if contains is not None:
        df = df.select([x for x in df.columns if contains in x])

    if regex is not None:
        df = df.select([x for x in df.columns if re.search(regex, x)])

    if drop_duplicates:
        df = df.dropDuplicates()

    return df


def split(
    df: DataFrame, col_in: str, col_out: str | None = None, split_on: str = " "
) -> DataFrame:
    """Split a string column into an array by separator; inplace or new.

    Splits a string column to array on specified separator. Option to
    return split column as new column or to split in place.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    col_in : str
        Column to be split to array.
    col_out : str, optional
        Output column for split strings, default value makes the split
        happen in place. Defaults to None.
    split_on : str, optional
        String or regex separator for split. Defaults to " ".

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with split array column.
    """
    if col_out is None:
        df = df.withColumn(
            col_in,
            F.when((F.col(col_in).isNull()) | (F.isnan(F.col(col_in))), None).otherwise(
                F.split(F.col(col_in), split_on)
            ),
        )

    else:
        df = df.withColumn(
            col_out,
            F.when((F.col(col_in).isNull()) | (F.isnan(F.col(col_in))), None).otherwise(
                F.split(F.col(col_in), split_on)
            ),
        )

    return df


def substring(
    df: DataFrame,
    out_col: str,
    target_col: str,
    start: int,
    length: int,
    *,
    from_end: bool = False,
) -> DataFrame:
    """Create new column extracting substring from another column.

    Creates a new column containing substring values from another
    column.

    Can either be a substring starting from the first character in the
    string if ``from_end`` is False, or from the last character in the
    string if ``from_end`` is True.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    out_col : str
        Column title for the new column which will store the substring
        values.
    target_col : str
        Column title for the target column to which the function is
        applied.
    start : int
        Index value of where the substring starts.
    length : int
        Length of substring being extracted to new column.
    from_end : bool, optional
        Option to reverse the string before applying substring start and
        length arguments. Defaults to False.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with new column made up of substring values.

    Examples
    --------
    >>> df3.show()
    +--------+---+
    |       a|  b|
    +--------+---+
    |tomatoes|  b|
    |potatoes|  c|
    +--------+---+

    >>> substring(
    ...     df3, out_col="substring", target_col="a", start=-2, length=3, from_end=False
    ... ).show()
    +--------+---+---------+
    |       a|  b|substring|
    +--------+---+---------+
    |tomatoes|  b|       es|
    |potatoes|  c|       es|
    +--------+---+---------+

    >>> substring(
    ...     df3, out_col="substring", target_col="a", start=2, length=3, from_end=False
    ... ).show()
    +--------+---+---------+
    |       a|  b|substring|
    +--------+---+---------+
    |tomatoes|  b|      oma|
    |potatoes|  c|      ota|
    +--------+---+---------+

    >>> substring(
    ...     df3, out_col="substring", target_col="a", start=2, length=3, from_end=True
    ... ).show()
    +--------+---+---------+
    |       a|  b|substring|
    +--------+---+---------+
    |tomatoes|  b|      toe|
    |potatoes|  c|      toe|
    +--------+---+---------+
    """
    if from_end is False:
        df = df.withColumn(out_col, F.substring(F.col(target_col), start, length))

    if from_end is True:
        df = (
            df.withColumn(target_col, F.reverse(F.col(target_col)))
            .withColumn(
                out_col,
                F.reverse(F.substring(F.col(target_col), start, length)),
            )
            .withColumn(target_col, F.reverse(F.col(target_col)))
        )

    return df


def suffix_columns(df: DataFrame, suffix: str, exclude: str | list[str]) -> DataFrame:
    """Rename columns with specified suffix string.

    Parameters
    ----------
     df : pyspark.sql.DataFrame
        A DataFrame containing columns to be renamed.
    suffix : str
        The suffix string that will be appended to column names.
    exclude : str | list[str]
        This argument either takes a list of column names or a string
        value that is a column name. These values are excluded from the
        renaming of columns.

    Returns
    -------
    pyspark.sql.DataFrame
        Dataframe with suffixed column names.

    Examples
    --------
    >>> df.show()
    +---+--------+-----------+-------+----------+---+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|
    +---+--------+-----------+-------+----------+---+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|
    +---+--------+-----------+-------+----------+---+

    >>> suffix_columns(df, suffix="_Simpsons").show()
    +-----------+-----------------+--------------------+----------------+------------+------------+
    |ID_Simpsons|Forename_Simpsons|Middle_name_Simpsons|Surname_Simpsons|DoB_Simpsons|Sex_Simpsons|
    +-----------+-----------------+--------------------+----------------+------------+------------+
    |          1|            Homer|                 Jay|         Simpson|  1983-05-12|           M|
    |          2|            Marge|              Juliet|         Simpson|  1983-03-19|           F|
    |          3|             Bart|               Jo-Jo|         Simpson|  2012-04-01|           M|
    |          3|             Bart|               Jo-Jo|         Simpson|  2012-04-01|           M|
    |          4|             Lisa|               Marie|         Simpson|  2014-05-09|           F|
    |          5|           Maggie|                null|         Simpson|  2021-01-12|           F|
    +-----------+-----------------+--------------------+----------------+------------+------------+

    >>> suffix_columns(df, suffix="_Simpsons", exclude="Surname").show()
    +-----------+-----------------+--------------------+-------+------------+------------+
    |ID_Simpsons|Forename_Simpsons|Middle_name_Simpsons|Surname|DoB_Simpsons|Sex_Simpsons|
    +-----------+-----------------+--------------------+-------+------------+------------+
    |          1|            Homer|                 Jay|Simpson|  1983-05-12|           M|
    |          2|            Marge|              Juliet|Simpson|  1983-03-19|           F|
    |          3|             Bart|               Jo-Jo|Simpson|  2012-04-01|           M|
    |          3|             Bart|               Jo-Jo|Simpson|  2012-04-01|           M|
    |          4|             Lisa|               Marie|Simpson|  2014-05-09|           F|
    |          5|           Maggie|                null|Simpson|  2021-01-12|           F|
    +-----------+-----------------+--------------------+-------+------------+------------+
    """
    if not isinstance(exclude, list):
        exclude = [exclude]

    old = [x for x in df.columns if x not in exclude]
    new = [x + suffix for x in old]

    rename = dict(zip(old, new))

    for old, new in rename.items():
        df = df.withColumnRenamed(old, new)

    return df


def union_all(*dfs: DataFrame, fill: Any = None):
    """Union a list of DataFrames to a single DataFrame.

    Where DataFrame columns are not consistent, creates columns to
    enable union with default null fill.

    Parameters
    ----------
    *dfs : pyspark.sql.DataFrame
        Any number of DataFrames to be combined.
    fill : typing.Any, optional
        A value to fill null values with when column names are
        inconsistent between the DataFrames being combined. Defaults to
        None.

    Returns
    -------
    pyspark.sql.DataFrame
        Single unioned DataFrame.

    Examples
    --------
    >>> df1.show()
    +---+--------+-----------+-------+----------+---+------------------------+
    |ID |Forename|Middle_name|Surname|DoB       |Sex|Profession              |
    +---+--------+-----------+-------+----------+---+------------------------+
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |Nuclear safety inspector|
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |Housewife               |
    +---+--------+-----------+-------+----------+---+------------------------+

    >>> df2.show()
    +---+--------+-----------+-------+----------+---+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|
    +---+--------+-----------+-------+----------+---+
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|
    +---+--------+-----------+-------+----------+---+

    >>> union_all(df1, df2).show(truncate=False)
    +---+--------+-----------+-------+----------+---+------------------------+
    |ID |Forename|Middle_name|Surname|DoB       |Sex|Profession              |
    +---+--------+-----------+-------+----------+---+------------------------+
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |Nuclear safety inspector|
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |Housewife               |
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |null                    |
    |4  |Lisa    |Marie      |Simpson|2014-05-09|F  |null                    |
    |5  |Maggie  |null       |Simpson|2021-01-12|F  |null                    |
    +---+--------+-----------+-------+----------+---+------------------------+

    >>> union_all(df1, df2, fill="too young").show(truncate=False)
    +---+--------+-----------+-------+----------+---+------------------------+
    |ID |Forename|Middle_name|Surname|DoB       |Sex|Profession              |
    +---+--------+-----------+-------+----------+---+------------------------+
    |1  |Homer   |Jay        |Simpson|1983-05-12|M  |Nuclear safety inspector|
    |2  |Marge   |Juliet     |Simpson|1983-03-19|F  |Housewife               |
    |3  |Bart    |Jo-Jo      |Simpson|2012-04-01|M  |too young               |
    |4  |Lisa    |Marie      |Simpson|2014-05-09|F  |too young               |
    |5  |Maggie  |null       |Simpson|2021-01-12|F  |too young               |
    +---+--------+-----------+-------+----------+---+------------------------+
    """
    if len(dfs) == 1:
        return dfs[0]

    columns = list({x for y in [df.columns for df in dfs] for x in y})

    out = dfs[0]

    add_columns = [x for x in columns if x not in out.columns]

    for col in add_columns:
        out = out.withColumn(col, F.lit(fill))

    for df in dfs[1:]:
        add_columns = [x for x in columns if x not in df.columns]

        for col in add_columns:
            df = df.withColumn(col, F.lit(fill))

        out = out.unionByName(df)

    return out


def window(
    df: DataFrame,
    window: str | list[str],
    target: str,
    mode: Literal["count", "countDistinct", "max", "min", "sum"],
    alias: str | None = None,
    *,
    drop_na: bool = False,
) -> DataFrame:
    """Add a column for a given ``mode`` over a given ``window``.

    Adds window column for ``count``, ``countDistinct``, ``min``,
    ``max``, or ``sum`` operations over window. Need to import the
    ``union_all`` function first.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame to which the function is applied.
    window : str | list[str]
        List of columns defining the window.
    target : str
        Name of target column for operations in string format.
    mode : typing.Literal["count", "countDistinct", "max", "min", "sum"]
        Operation performed on window.
    alias : str, optional
        Name of column for window function results. Defaults to None.
    drop_na : bool, optional
        If True, drops Null values from ``countDistinct`` window
        function when performing the operation. Defaults to False.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with window alias column appended to DataFrame showing
        results of the operation performed over a window of columns.

    Examples
    --------
    >>> df.show()
    +---+--------+-----------+-------+----------+---+-----------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|age_at_2022-12-06|
    +---+--------+-----------+-------+----------+---+-----------------+
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|               39|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|               39|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|               10|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|               10|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|                8|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|                1|
    +---+--------+-----------+-------+----------+---+-----------------+

    >>> window(
    ...     df,
    ...     window="ID",
    ...     target="Forename",
    ...     mode="count",
    ...     alias="forenames_per_ID",
    ...     drop_na=False,
    ... ).show()
    +---+--------+-----------+-------+----------+---+-----------------+----------------+
    | ID|Forename|Middle_name|Surname|       DoB|Sex|age_at_2022-12-06|forenames_per_ID|
    +---+--------+-----------+-------+----------+---+-----------------+----------------+
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|               10|               2|
    |  3|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|               10|               2|
    |  5|  Maggie|       null|Simpson|2021-01-12|  F|                1|               1|
    |  1|   Homer|        Jay|Simpson|1983-05-12|  M|               39|               1|
    |  4|    Lisa|      Marie|Simpson|2014-05-09|  F|                8|               1|
    |  2|   Marge|     Juliet|Simpson|1983-03-19|  F|               39|               1|
    +---+--------+-----------+-------+----------+---+-----------------+----------------+

    >>> window(
    ...     df,
    ...     window="ID",
    ...     target="Forename,
    ...     mode="countDistinct",
    ...     alias="distinct_forenames_per_ID",
    ...     drop_na=False,
    ... ).show()
    +---+-------------------------+--------+-----------+-------+----------+---+-----------------+
    | ID|distinct_forenames_per_ID|Forename|Middle_name|Surname|       DoB|Sex|age_at_2022-12-06|
    +---+-------------------------+--------+-----------+-------+----------+---+-----------------+
    |  3|                        1|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|               10|
    |  3|                        1|    Bart|      Jo-Jo|Simpson|2012-04-01|  M|               10|
    |  5|                        1|  Maggie|       null|Simpson|2021-01-12|  F|                1|
    |  1|                        1|   Homer|        Jay|Simpson|1983-05-12|  M|               39|
    |  4|                        1|    Lisa|      Marie|Simpson|2014-05-09|  F|                8|
    |  2|                        1|   Marge|     Juliet|Simpson|1983-03-19|  F|               39|
    +---+-------------------------+--------+-----------+-------+----------+---+-----------------+

    >>> window(
    ...     df,
    ...     window="Sex",
    ...     target="age_at_2022-12-09",
    ...     mode="min",
    ...     alias="youngest_per_sex",
    ...     drop_na=False,
    ... ).show()
    +---+----------------+---+--------+-----------+-------+----------+-----------------+
    |Sex|youngest_per_sex| ID|Forename|Middle_name|Surname|       DoB|age_at_2022-12-09|
    +---+----------------+---+--------+-----------+-------+----------+-----------------+
    |  F|               1|  2|   Marge|     Juliet|Simpson|1983-03-19|               39|
    |  F|               1|  4|    Lisa|      Marie|Simpson|2014-05-09|                8|
    |  F|               1|  5|  Maggie|       null|Simpson|2021-01-12|                1|
    |  M|              10|  1|   Homer|        Jay|Simpson|1983-05-12|               39|
    |  M|              10|  3|    Bart|      Jo-Jo|Simpson|2012-04-01|               10|
    |  M|              10|  3|    Bart|      Jo-Jo|Simpson|2012-04-01|               10|
    +---+----------------+---+--------+-----------+-------+----------+-----------------+

    >>> window(
    ...     df,
    ...     window="Sex",
    ...     target="age_at_2022-12-09",
    ...     mode="max",
    ...     alias="oldest_per_sex",
    ...     drop_na=False,
    ... ).show()
    +---+--------------+---+--------+-----------+-------+----------+-----------------+
    |Sex|oldest_per_sex| ID|Forename|Middle_name|Surname|       DoB|age_at_2022-12-09|
    +---+--------------+---+--------+-----------+-------+----------+-----------------+
    |  F|            39|  2|   Marge|     Juliet|Simpson|1983-03-19|               39|
    |  F|            39|  4|    Lisa|      Marie|Simpson|2014-05-09|                8|
    |  F|            39|  5|  Maggie|       null|Simpson|2021-01-12|                1|
    |  M|            39|  3|    Bart|      Jo-Jo|Simpson|2012-04-01|               10|
    |  M|            39|  1|   Homer|        Jay|Simpson|1983-05-12|               39|
    |  M|            39|  3|    Bart|      Jo-Jo|Simpson|2012-04-01|               10|
    +---+--------------+---+--------+-----------+-------+----------+-----------------+

    >>> window(
    ...     df,
    ...     window="Sex",
    ...     target="age_at_2022-12-09",
    ...     mode="sum",
    ...     alias="total_age_by_sex",
    ...     drop_na=False
    ... ).show()
    +---+----------------+---+--------+-----------+-------+----------+-----------------+
    |Sex|total_age_by_sex| ID|Forename|Middle_name|Surname|       DoB|age_at_2022-12-09|
    +---+----------------+---+--------+-----------+-------+----------+-----------------+
    |  F|              48|  2|   Marge|     Juliet|Simpson|1983-03-19|               39|
    |  F|              48|  4|    Lisa|      Marie|Simpson|2014-05-09|                8|
    |  F|              48|  5|  Maggie|       null|Simpson|2021-01-12|                1|
    |  M|              59|  1|   Homer|        Jay|Simpson|1983-05-12|               39|
    |  M|              59|  3|    Bart|      Jo-Jo|Simpson|2012-04-01|               10|
    |  M|              59|  3|    Bart|      Jo-Jo|Simpson|2012-04-01|               10|
    +---+----------------+---+--------+-----------+-------+----------+-----------------+

    See Also
    --------
    ``standardisation.standardise_null``
    """
    if not isinstance(window, list):
        window = [window]

    window_spec = Window.partitionBy(window)

    if mode == "count":
        if alias is not None:
            df = df.select(*df.columns, F.count(target).over(window_spec).alias(alias))

        else:
            df = df.select(*df.columns, F.count(target).over(window_spec))

    if mode == "countDistinct":
        df = df.fillna("<<<>>>", subset=window)

        if alias is not None:
            if drop_na is True:
                df = (
                    df.dropDuplicates(subset=window + [target])
                    .dropna(subset=[target])
                    .select(
                        *window + [target],
                        F.count(target).over(window_spec).alias(alias),
                    )
                    .drop(target)
                    .dropDuplicates()
                ).join(df, on=window, how="right")
            else:
                df = (
                    df.dropDuplicates(subset=window + [target])
                    .select(
                        *window + [target],
                        F.count(target).over(window_spec).alias(alias),
                    )
                    .drop(target)
                    .dropDuplicates()
                ).join(df, on=window, how="right")

        else:
            if drop_na is True:
                df = (
                    df.dropDuplicates(subset=window + [target])
                    .dropna(subset=[target])
                    .select(*window + [target], F.count(target).over(window_spec))
                    .drop(target)
                    .dropDuplicates()
                ).join(df, on=window, how="right")
            else:
                df = (
                    df.dropDuplicates(subset=window + [target])
                    .select(*window + [target], F.count(target).over(window_spec))
                    .drop(target)
                    .dropDuplicates()
                ).join(df, on=window, how="right")

    if mode == "min":
        if alias is not None:
            df_1 = df.dropna(subset=target).select(
                *df.columns, F.min(target).over(window_spec).alias(alias)
            )

            df_2 = df.where((F.col(target).isNull()) | F.isnan(F.col(target))).join(
                df_1.select(window), on=window, how="left_anti"
            )

            df = (
                union_all(df_1, df_2)
                .select(window + [alias])
                .dropDuplicates()
                .join(df, on=window, how="right")
            )

        else:
            df_1 = df.dropna(subset=target).select(
                *df.columns, F.min(target).over(window_spec)
            )

            df_2 = df.where((F.col(target).isNull()) | F.isnan(F.col(target))).join(
                df_1.select(window), on=window, how="left_anti"
            )

            df = (
                union_all(df_1, df_2)
                .drop(*[x for x in df.columns if x not in window])
                .dropDuplicates()
                .join(df, on=window, how="right")
            )

    if mode == "max":
        if alias is not None:
            df_1 = (
                df.dropna(subset=target)
                .select(*df.columns, F.max(target).over(window_spec).alias(alias))
                .select(window + [alias])
            )

            df_2 = (
                df.where((F.col(target).isNull()) | F.isnan(F.col(target)))
                .join(df_1.select(window), on=window, how="left_anti")
                .select(window)
            )

            df = (
                union_all(df_1, df_2)
                .select(window + [alias])
                .dropDuplicates()
                .join(df, on=window, how="right")
            )

        else:
            # needs alternative to selecting alias

            drop = [x for x in df.columns if x not in window]

            df_1 = (
                df.dropna(subset=target)
                .select(*df.columns, F.max(target).over(window_spec))
                .drop(*drop)
            )

            df_2 = (
                df.where((F.col(target).isNull()) | F.isnan(F.col(target)))
                .join(df_1.select(window), on=window, how="left_anti")
                .select(window)
            )

            df = (
                union_all(df_1, df_2)
                .drop(*[x for x in df.columns if x not in window])
                .dropDuplicates()
                .join(df, on=window, how="right")
            )

    if mode == "sum":
        if alias is not None:
            df_1 = (
                df.dropna(subset=target)
                .select(*df.columns, F.sum(target).over(window_spec).alias(alias))
                .select(window + [alias])
            )

            df_2 = (
                df.where((F.col(target).isNull()) | F.isnan(F.col(target)))
                .join(df_1.select(window), on=window, how="left_anti")
                .select(window)
            )

            df = (
                union_all(df_1, df_2)
                .select(window + [alias])
                .dropDuplicates()
                .join(df, on=window, how="right")
            )

        else:
            # needs alternative to selecting alias

            drop = [x for x in df.columns if x not in window]

            df_1 = (
                df.dropna(subset=target)
                .select(*df.columns, F.sum(target).over(window_spec))
                .drop(*drop)
            )

            df_2 = (
                df.where((F.col(target).isNull()) | F.isnan(F.col(target)))
                .join(df_1.select(window), on=window, how="left_anti")
                .select(window)
            )

            df = (
                union_all(df_1, df_2)
                .drop(*[x for x in df.columns if x not in window])
                .dropDuplicates()
                .join(df, on=window, how="right")
            )

    df = st.standardise_null(df, "^<<<>>>$", subset=window)

    return df
