import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType
from pyspark.testing import assertDataFrameEqual

from dlh_utils.dataframes import (
    clone_column,
    coalesced,
    concat,
    cut_off,
    date_diff,
    drop_columns,
    drop_nulls,
    explode,
    filter_window,
    index_select,
    literal_column,
    prefix_columns,
    rename_columns,
    select,
    split,
    substring,
    suffix_columns,
    union_all,
    window,
)


class TestCloneColumn:
    def test_clone_basic(self, spark: SparkSession) -> None:
        input_data = [("alice", 25), ("bob", 42)]
        input_df = spark.createDataFrame(input_data, schema=["name", "age"])
        expected_output = spark.createDataFrame(
            [("alice", 25, 25), ("bob", 42, 42)], schema=["name", "age", "age2"]
        )
        actual_output = clone_column(input_df, "age", "age2")
        assertDataFrameEqual(actual_output, expected_output)


class TestCoalesced:
    def test_expected(self, spark: SparkSession) -> None:
        input_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "extra": [None, None, None, "FO+ UR", None],
                    "lower": ["one", None, "one", "four", None],
                    "lowerNulls": ["one", "two", None, "four", None],
                    "upperNulls": ["ONE", "TWO", None, "FOU  R", None],
                    "value": [1, 2, 3, 4, 5],
                }
            )
        )
        expected_data: list[list[str | None | int]] = [
            [None, "one", "one", "ONE", 1, "one"],
            [None, None, "two", "TWO", 2, "two"],
            [None, "one", None, None, 3, "one"],
            ["FO+ UR", "four", "four", "FOU  R", 4, "FO+ UR"],
            [None, None, None, None, 5, "5"],
        ]
        expected_schema = StructType(
            [
                StructField("extra", StringType()),
                StructField("lower", StringType()),
                StructField("lowerNulls", StringType()),
                StructField("upperNulls", StringType()),
                StructField("value", LongType()),
                StructField("coalesced_col", StringType()),
            ]
        )
        expected_output = spark.createDataFrame(expected_data, expected_schema)
        actual_output = coalesced(input_df)
        assertDataFrameEqual(expected_output, actual_output)

    def test_expected_with_drop(self, spark: SparkSession) -> None:
        input_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "lower": ["one", None, "one", "four", None],
                    "value": [1, 2, 3, 4, 5],
                    "extra": [None, None, None, "FO+ UR", None],
                    "lowerNulls": ["one", "two", None, "four", None],
                    "upperNulls": ["ONE", "TWO", None, "FOU  R", None],
                }
            )
        )
        expected_data = [["one"], ["2"], ["one"], ["four"], ["5"]]
        expected_schema = StructType([StructField("coalesced_col", StringType())])
        expected_output = spark.createDataFrame(expected_data, expected_schema)
        actual_output = coalesced(input_df, drop=True)
        assertDataFrameEqual(expected_output, actual_output)


class TestConcat:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "firstname": [None, "Claire", "Josh", "Bob"],
                    "middle_name": ["Maria", None, "", "Greg"],
                    "lastname": ["Jones", None, "Smith", "Evans"],
                    "numeric": [1, 2, None, 4],
                    "after": ["Maria_Jones", "Claire", "Josh_Smith", "Bob_Greg_Evans"],
                }
            )
        )

        # Pandas replaces None with NaN in a numeric column. Convert
        # back to Null:
        test_df = test_df.replace(float("nan"), None)
        intended_schema = StructType(
            [
                StructField("firstname", StringType()),
                StructField("middle_name", StringType()),
                StructField("lastname", StringType()),
                StructField("numeric", DoubleType()),
                StructField("after", StringType()),
                StructField("fullname", StringType()),
            ]
        )
        intended_data: list[list[str | None | float]] = [
            [None, "Maria", "Jones", 1.0, "Maria_Jones", "Maria_Jones"],
            ["Claire", None, None, 2.0, "Claire", "Claire"],
            ["Josh", "", "Smith", None, "Josh_Smith", "Josh_Smith"],
            ["Bob", "Greg", "Evans", 4.0, "Bob_Greg_Evans", "Bob_Greg_Evans"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        result_df = concat(
            test_df,
            "fullname",
            sep="_",
            columns=["firstname", "middle_name", "lastname"],
        )
        assertDataFrameEqual(intended_df, result_df)


class TestCutOff:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {"strings": ["1", "2", "3", "4", "5"], "ints": [1, 2, 3, 4, 5]}
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame({"strings": ["3", "4", "5"], "ints": [3, 4, 5]})
        )

        # `cut_off` does not remove null values when the val is an Int
        # type.
        result_df = cut_off(test_df, threshold_column="ints", val=3, mode=">=")
        assertDataFrameEqual(intended_df, result_df)
        test_df_2 = spark.createDataFrame(
            pd.DataFrame(
                {"col1": [None, "15-05-1996", "16-04-1996", "17-06-1996", "18-05-1997"]}
            )
        ).withColumn("col1", sf.to_date("col1", "dd-MM-yyyy"))
        intended_df_2 = spark.createDataFrame(
            pd.DataFrame({"col1": ["18-05-1997"]})
        ).withColumn("col1", sf.to_date("col1", "dd-MM-yyyy"))
        result_df_2 = cut_off(test_df_2, "col1", "1997-01-15", ">=")
        assertDataFrameEqual(intended_df_2, result_df_2)
        intended_df_3 = spark.createDataFrame(
            pd.DataFrame({"strings": ["4", "5"], "ints": [4, 5]})
        )
        result_df3 = cut_off(test_df, threshold_column="ints", val=3, mode=">")
        assertDataFrameEqual(intended_df_3, result_df3)
        intended_df_4 = spark.createDataFrame(
            pd.DataFrame({"strings": ["1", "2", "3"], "ints": [1, 2, 3]})
        )
        result_df4 = cut_off(test_df, threshold_column="ints", val=4, mode="<")
        assertDataFrameEqual(intended_df_4, result_df4)
        intended_df_5 = spark.createDataFrame(
            pd.DataFrame({"strings": ["1", "2", "3"], "ints": [1, 2, 3]})
        )
        result_df5 = cut_off(test_df, threshold_column="ints", val=3, mode="<=")
        assertDataFrameEqual(intended_df_5, result_df5)


class TestDateDiff:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "dob": [
                        "1983-05-12",
                        "1983-03-19",
                        "2012-04-01",
                        "2012-04-01",
                        "2014-05-09",
                        "2021-01-12",
                    ],
                    "today": [
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                    ],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "dob": [
                        "1983-05-12",
                        "1983-03-19",
                        "2012-04-01",
                        "2012-04-01",
                        "2014-05-09",
                        "2021-01-12",
                    ],
                    "today": [
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                    ],
                    "Difference": [14600.0, 14653.96, 4048.0, 4048.0, 3280.0, 839.96],
                }
            )
        )
        result_df = date_diff(
            test_df, "dob", "today", in_date_format="yyyy-MM-dd", units="days"
        )
        assertDataFrameEqual(intended_df, result_df)
        intended_df_2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "dob": [
                        "1983-05-12",
                        "1983-03-19",
                        "2012-04-01",
                        "2012-04-01",
                        "2014-05-09",
                        "2021-01-12",
                    ],
                    "today": [
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                    ],
                    "Difference": [470.97, 472.71, 130.58, 130.58, 105.81, 27.1],
                }
            )
        )
        result_df2 = date_diff(
            test_df, "dob", "today", in_date_format="yyyy-MM-dd", units="months"
        )
        assertDataFrameEqual(intended_df_2, result_df2)
        intended_df_3 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "dob": [
                        "1983-05-12",
                        "1983-03-19",
                        "2012-04-01",
                        "2012-04-01",
                        "2014-05-09",
                        "2021-01-12",
                    ],
                    "today": [
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                        "2023-05-02",
                    ],
                    "Difference": [40.0, 40.15, 11.09, 11.09, 8.99, 2.3],
                }
            )
        )
        result_df3 = date_diff(
            test_df, "dob", "today", in_date_format="yyyy-MM-dd", units="years"
        )
        assertDataFrameEqual(intended_df_3, result_df3)


class TestDropColumns:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": ["ONE", "TWO", "THREE"],
                    "col2": ["one", "two", "three"],
                    "extra": ["One", "Two", "Three"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {"col2": ["one", "three", "two"], "extra": ["One", "Three", "Two"]}
            )
        )
        result_df = drop_columns(test_df, subset="col1")
        assertDataFrameEqual(intended_df, result_df)


class TestDropNulls:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "lower": [None, None, "one", "four", "five"],
                    "after": [None, None, "one", "four", "three"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame({"lower": ["one", "four"], "after": ["one", "four"]})
        )
        result_df = drop_nulls(test_df, subset="lower", val="five")
        assertDataFrameEqual(intended_df, result_df)


class TestExplode:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame({"check": ["iagsigajs"], "before1": ["a_b_c"]})
        ).select("check", "before1")
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "check": ["iagsigajs", "iagsigajs", "iagsigajs"],
                    "before1": ["b", "c", "a"],
                }
            )
        ).select("check", "before1")
        result_df = explode(test_df, "before1", "_")
        assertDataFrameEqual(intended_df, result_df)


class TestFilterWindow:
    def test_expected(self, spark: SparkSession) -> None:
        test_df1 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": ["a", "b", "c", "c", "d", "e", "d"],
                    "col2": [1, 1, 2, 2, 1, 1, 1],
                }
            )
        )
        intended_df1 = spark.createDataFrame(
            pd.DataFrame({"col1": ["e", "b", "a"], "col2": [1, 1, 1]})
        )
        result_df1 = filter_window(
            test_df1, "col1", "col2", "count", value=1, condition=True
        )
        assertDataFrameEqual(intended_df1, result_df1)
        test_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": ["a", "b", "c", "c", "d", "e", "d"],
                    "col2": [1, 1, 2, 3, 1, 1, 2],
                }
            )
        )
        intended_df2 = spark.createDataFrame(
            pd.DataFrame({"col1": ["d", "c"], "col2": [1, 2]})
        )
        result_df2 = filter_window(test_df2, "col1", "col2", "max", condition=False)
        assertDataFrameEqual(intended_df2, result_df2)


class TestIndexSelect:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [["a", "b", "c"], None, ["b", "c", "d"]],
                    "after": ["a", None, "b"],
                    "afterneg": ["c", None, "d"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [["a", "b", "c"], None, ["b", "c", "d"]],
                    "after": ["a", None, "b"],
                    "afterneg": ["c", None, "d"],
                    "test": ["a", None, "b"],
                }
            )
        )
        result_df = index_select(test_df, "before", "test", 0)
        assertDataFrameEqual(intended_df, result_df)


class TestLiteralColumn:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {"col1": ["one", None, "one", "four", None], "col2": [1, 2, 3, 4, 5]}
            )
        )
        intended_schema = StructType(
            [
                StructField("col1", StringType()),
                StructField("col2", LongType()),
                StructField("newStr", StringType(), nullable=False),
            ]
        )
        intended_data: list[list[str | int | None]] = [
            ["one", 1, "yes"],
            [None, 2, "yes"],
            ["one", 3, "yes"],
            ["four", 4, "yes"],
            [None, 5, "yes"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        result_df = literal_column(test_df, "newStr", "yes")
        assertDataFrameEqual(intended_df, result_df)


class TestPrefixColumns:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, None, "one", "four", "five"],
                    "col2": [None, None, "one", "four", "five"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, None, "one", "four", "five"],
                    "mrcol2": [None, None, "one", "four", "five"],
                }
            )
        )
        result_df = prefix_columns(test_df, prefix="mr", exclude="col1")
        assertDataFrameEqual(intended_df, result_df)


class TestRenameColumns:
    def test_columns_with_nones(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, None, "one", "four", "five"],
                    "col2": [None, None, "one", "four", "five"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "first": [None, None, "one", "four", "five"],
                    "second": [None, None, "one", "four", "five"],
                }
            )
        )
        result_df = rename_columns(
            test_df, rename_dict={"col1": "first", "col2": "second"}
        )
        assertDataFrameEqual(intended_df, result_df)

    def test_columns_with_lists(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "abefore": [["a", "b", "c"], None, ["b", "c", "d"]],
                    "bbefore": ["a", None, "b"],
                    "cbefore": ["c", None, "d"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "aafter": [["a", "b", "c"], None, ["b", "c", "d"]],
                    "bafter": ["a", None, "b"],
                    "cafter": ["c", None, "d"],
                }
            )
        )
        result_df = rename_columns(
            test_df,
            rename_dict={"abefore": "aafter", "bbefore": "bafter", "cbefore": "cafter"},
        )
        assertDataFrameEqual(intended_df, result_df)


class TestSelect:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "identifier": [1, 2, 3, 4],
                    "firstName": ["robert", "andrew", "carlos", "john"],
                    "firstLetter": ["r", "a", "c", "j"],
                    "first": ["x", "2", "3", "4"],
                    "numbers": [1, 2, 3, 4],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "firstName": ["robert", "andrew", "john", "carlos"],
                    "firstLetter": ["r", "a", "j", "c"],
                    "first": ["x", "2", "4", "3"],
                }
            )
        )
        result_df = select(test_df, startswith="first")
        assertDataFrameEqual(intended_df, result_df)


class TestSplit:
    def test_split_to_new_column(self, spark: SparkSession) -> None:
        input_df = spark.createDataFrame(pd.DataFrame({"before": ["a_b_c_d", None]}))
        expected_output = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["a_b_c_d", None],
                    "after": [["a", "b", "c", "d"], None],
                }
            )
        )
        actual_output = split(input_df, "before", col_out="after", split_on="_")
        assertDataFrameEqual(expected_output, actual_output)


class TestSubstring:
    def test_expected(self, spark: SparkSession) -> None:
        input_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "NEW": ["ONE", "TWO", "THREE", "FOUR"],
                    "start": ["ONE", "TWO", "THR", "FOU"],
                    "end": ["ENO", "OWT", "EER", "RUO"],
                }
            )
        )
        expected_output = spark.createDataFrame(
            pd.DataFrame(
                {
                    "NEW": ["ONE", "TWO", "THREE", "FOUR"],
                    "start": ["ONE", "TWO", "THR", "FOU"],
                    "end": ["ENO", "OWT", "EER", "RUO"],
                    "final": ["ONE", "TWO", "THR", "FOU"],
                }
            )
        )
        actual_output = substring(input_df, "final", "NEW", 1, 3)
        assertDataFrameEqual(expected_output, actual_output)


class TestSuffixColumns:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, None, "one", "four", "five"],
                    "col2": [None, None, "one", "four", "five"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, None, "one", "four", "five"],
                    "col2mr": [None, None, "one", "four", "five"],
                }
            )
        )
        result_df = suffix_columns(test_df, suffix="mr", exclude="col1")
        assertDataFrameEqual(intended_df, result_df)


class TestUnionAll:
    def test_expected(self, spark: SparkSession) -> None:
        test_df1 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, None, "one", "four", "five"],
                    "col2": [None, None, "one", "four", "three"],
                }
            )
        )
        test_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, "okay", "dfs", "few", "dfs"],
                    "col2": [None, None, "fdsa", "rew", "trt"],
                }
            )
        )
        test_df3 = spark.createDataFrame(
            pd.DataFrame({"col3": [None, "okay", "dfs", "few", "dfs"]})
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [
                        None,
                        None,
                        "one",
                        "four",
                        "five",
                        None,
                        "okay",
                        "dfs",
                        "few",
                        "dfs",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                    ],
                    "col2": [
                        None,
                        None,
                        "one",
                        "four",
                        "three",
                        None,
                        None,
                        "fdsa",
                        "rew",
                        "trt",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                    ],
                    "col3": [
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        "xd",
                        None,
                        "okay",
                        "dfs",
                        "few",
                        "dfs",
                    ],
                }
            )
        )
        result_df = union_all(test_df1, test_df2, test_df3, fill="xd")
        assertDataFrameEqual(intended_df, result_df)


class TestWindow:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": ["a", "b", "c", "c", "d", "e", "d"],
                    "col2": [1, 1, 2, 2, 1, 1, 1],
                }
            )
        )
        intended_schema = StructType(
            [
                StructField("col1", StringType()),
                StructField("col2", LongType()),
                StructField("new", LongType(), nullable=False),
            ]
        )
        intended_data: list[list[str | int]] = [
            ["c", 2, 2],
            ["c", 2, 2],
            ["a", 1, 1],
            ["b", 1, 1],
            ["e", 1, 1],
            ["d", 1, 2],
            ["d", 1, 2],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        result_df = window(
            test_df, window=["col1", "col2"], target="col2", mode="count", alias="new"
        )
        assertDataFrameEqual(intended_df, result_df)
        test_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": ["a", "b", "c", "c", "d", "e", "d"],
                    "col2": [1, 1, 1, 2, 1, 1, 2],
                }
            )
        )
        intended_schema2 = StructType(
            [
                StructField("col1", StringType()),
                StructField("new", LongType()),
                StructField("col2", LongType()),
            ]
        )
        intended_data2: list[list[str | int]] = [
            ["a", 1, 1],
            ["b", 1, 1],
            ["c", 1, 1],
            ["c", 1, 2],
            ["d", 1, 1],
            ["d", 1, 2],
            ["e", 1, 1],
        ]
        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)
        result_df2 = window(
            test_df2, window=["col1"], target="col2", mode="min", alias="new"
        ).orderBy("col1", "col2")
        assertDataFrameEqual(intended_df2, result_df2)
        intended_schema3 = StructType(
            [
                StructField("col1", StringType()),
                StructField("new", LongType()),
                StructField("col2", LongType()),
            ]
        )
        intended_data3: list[list[str | int]] = [
            ["a", 1, 1],
            ["b", 1, 1],
            ["c", 2, 1],
            ["c", 2, 2],
            ["d", 2, 1],
            ["d", 2, 2],
            ["e", 1, 1],
        ]
        intended_df3 = spark.createDataFrame(intended_data3, intended_schema3)
        result_df3 = window(
            test_df2, window=["col1"], target="col2", mode="max", alias="new"
        ).orderBy("col1", "col2")
        assertDataFrameEqual(intended_df3, result_df3)
        test_df4 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [
                        "a",
                        "b",
                        "c",
                        "c",
                        "d",
                        "e",
                        "d",
                        "c",
                        "c",
                        "c",
                        "d",
                        "d",
                    ],
                    "col2": [1, 1, 1, 2, 1, 1, 2, 5, 6, 7, 11, 12],
                }
            )
        )
        intended_schema4 = StructType(
            [
                StructField("col1", StringType()),
                StructField("new", LongType()),
                StructField("col2", LongType()),
            ]
        )
        intended_data4: list[list[str | int]] = [
            ["a", 1, 1],
            ["b", 1, 1],
            ["c", 7, 1],
            ["c", 7, 2],
            ["c", 7, 5],
            ["c", 7, 6],
            ["c", 7, 7],
            ["d", 12, 1],
            ["d", 12, 2],
            ["d", 12, 11],
            ["d", 12, 12],
            ["e", 1, 1],
        ]
        intended_df4 = spark.createDataFrame(intended_data4, intended_schema4)
        result_df4 = window(
            test_df4, window=["col1"], target="col2", mode="max", alias="new"
        ).orderBy("col1", "col2")
        assertDataFrameEqual(intended_df4, result_df4)
