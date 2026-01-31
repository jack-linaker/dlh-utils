import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.testing import assertDataFrameEqual

from dlh_utils.standardisation import (
    add_leading_zeros,
    age_at,
    align_forenames,
    cast_type,
    clean_forename,
    clean_hyphens,
    clean_surname,
    fill_nulls,
    group_single_characters,
    max_hyphen,
    max_white_space,
    reg_replace,
    remove_punctuation,
    replace,
    standardise_case,
    standardise_date,
    standardise_null,
    standardise_white_space,
    trim,
)


class TestAddLeadingZeros:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": ["1-2-12", "2-2-12", "3-2-12", "4-2-12", None],
                    "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                    "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                }
            )
        )
        result_df = add_leading_zeros(test_df, subset=["before1"], n=7)
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestAgeAt:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [
                StructField("ID", LongType()),
                StructField("Forename", StringType()),
                StructField("Surname", StringType()),
                StructField("DOB", StringType()),
            ]
        )
        test_data: list[list[int | str]] = [
            [1, "Homer", "Simpson", "1983-05-12"],
            [2, "Marge", "Simpson", "1993-03-19"],
            [3, "Bart", "Simpson", "2012-04-01"],
            [4, "Lisa", "Simpson", "2014-05-09"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        expected_schema = StructType(
            [
                StructField("ID", LongType()),
                StructField("Forename", StringType()),
                StructField("Surname", StringType()),
                StructField("DOB", StringType()),
                StructField("DoB_age_at_2022-11-03", IntegerType()),
            ]
        )
        expected_data: list[list[int | str]] = [
            [1, "Homer", "Simpson", "1983-05-12", 39],
            [2, "Marge", "Simpson", "1993-03-19", 29],
            [3, "Bart", "Simpson", "2012-04-01", 10],
            [4, "Lisa", "Simpson", "2014-05-09", 8],
        ]
        intended_df = spark.createDataFrame(expected_data, expected_schema)
        dates = ["2022-11-03"]
        result_df = age_at(test_df, "DoB", "yyyy-MM-dd", *dates)
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestAlignForenames:
    def test_expected_1(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "first_name": [
                        "David Joe",
                        "Dan James",
                        "Neil Oliver",
                        "Rich",
                        "Rachel",
                    ],
                    "middle_name": [" ", "Jim", "", "Fred", "Amy"],
                    "id": [101, 102, 103, 104, 105],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "first_name": ["David", "Dan", "Neil", "Rich", "Rachel"],
                    "middle_name": ["Joe", "James Jim", "Oliver", "Fred", "Amy"],
                    "id": [101, 102, 103, 104, 105],
                }
            )
        )
        result_df = align_forenames(test_df, "first_name", "middle_name", "id", sep=" ")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)

    def test_expected_2(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "identifier": [1, 2, 3, 4],
                    "firstName": [
                        "robert green",
                        "andrew",
                        "carlos senior",
                        "john wick",
                    ],
                    "middleName": [None, "hog", None, ""],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "identifier": [1, 2, 3, 4],
                    "firstName": ["robert", "andrew", "carlos", "john"],
                    "middleName": ["green", "hog", "senior", "wick"],
                }
            )
        )
        result_df = align_forenames(test_df, "firstName", "middleName", "identifier")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestCastType:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {"before": [None, "2", "3", "4", "5"], "after": [None, 2, 3, 4, 5]}
            )
        )
        intended_schema = StructType(
            [StructField("after", StringType()), StructField("before", StringType())]
        )
        intended_data: list[list[float | None]] = [
            [float("NaN"), None],
            [2.0, 2],
            [3.0, 3],
            [4.0, 4],
            [5.0, 5],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        # Check if it is string first.
        result_df = cast_type(test_df, subset="after", types="string")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_schema = StructType(
            [StructField("after", DoubleType()), StructField("before", IntegerType())]
        )
        intended_data = [[float("NaN"), None], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5]]
        intended_df2 = spark.createDataFrame(intended_data, intended_schema)

        # Check if columns are the same after various conversions.
        result_df2 = cast_type(test_df, subset="before", types="int")
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)


class TestCleanForename:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["MISS Maddie", "MR GEORGE", "DR Paul", "NO NAME"],
                    "after": [" Maddie", " GEORGE", " Paul", ""],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [" Maddie", " GEORGE", " Paul", ""],
                    "after": [" Maddie", " GEORGE", " Paul", ""],
                }
            )
        )
        result_df = clean_forename(test_df, "before")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestCleanHyphens:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, "", "th- ree", "--fo - ur", "fi -ve-"],
                    "before2": [None, "", "th- ree", "fo - ur", "fi -ve"],
                    "after": [None, "", "th-ree", "fo-ur", "fi-ve"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, "", "th-ree", "fo-ur", "fi-ve"],
                    "before2": [None, "", "th- ree", "fo - ur", "fi -ve"],
                    "after": [None, "", "th-ree", "fo-ur", "fi-ve"],
                }
            )
        )
        result_df = clean_hyphens(test_df, subset="before1")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestCleanSurname:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["O Leary", "VAN DER VAL", "SURNAME", "MC CREW"],
                    "after": ["OLeary", "VANDERVAL", "", "MCCREW"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["OLeary", "VANDERVAL", "", "MCCREW"],
                    "after": ["OLeary", "VANDERVAL", "", "MCCREW"],
                }
            )
        )
        result_df = clean_surname(test_df, "before")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestFillNulls:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["abcd", None, "fg", ""],
                    "numeric": [1, 2, None, 3],
                    "after": ["abcd", None, "fg", ""],
                    "after_numeric": [1, 2, 0, 3],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["abcd", "0", "fg", ""],
                    "numeric": [1.0, 2.0, 0.0, 3.0],
                    "after": ["abcd", "0", "fg", ""],
                    "after_numeric": [1, 2, 0, 3],
                }
            )
        )
        result_df = fill_nulls(test_df, 0)
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestGroupSingleCharacters:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [
                        None,
                        "",
                        "-t-h r e e",
                        "four",
                        "f i v e",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "t  e    n",
                        "e leve n",
                    ],
                    "before2": [
                        None,
                        "",
                        "-t-h r e e",
                        "four",
                        "f i v e",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "t  e    n",
                        "e leve n",
                    ],
                    "after": [
                        None,
                        "",
                        "-t-h ree",
                        "four",
                        "five",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "ten",
                        "e leve n",
                    ],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [
                        None,
                        "",
                        "-t-h ree",
                        "four",
                        "five",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "ten",
                        "e leve n",
                    ],
                    "before2": [
                        None,
                        "",
                        "-t-h r e e",
                        "four",
                        "f i v e",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "t  e    n",
                        "e leve n",
                    ],
                    "after": [
                        None,
                        "",
                        "-t-h ree",
                        "four",
                        "five",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "ten",
                        "e leve n",
                    ],
                }
            )
        )
        result_df = group_single_characters(test_df, subset="before1")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)

    def test_expected_include_terminals(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [
                        None,
                        "",
                        "-t-h r e e",
                        "four",
                        "f i v e",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "t  e    n",
                        "e leve n",
                    ],
                    "before2": [
                        None,
                        "",
                        "-t-h r e e",
                        "four",
                        "f i v e",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "t  e    n",
                        "e leve n",
                    ],
                    "after": [
                        None,
                        "",
                        "-t-h ree",
                        "four",
                        "five",
                        "six ",
                        " seven",
                        "eight",
                        "nine",
                        "ten",
                        "eleven",
                    ],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [
                        None,
                        "",
                        "-t-h ree",
                        "four",
                        "five",
                        "six ",
                        " seven",
                        "eight",
                        "nine",
                        "ten",
                        "eleven",
                    ],
                    "before2": [
                        None,
                        "",
                        "-t-h r e e",
                        "four",
                        "f i v e",
                        "six ",
                        " seven",
                        "eigh t",
                        "n ine",
                        "t  e    n",
                        "e leve n",
                    ],
                    "after": [
                        None,
                        "",
                        "-t-h ree",
                        "four",
                        "five",
                        "six ",
                        " seven",
                        "eight",
                        "nine",
                        "ten",
                        "eleven",
                    ],
                }
            )
        )
        result_df = group_single_characters(
            test_df, subset="before1", include_terminals=True
        )
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestMaxHyphen:
    def test_expected(self, spark: SparkSession) -> None:
        # max_hyphen gets rid of any hyphens that does not match or is
        # under the limit.
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agent-----john",
                    ],
                    "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                    "after4": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agentjohn",
                    ],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james--brad",
                        "tom--ridley",
                        "chicken-wing",
                        "agent--john",
                    ],
                    "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                    "after4": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agentjohn",
                    ],
                }
            )
        )
        result_df = max_hyphen(test_df, limit=2, subset=["before"])
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agent----john",
                    ],
                    "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                    "after4": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agentjohn",
                    ],
                }
            )
        )
        result_df2 = max_hyphen(test_df, limit=4, subset=["before"])
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)


class TestMaxWhiteSpace:
    def test_expected(self, spark: SparkSession) -> None:
        # max_white_space gets rid of any whitespace that does not match
        # or is under the limit.
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agent     john",
                    ],
                    "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after4": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after4": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                }
            )
        )
        result_df = max_white_space(test_df, limit=2, subset=["before"])
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                    "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after4": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                }
            )
        )
        result_df2 = max_white_space(test_df, limit=4, subset=["before"])
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)


class TestRegReplace:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, "hello str", "king strt", "king road"],
                    "col2": [None, "bond street", "queen street", "queen avenue"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "col1": [None, "bond street", "queen street", "queen avenue"],
                    "col2": [None, "bond street", "queen street", "queen avenue"],
                }
            )
        )
        result_df = reg_replace(
            test_df,
            replace_dict={
                "street": "\\bstr\\b|\\bstrt\\b",
                "avenue": "road",
                "bond": "hello",
                "queen": "king",
            },
        )
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestRemovePunct:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "after": ["ONE", "TWO", "THREE", "FOUR", "FI^VE"],
                    "before": [None, 'TW""O', "TH@REE", "FO+UR", "FI@^VE"],
                    "extra": [None, "TWO", "TH@REE", "FO+UR", "FI@^VE"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "after": ["ONE", "TWO", "THREE", "FOUR", "FI^VE"],
                    "before": [None, "TWO", "THREE", "FOUR", "FI^VE"],
                    "extra": [None, "TWO", "TH@REE", "FO+UR", "FI@^VE"],
                }
            )
        )
        result_df = remove_punctuation(test_df, keep="^", subset=["after", "before"])
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)


class TestReplace:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["a", None, "c", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, None, "f", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        result_df = replace(
            test_df, subset="before", replace_dict={"a": None, "c": "f"}
        )
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, None, "f", ""],
                    "before1": [None, "b", "f", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        result_df2 = replace(
            test_df, subset=["before", "before1"], replace_dict={"a": None, "c": "f"}
        )
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)

    def test_expected_with_join(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["a", None, "c", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        print("Test")
        test_df.show()
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["A", None, "f", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        print("Intended")
        intended_df.show()
        result_df = replace(
            test_df, subset="before", replace_dict={"a": "A", "c": "f"}, use_join=True
        )
        print("Result")
        result_df.show()
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)

    def test_expected_with_regex(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["alan", None, "betty", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["A", None, "Y", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        result_df = replace(
            test_df,
            subset="before",
            replace_dict={"^a": "A", "y$": "Y"},
            use_regex=True,
        )
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)

    def test_value_error_on_join_and_regex(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["alan", None, "betty", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        with pytest.raises(ValueError):
            replace(
                test_df,
                subset="before",
                replace_dict={"^a": "A", "y$": "Y"},
                use_regex=True,
                use_join=True,
            )

    def test_value_error_on_join_and_none(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["alan", None, "betty", ""],
                    "before1": ["a", "b", "c", "d"],
                    "after": [None, None, "f", ""],
                    "after1": [None, "b", "f", "d"],
                }
            )
        )
        with pytest.raises(ValueError):
            replace(
                test_df,
                subset="before",
                replace_dict={"^a": "A", None: "Y"},
                use_join=True,
            )


class TestStandardiseCase:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "upper": ["ONE", "TWO", "THREE"],
                    "lower": ["one", "two", "three"],
                    "title": ["One", "Two", "Three"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "upper": ["ONE", "TWO", "THREE"],
                    "lower": ["ONE", "TWO", "THREE"],
                    "title": ["One", "Two", "Three"],
                }
            )
        )
        result_df = standardise_case(test_df, subset="lower", val="upper")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "upper": ["one", "two", "three"],
                    "lower": ["one", "two", "three"],
                    "title": ["One", "Two", "Three"],
                }
            )
        )
        result_df2 = standardise_case(test_df, subset="upper", val="lower")
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)
        intended_df3 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "upper": ["ONE", "TWO", "THREE"],
                    "lower": ["One", "Two", "Three"],
                    "title": ["One", "Two", "Three"],
                }
            )
        )
        result_df3 = standardise_case(test_df, subset="lower", val="title")
        assertDataFrameEqual(intended_df3, result_df3, ignoreColumnOrder=True)


class TestStandardiseDate:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, "14-05-1996", "15-04-1996"],
                    "after": [None, "1996-05-14", "1996-04-15"],
                    "slashed": [None, "14/05/1996", "15/04/1996"],
                    "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, "1996-05-14", "1996-04-15"],
                    "after": [None, "1996-05-14", "1996-04-15"],
                    "slashed": [None, "14/05/1996", "15/04/1996"],
                    "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                }
            )
        )
        result_df = standardise_date(test_df, col_name="before")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, "14/05/1996", "15/04/1996"],
                    "after": [None, "1996-05-14", "1996-04-15"],
                    "slashed": [None, "14/05/1996", "15/04/1996"],
                    "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                }
            )
        )
        result_df2 = standardise_date(
            test_df, col_name="before", out_date_format="dd/MM/yyyy"
        )
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)
        intended_df3 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, "14-05-1996", "15-04-1996"],
                    "after": [None, "1996-05-14", "1996-04-15"],
                    "slashed": [None, "14/05/1996", "15/04/1996"],
                    "slashedReverse": [None, "14-05-1996", "15-04-1996"],
                }
            )
        )
        result_df3 = standardise_date(
            test_df,
            col_name="slashedReverse",
            in_date_format="yyyy/MM/dd",
            out_date_format="dd-MM-yyyy",
        )
        assertDataFrameEqual(intended_df3, result_df3, ignoreColumnOrder=True)
        intended_df4 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [None, "14/05/1996", "15/04/1996"],
                    "after": [None, "1996-05-14", "1996-04-15"],
                    "slashed": [None, "14/05/1996", "15/04/1996"],
                    "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                }
            )
        )
        result_df4 = standardise_date(
            test_df,
            col_name="before",
            in_date_format="dd-MM-yyyy",
            out_date_format="dd/MM/yyyy",
        )
        assertDataFrameEqual(intended_df4, result_df4, ignoreColumnOrder=True)


class TestStandardiseNull:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, "", "  ", "-999", "####", "KEEP"],
                    "before2": [None, "", "  ", "-999", "####", "KEEP"],
                    "after": [None, None, None, None, None, "KEEP"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, None, None, None, None, "KEEP"],
                    "before2": [None, "", "  ", "-999", "####", "KEEP"],
                    "after": [None, None, None, None, None, "KEEP"],
                }
            )
        )
        result_df = standardise_null(
            test_df, replace="^-[0-9]|^[#]+$|^$|^\\s*$", subset="before1", regex=True
        )
        assertDataFrameEqual(intended_df, result_df)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, "", "  ", None, "####", "KEEP"],
                    "before2": [None, "", "  ", "-999", "####", "KEEP"],
                    "after": [None, None, None, None, None, "KEEP"],
                }
            )
        )
        result_df2 = standardise_null(
            test_df, replace="-999", subset="before1", regex=False
        )
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)


class TestStandardiseWhiteSpace:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after": [
                        None,
                        "hello yes",
                        "hello yes",
                        "hello yes",
                        "hello yes",
                    ],
                    "before2": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after2": [
                        None,
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                    ],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        None,
                        "hello yes",
                        "hello yes",
                        "hello yes",
                        "hello yes",
                    ],
                    "after": [
                        None,
                        "hello yes",
                        "hello yes",
                        "hello yes",
                        "hello yes",
                    ],
                    "before2": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after2": [
                        None,
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                    ],
                }
            )
        )
        result_df = standardise_white_space(test_df, subset="before", wsl="one")
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after": [
                        None,
                        "hello yes",
                        "hello yes",
                        "hello yes",
                        "hello yes",
                    ],
                    "before2": [
                        None,
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                    ],
                    "after2": [
                        None,
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                    ],
                }
            )
        )
        result_df2 = standardise_white_space(test_df, subset="before2", fill="_")
        assertDataFrameEqual(intended_df2, result_df2, ignoreColumnOrder=True)

    def test_expected_wsl_none(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after": [
                        None,
                        "hello yes",
                        "hello yes",
                        "hello yes",
                        "hello yes",
                    ],
                    "before2": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after2": [
                        None,
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                    ],
                }
            )
        )
        intended_df3 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        None,
                        "hello  yes",
                        "hello yes",
                        "hello   yes",
                        "hello yes",
                    ],
                    "after": [
                        None,
                        "hello yes",
                        "hello yes",
                        "hello yes",
                        "hello yes",
                    ],
                    "before2": [
                        None,
                        "helloyes",
                        "helloyes",
                        "helloyes",
                        "helloyes",
                    ],
                    "after2": [
                        None,
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                        "hello_yes",
                    ],
                }
            )
        )
        result_df3 = standardise_white_space(test_df, subset="before2", wsl="none")
        assertDataFrameEqual(intended_df3, result_df3, ignoreColumnOrder=True)


class TestTrim:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, "", " th re e", "  four ", "  f iv  e "],
                    "before2": [None, " ", " th re e", "  four ", "  f iv  e "],
                    "numeric": [1, 2, 3, 4, 5],
                    "after": [None, "", "th re e", "four", "f iv  e"],
                }
            )
        )
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before1": [None, "", "th re e", "four", "f iv  e"],
                    "before2": [None, " ", " th re e", "  four ", "  f iv  e "],
                    "numeric": [1, 2, 3, 4, 5],
                    "after": [None, "", "th re e", "four", "f iv  e"],
                }
            )
        )
        result_df = trim(test_df, subset=["before1", "numeric", "after"])
        assertDataFrameEqual(intended_df, result_df, ignoreColumnOrder=True)
