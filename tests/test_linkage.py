import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.testing import assertDataFrameEqual

from dlh_utils.linkage import (
    alpha_name,
    assert_unique,
    assert_unique_matches,
    blocking,
    clerical_sample,
    cluster_number,
    deterministic_linkage,
    difflib_sequence_matcher,
    extract_mk_variables,
    jaro,
    jaro_winkler,
    matchkey_counts,
    matchkey_dataframe,
    matchkey_join,
    metaphone,
    mk_dropna,
    order_matchkeys,
    soundex,
    std_lev_score,
)


class TestAlphaName:
    def test_expected_1(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("ID", IntegerType()), StructField("Forename", StringType())]
        )
        test_data: list[list[int | str]] = [
            [1, "Homer"],
            [2, "Marge"],
            [3, "Bart"],
            [4, "Lisa"],
            [5, "Maggie"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        intended_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("alpha_name", StringType()),
            ]
        )
        intended_data: list[list[int | str]] = [
            [1, "Homer", "EHMOR"],
            [2, "Marge", "AEGMR"],
            [3, "Bart", "ABRT"],
            [4, "Lisa", "AILS"],
            [5, "Maggie", "AEGGIM"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        result_df = alpha_name(test_df, "Forename", "alpha_name")
        assertDataFrameEqual(intended_df, result_df)

    def test_expected_2(self, spark: SparkSession) -> None:
        test_schema2 = StructType(
            [StructField("ID", IntegerType()), StructField("Name", StringType())]
        )
        test_data2: list[list[int | str | None]] = [
            [1, "Romer, Bogdan"],
            [2, "Margarine"],
            [3, None],
            [4, "Nisa"],
            [5, "Moggie"],
        ]
        test_df2 = spark.createDataFrame(test_data2, test_schema2)
        intended_schema2 = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Name", StringType()),
                StructField("alpha_name", StringType()),
            ]
        )  # Note alpha_name is always returned as nullable=false
        intended_data2: list[list[int | str | None]] = [
            [1, "Romer, Bogdan", " ,ABDEGMNOORR"],
            [2, "Margarine", "AAEGIMNRR"],
            [3, None, None],
            [4, "Nisa", "AINS"],
            [5, "Moggie", "EGGIMO"],
        ]
        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)
        result_df2 = alpha_name(test_df2, "Name", "alpha_name")
        assertDataFrameEqual(intended_df2, result_df2)


class TestAssertUnique:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("colA", IntegerType()), StructField("colB", IntegerType())]
        )
        test_data = [[1, 1], [1, 2]]
        df = spark.createDataFrame(test_data, test_schema)
        try:
            assert_unique(df, "colA")
        except AssertionError:
            pass
        assert_unique(df, ["colB"])


class TestAssertUniqueMatches:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {"id_l": ["1", "2", "3", "4", "5"], "id_r": ["a", "b", "c", "d", "e"]}
            )
        )
        intended_df = None
        result_df = assert_unique_matches(test_df, "id_l", "id_r")
        assert result_df == intended_df
        x = 0
        try:
            assert_unique_matches(test_df, "id_l", "id_r")
        except:
            x = 1
        assert x == 0
        df = spark.createDataFrame(
            pd.DataFrame(
                {"id_l": ["1", "1", "3", "4", "5"], "id_r": ["a", "b", "c", "d", "d"]}
            )
        )
        x = 0
        try:
            assert_unique_matches(df, "id_l", "id_r")
        except:
            x = 1
        assert x == 1


class TestBlocking:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [
                StructField("ID_1", IntegerType()),
                StructField("age_df1", IntegerType()),
                StructField("sex_df1", StringType()),
                StructField("pc_df1", StringType()),
            ]
        )
        test_data: list[list[int | str]] = [
            [1, 1, "Male", "gu1111"],
            [2, 1, "Female", "gu1211"],
            [3, 56, "Male", "gu2111"],
        ]
        df1 = spark.createDataFrame(test_data, test_schema)
        test_schema = StructType(
            [
                StructField("ID_2", IntegerType()),
                StructField("age_df2", IntegerType()),
                StructField("sex_df2", StringType()),
                StructField("pc_df2", StringType()),
            ]
        )
        test_data = [
            [6, 2, "Female", "gu1211"],
            [5, 56, "Male", "gu1411"],
            [4, 7, "Female", "gu1111"],
        ]
        df2 = spark.createDataFrame(test_data, test_schema)
        id_vars = ["ID_1", "ID_2"]
        blocks = {"pc_df1": "pc_df2"}
        result_df = blocking(df1, df2, blocks, id_vars)
        expected_schema = StructType(
            [
                StructField("ID_1", IntegerType()),
                StructField("age_df1", IntegerType()),
                StructField("sex_df1", StringType()),
                StructField("pc_df1", StringType()),
                StructField("ID_2", IntegerType()),
                StructField("age_df2", IntegerType()),
                StructField("sex_df2", StringType()),
                StructField("pc_df2", StringType()),
            ]
        )
        expected_data: list[list[int | str]] = [
            [1, 1, "Male", "gu1111", 4, 7, "Female", "gu1111"],
            [2, 1, "Female", "gu1211", 6, 2, "Female", "gu1211"],
        ]
        expected_df = spark.createDataFrame(expected_data, expected_schema)
        assertDataFrameEqual(expected_df, result_df)


class TestClericalSample:
    def test_expected(self, spark: SparkSession) -> None:
        df_l = spark.createDataFrame(
            pd.DataFrame(
                {
                    "l_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
                    "l_first_name": ["aa", None, "ab", "bb", "aa", "ax", "cr", "cd"],
                    "l_last_name": ["fr", "gr", None, "ga", "gx", "mx", "ra", "ga"],
                }
            )
        )
        df_r = spark.createDataFrame(
            pd.DataFrame(
                {
                    "r_id": ["1", "2", "3", "4", "5", "6", "7", "8"],
                    "r_first_name": ["ax", None, "ad", "bd", "ar", "ax", "cr", "cd"],
                    "r_last_name": ["fr", "gr", "fa", "ga", "gx", "mx", "ra", None],
                }
            )
        )
        mks = [
            [
                df_l["l_first_name"] == df_r["r_first_name"],
                df_l["l_last_name"] == df_r["r_last_name"],
            ],
            [
                sf.substring(df_l["l_first_name"], 1, 1)
                == sf.substring(df_r["r_first_name"], 1, 1),
                df_l["l_last_name"] == df_r["r_last_name"],
            ],
            [
                sf.substring(df_l["l_first_name"], 1, 1)
                == sf.substring(df_r["r_first_name"], 1, 1),
                sf.substring(df_l["l_last_name"], 1, 1)
                == sf.substring(df_r["r_last_name"], 1, 1),
            ],
        ]
        linked_ids = deterministic_linkage(df_l, df_r, "l_id", "r_id", mks, None)
        mk_df = matchkey_dataframe(mks, spark)
        result = clerical_sample(linked_ids, mk_df, df_l, df_r, "l_id", "r_id", n_ids=3)
        result_agg = result.groupby("matchkey").count()
        intended_schema = StructType(
            [
                StructField("matchkey", IntegerType(), nullable=False),
                StructField("count", LongType(), nullable=False),
            ]
        )
        intended_data = [[1, 2], [2, 3]]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assertDataFrameEqual(result_agg, intended_df)


class TestClusterNumber:
    def test_expected(self, spark: SparkSession) -> None:
        """Test may fail due to lack of `graphframes`.

        If this test fails because of the graphframes package not being found,
        make sure you have both graphframes and graphframes_wrapper installed
        via pip3.
        """
        input_data = [
            ["1a", "2b"],
            ["3a", "3b"],
            ["2a", "1b"],
            ["3a", "7b"],
            ["1a", "8b"],
            ["2a", "9b"],
        ]
        input_df = spark.createDataFrame(input_data, schema=["id1", "id2"])
        expected_data: list[list[str | int]] = [
            ["1a", "2b", 2],
            ["1a", "8b", 2],
            ["2a", "1b", 1],
            ["2a", "9b", 1],
            ["3a", "3b", 3],
            ["3a", "7b", 3],
        ]
        expected_output = spark.createDataFrame(
            expected_data, schema="id1: string, id2: string, Cluster_Number: int"
        )
        actual_output = cluster_number(input_df, id_1="id1", id_2="id2")
        assert actual_output is not None
        assertDataFrameEqual(actual_output, expected_output)


class TestDeterministicLinkage:
    def test_deterministic_linkage(self, spark: SparkSession) -> None:
        df_l = spark.createDataFrame(
            pd.DataFrame(
                {
                    "l_id": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "aa",
                        "ba",
                        "ab",
                        "bb",
                        "aa",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        df_r = spark.createDataFrame(
            pd.DataFrame(
                {
                    "r_id": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "ax",
                        "bx",
                        "ad",
                        "bd",
                        "ar",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        mks = [
            [
                df_l["first_name"] == df_r["first_name"],
                df_l["last_name"] == df_r["last_name"],
            ],
            [
                sf.substring(df_l["first_name"], 1, 1)
                == sf.substring(df_r["first_name"], 1, 1),
                df_l["last_name"] == df_r["last_name"],
            ],
            [
                sf.substring(df_l["first_name"], 1, 1)
                == sf.substring(df_r["first_name"], 1, 1),
                sf.substring(df_l["last_name"], 1, 1)
                == sf.substring(df_r["last_name"], 1, 1),
            ],
        ]
        result_df = deterministic_linkage(df_l, df_r, "l_id", "r_id", mks, out_dir=None)
        intended_schema = StructType(
            [
                StructField("l_id", StringType()),
                StructField("r_id", StringType()),
                StructField("matchkey", IntegerType(), nullable=False),
            ]
        )
        intended_data: list[list[str | int]] = [
            ["10", "10", 1],
            ["11", "11", 1],
            ["12", "12", 1],
            ["13", "13", 1],
            ["14", "14", 1],
            ["15", "15", 1],
            ["16", "16", 1],
            ["17", "17", 1],
            ["18", "18", 1],
            ["19", "19", 1],
            ["20", "20", 1],
            ["6", "6", 1],
            ["7", "7", 1],
            ["8", "8", 1],
            ["9", "9", 1],
            ["1", "1", 2],
            ["2", "2", 2],
            ["3", "3", 2],
            ["4", "4", 2],
            ["5", "5", 2],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assertDataFrameEqual(result_df, intended_df, ignoreColumnOrder=True)


class TestDifflibSequenceMatcher:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("string1", StringType()), StructField("string2", StringType())]
        )
        test_data = [
            ["David", "Emily"],
            ["Idrissa", "Emily"],
            ["Edward", "Emily"],
            ["Gordon", "Emily"],
            ["Emma", "Emily"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        result_df = test_df.withColumn(
            "difflib", difflib_sequence_matcher(sf.col("string1"), sf.col("string2"))
        )
        intended_schema = StructType(
            [
                StructField("string1", StringType()),
                StructField("string2", StringType()),
                StructField("difflib", FloatType()),
            ]
        )
        intended_data: list[list[str | float]] = [
            ["David", "Emily", 0.2],
            ["Idrissa", "Emily", 0.16666667],
            ["Edward", "Emily", 0.18181819],
            ["Gordon", "Emily", 0.0],
            ["Emma", "Emily", 0.44444445],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assertDataFrameEqual(intended_df, result_df)


class TestExtractMkVariables:
    def test_expected(self, spark: SparkSession) -> None:
        test_df_l = spark.createDataFrame(
            pd.DataFrame(
                {
                    "l_id": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "aa",
                        "ba",
                        "ab",
                        "bb",
                        "aa",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        test_df_r = spark.createDataFrame(
            pd.DataFrame(
                {
                    "r_id": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "ax",
                        "bx",
                        "ad",
                        "bd",
                        "ar",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        mks = [
            [
                test_df_l["first_name"] == test_df_r["first_name"],
                test_df_l["last_name"] == test_df_r["last_name"],
            ],
            [
                sf.substring(test_df_l["first_name"], 1, 1)
                == sf.substring(test_df_r["first_name"], 1, 1),
                test_df_l["last_name"] == test_df_r["last_name"],
            ],
            [
                sf.substring(test_df_l["first_name"], 1, 1)
                == sf.substring(test_df_r["first_name"], 1, 1),
                sf.substring(test_df_l["last_name"], 1, 1)
                == sf.substring(test_df_r["last_name"], 1, 1),
            ],
        ]
        intended_list = sorted(["first_name", "last_name"])
        result_list = sorted(extract_mk_variables(test_df_l, mks))
        assert result_list == intended_list


class TestJaro:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("string1", StringType()), StructField("string2", StringType())]
        )
        test_data = [
            ["Hello", "HHheello"],
            ["Hello", " h e l l o"],
            ["Hello", "olleH"],
            ["Hello", "H1234"],
            ["Hello", "1234"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        result = test_df.withColumn(
            "jaro", jaro(test_df["string1"], test_df["string2"])
        )
        assert sorted(result.toPandas().loc[:, "jaro"].tolist()) == sorted(
            [0.875, 0.6333333253860474, 0.6000000238418579, 0.46666666865348816, 0.0]
        )


class TestJaroWinkler:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("string1", StringType()), StructField("string2", StringType())]
        )
        test_data = [
            ["Hello", "HHheello"],
            ["Hello", " h e l l o"],
            ["Hello", "olleH"],
            ["Hello", "H1234"],
            ["Hello", "1234"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        result = test_df.withColumn(
            "jaro_winkler", jaro_winkler(test_df["string1"], test_df["string2"])
        )
        assert sorted(result.toPandas().loc[:, "jaro_winkler"].tolist()) == sorted(
            [
                0.887499988079071,
                0.6333333253860474,
                0.6000000238418579,
                0.46666666865348816,
                0.0,
            ]
        )


class TestMatchkeyCounts:
    def test_expected(self, spark: SparkSession) -> None:
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "matchkey": ["1", "1", "3", "4", "4"],
                    "id_r": ["a", "b", "c", "d", "e"],
                }
            )
        ).select("matchkey", "id_r")
        intended_df = spark.createDataFrame(
            pd.DataFrame({"matchkey": ["1", "4", "3"], "count": [2, 2, 1]})
        ).select("matchkey", "count")
        result_df = matchkey_counts(test_df)
        assertDataFrameEqual(intended_df, result_df)


class TestMatchkeyDataframe:
    def test_creates_matchkey_descriptions_dataframe(self, spark: SparkSession) -> None:
        df_l = spark.createDataFrame(
            pd.DataFrame(
                {
                    "first_name": ["test"] * 10,
                    "last_name": ["test"] * 10,
                    "uprn": ["test"] * 10,
                    "date_of_birth": ["test"] * 10,
                }
            )
        )
        df_r = spark.createDataFrame(
            pd.DataFrame(
                {
                    "first_name": ["test"] * 10,
                    "last_name": ["test"] * 10,
                    "uprn": ["test"] * 10,
                    "date_of_birth": ["test"] * 10,
                }
            )
        )
        expected_data: list[list[int | str]] = [
            [
                1,
                "[(first_name=first_name),(last_name=last_name),(uprn=uprn),"
                "(date_of_birth=date_of_birth)]",
            ],
            [
                2,
                "[(substring(first_name,0,2)=substring(first_name,0,2)),"
                "(substring(last_name,0,2)=substring(last_name,0,2)),"
                "(uprn=uprn),(date_of_birth=date_of_birth)]",
            ],
        ]
        expected_schema = StructType(
            [
                StructField("matchkey", LongType()),
                StructField("description", StringType()),
            ]
        )
        mks = [
            [
                df_l["first_name"] == df_r["first_name"],
                df_l["last_name"] == df_r["last_name"],
                df_l["uprn"] == df_r["uprn"],
                df_l["date_of_birth"] == df_r["date_of_birth"],
            ],
            [
                sf.substring(df_l["first_name"], 0, 2)
                == sf.substring(df_r["first_name"], 0, 2),
                sf.substring(df_l["last_name"], 0, 2)
                == sf.substring(df_r["last_name"], 0, 2),
                df_l["uprn"] == df_r["uprn"],
                df_l["date_of_birth"] == df_r["date_of_birth"],
            ],
        ]
        expected = spark.createDataFrame(expected_data, expected_schema)
        actual = matchkey_dataframe(mks, spark)
        assertDataFrameEqual(actual, expected)


class TestMatchkeyJoin:
    def test_expected(self, spark: SparkSession) -> None:
        test_df_1 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "l_id": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "aa",
                        "ba",
                        "ab",
                        "bb",
                        "aa",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        test_df_2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "r_id": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "ax",
                        "bx",
                        "ad",
                        "bd",
                        "ar",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        mks = [
            [
                test_df_1["first_name"] == test_df_2["first_name"],
                test_df_1["last_name"] == test_df_2["last_name"],
            ],
            [
                sf.substring(test_df_1["first_name"], 1, 1)
                == sf.substring(test_df_2["first_name"], 1, 1),
                test_df_1["last_name"] == test_df_2["last_name"],
            ],
            [
                sf.substring(test_df_1["first_name"], 1, 1)
                == sf.substring(test_df_2["first_name"], 1, 1),
                sf.substring(test_df_1["last_name"], 1, 1)
                == sf.substring(test_df_2["last_name"], 1, 1),
            ],
        ]
        intended_schema = StructType(
            [
                StructField("l_id", StringType()),
                StructField("r_id", StringType()),
                StructField("matchkey", IntegerType(), nullable=False),
            ]
        )
        intended_data: list[list[str | int]] = [
            ["7", "7", 1],
            ["15", "15", 1],
            ["8", "8", 1],
            ["5", "5", 1],
            ["18", "18", 1],
            ["9", "9", 1],
            ["10", "10", 1],
            ["12", "12", 1],
            ["13", "13", 1],
            ["14", "14", 1],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        result_df = matchkey_join(test_df_1, test_df_2, "l_id", "r_id", mks[2], 1)
        assertDataFrameEqual(intended_df, result_df)


class TestMetaphone:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("ID", IntegerType()), StructField("Forename", StringType())]
        )
        test_data: list[list[int | str]] = [
            [1, "David"],
            [2, "Idrissa"],
            [3, "Edward"],
            [4, "Gordon"],
            [5, "Emma"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        result_df = metaphone(test_df, "Forename", "metaname")
        intended_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("metaname", StringType()),
            ]
        )
        intended_data = [
            [1, "David", "TFT"],
            [2, "Idrissa", "ITRS"],
            [3, "Edward", "ETWRT"],
            [4, "Gordon", "KRTN"],
            [5, "Emma", "EM"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assertDataFrameEqual(intended_df, result_df)


class TestMkDropna:
    def test_expected(self, spark: SparkSession) -> None:
        test_df_l = spark.createDataFrame(
            pd.DataFrame(
                {
                    "l_id": ["1", "2", None, None, None, "6", "7", "8"],
                    "first_name": ["aa", None, "ab", "bb", "aa", "ax", "cr", "cd"],
                    "last_name": ["fr", "gr", None, "ga", "gx", "mx", "ra", "ga"],
                }
            )
        )
        test_df_r = spark.createDataFrame(
            pd.DataFrame(
                {
                    "r_id": ["1", "2", "3", "4", None, None, None, "8"],
                    "first_name": ["ax", None, "ad", "bd", "ar", "ax", "cr", "cd"],
                    "last_name": ["fr", "gr", "fa", "ga", "gx", "mx", "ra", None],
                }
            )
        )
        mks = [
            [
                test_df_l["first_name"] == test_df_r["first_name"],
                test_df_l["last_name"] == test_df_r["last_name"],
            ],
            [
                sf.substring(test_df_l["first_name"], 1, 1)
                == sf.substring(test_df_r["first_name"], 1, 1),
                test_df_l["last_name"] == test_df_r["last_name"],
            ],
            [
                sf.substring(test_df_l["first_name"], 1, 1)
                == sf.substring(test_df_r["first_name"], 1, 1),
                sf.substring(test_df_l["last_name"], 1, 1)
                == sf.substring(test_df_r["last_name"], 1, 1),
            ],
        ]
        result_df = mk_dropna(df=test_df_l, match_key=mks)
        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "l_id": ["1", None, None, "6", "7", "8"],
                    "first_name": ["aa", "bb", "aa", "ax", "cr", "cd"],
                    "last_name": ["fr", "ga", "gx", "mx", "ra", "ga"],
                }
            )
        )
        assertDataFrameEqual(intended_df, result_df)


class TestOrderMatchkeys:
    def test_expected(self, spark: SparkSession) -> None:
        dfo = spark.createDataFrame(
            pd.DataFrame(
                {
                    "uprn": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "aa",
                        "ba",
                        "ab",
                        "bb",
                        "aa",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        dffn = spark.createDataFrame(
            pd.DataFrame(
                {
                    "uprn": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "20",
                    ],
                    "first_name": [
                        "ax",
                        "bx",
                        "ad",
                        "bd",
                        "ar",
                        "ax",
                        "cr",
                        "cd",
                        "dc",
                        "dx",
                        "ag",
                        "rd",
                        "rf",
                        "rg",
                        "rr",
                        "dar",
                        "dav",
                        "dam",
                        "dax",
                        "dev",
                    ],
                    "last_name": [
                        "fr",
                        "gr",
                        "fa",
                        "ga",
                        "gx",
                        "mx",
                        "ra",
                        "ga",
                        "fg",
                        "gx",
                        "mr",
                        "pr",
                        "ar",
                        "to",
                        "lm",
                        "pr",
                        "pf",
                        "se",
                        "xr",
                        "xf",
                    ],
                }
            )
        )
        mks = [
            [
                sf.substring(dfo["first_name"], 1, 1)
                == sf.substring(dffn["first_name"], 1, 1),
                sf.substring(dfo["last_name"], 1, 1)
                == sf.substring(dffn["last_name"], 1, 1),
            ],
            [
                sf.substring(dfo["first_name"], 1, 1)
                == sf.substring(dffn["first_name"], 1, 1),
                dfo["last_name"] == dffn["last_name"],
            ],
            [
                dfo["first_name"] == dffn["first_name"],
                dfo["last_name"] == dffn["last_name"],
            ],
        ]
        intended_list = [
            [
                dfo["first_name"] == dffn["first_name"],
                dfo["last_name"] == dffn["last_name"],
            ],
            [
                sf.substring(dfo["first_name"], 1, 1)
                == sf.substring(dffn["first_name"], 1, 1),
                dfo["last_name"] == dffn["last_name"],
            ],
            [
                sf.substring(dfo["first_name"], 1, 1)
                == sf.substring(dffn["first_name"], 1, 1),
                sf.substring(dfo["last_name"], 1, 1)
                == sf.substring(dffn["last_name"], 1, 1),
            ],
        ]
        result_list = order_matchkeys(dfo, dffn, mks)
        assert result_list[0] and intended_list[0]
        assert result_list[1] and intended_list[1]
        assert result_list[2] and intended_list[2]


class TestSoundex:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [StructField("ID", IntegerType()), StructField("Forename", StringType())]
        )
        test_data: list[list[int | str]] = [
            [1, "Homer"],
            [2, "Marge"],
            [3, "Bart"],
            [4, "Lisa"],
            [5, "Maggie"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        result_df = soundex(test_df, "Forename", "forename_soundex")
        intended_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("forename_soundex", StringType()),
            ]
        )
        intended_data: list[list[int | str]] = [
            [1, "Homer", "H560"],
            [2, "Marge", "M620"],
            [3, "Bart", "B630"],
            [4, "Lisa", "L200"],
            [5, "Maggie", "M200"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assertDataFrameEqual(intended_df, result_df)

    def test_expected_2(self, spark: SparkSession) -> None:
        test_schema2 = StructType([StructField("Surname", StringType())])
        test_data2: list[list[str | None]] = [
            ["McDonald"],
            [None],
            ["MacDonald"],
            ["MacDougall"],
        ]
        test_df2 = spark.createDataFrame(test_data2, test_schema2)
        result_df2 = soundex(test_df2, "Surname", "soundex")
        intended_schema2 = StructType(
            [StructField("Surname", StringType()), StructField("soundex", StringType())]
        )
        intended_data2: list[list[str | None]] = [
            ["McDonald", "M235"],
            [None, None],
            ["MacDonald", "M235"],
            ["MacDougall", "M232"],
        ]
        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)
        assertDataFrameEqual(intended_df2, result_df2)


class TestStdLevScore:
    def test_expected(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("Forename_2", StringType()),
            ]
        )
        test_data: list[list[int | str]] = [
            [1, "Homer", "Milhouse"],
            [2, "Marge", "Milhouse"],
            [3, "Bart", "Milhouse"],
            [4, "Lisa", "Milhouse"],
            [5, "Maggie", "Milhouse"],
        ]
        test_df = spark.createDataFrame(test_data, test_schema)
        result_df = test_df.withColumn(
            "forename_lev", std_lev_score(sf.col("Forename"), sf.col("Forename_2"))
        )
        intended_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("Forename_2", StringType()),
                StructField("forename_lev", DoubleType()),
            ]
        )
        intended_data: list[list[int | str | float]] = [
            [1, "Homer", "Milhouse", 1 / 8],
            [2, "Marge", "Milhouse", 2 / 8],
            [3, "Bart", "Milhouse", 0 / 8],
            [4, "Lisa", "Milhouse", 2 / 8],
            [5, "Maggie", "Milhouse", 2 / 8],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)
        assertDataFrameEqual(intended_df, result_df)

    def test_expected_2(self, spark: SparkSession) -> None:
        test_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("Forename_2", StringType()),
            ]
        )
        test_data2: list[list[int | str | None]] = [
            [1, "Homer", "Milhouse"],
            [2, "Marge", "Milhouse"],
            [3, "Bart", "Milhouse"],
            [4, "Lisa", "Milhouse"],
            [5, "Maggie", "Milhouse"],
            [6, None, "Milhouse"],
            [7, "Milhouse", None],
            [8, "Milhouse", "Milhouse"],
        ]
        test_df2 = spark.createDataFrame(test_data2, test_schema)
        result_df2 = test_df2.withColumn(
            "forename_lev", std_lev_score(sf.col("Forename"), sf.col("Forename_2"))
        )
        intended_schema = StructType(
            [
                StructField("ID", IntegerType()),
                StructField("Forename", StringType()),
                StructField("Forename_2", StringType()),
                StructField("forename_lev", DoubleType()),
            ]
        )
        intended_data2: list[list[int | str | None | float]] = [
            [1, "Homer", "Milhouse", 1 / 8],
            [2, "Marge", "Milhouse", 2 / 8],
            [3, "Bart", "Milhouse", 0 / 8],
            [4, "Lisa", "Milhouse", 2 / 8],
            [5, "Maggie", "Milhouse", 2 / 8],
            [6, None, "Milhouse", None],
            [7, "Milhouse", None, None],
            [8, "Milhouse", "Milhouse", 1 / 1],
        ]
        intended_df2 = spark.createDataFrame(intended_data2, intended_schema)
        assertDataFrameEqual(intended_df2, result_df2)
