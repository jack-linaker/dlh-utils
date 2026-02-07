from functools import partial
from pathlib import Path

import pandas as pd

from dlh_utils.formatting import (
    apply_styles,
    export_to_excel,
    style_colour_gradient,
    style_map_values,
    style_on_condition,
    style_on_cutoff,
)


class TestApplyStyles:
    def test_one_style_one_column(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        # Default behaviour.
        sdf = apply_styles(df, {style_on_cutoff: "numeric_A"})
        applied = getattr(sdf, "_applied_styles", [])

        # Only one style was applied.
        assert len(applied) == 1

        # Style was `style_on_cutoff`.
        assert applied[0][0] is style_on_cutoff

        # Style was applied to numeric_A only.
        assert applied[0][1] == ["numeric_A"]

    def test_one_style_multiple_columns(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        # Default behaviour with multiple columns.
        sdf = apply_styles(df, {style_on_condition: ["numeric_A", "numeric_B"]})
        applied = getattr(sdf, "_applied_styles", [])

        # One style was applied.
        assert len(applied) == 1

        # Styles were `style_on_condition`.
        assert applied[0][0] is style_on_condition

        # Styles were applied to the appropriate columns.
        assert set(applied[0][1]) == {"numeric_A", "numeric_B"}

    def test_partial_function(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )
        function = partial(style_on_condition, property="color")
        sdf = apply_styles(df, {function: "numeric_A"})
        applied = getattr(sdf, "_applied_styles", [])

        # One style was applied.
        assert len(applied) == 1

        # The right function was applied.
        assert applied[0][0] is function

        # Style was applied to numeric_A only.
        assert applied[0][1] == ["numeric_A"]

    def test_custom_function(self) -> None:
        df = pd.DataFrame(
            {
                "firstname": [None, "Claire", "Josh", "Bob"],
                "lastname": ["Jones", None, "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, None, 4.3],
                "numeric_B": [-500, 221, None, 0],
            }
        )

        def udf(x: str) -> bool:
            return "a" in x.lower()

        sdf = apply_styles(df, {udf: "lastname"})
        applied = getattr(sdf, "_applied_styles", [])

        # One style was applied.
        assert len(applied) == 1

        # The right function was applied.
        assert applied[0][0] is udf

        # Style was applied to lastname only.
        assert applied[0][1] == ["lastname"]


class TestExportToExcel:
    def test_export_one_sheet_df_no_formatting(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Alan", "Claire", "Josh", "Bob"],
                "lastname": ["Jones", "Llewelyn", "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, 0, 4.3],
                "numeric_B": [-500, 221, 1, 0],
            }
        )
        out_file = tmp_path / "test.xlsx"
        wb = export_to_excel({"Sheet1": df}, local_path=str(out_file))
        assert len(wb.worksheets) == 1
        for i, c in enumerate(df.columns):
            assert (wb["Sheet1"].cell(1, i + 1).value) == c
        for row_num in range(df.shape[0]):
            for i, c in enumerate(df.columns):
                a = wb["Sheet1"].cell(row_num + 2, i + 1).value
                b = df.loc[row_num, c]
                assert a == b

    def test_export_list_of_columns_one_sheet_df(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Alan", "Claire", "Josh", "Bob"],
                "lastname": ["Jones", "Llewelyn", "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, 0, 4.3],
                "numeric_B": [-500, 221, 1, 0],
            }
        )
        out_file = tmp_path / "test.xlsx"
        wb = export_to_excel(
            {"Sheet1": df}, columns=["firstname", "numeric_A"], local_path=str(out_file)
        )
        assert len(wb.worksheets) == 1
        for i, c in enumerate(["firstname", "numeric_A"]):
            assert (wb["Sheet1"].cell(1, i + 1).value) == c
        for row_num in range(df.shape[0]):
            for i, c in enumerate(["firstname", "numeric_A"]):
                a = wb["Sheet1"].cell(row_num + 2, i + 1).value
                b = df.loc[row_num, c]
                assert a == b

    def test_export_two_dfs_two_sheets(self, tmp_path: Path) -> None:
        df_a = pd.DataFrame(
            {
                "firstname": ["Alan", "Claire", "Josh", "Bob"],
                "lastname": ["Jones", "Llewelyn", "Jackson", "Evans"],
                "numeric_A": [1.98, -2.1, 0, 4.3],
                "numeric_B": [-500, 221, 1, 0],
            }
        )
        df_b = pd.DataFrame(
            {
                "firstname": ["Anne", "Betty", "Carlo", "Daphne"],
                "lastname": ["Abbot", "Benson", "Carruthers", "De Morgan"],
                "numeric_A": [1, 2, 3, 4],
                "numeric_B": [91, 92, 93, 94],
            }
        )
        out_file = tmp_path / "test.xlsx"
        wb = export_to_excel(
            {"Dataframe A": df_a, "Dataframe B": df_b}, local_path=str(out_file)
        )
        expected_number_of_sheets = 2
        assert len(wb.worksheets) == expected_number_of_sheets
        for i, c in enumerate(df_a.columns):
            assert (wb["Dataframe A"].cell(1, i + 1).value) == c
        for row_num in range(df_a.shape[0]):
            for i, c in enumerate(df_a.columns):
                a = wb["Dataframe A"].cell(row_num + 2, i + 1).value
                b = df_a.loc[row_num, c]
                assert a == b
        for i, c in enumerate(df_b.columns):
            assert (wb["Dataframe B"].cell(1, i + 1).value) == c
        for row_num in range(df_b.shape[0]):
            for i, c in enumerate(df_b.columns):
                a = wb["Dataframe B"].cell(row_num + 2, i + 1).value
                b = df_b.loc[row_num, c]
                assert a == b

    def test_export_one_sheet_df_specify_columns(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Anne", "Betty", "Carlo", "Daphne"],
                "lastname": ["Abbot", "Benson", "Carruthers", "De Morgan"],
                "numeric_A": [1, 2, 3, 4],
                "numeric_B": [91, 92, 93, 94],
            }
        )
        column_list = ["numeric_B", "lastname"]
        out_file = tmp_path / "test.xlsx"
        wb = export_to_excel(
            {"Dataframe C": df},
            local_path=str(out_file),
            columns={"Dataframe C": column_list},
        )
        assert len(wb.worksheets) == 1
        for i, c in enumerate(column_list):
            assert (wb["Dataframe C"].cell(1, i + 1).value) == c
        for row_num in range(df.shape[0]):
            for i, c in enumerate(column_list):
                a = wb["Dataframe C"].cell(row_num + 2, i + 1).value
                b = df.loc[row_num, c]
                assert a == b

    def test_export_one_sheet_df_with_basic_formatting(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "firstname": ["Anne", "Betty", "Carlo", "Daphne"],
                "lastname": ["Abbot", "Benson", "Carruthers", "De Morgan"],
                "numeric_A": [-1, 2, -3, 4],
                "numeric_B": [91, 92, 93, 94],
            }
        )
        out_file = tmp_path / "test.xlsx"
        wb = export_to_excel(
            {"Dataframe D": df},
            local_path=str(out_file),
            styles={"Dataframe D": {style_on_cutoff: "numeric_A"}},
        )
        assert len(wb.worksheets) == 1
        assert wb["Dataframe D"].cell(2, 3).has_style
        assert wb["Dataframe D"].cell(2, 3).fill.fgColor.rgb == "00FF0000"


class TestStyleColourGradient:
    def test_min_value(self) -> None:
        result = style_colour_gradient(
            1, 1, 10, min_colour="FF0000", max_colour="00DDFF", error_colour=None
        )
        assert result == "background-color : #FF0000;"

    def test_max_value(self) -> None:
        result = style_colour_gradient(
            10,
            1,
            10,
            min_colour="FF0000",
            max_colour="00DDFF",
            error_colour=None,
            property="color",
        )
        assert result == "color : #00DDFF;"

    def test_intermediate_value(self) -> None:
        result = style_colour_gradient(
            4, 1, 10, min_colour="FF0000", max_colour="00DDFF", error_colour=None
        )
        assert result == "background-color : #AA4955;"

    def test_error_value(self) -> None:
        result = style_colour_gradient(
            "ERROR",
            1,
            10,
            min_colour="FF0000",
            max_colour="00DDFF",
            error_colour="AAAAAA",
        )
        assert result == "background-color : #AAAAAA;"


class TestStyleMapValues:
    def test_numeric(self) -> None:
        result = style_map_values(1, {0: "black", 1: "red"})
        assert result == "background-color : red;"

    def test_boolean(self) -> None:
        result = style_map_values(
            value=True,
            mapping_dictionary={True: "black", False: "red"},
            property="color",
        )
        assert result == "color : black;"

        # NB 1 is truthy in Python, so this is expected behaviour.
        result = style_map_values(1, {True: "black", False: "red"}, property="color")
        assert result == "color : black;"

    def test_default_style(self) -> None:
        result = style_map_values(
            2, {0: "black", 1: "red"}, property="color", default_style="green"
        )
        assert result == "color : green;"

    def test_error_style(self) -> None:
        """Test that the function applies the error style correctly.

        A TypeError arises because [1] is a list, which cannot be used
        to look up a key in a dictionary. This is "swallowed" by
        style_map_values, which instead returns the error_style.
        """
        result = style_map_values(
            [1], {True: "black", False: "red"}, property="color", error_style="green"
        )
        assert result == "color : green;"


class TestStyleOnCondition:
    def test_default_behaviour(self) -> None:
        result = style_on_condition(0)
        assert result == "font-weight : bold;"
        result = style_on_condition(1)
        assert result == "font-weight : normal;"

    def test_custom_property(self) -> None:
        result = style_on_condition(
            1,
            property="color",
            true_style="'red'",
            false_style="'blue'",
            error_style=None,
        )
        assert result == "color : 'blue';"

    def test_custom_condition(self) -> None:
        result = style_on_condition(10, condition=lambda x: x % 2 == 0)
        assert result == "font-weight : bold;"


class TestStyleOnCutoff:
    def test_default_behaviour(self) -> None:
        result = style_on_cutoff(5)
        assert result == "background-color : green;"
        result = style_on_cutoff(0)
        assert result == "background-color : white;"
        result = style_on_cutoff(-5)
        assert result == "background-color : red;"
        result = style_on_cutoff("ERROR")
        assert result == "background-color : black;"

    def test_custom_behaviour(self) -> None:
        result = style_on_cutoff(
            5,
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color",
        )
        assert result == "color : #AABBAA;"
        result = style_on_cutoff(
            3,
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color",
        )
        assert result == "color : #33DD33;"
        result = style_on_cutoff(
            -5,
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color",
        )
        assert result == "color : #00FF00;"
        result = style_on_cutoff(
            "ERROR",
            cutoff=3,
            negative_style="#00FF00",
            positive_style="#AABBAA",
            zero_style="#33DD33",
            error_style="green",
            property="color",
        )
        assert result == "color : green;"
