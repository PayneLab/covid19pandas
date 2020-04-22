#   Copyright 2018 Samuel Payne sam_payne@byu.edu
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import covid19pandas as cod
import covid19pandas.exceptions as codex

import pandas as pd
import numpy as np
import datetime
import pytest

formats = ["wide", "long"]
jhu_data_types = ["all", "cases", "deaths", "recovered"]
jhu_regions = ["global", "us"]

nyt_data_types = ["all", "cases", "deaths"]
nyt_county_options = [True, False]

@pytest.mark.filterwarnings("ignore::covid19pandas.exceptions.FileNotUpdatedWarning")
class TestSelectors:

    @classmethod
    def setup_class(cls):
        """Ensures that all data tables have been recently downloaded, so we can skip the update in all our tests to improve speed."""
        cod.get_data_jhu(data_type="all", region="global", update=True)
        cod.get_data_jhu(data_type="all", region="us", update=True)

        cod.get_data_nyt(data_type="all", counties=False, update=True)
        cod.get_data_nyt(data_type="all", counties=True, update=True)

        # Set pandas display options for when we print tables
        pd.options.display.max_rows = 60
        pd.options.display.max_columns = 10
        pd.options.display.width = None
        pd.options.display.min_rows = None

    # -------------------------------------------------------------------------------------------------------------
    # Tests for select_top_x_regions
    # -------------------------------------------------------------------------------------------------------------
    def test_select_top_x_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    if (region == "us" and data_type == "recovered") or (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=False)

                        if data_type == "all":
                            compare_by_types = set(jhu_data_types)
                            compare_by_types.remove("all")
                            if region == "us":
                                compare_by_types.remove("recovered")

                            for compare_by_type in compare_by_types:
                                self._check_select_top_x(df, compare_by_type)
                        else:
                            self._check_select_top_x(df, data_type)

    def test_select_top_x_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)

                        if data_type == "all":
                            for compare_by_type in [type for type in nyt_data_types if type != "all"]:
                                self._check_days_since(df, compare_by_type)
                        else:
                            self._check_select_top_x(df, data_type)

    # -------------------------------------------------------------------------------------------------------------
    # Tests for select_regions
    # -------------------------------------------------------------------------------------------------------------
    def test_select_regions_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    if (region == "us" and data_type == "recovered") or (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=False)
                        self._check_select_regions(df)

    def test_select_regions_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)
                        self._check_select_regions(df)

    # -------------------------------------------------------------------------------------------------------------
    # Tests for calc_x_day_avg
    # -------------------------------------------------------------------------------------------------------------
    def test_calc_x_day_avg_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    if (region == "us" and data_type == "recovered") or (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=False)
                        self._check_calc_x_day_avg(df, data_type)

    def test_calc_x_day_avg_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)
                        self._check_calc_x_day_avg(df, data_type)

    # -------------------------------------------------------------------------------------------------------------
    # Tests for calc_daily_change
    # -------------------------------------------------------------------------------------------------------------
    def test_calc_daily_change_jhu_long(self):
        for data_type in jhu_data_types:
            for region in jhu_regions:
                if region == "us" and data_type == "recovered":
                    pass # Invalid table parameter combination
                else:
                    df = cod.get_data_jhu(format="long", data_type=data_type, region=region, update=False)
                    self._check_daily_change(df, data_type, format="long")

    def test_calc_daily_change_jhu_wide(self):
        for data_type in jhu_data_types:
            for region in jhu_regions:
                if (region == "us" and data_type == "recovered") or data_type == "all":
                    pass # Invalid table parameter combination
                else:
                    df = cod.get_data_jhu(format="wide", data_type=data_type, region=region, update=False)
                    self._check_daily_change(df, data_type, format="wide")

    def test_calc_daily_change_nyt_long(self):
        for data_type in nyt_data_types:
            for county_option in nyt_county_options:
                df = cod.get_data_nyt(format="long", data_type=data_type, counties=county_option, update=False)
                self._check_daily_change(df, data_type, format="long")

    def test_calc_daily_change_nyt_wide(self):
        for data_type in nyt_data_types:
            for county_option in nyt_county_options:
                if data_type == "all":
                    pass # Invalid table parameter combination
                else:
                    df = cod.get_data_nyt(format="wide", data_type=data_type, counties=county_option, update=False)
                    self._check_daily_change(df, data_type, format="wide")
    # -------------------------------------------------------------------------------------------------------------
    # Tests for calc_days_since_min_count
    # -------------------------------------------------------------------------------------------------------------
    def test_calc_days_since_min_count_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    if (region == "us" and data_type == "recovered") or (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=False)
                        if data_type == "all":

                            count_by_types = set(jhu_data_types)
                            count_by_types.remove("all")
                            if region == "us":
                                count_by_types.remove("recovered")

                            for count_by_type in count_by_types:
                                self._check_days_since(df, count_by_type)
                        else:
                            self._check_days_since(df, data_type)

    def test_calc_days_since_min_count_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)
                        if data_type == "all":
                            for count_by_type in [type for type in nyt_data_types if type != "all"]:
                                self._check_days_since(df, count_by_type)
                        else:
                            self._check_days_since(df, data_type)

    # -------------------------------------------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------------------------------------------
    def _check_select_top_x(self, df, data_type):

        # Search for defined region cols (based on data source)
        if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
            region_col = "Country/Region"
            exclude = ["US", "China"]
        elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
            region_col = "Province_State"
            exclude = ["Washington", "Illinois"]
        elif {"state"}.issubset(df.columns): # NYT USA state only or states and counties table.
            region_col = "state"
            exclude = ["Washington", "Illinois"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        # Call the function
        top = cod.select_top_x_regions(df, region_col=region_col, data_type=data_type, x=10, combine_subregions=True)        
        top_uncombined = cod.select_top_x_regions(df, region_col=region_col, data_type=data_type, x=10, combine_subregions=False)        
        top_with_exclusions = cod.select_top_x_regions(df, region_col=region_col, data_type=data_type, x=10, combine_subregions=True, exclude=exclude)        

        # Just print the output for now. We'll add more intense tests later.
        print(top)
        print(top_uncombined)
        print(top_with_exclusions)

    def _check_select_regions(self, df):

        # Search for defined region cols (based on data source)
        if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
            region_col = "Country/Region"
            regions = ["US", "China", "Turkey"]
        elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
            region_col = "Province_State"
            regions = ["Washington", "New York", "Arizona"]
        elif {"state"}.issubset(df.columns): # NYT USA state only or states and counties table.
            region_col = "state"
            regions = ["Washington", "New York", "Arizona"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        # Call the function
        selected = cod.select_regions(df, region_col=region_col, regions=regions, combine_subregions=True)
        selected_uncombined = cod.select_regions(df, region_col=region_col, regions=regions, combine_subregions=False)

        # Just print the output for now. We'll add more intense tests later.
        print(selected)
        print(selected_uncombined)

    def _check_calc_x_day_avg(self, df):
        pass

    def _check_daily_change(self, df, data_type, format):
        """Verifies that when df is passed to calc_daily_change, the daily count columns generated are correct.

        df (pandas.DataFrame): A dataframe from the package.
        data_type (str): The data type the table is for. Either "cases", "deaths", "recovered", or "all".
        format (str): The format of the table. Either "wide" or "long".

        Returns:
        None
        """
        # Search for defined grouping cols (based on data source and region)
        if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
            group_cols = ["Province/State", "Country/Region"]
        elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
            group_cols = ["Combined_Key"]
        elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
            group_cols = ["county", "state"]
        elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
            group_cols = ["state"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        if data_type == "all":

            data_types = ["cases", "deaths"]
            if "recovered" in df.columns:
                data_types.append("recovered")

        elif data_type in ["cases", "deaths", "recovered"]:
            data_types = [data_type]
        else:
            raise ParameterError(f"{data_type} is not a valid data type. Pass 'cases', 'deaths', or 'recovered'.")

        if format == "long":
            daily = cod.calc_daily_change(df, data_type)
            both = cod.calc_daily_change(df, data_type, keep_cumulative=True)

            for iter_data_type in data_types:
                if len(group_cols) == 1:
                    group_col = group_cols[0]
                    for group in df[group_col].drop_duplicates():
                        group_df = df[df[group_col] == group]
                        group_daily = daily[daily[group_col] == group]
                        group_both = both[both[group_col] == group]

                        assert group_daily["daily_" + iter_data_type].equals(pd.Series(group_df[iter_data_type] - np.insert(group_df[iter_data_type].values[:-1], 0, 0)))

                elif len(group_cols) == 2:
                    group_col1 = group_cols[0]
                    group_col2 = group_cols[1]
                    for group1 in df[group_col1].drop_duplicates():
                        for group2 in df[group_col2].drop_duplicates():
                            group_df = df[(df[group_col1] == group1) & (df[group_col2] == group2)]
                            group_daily = daily[(daily[group_col1] == group1) & (daily[group_col2] == group2)]
                            group_both = both[(both[group_col1] == group1) & (both[group_col2] == group2)]

                            assert group_daily["daily_" + iter_data_type].equals(pd.Series(group_df[iter_data_type] - np.insert(group_df[iter_data_type].values[:-1], 0, 0))) # Check the daily calculation against the cumulative col in the original df
                            assert group_both["daily_" + iter_data_type].equals(pd.Series(group_both[iter_data_type] - np.insert(group_both[iter_data_type].values[:-1], 0, 0))) # Check the daily calculation against the cumulative col in the same df
                            assert group_both["daily_" + iter_data_type].equals(pd.Series(group_df[iter_data_type] - np.insert(group_df[iter_data_type].values[:-1], 0, 0))) # Check the daily calculation against the cumulative col in the original df

                else:
                    raise Exception(f"Unexpected length of group_cols: '{len(group_cols)}'. group_cols:\n{group_cols}")

        elif format == "wide":
            daily = cod.calc_daily_change(df, data_type)
            date_cols = [col for col in df.columns if issubclass(type(col), datetime.date)]

            id_cols = [col for col in df.columns if not issubclass(type(col), datetime.date)]
            df = df.sort_values(by=id_cols)
            daily = daily.sort_values(by=id_cols)

            for i in range(1, len(date_cols)):
                day = date_cols[i]
                prev_day = date_cols[i - 1]
                assert np.equal(daily[day].values, (df[day] - df[prev_day]).values).all()

        else:
            raise Exception(f"Invalid format '{format}'")

    def _check_days_since(self, df, data_type):
        """Verifies that when df is passed to calc_days_since_min_count, the functions works.

        df (pandas.DataFrame): A dataframe from the package.
        data_type (str): The data type the table is for. Either "cases", "deaths", "recovered", or "all".

        Returns:
        None
        """

        # Search for defined grouping cols (based on data source and region)
        if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
            group_cols = ["Province/State", "Country/Region"]
        elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
            group_cols = ["Combined_Key"]
        elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
            group_cols = ["county", "state"]
        elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
            group_cols = ["state"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        # Call the function
        ct = cod.calc_days_since_min_count(df, data_type, min_count=100, group_by=group_cols)

        # Just print the output for now. We'll add more intense tests later.
        print(ct)
