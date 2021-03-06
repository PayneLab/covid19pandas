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
from test_getters import _check_gotten

import pandas as pd
import numpy as np
import datetime
import pytest
import math

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
                                self._check_select_top_x(df, format, compare_by_type, num_regions=1) # Don't keep others
                                self._check_select_top_x(df, format, compare_by_type, num_regions=1, other_to_keep=[col for col in compare_by_types if col != compare_by_type]) # Keep others

                                self._check_select_top_x(df, format, compare_by_type, num_regions=2) # Don't keep others
                                self._check_select_top_x(df, format, compare_by_type, num_regions=2, other_to_keep=[col for col in compare_by_types if col != compare_by_type]) # Keep others
                        else:
                            self._check_select_top_x(df, format, data_type, num_regions=1)
                            self._check_select_top_x(df, format, data_type, num_regions=2)

    def test_select_top_x_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)

                        if data_type == "all":
                            compare_by_types = set(nyt_data_types)
                            compare_by_types.remove("all")
                            for compare_by_type in compare_by_types:
                                self._check_select_top_x(df, format, compare_by_type, num_regions=1) # Don't keep others
                                self._check_select_top_x(df, format, compare_by_type, num_regions=1, other_to_keep=[col for col in compare_by_types if col != compare_by_type]) # Keep others

                                # It only will work to do more than 1 grouping col if we're using the states and counties table, because the just states table only has one grouping col
                                if county_option:
                                    self._check_select_top_x(df, format, compare_by_type, num_regions=2) # Don't keep others
                                    self._check_select_top_x(df, format, compare_by_type, num_regions=2, other_to_keep=[col for col in compare_by_types if col != compare_by_type]) # Keep others
                        else:
                            self._check_select_top_x(df, format, data_type, num_regions=1)

                            # It only will work to do more than 1 grouping col if we're using the states and counties table, because the just states table only has one grouping col
                            if county_option:
                                self._check_select_top_x(df, format, data_type, num_regions=2)

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

                        if data_type == "all":
                            cols_to_keep = {"cases", "deaths", "recovered"}
                            if region == "us":
                                cols_to_keep.remove("recovered")
                            cols_to_keep = sorted(cols_to_keep) # Convert it back to a list
                        else:
                            cols_to_keep = [data_type]
                        self._check_select_regions(df, format, cols_kept=cols_to_keep)

    def test_select_regions_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)

                        if data_type == "all":
                            cols_to_keep = ["cases", "deaths"]
                        else:
                            cols_to_keep = [data_type]
                        self._check_select_regions(df, format, cols_kept=cols_to_keep)

    # -------------------------------------------------------------------------------------------------------------
    # Tests for calc_x_day_rolling_mean
    # -------------------------------------------------------------------------------------------------------------
    def test_calc_x_day_rolling_mean_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    if (region == "us" and data_type == "recovered") or (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=False)

                        if data_type == "all":
                            input_data_types = set(jhu_data_types)
                            input_data_types.remove("all")
                            if region == "us":
                                input_data_types.remove("recovered")

                            for input_data_type in input_data_types:
                                self._check_calc_x_day_rolling_mean(df, format, data_type=input_data_type, other_input_data_types=[col for col in input_data_types if col != input_data_type])

                        # Note that we still also perform this test if data_type == "all" because we can also calculate the x day mean for all columns.
                        self._check_calc_x_day_rolling_mean(df, format, data_type)

    def test_calc_x_day_rolling_mean_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)

                        if data_type == "all":
                            input_data_types = set(nyt_data_types)
                            input_data_types.remove("all")

                            for input_data_type in input_data_types:
                                self._check_calc_x_day_rolling_mean(df, format, data_type=input_data_type, other_input_data_types=[col for col in input_data_types if col != input_data_type])
                                
                        # Note that we still also perform this test if data_type == "all" because we can also calculate the x day mean for all columns.
                        self._check_calc_x_day_rolling_mean(df, format, data_type)

    # -------------------------------------------------------------------------------------------------------------
    # Tests for calc_daily_change
    # -------------------------------------------------------------------------------------------------------------
    def test_calc_daily_change_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    if (region == "us" and data_type == "recovered") or (format == "wide" and data_type == "all"):
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=False)

                        if data_type == "all":
                            input_data_types = set(jhu_data_types)
                            input_data_types.remove("all")
                            if region == "us":
                                input_data_types.remove("recovered")

                            for input_data_type in input_data_types:
                                self._check_daily_change(df, format=format, data_type=input_data_type, other_data_types=[col for col in input_data_types if col != input_data_type])

                        # Note that we still also perform this test if data_type == "all" because we can also calculate daily change for all columns.
                        self._check_daily_change(df, format=format, data_type=data_type)

    def test_calc_daily_change_long_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    if format == "wide" and data_type == "all":
                        pass # Invalid table parameter combination
                    else:
                        df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=False)

                        if data_type == "all":
                            input_data_types = set(nyt_data_types)
                            input_data_types.remove("all")

                            for input_data_type in input_data_types:
                                self._check_daily_change(df, format=format, data_type=input_data_type, other_data_types=[col for col in input_data_types if col != input_data_type])

                        # Note that we still also perform this test if data_type == "all" because we can also calculate daily change for all columns.
                        self._check_daily_change(df, format=format, data_type=data_type)

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
                                self._check_days_since(df, format, count_by_type)
                        else:
                            self._check_days_since(df, format, data_type)

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
                                self._check_days_since(df, format, count_by_type)
                        else:
                            self._check_days_since(df, format, data_type)

    # -------------------------------------------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _check_select_top_x(df, format, data_type, num_regions, other_to_keep=[]):

        if num_regions == 1:
            # Search for defined region cols (based on data source)
            if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
                region_col = "Country/Region"
                exclude = ["US", "China"]
            elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
                region_col = "Province_State"
                exclude = ["New York", "Illinois"]
            elif {"state"}.issubset(df.columns): # NYT USA state only or states and counties table.
                region_col = "state"
                exclude = ["Washington", "Illinois"]
            else:
                raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

            if format == "wide":
                group_cols = [region_col]
            else: # format == "long"
                group_cols = ["date", region_col]

            num_top = 10
            # Call the function
            outs = {
                "top_others_kept": cod.select_top_x_regions(df, region_cols=region_col, data_col=data_type, x=num_top, combine_subregions=True, other_data_cols=other_to_keep),
                "top_uncombined": cod.select_top_x_regions(df, region_cols=region_col, data_col=data_type, x=num_top, combine_subregions=False, other_data_cols=other_to_keep),
                "top_with_exclusions": cod.select_top_x_regions(df, region_cols=region_col, data_col=data_type, x=num_top, combine_subregions=True, other_data_cols=other_to_keep, exclude=exclude),
            }

            # Run basic table checks
            for name, out in outs.items():

                if name == "top_uncombined" and {"Admin2"}.issubset(df.columns):
                    _check_gotten(out, format, group_cols=group_cols + ["Admin2"]) # If it's the JHU U.S. table, we need to add "Admin2" as a group col, but only for the uncombined table.
                elif name == "top_uncombined" and {"Province/State"}.issubset(df.columns):
                    _check_gotten(out, format, group_cols=group_cols + ["Province/State"]) # If it's the JHU global table, we need to add "Province/State" as a group col, but only for the uncombined table.
                elif name == "top_uncombined" and {"county"}.issubset(df.columns):
                    _check_gotten(out, format, group_cols=group_cols + ["county"]) # If it's the NYT county table, we need to add "county" as a group col, but only for the uncombined table.
                else:
                    _check_gotten(out, format, group_cols=group_cols)

            # Make sure that the data values weren't changed, if we didn't aggregate
            if format == "wide":
                for name, out in outs.items():
                    df_dates = df.columns[df.columns.map(lambda col: issubclass(type(col), datetime.date))]
                    out_dates = out.columns[out.columns.map(lambda col: issubclass(type(col), datetime.date))]
                    assert df_dates.equals(out_dates)

                    if name == "top_uncombined":
                        for date in df_dates:
                            for region in out[region_col].unique():
                                assert out.loc[out[region_col] == region, date].equals(df.loc[df[region_col] == region, date])
            else:
                for name, out in outs.items():
                    assert data_type in out.columns
                    if name == "top_uncombined":
                        for region in out[region_col].unique():
                            assert out.loc[out[region_col] == region, data_type].equals(df.loc[df[region_col] == region, data_type])

            # If we had other cols to keep, make sure they were kept, and are equal to their original values.
            for keep in other_to_keep:
                for name, out in outs.items():
                    assert keep in out.columns
                    if name == "top_uncombined":
                        for region in out[region_col].unique():
                            assert out.loc[out[region_col] == region, keep].equals(df.loc[df[region_col] == region, keep])

            # Check that the excluded countries aren't in the list
            assert not outs["top_with_exclusions"][region_col].isin(exclude).any()

            # Check that length of combined table is x * len(unique(dates))
            if format == "wide":
                for name, out in outs.items():
                    if name != "top_uncombined":
                        assert out.shape[0] == num_top
                        assert out.shape[0] == out[region_col].unique().size
            else:
                for name, out in outs.items():
                    if name == "top_uncombined":
                        assert out.shape[0] == df[region_col].isin(out[region_col]).sum()
                    else:
                        assert out.shape[0] <= num_top * out["date"].unique().size # We check <= because some of the regions may not have counts for all days at the beginning
                        assert num_top == out[region_col].unique().size
        elif num_regions == 2:

            # Search for defined region cols (based on data source)
            if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
                region_cols = ["Country/Region", "Province/State"]
                exclude = ["US", "China"]
            elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
                region_cols = ["Province_State", "Admin2"]
                exclude = ["New York", "Illinois"]
            elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
                region_cols = ["county", "state"]
                exclude = ["Washington", "Illinois"]
            elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
                raise Exception("Can't do more than one region col for NYT states only table.")
            else:
                raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

            num_top = 10
            # Call the function
            outs = {
                "top_others_kept": cod.select_top_x_regions(df, region_cols=region_cols, data_col=data_type, x=num_top, combine_subregions=True, other_data_cols=other_to_keep),
                "top_uncombined": cod.select_top_x_regions(df, region_cols=region_cols, data_col=data_type, x=num_top, combine_subregions=False, other_data_cols=other_to_keep),
                "top_with_exclusions": cod.select_top_x_regions(df, region_cols=region_cols, data_col=data_type, x=num_top, combine_subregions=True, other_data_cols=other_to_keep, exclude=exclude),
            }

            # Run basic table checks
            if format == "wide":
                group_cols = region_cols
            else: # format == "long"
                group_cols = ["date"] + region_cols

            for name, out in outs.items():
                _check_gotten(out, format, group_cols=group_cols)

            # For these tests, we need any NaNs in the region cols to be filled with strings so they can compare equal.
            for name in outs.keys():
                out = outs[name]
                for region_col in region_cols:
                    out[region_col] = out[region_col].fillna("n/a")

                outs[name] = out
                    
            for region_col in region_cols:
                df[region_col] = df[region_col].fillna("n/a")

            # Make sure that the data values weren't changed, if we didn't aggregate
            if format == "wide":
                for name, out in outs.items():
                    df_dates = df.columns[df.columns.map(lambda col: issubclass(type(col), datetime.date))]
                    out_dates = out.columns[out.columns.map(lambda col: issubclass(type(col), datetime.date))]
                    assert df_dates.equals(out_dates)

                    if name == "top_uncombined":
                        for date in df_dates:
                            region_combos = out[region_cols].drop_duplicates(keep="first")
                            rcol1 = region_cols[0]
                            rcol2 = region_cols[1]

                            for i in range(0, region_combos.index.size):
                                rval1 = region_combos.iloc[i, 0]
                                rval2 = region_combos.iloc[i, 1]

                                assert out.loc[(out[rcol1] == rval1) & (out[rcol2] == rval2), date].equals(df.loc[(df[rcol1] == rval1) & (df[rcol2] == rval2), date])
            else:
                for name, out in outs.items():
                    assert data_type in out.columns
                    if name == "top_uncombined":
                        region_combos = out[region_cols].drop_duplicates(keep="first")
                        rcol1 = region_cols[0]
                        rcol2 = region_cols[1]

                        for i in range(0, region_combos.index.size):
                            rval1 = region_combos.iloc[i, 0]
                            rval2 = region_combos.iloc[i, 1]

                            assert out.loc[(out[rcol1] == rval1) & (out[rcol2] == rval2), data_type].equals(df.loc[(df[rcol1] == rval1) & (df[rcol2] == rval2), data_type])

            # If we had other cols to keep, make sure they were kept, and are equal to their original values.
            for keep in other_to_keep:
                for name, out in outs.items():
                    assert keep in out.columns
                    if name == "top_uncombined":
                        region_combos = out[region_cols].drop_duplicates(keep="first")
                        rcol1 = region_cols[0]
                        rcol2 = region_cols[1]

                        for i in range(0, region_combos.index.size):
                            rval1 = region_combos.iloc[i, 0]
                            rval2 = region_combos.iloc[i, 1]

                            assert out.loc[(out[rcol1] == rval1) & (out[rcol2] == rval2), keep].equals(df.loc[(df[rcol1] == rval1) & (df[rcol2] == rval2), keep])

            # Check that the excluded countries aren't in the list
            for region_col in region_cols:
                assert not outs["top_with_exclusions"][region_col].isin(exclude).any()

            # Check that length of combined table is x * len(unique(dates))
            if format == "wide":
                for name, out in outs.items():
                    if name != "top_uncombined":
                        assert out.shape[0] == num_top
                        assert out.shape[0] == out[region_cols].drop_duplicates(keep="first").index.size
            else:
                for name, out in outs.items():
                    if name == "top_uncombined":
                        rcol1 = region_cols[0]
                        rcol2 = region_cols[1]
                        assert out.shape[0] == df[region_cols].isin({rcol1: out[rcol1], rcol2: out[rcol2]}).all(axis="columns").sum()
                    else:
                        assert out.shape[0] <= num_top * out["date"].unique().size # We check <= because some of the regions may not have counts for all days at the beginning
                        assert num_top == out[region_cols].drop_duplicates(keep="first").index.size
        else: 
            raise Exception("Test doesn't support that number of regions. Options are 1 or 2.")

    @staticmethod
    def _check_select_regions(df, format, cols_kept):

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
        dfs = {
            "selected": cod.select_regions(df, region_col=region_col, regions=regions, combine_subregions=True, data_cols=cols_kept),
            "selected_uncombined": cod.select_regions(df, region_col=region_col, regions=regions, combine_subregions=False, data_cols=cols_kept),
        }

        # Run basic table checks
        for name, out in dfs.items():
            if name == "selected":
                if format == "long":
                    _check_gotten(out, format, group_cols=["date", region_col])
                else: 
                    _check_gotten(out, format, group_cols=[region_col])
            else: # name == "selected_uncombined"
                _check_gotten(out, format)

        # Make sure that only the regions we specified exist in the region col
        for out in dfs.values():
            assert out[region_col].isin(regions).all()

        # Make sure cols_kept were kept
        for name, out in dfs.items():
            if format == "wide":
                df_dates = df.columns[df.columns.map(lambda col: issubclass(type(col), datetime.date))]
                out_dates = out.columns[out.columns.map(lambda col: issubclass(type(col), datetime.date))]
                assert df_dates.equals(out_dates)

                if name == "selected_uncombined":
                    for date in df_dates:
                        assert out[date].equals(df.loc[df[region_col].isin(regions), date])

            else: # format == "long"
                for col in cols_kept:
                        assert col in out.columns
                        if name == "selected_uncombined":
                            assert out[col].equals(df.loc[df[region_col].isin(regions), col])

    @staticmethod
    def _check_calc_x_day_rolling_mean(df, format, data_type, other_input_data_types=[]):

        # Process the data_type parameter
        if data_type == "all":

            data_types = ["cases", "deaths"]
            if "recovered" in df.columns:
                data_types.append("recovered")

        elif data_type in ["cases", "deaths", "recovered"]:
            data_types = [data_type]
        else:
            raise ParameterError(f"{data_type} is not a valid data type. Pass 'cases', 'deaths', 'recovered', or 'all'.")

        # Decide on our region_cols
        # Search for defined region cols (based on data source)
        if {"Province/State", "Country/Region"}.issubset(df.columns): # JHU global table
            region_cols = ["Country/Region", "Province/State"]
        elif {"Combined_Key"}.issubset(df.columns): # JHU USA table
            region_cols = ["Province_State", "Admin2"]
        elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
            region_cols = ["county", "state"]
        elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
            region_cols = ["state"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        mean_range = 3
        dfs = {
            "centered": cod.calc_x_day_rolling_mean(df, data_cols=data_types, region_cols=region_cols, x=mean_range, center=True),
            "not_centered": cod.calc_x_day_rolling_mean(df, data_cols=data_types, region_cols=region_cols, x=mean_range, center=False),
        }

        # Run basic table checks
        if format == "long":
            unique_cols = ["date"] + region_cols
        else: # format == "wide"
            unique_cols = region_cols
        for name, out in dfs.items():
            _check_gotten(out, format, group_cols=unique_cols)

        # Check that data_types got averaged
        for out in dfs.values():
            for data_type in data_types:
                if format == "long":
                    assert f"mean_{data_type}" in out.columns
                else: # format == "wide"
                    assert not df.equals(out)

        # Make sure other_input_data_types are unchanged
        if format == "long":
            for other_input in other_input_data_types:
                for out in dfs.values():
                    assert out[other_input].equals(df[other_input])

    @staticmethod
    def _check_daily_change(df, format, data_type, other_data_types=[]):
        """Verifies that when df is passed to calc_daily_change, the daily count columns generated are correct.

        df (pandas.DataFrame): A dataframe from the package.
        format (str): The format of the table. Either "wide" or "long".
        data_type (str): The data type the table is for. Either "cases", "deaths", "recovered", or "all".
        other_data_types (list of str, optional): Other data types for which the daily change isn't calculated, and which should be unchanged by the function.

        Returns:
        None
        """

        # Search for defined grouping cols (based on data source and region)
        if {"Combined_Key"}.issubset(df.columns): # JHU table
            group_cols = ["Combined_Key"]
        elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
            group_cols = ["county", "state"]
        elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
            group_cols = ["state"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        if format == "long":
            if data_type == "all":

                data_types = ["cases", "deaths"]
                if "recovered" in df.columns:
                    data_types.append("recovered")

            elif data_type in ["cases", "deaths", "recovered"]:
                data_types = [data_type]
            else:
                raise ParameterError(f"{data_type} is not a valid data type. Pass 'cases', 'deaths', or 'recovered'.")

            daily = cod.calc_daily_change(df, data_types, region_cols=group_cols)

            for other_data_type in other_data_types:
                assert daily[other_data_type].equals(df[other_data_type])

            # Run basic table checks
            _check_gotten(daily, format, allow_negs=True)

            # Check that no columns were lost
            assert df.columns.isin(daily.columns).all()

            # Check daily change calculations against original cumulative columns
            for iter_data_type in data_types:
                if len(group_cols) == 1:
                    group_col = group_cols[0]

                    for group in df[group_col].drop_duplicates():
                        row_filter = df[group_col] == group
                        group_df = df[row_filter]
                        group_daily = daily[row_filter]

                        assert group_daily["daily_" + iter_data_type].equals(pd.Series(group_daily[iter_data_type] - np.insert(group_daily[iter_data_type].values[:-1], 0, 0))) # Check the daily calculation against the cumulative col in the same df
                        assert group_daily[iter_data_type].equals(group_df[iter_data_type]) # Check the cumulative col against the one in the original df

                elif len(group_cols) == 2:
                    group_col1 = group_cols[0]
                    group_col2 = group_cols[1]

                    existing_groups = df[group_cols].drop_duplicates(keep="first")
                    for i in range(0, existing_groups.index.size):

                        group1 = existing_groups.iloc[i, 0]
                        group2 = existing_groups.iloc[i, 1]

                        df_filter = (df[group_col1] == group1) & (df[group_col2] == group2)

                        if df_filter.any():
                            group_df = df[df_filter]
                            group_daily = daily[df_filter]

                            assert group_daily["daily_" + iter_data_type].equals(pd.Series(group_daily[iter_data_type] - np.insert(group_daily[iter_data_type].values[:-1], 0, 0))) # Check the daily calculation against the cumulative col in the same df
                            assert group_daily[iter_data_type].equals(group_df[iter_data_type]) # Check the cumulative col against the one in the original df
                        else:
                            raise Exception("That was unexpected.")

                else:
                    raise Exception(f"Unexpected length of group_cols: '{len(group_cols)}'. group_cols:\n{group_cols}")

        elif format == "wide":
            daily = cod.calc_daily_change(df, data_type, region_cols=group_cols)

            # Run basic table checks
            _check_gotten(daily, format, allow_negs=True)

            date_cols = [col for col in df.columns if issubclass(type(col), datetime.date)]

            # The first day should be the same in both the daily and original dfs, since all cases/deaths/recovered were "new"
            assert np.equal(df[date_cols[0]], daily[date_cols[0]]).all()

            # Check that each daily change equals that day's columns minus the previous day's column, element-wise, in the original df
            for i in range(1, len(date_cols)):
                day = date_cols[i]
                prev_day = date_cols[i - 1]
                assert np.equal(daily[day].values, (df[day] - df[prev_day]).values).all()

        else:
            raise Exception(f"Invalid format '{format}'")

    @staticmethod
    def _check_days_since(df, format, data_type):
        """Verifies that when df is passed to calc_days_since_min_count, the functions works.

        df (pandas.DataFrame): A dataframe from the package.
        format (str): The format of the table. Either "wide" or "long".
        data_type (str): The data type the table is for. Either "cases", "deaths", "recovered", or "all".

        Returns:
        None
        """

        # Search for defined grouping cols (based on data source and region)
        if {"Combined_Key"}.issubset(df.columns): # JHU table
            group_cols = ["Combined_Key"]
        elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
            group_cols = ["county", "state"]
        elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
            group_cols = ["state"]
        else:
            raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        # Call the function
        min_count = 100
        ct = cod.calc_days_since_min_count(df, data_type, region_cols=group_cols, min_count=min_count)

        # Run basic table checks
        _check_gotten(ct, format="long") # The calc_days_since_min_count function only outputs table in long format, even if given wide format as input

        # Check that all values in data type col are >= min count
        assert (ct[data_type] >= min_count).all()

        # Check that all values in days_since_{min_count}_{data_type} are <= number of days in original df
        if format == "long":
            num_days = df["date"].unique().size
        else: # format == "wide"
            num_days = df.columns.map(lambda col: issubclass(type(col), datetime.date)).to_series().astype(bool).sum()

        assert (ct[f"days_since_{min_count}_{data_type}"] <= num_days).all()
