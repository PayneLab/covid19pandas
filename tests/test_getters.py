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

import pytest
import pandas as pd
import numpy as np
import datetime

formats = ["long", "wide"]
jhu_data_types = ["all", "cases", "deaths", "recovered"]
jhu_regions = ["global", "us"]
update_options = [True, False]

nyt_data_types = ["all", "cases", "deaths"]
nyt_county_options = [True, False]

class TestGetters:

    def test_get_data_jhu(self):
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    for update_option in update_options:

                        # Check that logic errors get caught
                        if region == "us" and data_type == "recovered":
                            with pytest.raises(codex.ParameterError) as excinfo:
                                cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)
                            assert str(excinfo.value) == "JHU does not provide recovery data for US states/counties."

                        elif format == "wide" and data_type == "all":
                            with pytest.raises(codex.ParameterError) as excinfo:
                                cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)
                            assert str(excinfo.value) == "'wide' table format only allows one data type. You requested 'all'. Please pass 'cases', 'deaths', or 'recovered'."

                        else:
                            if not update_option:
                                with pytest.warns(codex.FileNotUpdatedWarning):
                                    df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)
                            else:
                                df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)

                            _check_gotten(df, format)

    def test_get_data_nyt(self):
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    for update_option in update_options:

                        # Check that logic errors get caught
                        if format == "wide" and data_type == "all":
                            with pytest.raises(codex.ParameterError) as excinfo:
                                cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=update_option)
                            assert str(excinfo.value) == "'wide' table format only allows one data type. You requested 'all'. Please pass 'cases', 'deaths', or 'recovered'."
                        else:
                            if not update_option:
                                with pytest.warns(codex.FileNotUpdatedWarning):
                                    df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=update_option)
                            else:
                                df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=update_option)

                            _check_gotten(df, format)


    def test_deprecated_getters(self):
        with pytest.warns(codex.DeprecatedWarning):
            df = cod.get_cases()
        assert df.shape[0] > 0 and df.shape[1] > 0

        with pytest.warns(codex.DeprecatedWarning):
            df = cod.get_deaths()
        assert df.shape[0] > 0 and df.shape[1] > 0

        with pytest.warns(codex.DeprecatedWarning):
            df = cod.get_recovered()
        assert df.shape[0] > 0 and df.shape[1] > 0

# Help functions
def _check_gotten(df, format, group_cols=None, allow_negs=False):
    """Standard checks to verify integrity of gotten table."""

    # Check dimensions
    assert df.shape[0] > 0 and df.shape[1] > 0

    if group_cols is None:
        # Search for defined id cols (based on data source and region)
        if {"Combined_Key"}.issubset(df.columns): # JHU table
            group_cols = ["Combined_Key"]
        elif {"county", "state"}.issubset(df.columns): # NYT USA state and county table
            group_cols = ["county", "state"]
        elif {"state"}.issubset(df.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not that table.
            group_cols = ["state"]
        else:
            raise Exception("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{df.columns}")

        if format == "long":
            group_cols = ["date"] + group_cols

        if "UID" in df.columns:
            if format == "long":
                assert not df.duplicated(subset=["date", "UID"]).any()
            else:
                assert not df.duplicated(subset="UID").any()

    # Check that there aren't duplicates
    assert not df.duplicated(subset=group_cols).any()

    # Check for proper date types
    if format == "long":
        assert df["date"].dtype == np.dtype('datetime64[ns]')
    elif format == "wide":
        assert df.columns.map(lambda x: issubclass(type(x), datetime.date)).any()

    if not allow_negs:
        # Check that there aren't negative counts
        for col in df.columns[~df.columns.isin(["Lat", "Long"])]:
            if np.issubdtype(df[col].dtype, np.number):
                assert not (df[col] < 0).any()
