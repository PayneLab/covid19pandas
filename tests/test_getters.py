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

formats = ["long", "wide"]
jhu_data_types = ["all", "cases", "deaths", "recovered"]
jhu_regions = ["global", "us"]
update_options = [True, False]

nyt_data_types = ["all", "cases", "deaths"]
nyt_county_options = [True, False]

class TestGetters:

    def test_get_data_jhu():
        for format in formats:
            for data_type in jhu_data_types:
                for region in jhu_regions:
                    for update_option in update_options:

                        # Check that logic errors get caught
                        if region == "us" and data_type == "recovered":
                            with pytest.raises(codex.ParameterError) as excinfo:
                                cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)
                            assert str(excinfo.value) == "JHU does not provide recovery data for US states/counties.":

                        elif format == "wide" and data_type == "all":
                            with pytest.raises(codex.ParameterError) as excinfo:
                                cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)
                            assert str(excinfo.value) == "'wide' table format only allows one data type. You requested 'all'. Please pass 'cases', 'deaths', or 'recovered'.":

                        else:
                            df = cod.get_data_jhu(format=format, data_type=data_type, region=region, update=update_option)
                            assert df.shape[0] > 0 and df.shape[1] > 0:
                            df.to_csv("out.txt", mode="a")

    def test_get_data_nyt():
        for format in formats:
            for data_type in nyt_data_types:
                for county_option in nyt_county_options:
                    for update_option in update_options:

                        # Check that logic errors get caught
                        if format == "wide" and data_type == "all":
                            with pytest.raises(codex.ParameterError) as excinfo:
                                cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=update_option)
                            assert str(excinfo.value) == "'wide' table format only allows one data type. You requested 'all'. Please pass 'cases', 'deaths', or 'recovered'.":
                        else:
                            df = cod.get_data_nyt(format=format, data_type=data_type, counties=county_option, update=update_option)
                            assert df.shape[0] > 0 and df.shape[1] > 0:
                            df.to_csv("out.txt", mode="a")

    def test_deprecated_getters():
        df = cod.get_cases()
        assert df.shape[0] > 0 and df.shape[1] > 0:
        df = cod.get_deaths()
        assert df.shape[0] > 0 and df.shape[1] > 0:
        df = cod.get_recovered()
        assert df.shape[0] > 0 and df.shape[1] > 0:
