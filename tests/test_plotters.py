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
import math

import matplotlib.pyplot as plt

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

    def test_plot_lines_jhu(self):
        """Test the plot_lines function with the JHU data."""
        for data_type in jhu_data_types:
            for region in jhu_regions:
                if data_type == "recovered" and region == "us":
                    continue # Invalid parameter combo

                df = cod.get_data_jhu(format="long", data_type=data_type, region=region, update=False)

                if data_type == "all":
                    if region == "global":
                        plot_types = ["cases", "deaths", "recovered"]
                    else: # region == "us"
                        plot_types = ["cases", "deaths"]
                else:
                    plot_types = [data_type]

                if region == "global":
                    region_col = "Country/Region"
                else: # region == "us"
                    region_col = "Province_State"

                for plot_type in plot_types:
                    top_ten = cod.select_top_x_regions(data=df, data_col=plot_type, region_cols=region_col, x=10)
                    fig, ax = cod.plot_lines(data=top_ten, x_col="date", y_col=plot_type, group_col=region_col)
                    plt.show()
        
    def test_plot_lines_nyt(self):
        """Test the plot_lines function with the NYT data."""
        for data_type in nyt_data_types:
            for county_option in nyt_county_options:
                df = cod.get_data_nyt(format="long", data_type=data_type, counties=county_option, update=False)

                if data_type == "all":
                    plot_types = ["cases", "deaths"]
                else:
                    plot_types = [data_type]

                region_col = "state"

                for plot_type in plot_types:
                    top_ten = cod.select_top_x_regions(data=df, data_col=plot_type, region_cols=region_col, x=10)
                    fig, ax = cod.plot_lines(data=top_ten, x_col="date", y_col=plot_type, group_col=region_col)
                    plt.show()

    def test_plot_lines_two_y_jhu(self):
        """Test the plot_lines function with the JHU data."""
        for data_type in jhu_data_types:
            for region in jhu_regions:
                if data_type == "recovered" and region == "us":
                    continue # Invalid parameter combo

                df = cod.get_data_jhu(format="long", data_type=data_type, region=region, update=False)

                if data_type == "all":
                    if region == "global":
                        plot_types = ["cases", "deaths", "recovered"]
                    else: # region == "us"
                        plot_types = ["cases", "deaths"]
                else:
                    plot_types = [data_type]

                if region == "global":
                    region_col = "Country/Region"
                    region = "US"
                else: # region == "us"
                    region_col = "Province_State"
                    region = "New York"
                    
                for plot_type in plot_types:
                    country_df = cod.select_regions(data=df, region_col=region_col, regions=region, combine_subregions=True, data_cols=plot_type)
                    with_daily = cod.calc_daily_change(data=country_df, data_cols=plot_type, region_cols=region_col)
                    fig, ax = cod.plot_lines_two_y(data=with_daily, x_col="date", y1_col=f"daily_{plot_type}", y2_col=plot_type)
                    plt.show()
        
    def test_plot_lines_two_y_nyt(self):
        """Test the plot_lines function with the NYT data."""
        for data_type in nyt_data_types:
            for county_option in nyt_county_options:
                df = cod.get_data_nyt(format="long", data_type=data_type, counties=county_option, update=False)

                if data_type == "all":
                    plot_types = ["cases", "deaths"]
                else:
                    plot_types = [data_type]

                region_col = "state"
                region = "California"

                for plot_type in plot_types:
                    country_df = cod.select_regions(data=df, region_col=region_col, regions=region, combine_subregions=True, data_cols=plot_type)
                    with_daily = cod.calc_daily_change(data=country_df, data_cols=plot_type, region_cols=region_col)
                    fig, ax = cod.plot_lines_two_y(data=with_daily, x_col="date", y1_col=f"daily_{plot_type}", y2_col=plot_type)
                    plt.show()
