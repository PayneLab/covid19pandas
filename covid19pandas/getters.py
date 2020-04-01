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

import pandas as pd
import os
import warnings
import datetime
from .download import download_github_file
from .exceptions import *

# Old getters
def get_cases():
    # Deprecated warning
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    warnings.warn("This function is deprecated. Use get_data_jhu instead; see tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs/>.", DeprecatedWarning, stacklevel=2)
    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return _get_table(url, "time_series_covid19_confirmed_global.csv", source="jhu", update=True)

def get_deaths():
    # Deprecated warning
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    warnings.warn("This function is deprecated. Use get_data_jhu instead; see tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs/>.", DeprecatedWarning, stacklevel=2)
    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return _get_table(url, "time_series_covid19_deaths_global.csv", source="jhu", update=True)

def get_recovered():
    # Deprecated warning
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    warnings.warn("This function is deprecated. Use get_data_jhu instead; see tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs/>.", DeprecatedWarning, stacklevel=2)
    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return _get_table(url, "time_series_covid19_recovered_global.csv", source="jhu", update=True)

# New getters
def get_data_jhu(format="long", data_type="all", region="global", update=True):

    region = region.lower()
    format = format.lower()
    data_type = data_type.lower()

    # Parameter checks
    if format not in ("long", "wide"):
        raise ParameterError(f"Invalid argument for 'format' parameter. You passed {format}. Valid options are 'long' or 'wide'.")
    if region not in ("global", "us"):
        raise ParameterError(f"Invalid argument for 'region' parameter. You passed {region}. Valid options are 'global' or 'us'.")
    if data_type not in ("all", "cases", "deaths", "recovered"):
        raise ParameterError(f"Invalid argument for 'data_type' parameter. You passed {data_type}. Valid options are 'all', 'cases', 'deaths', or 'recovered'.")

    # Logic checks
    if region == "us" and data_type == "recovered":
        raise ParameterError("JHU does not provide recovery data for US states/counties.")
    if format == "wide" and data_type == "all":
        raise ParameterError("'wide' table format only allows one data type. You requested 'all'. Please pass 'cases', 'deaths', or 'recovered'.")


    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    file_names = {
        "global": {
            "cases": "time_series_covid19_confirmed_global.csv",
            "deaths": "time_series_covid19_deaths_global.csv",
            "recovered": "time_series_covid19_recovered_global.csv",
        },
         "us": {
            "cases": "time_series_covid19_confirmed_US.csv",
            "deaths": "time_series_covid19_deaths_US.csv",
        }
    }

    if format == "wide":
        print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
        return _get_table(base_url, file_names[region][data_type], source="jhu", update=update)

    # Get the requested table types
    dfs = {}
    if data_type == "all":
        for iter_data_type in file_names[region].keys():
             dfs[iter_data_type] = _get_table(base_url, file_names[region][iter_data_type], source="jhu", update=update)
    else:
         dfs[data_type] = _get_table(base_url, file_names[region][data_type], source="jhu", update=update)

    # Gather the tables into long format (a la tidyr), and join into one table
    all_df = None
    for iter_data_type, df in dfs.items():

        id_cols = [col for col in df.columns if not isinstance(col, datetime.date)]
        df = pd.melt(df, id_vars=id_cols, var_name="date", value_name=iter_data_type)
        df = df[df[iter_data_type] != 0] # Drop rows of zeros

        id_cols.append("date")
        if all_df is None:
            all_df = pd.DataFrame(columns=id_cols)
            all_df = all_df.set_index(id_cols)

        df = df.set_index(id_cols)
        all_df = all_df.join(df, how="outer")

    if region == "global":
        all_df = all_df.sort_index(level=["date", "Country/Region", "Province/State"])
    elif region == "us":
        all_df = all_df.sort_index(level=["date", "UID"])

    # Reorder index so date is first
    idx_names = list(all_df.index.names)
    idx_names.remove("date")
    new_idx_name_order = ["date"] + idx_names
    all_df = all_df.reorder_levels(new_idx_name_order)

    all_df = all_df.fillna(0)
    all_df = all_df.reset_index()

    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return all_df

def get_data_nyt(format="long", data_type="all", counties=False, update=True):

    format = format.lower()
    data_type = data_type.lower()

    # Parameter checks
    if format not in ("long", "wide"):
        raise ParameterError(f"Invalid argument for 'format' parameter. You passed {format}. Valid options are 'long' or 'wide'.")
    if data_type not in ("all", "cases", "deaths"):
        raise ParameterError(f"Invalid argument for 'data_type' parameter. You passed {data_type}. Valid options are 'all', 'cases', or 'deaths'.")

    # Logic checks
    if format == "wide" and data_type == "all":
        raise ParameterError("'wide' table format only allows one data type. You requested 'all'. Please pass 'cases', 'deaths', or 'recovered'.")

    # Get either counties or states table
    base_url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/"
    if counties:
        df = _get_table(base_url, "us-counties.csv", source="nyt", update=update)
    else: # states
        df = _get_table(base_url, "us-states.csv", source="nyt", update=update)

    # Drop unrequested columns, if needed
    if data_type == "cases":
        df = df.drop(columns="deaths")
    elif data_type == "deaths":
        df = df.drop(columns="cases")

    if format == "long":
        print("These data were obtained from The New York Times (https://github.com/nytimes/covid-19-data).")
        return df

    # Spread table into wide format, a la tidyr
    id_cols = [col for col in df.columns if col != data_type]
    df = df.set_index(id_cols)
    df = df.unstack(level=0, fill_value=0)
    df.columns = df.columns.droplevel(0)
    df.columns.name = None
    df = df.sort_index(level="state")
    df = df.reset_index()

    print("These data were obtained from The New York Times (https://github.com/nytimes/covid-19-data).")
    return df

# Helper functions

def _get_table(base_url, file_name, source, update):

    # Construct the url and path for the file
    path_here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path_here, "data", source, file_name)

    if update:
        # Download the latest version of the file
        url = base_url + file_name
        try:
            download_github_file(url, path)
        except NoInternetError:
            warnings.warn("Insufficient internet to update data files. Data from most recent download will be used.", FileNotUpdatedWarning, stacklevel=3)
    else:
        warnings.warn("You chose to not update data files. Data from most recent download will be used. To update files instead, pass True to the 'update' parameter.", FileNotUpdatedWarning, stacklevel=3)

    if not os.path.isfile(path):
        raise FileDoesNotExistError("Data file has not been downloaded previously, and current internet connection is not sufficient to download it. Try again when you have a better internet connection.")

    df = pd.read_csv(path)

    # Formatting fixes
    if source == "jhu":
        df.columns = df.columns.map(lambda x: pd.to_datetime(x, errors="ignore")).map(lambda x: x.date() if isinstance(x, pd.Timestamp) else x)
    if "Long_" in df.columns:
        df = df.rename(columns={"Long_": "Long"})

    return df
