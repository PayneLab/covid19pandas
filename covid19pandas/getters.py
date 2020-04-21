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

"""
For more help, see our tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs>.
"""

import pandas as pd
import os
import warnings
import datetime
from .download import download_github_file
from .selectors import _wide_to_long, _long_to_wide
from .exceptions import *

# Old getters
def get_cases():
    """***DEPRECATED - Use get_data_jhu instead.***
    Get most recent case counts from JHU."""
    # Deprecated warning
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    warnings.warn("This function is deprecated. Use get_data_jhu instead; see tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs/>.", DeprecatedWarning, stacklevel=2)
    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return _get_table(url, "time_series_covid19_confirmed_global.csv", source="jhu", update=True)

def get_deaths():
    """***DEPRECATED - Use get_data_jhu instead.***
    Get most recent fatality counts from JHU."""
    # Deprecated warning
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    warnings.warn("This function is deprecated. Use get_data_jhu instead; see tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs/>.", DeprecatedWarning, stacklevel=2)
    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return _get_table(url, "time_series_covid19_deaths_global.csv", source="jhu", update=True)

def get_recovered():
    """***DEPRECATED - Use get_data_jhu instead.***
    Get most recent recovered counts from JHU."""
    # Deprecated warning
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    warnings.warn("This function is deprecated. Use get_data_jhu instead; see tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs/>.", DeprecatedWarning, stacklevel=2)
    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return _get_table(url, "time_series_covid19_recovered_global.csv", source="jhu", update=True)

# New getters
def get_data_jhu(format="long", data_type="all", region="global", update=True):
    """Get the most current data tables from JHU (https://github.com/CSSEGISandData/COVID-19).

    Parameters:
    format (str, optional): Format to return the tables in. Pass either "long" or "wide". See https://en.wikipedia.org/wiki/Wide_and_narrow_data for details on the two formats. Default "long".
    data_type (str, optional): The type of data to get. Either "cases", "deaths", "recovered", or "all". Default "all".
    region (str, optional): The region to get data for. Either "global" or "us" (meaning United States). Default "global".
    update (bool, optional): Whether to download the latest tables from the Internet. Otherwise, will attempt to use previously downloaded tables, if they exist. Default True.

    Returns:
    pandas.DataFrame: The requested data table.
    """

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
        df = _get_table(base_url, file_names[region][data_type], source="jhu", update=update)

    else: # format == "long":
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

            df = _wide_to_long(df, iter_data_type)
            id_cols = df.columns[df.columns != iter_data_type].tolist()
            df = df.set_index(id_cols)

            if all_df is None:
                all_df = pd.DataFrame(columns=id_cols)
                all_df = all_df.set_index(id_cols)

            all_df = all_df.join(df, how="outer")

        if region == "global":
            all_df = all_df.sort_index(level=["date", "Country/Region", "Province/State"])
        elif region == "us":
            all_df = all_df.sort_index(level=["date", "Country_Region", "Province_State", "Admin2"])

        all_df = all_df.fillna(0)
        all_df = all_df.astype('int64')
        all_df = all_df.reset_index()
        all_df = all_df.drop_duplicates(keep="first") # Duplicate rows may have been created by the joins if there were NaNs in any of the id_cols
        df = all_df

    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return df

def get_data_nyt(format="long", data_type="all", counties=False, update=True):
    """Get the most current data tables from NYT (https://github.com/nytimes/covid-19-data).

    Parameters:
    format (str, optional): Format to return the tables in. Pass either "long" or "wide". See https://en.wikipedia.org/wiki/Wide_and_narrow_data for details on the two formats. Default "long".
    data_type (str, optional): The type of data to get. Either "cases", "deaths", or "all". Default "all".
    counties (bool, optional): Whether to get county-level data instead of state-level data. Default False.
    update (bool, optional): Whether to download the latest tables from the Internet. Otherwise, will attempt to use previously downloaded tables, if they exist. Default True.

    Returns:
    pandas.DataFrame: The requested data table.
    """

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

    if format == "wide":
        # Spread table into wide format, a la tidyr
        df = _long_to_wide(df, data_type, sort_by="state")

    print("These data were obtained from The New York Times (https://github.com/nytimes/covid-19-data).")
    return df

# Helper functions

def _get_table(base_url, file_name, source, update):
    """Get a table.

    Parameters:
    base_url (str): The raw.githubusercontent.com URL to the folder that contains the file we want.
    file_name (str): The name of the file we want from the folder specified by the URL.
    update (bool): Whether to re-download the table from the Internet. Otherwise, will load a previously downloaded copy, if it exists.

    Returns:
    pandas.DataFrame: The requested DataFrame.
    """

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
        df.columns = df.columns.map(lambda x: pd.to_datetime(x, errors="ignore"))
        df = df.replace(to_replace="Taiwan*", value="Taiwan", regex=False)
        df = df.rename(columns={"Long_": "Long"}, errors="ignore")
        if "Province/State" in df.columns and "Country/Region" in df.columns:
            df = df[~((df["Province/State"] == "Recovered") & (df["Country/Region"] == "Canada"))]

    if source == "nyt":
        df = df.astype({"date": 'datetime64'})

    return df
