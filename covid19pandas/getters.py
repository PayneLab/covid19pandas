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
import numpy as np
import os
import warnings
import datetime

from .download import download_github_file
from .exceptions import FileDoesNotExistError, NoInternetError, ParameterError, DeprecatedWarning, FileNotUpdatedWarning
from .utils import _wide_to_long, _long_to_wide

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

    if region == "global":
        id_cols = ["Province/State", "Country/Region"]
    else: # region == "us"
        id_cols = ["Combined_Key"]

    if format == "wide":
        df = _get_table(base_url, file_names[region][data_type], source="jhu", update=update)

        # Drop identifier columns besides the one we'll use to join on with the location table.
        date_cols = [col for col in df.columns if issubclass(type(col), datetime.date)]
        df = df[id_cols + date_cols]

    else: # format == "long":
        # Get the requested table types
        dfs = {}
        if data_type == "all":
            for iter_data_type in file_names[region].keys():
                 dfs[iter_data_type] = _get_table(base_url, file_names[region][iter_data_type], source="jhu", update=update)
        else:
             dfs[data_type] = _get_table(base_url, file_names[region][data_type], source="jhu", update=update)

        # Gather the tables into long format (a la tidyr), and join into one table
        date_and_id_cols = ["date"] + id_cols
        all_df = None
        for iter_data_type, df in dfs.items():

            df = _wide_to_long(df, iter_data_type)
            df = df[date_and_id_cols + [iter_data_type]] # Drop identifier columns besides the one we'll use to join on with the location table.

            # Temporarily fill NaNs in the id cols with a string so they can be equal in join key comparisons
            for id_col in id_cols:
                df[id_col] = df[id_col].fillna("n/a")

            if all_df is None:
                all_df = pd.DataFrame(columns=date_and_id_cols)
                all_df = all_df.set_index(date_and_id_cols)

            # Set the date and id cols as the index of df, then join on them to all_df
            df = df.set_index(date_and_id_cols)
            all_df = all_df.join(df, how="outer")

        all_df = all_df.fillna(0)
        all_df = all_df.astype('int64')
        all_df = all_df.reset_index()

        # Put the NaNs back into the id cols
        for id_col in id_cols:
            all_df[id_col] = all_df[id_col].replace(to_replace="n/a", value=np.nan)

        df = all_df

    # Get the location data to join in
    loc_table = get_jhu_location_data(update=update)
    if region == "global":
        loc_table = loc_table.rename(columns={"Country_Region": "Country/Region", "Province_State": "Province/State"})
        loc_table = loc_table[pd.isnull(loc_table["Admin2"])] # Drop location data for individual US counties--we only want state level data, to avoid duplicate rows

    # Merge in the location data
    df = loc_table.merge(df, on=id_cols, how="right", suffixes=(False, False), validate="one_to_many")

    # If it's the global table, drop the FIPS and Admin2 columns--they're only relevant for the US table
    if region == "global":
        df = df.drop(columns=["FIPS", "Admin2"])

    # If long format, make the date columns the first column in the dataframe
    if format == "long":
        df_col_list = df.columns.tolist()
        df_col_list.remove("date")
        df = df[["date"] + df_col_list]

    # Sort the table
    if region == "global":
        sort_cols = ["Country/Region", "Province/State"]
    else: # region == "us"
        sort_cols = ["Country_Region", "Province_State", "Admin2"]

    if format == "long":
        sort_cols = ["date"] + sort_cols

    df = df.sort_values(by=sort_cols)
    df = df.reset_index(drop=True) # So the range index is still in ascending order after sorting

    print("These data were obtained from Johns Hopkins University (https://github.com/CSSEGISandData/COVID-19).")
    return df

def get_jhu_location_data(update=True):
    """Get the location data table from JHU (see https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv).

    Parameters:
    update (bool, optional): Whether to try updating the table. Default True.

    Returns:
    pandas.DataFrame: The location data table from JHU.
    """

    loc_table_base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/"
    loc_table_name = "UID_ISO_FIPS_LookUp_Table.csv"
    loc_table = _get_table(loc_table_base_url, loc_table_name, source="jhu", update=update)
    return loc_table

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

        if "Admin2" in df.columns:
            df = df[~(df["Combined_Key"] == "Southwest, Utah, US")] # This column is a typo, and has all zeros

        if "Combined_Key" in df.columns:
            df["Combined_Key"] = df["Combined_Key"].str.replace(" ", "") # Spacing isn't consistent for this column, so we'll nix all spaces


    if source == "nyt":
        df = df.astype({"date": 'datetime64'})

    return df

# Deprecated getters
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
