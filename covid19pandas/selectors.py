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
These functions are for manipulating tables gotten by the data getter functions. For more help, see our tutorials at <https://github.com/PayneLab/covid19pandas/tree/master/docs>.
"""

import pandas as pd
import numpy as np
import os
import warnings
import datetime
from .exceptions import *

def select_top_x_regions(data, region_col, data_type, x, combine_subregions, exclude=[]):
    """Select the top x regions with the most cases, deaths, or recoveries.

    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designations you want to group by.
    data_type (str): The data type you want to rank regions by, e.g. "cases", "deaths", or "recovered". Or a different column name if you like.
    x (int): The number of top regions to keep. Default 10.
    combine_subregions (bool): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Otherwise, keeps the region broken into subregions. Default True.
    exclude (list of str, optional): A list of regions to exclude from the selection. If an excluded region made the cut, the next highest region will take its place.

    Returns:
    pandas.DataFrame: Counts for the top x regions.
    """
    if "date" not in data.columns:
        long_data = _wide_to_long(data, data_type) # If they give us a wide format table, convert it to long format.
    else:
        long_data = data.copy()

    if data_type not in long_data.columns:
        raise ParameterError(f"There is no '{data_type}' column in the dataframe you passed. Existing columns: \n{data.columns}")

    last_day = long_data["date"].max() # Get the last recorded day

    current_ct = long_data[long_data['date'] == last_day] # Pull all records for that day
    current_ct = long_data[['date', region_col, data_type]]
    current_ct = current_ct[~current_ct[region_col].isin(exclude)] # Select only columns where the region_col value is not in the exclude list
    current_ct = current_ct.groupby(region_col).aggregate(np.sum) # Sum all counts for today in subregions within that regions

    top_x_names = current_ct.sort_values(by=data_type).tail(x) # Get the names of the top regions
    top_x_cts = data[data[region_col].isin(top_x_names.index)] # Filter out other regions

    if combine_subregions:
        cols_to_drop = [col for col in top_x_cts.columns if col not in ["date", region_col, data_type] and not issubclass(type(col), datetime.date)]
        top_x_cts = top_x_cts.drop(columns=cols_to_drop)
        id_cols = top_x_cts.columns[top_x_cts.columns.isin(["date", region_col])].tolist()
        top_x_cts = top_x_cts.groupby(id_cols).aggregate(np.sum).reset_index() # Sum up total cases per day, per country

    return top_x_cts

def select_regions(data, region_col, regions, combine_subregions=True):
    """Select all data for particular regions within a table, optionally summing counts for subregions into one count for each region for each day.
    
    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designation you're specifying by. E.g., if you want to select particular states, pass the name of the state column
    regions (list of str): The regions to select.
    combine_subregions (bool, optional): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Otherwise, keeps the region broken into subregions. Default True.

    Returns:
    pandas.DataFrame: The data for the specified regions.
    """

def calc_x_day_avg(data, x=3):
    """Take a table of daily counts, and average the counts for each set of x consecutive days (e.g., a 3 day average).

    Parameters:
    data (pandas.DataFrame): The data to average.
    x (int, optional): The number of days to put into each averaged group. Default 3.

    Returns:
    pandas.DataFrame: The table, averaged over the specified number of days.
    """
    pass
#
#def calc_daily_change(data, data_type="all", keep_cumulative=False):
#    """Get the daily change in the number of cases/deaths/recoveries, instead of cumulative counts.
#    
#    Parameters:
#    data (pandas.DataFrame): The cumulative counts from which to calculate the daily change.
#    data_type (str): When your table contains multiple count types (e.g. cases, deaths, recovered), use this parameter to specify which columns you want to calculate the daily change for. Other columns will be left unchanged. Default "all".
#    keep_cumulative (bool, optional): Whether to keep the original column of cumulative counts next to the new daily change column; otherwise drop it. Default False.
#    
#    Returns:
#    pandas.DataFrame: The same table, but with daily change in counts.
#    """
#temp_cases = list(data["cases"])
#temp_cases = [0] + temp_cases[:-1] # Offset all values by 1 row
#daily_new_cases = data["cases"] - temp_cases
#data.insert(1, "new_cases", daily_new_cases)
#
#
#def replace_date_with_days_from_min_count(data, data_type, min_count, drop_date=True):
#    """Create a column where the value for each row is the number of days since the country/region in that row had a particular count of cases, deaths, or recoveries. You can then index by this column to compare how different countries were doing after similar amounts of time from first having infections.
#
#    Parameters:
#    
#    Returns:
#    pandas.DataFrame: The original table, with days since the xth case/death/recovery.
#    """
#country_day_cts = pd.DataFrame(columns=["day", "Country/Region", data_type]) # We'll append everything to this
#for country in top_names.index:
#
#        country_tbl = country_groups.get_group(country)
#        country_tbl = country_tbl[country_tbl[data_type] >= min_count] # Select only days with 100 or more cases
#        assert(country_tbl["date"].duplicated().sum() == 0) # Verify there are no duplicate days
#
#        day_number_col = range(0, len(country_tbl.index)) # Generate a column of day number since 100 cases
#        country_tbl.insert(loc=0, column="day", value=day_number_col) # Insert the column
#        country_tbl = country_tbl.drop(columns="date") # We don't need the date column anymore
#
#        country_day_cts = pd.concat([country_day_cts, country_tbl]).sort_values(by="day")
#
#    country_day_cts = country_day_cts.apply(pd.to_numeric, errors="ignore")

# Helper functions
def _wide_to_long(data, data_type):
    """Convert a dataframe from wide format to long format.

    Parameters:
    data (pandas.DataFrame): The dataframe to convert. Must have dates in at least some of the columns.
    data_type (str): The name of the data type the table contains. Either "cases", "deaths", or "recovered".

    Returns:
    pandas.DataFrame: The dataframe in long format.
    """
    if not data.columns.map(lambda x: issubclass(type(x), datetime.date)).any():
        raise ParameterError("Invalid table format. Must either have a 'date' column, or have dates as the columns.")

    id_cols = [col for col in data.columns if not issubclass(type(col), datetime.date)]
    data = pd.melt(data, id_vars=id_cols, var_name="date", value_name=data_type)
    data = data[data[data_type] != 0] # Drop rows of zeros

    id_cols.append("date")
    data = data.set_index(id_cols)

    # Reorder index so date is first
    idx_names = list(data.index.names)
    idx_names.remove("date")
    new_idx_name_order = ["date"] + idx_names
    data = data.reorder_levels(new_idx_name_order)

    # Convert index into just columns
    data = data.reset_index()

    return data

def _long_to_wide(data, data_type):
    """Convert a dataframe from long format to wide format.

    Parameters:
    data (pandas.DataFrame): The dataframe to convert. Must have a column called "date".
    data_type (str): The name of the data type to keep when we pivot. Either "cases", "deaths", or "recovered".

    Returns:
    pandas.DataFrame: The dataframe in wide format.
    """
    # If there are multiple data type columns, only keep the one specified
    possible_data_types = ["cases", "deaths", "recovered"]
    cols_to_drop = [col for col in possible_data_types if col != data_type and col in data.columns]
    data = data.drop(columns=cols_to_drop)

    # Spread the table, a la tidyr
    id_cols = [col for col in data.columns if col != data_type]
    data = data.set_index(id_cols)
    data = data.unstack(level=0, fill_value=0)
    data.columns = data.columns.droplevel(0)
    data.columns.name = None
    data = data.reset_index()

    return data
