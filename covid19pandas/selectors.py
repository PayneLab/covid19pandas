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

from .exceptions import ParameterError
from .getters import get_jhu_location_data
from .utils import _long_to_wide, _wide_to_long, _offset_subtract

def select_top_x_regions(data, region_col, data_type, x, combine_subregions, other_data_cols_to_keep, exclude=[]):
    """Select the top x regions with the most cases, deaths, recoveries, or count of another data type.

    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designations you want to group by.
    data_type (str): The data type you want to rank regions by, e.g. "cases", "deaths", or "recovered". Or a different column name if you like.
    x (int): The number of top regions to keep. Default 10.
    combine_subregions (bool): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Otherwise, keeps the region broken into subregions. Default True.
    other_data_cols_to_keep (list of str, optional): A list of other data columns in the table that you want to be summed for each region group instead of dropped, if combine_subregions is True. We drop other columns by default, because numerical columns like Latitude and Longitude or FIPS would be messed up by the aggregation. This parameter has no effect if combine_subregions is False.
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
    current_ct = current_ct.fillna("n/a") # Fill NaNs so they aren't excluded in groupby
    current_ct = current_ct.groupby(region_col).aggregate(np.sum) # Sum all counts for today in subregions within that regions

    top_x_names = current_ct.sort_values(by=data_type).tail(x) # Get the names of the top regions
    top_x_cts = data[data[region_col].isin(top_x_names.index)] # Filter out other regions

    if combine_subregions:
        # Make sure that the other_data_cols_to_keep columns all exist
        not_in = [col for col in other_data_cols_to_keep if not col in data.columns]
        if len(not_in) > 0:
            raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the other_data_cols_to_keep parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

        # Drop columns that would be messed up by the groupby
        cols_to_not_drop = ["date", region_col, data_type] + other_data_cols_to_keep
        cols_to_drop = [col for col in top_x_cts.columns if col not in cols_to_not_drop and not issubclass(type(col), datetime.date)]
        top_x_cts = top_x_cts.drop(columns=cols_to_drop)

        # Determine the id cols to group by, and fill NaNs in them so those aren't excluded in groupby
        id_cols = top_x_cts.columns[top_x_cts.columns.isin(["date", region_col])].tolist()
        for id_col in id_cols:
            top_x_cts[id_col] = top_x_cts[id_col].fillna("n/a") 

        top_x_cts = top_x_cts.groupby(id_cols).aggregate(np.sum).reset_index() # Sum up total counts per day for each country
        top_x_cts = top_x_cts.replace(to_replace="n/a", value=np.nan) # Put the NaNs back in

    return top_x_cts

def select_regions(data, region_col, regions, combine_subregions, data_cols_to_keep=[]):
    """Select all data for particular regions within a table, optionally summing counts for subregions into one count for each region for each day.
    
    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designation you're specifying by. E.g., if you want to select particular states, pass the name of the state column
    regions (str or list of str): The regions to select.
    combine_subregions (bool): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Default True. Otherwise, keeps the regions broken into subregions. 
    data_cols_to_keep (str or list of str, optional): Only required for long format tables. This is the data column(s) in the table that you want to be summed for each region group instead of dropped, if combine_subregions is True. We drop other columns by default, because numerical columns like Latitude and Longitude or FIPS would be messed up by the aggregation. This parameter has no effect if combine_subregions is False or you pass a wide format table; default is an empty list.

    Returns:
    pandas.DataFrame: The data for the specified regions.
    """
    # Allow them to pass either a string for one column, or a list of str for several columns.
    if isinstance(regions, str):
        regions = [regions]
    if isinstance(data_cols_to_keep, str):
        data_cols_to_keep = [data_cols_to_keep]

    # Select the data
    data = data[data[region_col].isin(regions)]

    # Aggregate, if desired
    if combine_subregions:
        if "date" in data.columns: # Long format table
            group_cols = ["date", region_col]

            # Make sure that the data_cols_to_keep columns all exist
            not_in = [col for col in data_cols_to_keep if not col in data.columns]
            if len(not_in) > 0:
                raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the data_cols_to_keep parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

        else:
            group_cols = [region_col] # Wide format table

        # Drop columns that would be messed up by the groupby
        cols_to_not_drop = group_cols + data_cols_to_keep
        cols_to_drop = [col for col in data.columns if col not in cols_to_not_drop and not issubclass(type(col), datetime.date)]
        data = data.drop(columns=cols_to_drop)

        # Fill NaNs in the group cols so they aren't excluded in groupby and joins
        for group_col in group_cols:
            data[group_col] = data[group_col].fillna("n/a") 

        data = data.groupby(group_cols).aggregate(np.sum)
        data = data.reset_index()

        for group_col in group_cols:
            data[group_col] = data[group_col].replace(to_replace="n/a", value=np.nan) # Put the NaNs back in

    return data

def calc_x_day_rolling_mean(data, data_types, x, center=True):
    """Calculate a centered rolling mean with x days for each number in a count.

    Parameters:
    data (pandas.DataFrame): The data to calculate the rolling means for.
    data_types (str or list of str): The data columns in your table that you want to calculate the x day rolling means for. If you pass a wide format table, this parameter is meaningless since the data is obviously just whatever is in the date columns, so you can just pass an empty list for this parameter in that case.
    x (int): The number of days to calculate the means over.
    center (bool, optional): Whether to center the window on each value. Default True.

    Returns:
    pandas.DataFrame: The table, with rolling means calculated over the specified number of days.
    """

    # Convert from str to list input if needed
    if isinstance(data_types, str):
        data_types = [data_types]

    # Search for defined location id cols (based on data source and region)
    jhu = False
    if {"Combined_Key"}.issubset(data.columns): # JHU table
        id_cols = ["Combined_Key"]
        jhu = True
    elif {"county", "state"}.issubset(data.columns): # NYT USA state and county table
        id_cols = ["county", "state"]
    elif {"state"}.issubset(data.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not the state/county table.
        id_cols = ["state"]
    else:
        raise ParameterError("The dataframe you passed does not contain any of the standard location identification columns. Must contain one of these sets of columns: \n\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{data.columns}")

    # Deal with wide format tables
    wide = False
    if "date" not in data.columns:
        wide = True
        data = _wide_to_long(data, "generic_data_type") # We use generic because if it's a wide table, we know there's only one data type, but we don't know what it is
        data_types = ["generic_data_type"]

    # Make sure that the data_types columns all exist
    not_in = [col for col in data_types if not col in data.columns]
    if len(not_in) > 0:
        raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the data_types parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

    # Fill NaNs in the grouping columns, so they don't get messed up in groupby or join operations
    for id_col in id_cols:
        data[id_col] = data[id_col].fillna("n/a")

    # For each data_type, group by the id cols and calculate a rolling mean with a window x days wide, then join back into the original table
    data_date_idx = data.set_index("date") # So that the groupby and rolling calculations will work properly
    means_cols = []

    for data_type in data_types:
        means = data_date_idx.groupby(id_cols)[data_type].rolling(window=x, min_periods=1, center=center).mean()

        col_name = f"{data_type}_mean"
        means.name = col_name
        means_cols.append(col_name)

        data = data.join(means, on=id_cols + ["date"])

    # Put the NaNs back in
    for id_col in id_cols:
        data[id_col] = data[id_col].replace(to_replace="n/a", value=np.nan)

    if wide:

        # Convert back
        meaned_cols_to_keep = ["date"] + id_cols + means_cols
        data = data[meaned_cols_to_keep]
        data = data.drop_duplicates(keep="first")

        if jhu: # Join back in the location columns

            data = data.set_index(id_cols)
            loc_table = get_jhu_location_data(update=False) # That way it will match whatever it was when they last downloaded the JHU data, because loading JHU data automatically loads the location table.
            data = loc_table.merge(data, on=id_cols, how="right", suffixes=(False, False), validate="one_to_many")

            # If it's the global table, drop the FIPS and Admin2 columns--they're only relevant for the US table
            cols_to_check = ["FIPS", "Admin2"]
            for col in cols_to_check:
                if data[col].isnull().all():
                    data = data.drop(columns=col)

        data = _long_to_wide(data, data_type=means_cols[0])

    return data

def calc_daily_change(data, data_types):
    """Get the daily change for a cumulative count within each region. Original cumulative counts are not dropped.
    
    Parameters:
    data (pandas.DataFrame): The cumulative counts from which to calculate the daily change.
    data_type (str or list of str): The column(s) you want to calculate the daily change for. Other columns will be left unchanged.
    
    Returns:
    pandas.DataFrame: The same table, but with daily change in counts. The column is named "'daily_' + data_type" for each data type.
    """
    wide = False
    if "date" not in data.columns:
        wide = True

    # Convert from str to list input if needed
    if isinstance(data_types, str):
        data_types = [data_types]

    # Search for defined grouping cols (based on data source and region)
    if {"Combined_Key"}.issubset(data.columns): # JHU table
        group_cols = ["Combined_Key"]
    elif {"county", "state"}.issubset(data.columns): # NYT USA state and county table
        group_cols = ["county", "state"]
    elif {"state"}.issubset(data.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not the state/county table.
        group_cols = ["state"]
    else:
        raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{data.columns}")

    if wide:
        if not data.columns.map(lambda x: issubclass(type(x), datetime.date)).any():
            raise ParameterError("Invalid table format. Must either have a 'date' column, or have dates as the columns.")

        cols_are_dates_filter = data.columns.map(lambda x: issubclass(type(x), datetime.date)).to_series().astype(bool)
        id_cols = data.columns[~cols_are_dates_filter].tolist()
        date_cols = data.columns[cols_are_dates_filter].tolist()

        new_data = data[id_cols]
        new_data.insert(loc=len(new_data.columns), column=date_cols[0], value=data[date_cols[0]]) # All counts on first day were new
        for i in range(1, len(date_cols)):
            day = date_cols[i]
            prev_day = date_cols[i - 1]
            new_data.insert(loc=len(new_data.columns), column=day, value=data[day] - data[prev_day])

        data = new_data

    else: # It's a long format table
        for data_type in data_types:

            if data_type not in data.columns:
                raise ParameterError(f"There is no '{data_type}' column in the dataframe you passed. Existing columns: \n{data.columns}")

            # Duplicate grouping cols, since they'll be lost when used for grouping
            for group_col in group_cols:
                data = data.assign(**{group_col + "_group": data[group_col]})

            # Add the suffix to the group_cols list, so we group by (and lose) the duplicated columns
            suffix_group_cols = [col + "_group" for col in group_cols]

            # Duplicate the count col so we can keep the cumulative counts
            daily_col = "daily_" + data_type
            data = data.assign(**{daily_col: data[data_type]})

            # Put all columns besides the duplicates we created into the index, so they aren't affected by the groupby
            id_cols = data.columns[~data.columns.isin(suffix_group_cols + [daily_col])].tolist()
            data = data.set_index(id_cols)

            # Fill NaNs in grouping cols (fillna excludes index)
            data = data.fillna("n/a")

            # Group by location and calculate daily counts with our helper function _offset_subtract
            data = data.groupby(suffix_group_cols).transform(_offset_subtract)

            # Put back in any remaining NaNs
            data = data.replace(to_replace="n/a", value=np.nan)

            # Take the other columns back out of the index
            data = data.reset_index()

    return data


def calc_days_since_min_count(data, data_type, min_count, group_by):
    """Create a column where the value for each row is the number of days since the country/region in that row had a particular count of a data type, e.g. cases, deaths, or recoveries. You can then index by this column to compare how different countries were doing after similar amounts of time from first having infections.

    Parameters:
    data (pandas.DataFrame): The dataframe to do the calculation for.
    data_type (str): The data type you want the days since the minimum count of. If other data types are present in the table, they will also be kept for days that pass the cutoff in this data type.
    min_count (int): The minimum count for your data type at which you want to start counting from for each country/region.
    group_by (str or list of str): The column(s) that uniquely identify each region for each day.
    
    Returns:
    pandas.DataFrame: The original table, with days since the xth case/death/recovery. Note: This function only outputs data in long format tables, since wide format tables would be messy with this transformation.
    """
    date_col = "date"
    if isinstance(group_by, str): 
        group_by = [group_by]

    # If they give us a wide format table, convert it to long format.
    if "date" not in data.columns:
        data = _wide_to_long(data, data_type) 

    # Drop all rows for days that don't meet the minimum count
    data = data[data[data_type] >= min_count] 

    # Check no duplicate dates in each group
    if data.duplicated(subset=[date_col] + group_by).any():
        raise ParameterError("The combination of grouping columns you passed does not uniquely identify each row for each day. Either pass a different set of grouping columns, or aggregate the counts for each combination of day and grouping columns before using this function.")

    # Duplicate grouping cols, since they'll be lost when used for grouping
    for group_col in group_by:
        data = data.assign(**{group_col + "_group": data[group_col]})

    # Add the suffix to the group_cols list, so we group by (and lose) the duplicated columns
    suffix_group_cols = [col + "_group" for col in group_by]

    # Duplicate the date col so we can keep the original dates if desired
    days_since_col = f"days_since_{min_count}_{data_type}"
    data = data.assign(**{days_since_col: data[date_col]})

    # Put the non-grouping and non-date columns in the index, so they don't get changed
    data = data.sort_values(by="date") # This will make sure eveything is in the right order to generate days since the cutoff
    id_cols = data.columns[~data.columns.isin(suffix_group_cols + [days_since_col])].tolist()
    data = data.set_index(id_cols)

    # Fill NaNs in grouping cols (fillna excludes index)
    data = data.fillna("n/a")

    # Separate the groups, and calculate the number of days since the specified count
    data = data.groupby(suffix_group_cols).transform(lambda col: pd.Series(data=range(0, len(col)), index=col.index))

    # Put back in any remainings NaNs
    data = data.replace(to_replace="n/a", value=np.nan)

    # Take the other columns back out of the index
    data = data.reset_index()

    # Sort the table
    data = data.sort_values(by=[date_col] + group_by)

    return data
