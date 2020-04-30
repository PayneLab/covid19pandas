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
from .utils import _long_to_wide, _wide_to_long, _offset_subtract

def select_top_x_regions(data, data_col, region_cols, x, combine_subregions=True, other_data_cols=[], exclude=[]):
    """Select the top x regions with the most cases, deaths, recoveries, or count of another data type.

    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    data_col (str): The data column you want to rank regions by.
    region_cols (str or list of str): The name(s) of the column(s) that contain the region designations you want to group by.
    x (int): The number of top regions to keep.
    combine_subregions (bool, optional): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Otherwise, keeps the region broken into subregions. Default True.
    other_data_cols (list of str, optional): A list of other data columns in the table that you want to be summed for each region group instead of dropped, if combine_subregions is True. This parameter has no effect if combine_subregions is False. Default empty list.
    exclude (list of str, optional): A list of regions to exclude from the selection. If you passed multiple region cols, a region with a value in any of those columns that matches a value in this list will be excluded. If an excluded region made the cut, the next highest region will take its place. Default empty list.

    Returns:
    pandas.DataFrame: Counts for the top x regions.
    """

    # If they give us a wide format table, make a long format copy for determining the top x regions
    if "date" not in data.columns:
        long_data = _wide_to_long(data, data_col) 
    else:
        long_data = data.copy()

    # Check that data_col is in the dataframe
    if data_col not in long_data.columns:
        raise ParameterError(f"There is no '{data_col}' column in the dataframe you passed. Existing columns: \n{data.columns}")

    # Process string input for region grouping cols
    if isinstance(region_cols, str):
        region_cols = [region_cols]

    # Find the top x regions
    # Get the last recorded day
    last_day = long_data["date"].max() 

    current_ct = long_data[long_data['date'] == last_day] # Pull all records for that day
    current_ct = long_data[['date', data_col] + region_cols]

    # Select only rows where region_cols values are not in the exclude list
    for region_col in region_cols: 
        current_ct = current_ct[~current_ct[region_col].isin(exclude)] 

    # Fill NaNs so they aren't excluded in groupby and can match in joins
    for region_col in region_cols: 
        current_ct[region_col] = current_ct[region_col].fillna("n/a") 
        data[region_col] = data[region_col].fillna("n/a")

    # Sum all counts for today in subregions within regions. Now we have to totals for all the regions for the most recent day.
    current_ct = current_ct.groupby(region_cols).aggregate(np.sum) 

    # Sort the table by the data_col, then select the last x names--they are the top x regions
    top_x_names = current_ct.sort_values(by=data_col).tail(x) 
    top_x_names = top_x_names.drop(columns=data_col) # So that we don't have column overlap when joining with the original data table. So now top_x_names is a dataframe with just an index and no columns, but still joinable

    # Select data for the top regions
    data = data.join(top_x_names, on=region_cols, how="inner")

    # Put the NaNs back in
    for region_col in region_cols: 
        data[region_col] = data[region_col].replace(to_replace="n/a", value=np.nan)

    # If it's long format, sort everything by date first again.
    if "date" in data.columns:
        data = data.sort_values(by=["date"] + region_cols)

    if combine_subregions:
        # Make sure that the other_data_cols columns all exist
        not_in = [col for col in other_data_cols if not col in data.columns]
        if len(not_in) > 0:
            raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the other_data_cols parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

        # Drop columns that would be messed up by the groupby
        cols_to_not_drop = ["date", data_col] + region_cols + other_data_cols
        cols_to_drop = [col for col in data.columns if col not in cols_to_not_drop and not issubclass(type(col), datetime.date)]
        data = data.drop(columns=cols_to_drop)

        # Determine the id cols to group by, and fill NaNs in them so those aren't excluded in groupby
        id_cols = data.columns[data.columns.isin(["date"] + region_cols)].tolist()
        for id_col in id_cols:
            data[id_col] = data[id_col].fillna("n/a") 

        # Sum up total counts per day for each country
        data = data.groupby(id_cols).aggregate(np.sum).reset_index() 

        # Put the NaNs back in
        for id_col in id_cols:
            data[id_col] = data[id_col].replace(to_replace="n/a", value=np.nan) 

    return data

def select_regions(data, region_col, regions, combine_subregions=False, data_cols=[]):
    """Select all data for particular regions within a table, optionally summing counts for subregions into one count for each region for each day.
    
    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designation you're specifying by. E.g., if you want to select particular states, pass the name of the state column
    regions (str or list of str): The regions to select.
    combine_subregions (bool): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Default False.
    data_cols (str or list of str, optional): Only required when passing long format tables and combine_subregions is True. These are the data column(s) in the table that you want to be summed for each region group instead of dropped, if combine_subregions is True. Default is an empty list.

    Returns:
    pandas.DataFrame: The data for the specified regions.
    """
    # Allow them to pass either a string for one column, or a list of str for several columns.
    if isinstance(regions, str):
        regions = [regions]
    if isinstance(data_cols, str):
        data_cols = [data_cols]

    # Select the data
    data = data[data[region_col].isin(regions)].copy()

    # Check that there at least one row matched
    if data.shape[0] < 1:
        raise ParameterError(f"No rows in the dataframe have any of the values {regions} in the column '{region_col}'.")

    # Aggregate, if desired
    if combine_subregions:
        if "date" in data.columns: # Long format table
            group_cols = ["date", region_col]

            # Make sure that the data_cols columns all exist
            not_in = [col for col in data_cols if not col in data.columns]
            if len(not_in) > 0:
                raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the data_cols parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

        else:
            group_cols = [region_col] # Wide format table

        # Drop columns that would be messed up by the groupby
        cols_to_not_drop = group_cols + data_cols
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

def calc_x_day_rolling_mean(data, data_cols, region_cols, x, center=False):
    """Calculate a centered rolling mean with x days for each number in a count.

    Parameters:
    data (pandas.DataFrame): The data to calculate the rolling means for.
    data_cols (str or list of str): The data columns in your table that you want to calculate the x day rolling means for.
    region_cols (str or list of str): Column(s) that uniquely identify each region for each day.
    x (int): The number of days to calculate the means over.
    center (bool, optional): Whether to center the window on each value, instead of having the value at the right side of the window. Default False.

    Returns:
    pandas.DataFrame: The table, with rolling means calculated over the specified number of days.
    """

    # Convert from str to list input if needed
    if isinstance(data_cols, str):
        data_cols = [data_cols]

    if isinstance(region_cols, str):
        region_cols = [region_cols]

    # Deal with wide format tables
    wide = False
    if "date" not in data.columns:
        wide = True
        data = _wide_to_long(data, "generic_data_col") # We use generic because if it's a wide table, we know there's only one data type, but we don't know what it is
        data_cols = ["generic_data_col"]

    # Check that the provided region_cols uniquely identify each row for each date
    if data.duplicated(subset=["date"] + region_cols).any():
        raise ParameterError(f"The region_cols you passed do not uniquely identify each row for each day. You passed {region_cols}.")

    # Make sure that the data_cols columns all exist
    not_in = [col for col in data_cols if not col in data.columns]
    if len(not_in) > 0:
        raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the data_cols parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

    # Fill NaNs in the grouping columns, so they don't get messed up in groupby or join operations
    for region_col in region_cols:
        data[region_col] = data[region_col].fillna("n/a")

    # For each data_col, group by the id cols and calculate a rolling mean with a window x days wide, then join back into the original table
    data_date_idx = data.set_index("date") # So that the groupby and rolling calculations will work properly
    means_cols = []

    for data_col in data_cols:
        means = data_date_idx.groupby(region_cols)[data_col].rolling(window=x, min_periods=1, center=center).mean()

        # Note that we follow the standard of adding the transformation descriptor ("mean_" in this case) to the beginning of the column name so that when we compose different calc functions, the order of composition is apparent.
        col_name = f"mean_{data_col}"
        means.name = col_name
        means_cols.append(col_name)

        data = data.join(means, on=region_cols + ["date"])

    # Put the NaNs back in
    for region_col in region_cols:
        data[region_col] = data[region_col].replace(to_replace="n/a", value=np.nan)

    if wide:
        data = data.drop(columns="generic_data_col")
        data = _long_to_wide(data, data_type=means_cols[0])

    return data

def calc_daily_change(data, data_cols, region_cols):
    """Get the daily change for a cumulative count within each region. Original cumulative counts are not dropped.
    
    Parameters:
    data (pandas.DataFrame): The cumulative counts from which to calculate the daily change.
    data_col (str or list of str): The column(s) you want to calculate the daily change for. Other columns will be left unchanged.
    region_cols (str or list of str): Column(s) that uniquely identify each region for each day.
    
    Returns:
    pandas.DataFrame: The same table, but with daily change in counts. The column is named "'daily_' + data_col" for each data type.
    """
    wide = False
    if "date" not in data.columns:
        wide = True

    # Convert from str to list input if needed
    if isinstance(data_cols, str):
        data_cols = [data_cols]

    if isinstance(region_cols, str):
        region_cols = [region_cols]

    # Check that the provided region_cols uniquely identify each row for each date
    if wide:
        unique_groups = region_cols
    else: # long format table
        unique_groups = ["date"] + region_cols

    if data.duplicated(subset=unique_groups).any():
        raise ParameterError(f"The region_cols you passed do not uniquely identify each row for each day. You passed {region_cols}.")

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
        for data_col in data_cols:

            if data_col not in data.columns:
                raise ParameterError(f"There is no '{data_col}' column in the dataframe you passed. Existing columns: \n{data.columns}")

            # Duplicate grouping cols, since they'll be lost when used for grouping
            for region_col in region_cols:
                data = data.assign(**{region_col + "_group": data[region_col]})

            # Add the suffix to the region_cols list, so we group by (and lose) the duplicated columns
            suffix_region_cols = [col + "_group" for col in region_cols]

            # Duplicate the count col so we can keep the cumulative counts
            # Note that we follow the standard of adding the transformation descriptor ("daily_" in this case) to the beginning of the column name so that when we compose different calc functions, the order of composition is apparent.
            daily_col = "daily_" + data_col
            data = data.assign(**{daily_col: data[data_col]})

            # Put all columns besides the duplicates we created into the index, so they aren't affected by the groupby
            id_cols = data.columns[~data.columns.isin(suffix_region_cols + [daily_col])].tolist()
            data = data.set_index(id_cols)

            # Fill NaNs in grouping cols (fillna excludes index)
            data = data.fillna("n/a")

            # Group by location and calculate daily counts with our helper function _offset_subtract
            data = data.groupby(suffix_region_cols).transform(_offset_subtract)

            # Put back in any remaining NaNs
            data = data.replace(to_replace="n/a", value=np.nan)

            # Take the other columns back out of the index
            data = data.reset_index()

    return data


def calc_days_since_min_count(data, data_col, region_cols, min_count):
    """Create a column where the value for each row is the number of days since the country/region in that row had a particular count of a data type, e.g. cases, deaths, or recoveries. You can then index by this column to compare how different countries were doing after similar amounts of time from first having infections.

    Parameters:
    data (pandas.DataFrame): The dataframe to do the calculation for.
    data_col (str): The data type you want the days since the minimum count of. If other data types are present in the table, they will also be kept for days that pass the cutoff in this data type.
    region_cols (str or list of str): Column(s) that uniquely identify each region for each day.
    min_count (int): The minimum count for your data type at which you want to start counting from for each country/region.
    
    Returns:
    pandas.DataFrame: The original table, with days since the xth case/death/recovery. Note: This function only outputs data in long format tables, since wide format tables would be messy with this transformation.
    """
    date_col = "date"
    if isinstance(region_cols, str): 
        region_cols = [region_cols]

    # If they give us a wide format table, convert it to long format.
    if "date" not in data.columns:
        data = _wide_to_long(data, data_col) 

    # Drop all rows for days that don't meet the minimum count
    data = data[data[data_col] >= min_count] 

    # Check no duplicate dates in each group
    if data.duplicated(subset=[date_col] + region_cols).any():
        raise ParameterError("The combination of grouping columns you passed does not uniquely identify each row for each day. Either pass a different set of grouping columns, or aggregate the counts for each combination of day and grouping columns before using this function.")

    # Duplicate grouping cols, since they'll be lost when used for grouping
    for region_col in region_cols:
        data = data.assign(**{region_col + "_group": data[region_col]})

    # Add the suffix to the group_cols list, so we group by (and lose) the duplicated columns
    suffix_group_cols = [col + "_group" for col in region_cols]

    # Duplicate the date col so we can keep the original dates if desired
    days_since_col = f"days_since_{min_count}_{data_col}"
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
    data = data.sort_values(by=[date_col] + region_cols)

    return data
