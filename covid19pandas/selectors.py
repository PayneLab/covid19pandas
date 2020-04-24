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

def select_regions(data, region_col, regions, combine_subregions, data_cols_to_keep):
    """Select all data for particular regions within a table, optionally summing counts for subregions into one count for each region for each day.
    
    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designation you're specifying by. E.g., if you want to select particular states, pass the name of the state column
    regions (str or list of str): The regions to select.
    combine_subregions (bool, optional): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Default True. Otherwise, keeps the regions broken into subregions. 
    data_cols_to_keep (str or list of str): The data column(s) in the table that you want to be summed for each region group instead of dropped, if combine_subregions is True. We drop other columns by default, because numerical columns like Latitude and Longitude or FIPS would be messed up by the aggregation. This parameter has no effect if combine_subregions is False.

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
        else:
            group_cols = [region_col] # Wide format table

        # Make sure that the data_cols_to_keep columns all exist
        not_in = [col for col in data_cols_to_keep if not col in data.columns]
        if len(not_in) > 0:
            raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the data_cols_to_keep parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

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

def calc_x_day_mean(data, x, keep_originals, data_types):
    """Take a table of daily counts, and calculate the mean of the counts for each set of x consecutive days (e.g., a 3 day mean).

    Parameters:
    data (pandas.DataFrame): The data to calculate the means of.
    x (int): The number of days to calculate the means over.
    keep_originals (bool): Whether to keep the original values. Otherwise, the function just returns the means.
    data_types (list of str): A list of the data columns in your table that you want to calculate the x day group means for.
    Returns:
    pandas.DataFrame: The table, with means calculated over the specified number of days.
    """
    wide = False
    if "date" not in data.columns:
        wide = True

    if wide and keep_originals:
        raise ParameterError("You appear to have passed a wide format table, as there is no column of dates in the table. It is not possible to keep the original counts for a wide format table. Either pass a long format table, or pass keep_originals=False.")

    # Search for defined location id cols (based on data source and region)
    if {"Combined_Key"}.issubset(data.columns): # JHU table
        id_cols = ["Combined_Key"]
    elif {"county", "state"}.issubset(data.columns): # NYT USA state and county table
        id_cols = ["county", "state"]
    elif {"state"}.issubset(data.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not the state/county table.
        id_cols = ["state"]
    else:
        raise ParameterError("The dataframe you passed does not contain any of the standard location identification columns. Must contain one of these sets of columns: \n\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{data.columns}")

    if wide:
        data = _wide_to_long(data, "generic_data_type") # We use generic because if it's a wide table, we know there's only one data type, but we don't know what it is
        data_types = ["generic_data_type"]

    # Make sure that the data_types columns all exist
    not_in = [col for col in data_types if not col in data.columns]
    if len(not_in) > 0:
        raise ParameterError(f"The dataframe you passed does not contain all of the data types you passed to the data_types parameter. These are the missing columns:\n{not_in}\n\nYour dataframe's columns are:\n{data.columns}")

    # Extract date, id, and data cols
    cols_to_keep = [col for col in data.columns if col == "date" or col in id_cols + data_types]        
    means = data[cols_to_keep]
    
    # Get list of unique dates
    dates = means["date"].drop_duplicates(keep="first")

    # Generate a sequence assigning group numbers to each date, and make it a dataframe
    num_groups = len(dates) // x + 1
    group_nums = np.repeat(range(0, num_groups), x)
    group_nums = group_nums[0: len(dates)]
    groups = pd.DataFrame({"group_num": group_nums}, index=dates)
    
    # Use drop_duplicates keep parameter to get last and first days for each group
    first_days = groups["group_num"].drop_duplicates(keep="first")
    last_days = groups["group_num"].drop_duplicates(keep="last")

    # Swap the index and values for each of those two series, so we can join on the day numbers
    first_days = pd.Series(first_days.index, index=first_days, name="mean_group_start")
    last_days = pd.Series(last_days.index, index=last_days, name="mean_group_end")

    # Join those series into the group nums dataframe, joining on group num
    groups = groups.join(first_days, on="group_num")
    groups = groups.join(last_days, on="group_num")

    # Drop the group num column from the group dataframe
    groups = groups.drop(columns="group_num")

    # Join the group first and last days dataframe to the selected data and to the original dataframe, joining on date.
    means = means.join(groups, on="date")
    data = data.join(groups, on="date")

    # Fill NaNs in the grouping columns, so they don't get messed up in groupby or join operations
    for id_col in id_cols:
        means[id_col] = means[id_col].fillna("n/a")
        data[id_col] = data[id_col].fillna("n/a")

    # Group by first day and id cols, and aggregate
    group_cols = id_cols + ["mean_group_start"]
    means = means.groupby(group_cols).aggregate(np.mean)

    # Rename the mean columns
    means = means.rename(columns=lambda name, x=x: f"{name}_mean{x}days")

    # Join the mean data into the original dataframe
    data = data.join(means, on=group_cols) # Join the means into the original dataframe
    
    # Put the NaNs back in
    for id_col in id_cols:
        data[id_col] = data[id_col].replace(to_replace="n/a", value=np.nan)

    if not keep_originals:
        cols_to_drop = ["date"] + data_types
        data = data.drop(columns=cols_to_drop)
        data = data.drop_duplicates()

    if wide: # Therefore, keep_originals is False, and the above block that drops columns was executed
        data = data.drop(columns="mean_group_start") # We'll use mean_group_end for the column indices
        data = _long_to_wide(data, data_type=f"generic_data_type_mean{x}days", date_col="mean_group_end")

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

# Helper functions
def _wide_to_long(data, data_type):
    """Convert a dataframe from wide format to long format.

    Parameters:
    data (pandas.DataFrame): The dataframe to convert. Must have dates in at least some of the columns.
    data_type (str): The name of the data type the table contains.

    Returns:
    pandas.DataFrame: The dataframe in long format.
    """
    if not data.columns.map(lambda x: issubclass(type(x), datetime.date)).any():
        raise ParameterError("Invalid table format. Must either have a 'date' column, or have dates as the columns.")

    id_cols = [col for col in data.columns if not issubclass(type(col), datetime.date)]
    data = pd.melt(data, id_vars=id_cols, var_name="date", value_name=data_type)

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

def _long_to_wide(data, data_type, date_col="date", other_data_types_to_drop=[], sort_by=None):
    """Convert a dataframe from long format to wide format.

    Parameters:
    data (pandas.DataFrame): The dataframe to convert.
    data_type (str): The name of the data type to keep when we pivot.
    date_col (str, optional): The name of the column with the dates in it. Default "date".
    other_data_types_to_drop (list of str, optional): A list of other data_type columns that may exist in the table, which will be dropped. Note that if data_type is included in this list, it will actually not be dropped.
    sort_by (str, optional): The name of one of the indexing columns to sort the dataframe by before returning it. Default of None causes no extra sorting to be performed.

    Returns:
    pandas.DataFrame: The dataframe in wide format.
    """
    # If there are multiple data type columns, only keep the one specified
    cols_to_drop = [col for col in other_data_types_to_drop if col != data_type and col in data.columns]
    data = data.drop(columns=cols_to_drop)

    # Spread the table, a la tidyr
    id_cols = [col for col in data.columns if col != data_type]
    data = data.set_index(id_cols) # Putting these in the index keeps them from being spread
    data = data.unstack(level=date_col, fill_value=0)
    data.columns = data.columns.droplevel(0)
    data.columns.name = None
    if sort_by is not None:
        data = data.sort_index(level=sort_by)
    data = data.reset_index() # Take the saved columns out of the index

    return data

def _offset_subtract(col):
    """Takes a column, creates a copy with all values moved down by one (last value dropped, zero inserted at start), then subtracts the copy from the original and returns the result. For use with dataframe.groupby(...).transform().

    col (pandas.Series): The column to transform.

    Returns:
    pandas.Series: The transformed column.
    """
    offset = col.values[:-1]
    offset = np.insert(offset, 0, 0)
    return col - offset
