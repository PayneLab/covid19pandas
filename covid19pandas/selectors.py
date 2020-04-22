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
    current_ct = current_ct.fillna("n/a") # Fill NaNs so they aren't excluded in groupby
    current_ct = current_ct.groupby(region_col).aggregate(np.sum) # Sum all counts for today in subregions within that regions

    top_x_names = current_ct.sort_values(by=data_type).tail(x) # Get the names of the top regions
    top_x_cts = data[data[region_col].isin(top_x_names.index)] # Filter out other regions

    if combine_subregions:
        cols_to_drop = [col for col in top_x_cts.columns if col not in ["date", region_col, data_type] and not issubclass(type(col), datetime.date)]
        top_x_cts = top_x_cts.drop(columns=cols_to_drop)
        id_cols = top_x_cts.columns[top_x_cts.columns.isin(["date", region_col])].tolist()

        for id_col in id_cols:
            top_x_cts[id_col] = top_x_cts[id_col].fillna("n/a") # Fill NaNs so they aren't excluded in groupby
        top_x_cts = top_x_cts.groupby(id_cols).aggregate(np.sum).reset_index() # Sum up total cases per day, per country
        top_x_cts = top_x_cts.replace(to_replace="n/a", value=np.nan) # Put the NaNs back in

    return top_x_cts

def select_regions(data, region_col, regions, combine_subregions):
    """Select all data for particular regions within a table, optionally summing counts for subregions into one count for each region for each day.
    
    Parameters:
    data (pandas.DataFrame): The dataframe from which to select data.
    region_col (str): The name of the column that contains the region designation you're specifying by. E.g., if you want to select particular states, pass the name of the state column
    regions (str or list of str): The regions to select.
    combine_subregions (bool, optional): When a particular region has different subregions, whether to sum the daily counts for all those subregions into one count for the region for each day. Otherwise, keeps the region broken into subregions. Default True.

    Returns:
    pandas.DataFrame: The data for the specified regions.
    """
    # Allow them to pass either a string for one column, or a list of str for several columns.
    if isinstance(regions, str):
        regions = [regions]

    # Select the data
    data = data[data[region_col].isin(regions)]

    # Aggregate, if desired
    if combine_subregions:
        if "date" in data.columns: # Long format table
            group_cols = ["date", region_col]
        else:
            group_cols = [region_col] # Wide format table

        for group_col in group_cols:
            data[group_col] = data[group_col].fillna("n/a") # Fill NaNs so they aren't excluded in groupby and joins
        data = data.groupby(group_cols).aggregate(np.sum)
        data = data.reset_index()

        for group_col in group_cols:
            data[group_col] = data[group_col].replace(to_replace="n/a", value=np.nan) # Put the NaNs back in

        # Drop columns that would've been messed up by the aggregation
        cols_to_not_drop = group_cols + ["cases", "deaths", "recovered"]
        cols_to_drop = [col for col in data.columns if col not in cols_to_not_drop and not issubclass(type(col), datetime.date)]
        data = data.drop(columns=cols_to_drop)

    return data

def calc_x_day_avg(data, x, keep_unaveraged):
    """Take a table of daily counts, and average the counts for each set of x consecutive days (e.g., a 3 day average).

    Parameters:
    data (pandas.DataFrame): The data to average.
    x (int): The number of days to put into each averaged group.
    keep_unaveraged (bool): Whether to keep the unaveraged values. Otherwise, the function just returns the averaged values.

    Returns:
    pandas.DataFrame: The table, averaged over the specified number of days.
    """
    wide = False
    if "date" not in data.columns:
        wide = True

    # Search for defined location id cols (based on data source and region)
    if {"Province/State", "Country/Region"}.issubset(data.columns): # JHU global table
        id_cols = ["Province/State", "Country/Region"]
    elif {"Combined_Key"}.issubset(data.columns): # JHU USA table
        id_cols = ["Combined_Key"]
    elif {"county", "state"}.issubset(data.columns): # NYT USA state and county table
        id_cols = ["county", "state"]
    elif {"state"}.issubset(data.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not the state/county table.
        id_cols = ["state"]
    else:
        raise ParameterError("The dataframe you passed does not contain any of the standard location identification columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{data.columns}")

    if wide:

        # Make id cols the index
        averages = data.set_index(id_cols)

        # Drop cols besides date cols
        cols_to_drop = [col for col in data.columns if not issubclass(type(col), datetime.date)]
        averages = data.drop(columns=cols_to_drop)

        # Stack the date cols into the index

        # Get list of unique dates

        # Generate a sequence assigning group numbers to each date, and make it a series

        # Use drop_duplicates keep parameter to get last and first days for each group

        # Join that series to the data, both indexed by date

        # Group by first day and id cols, and aggregate

        # Unstack the date level


    else: # Long format table

        # Extract date, data, and id cols
        data_cols = ["cases", "deaths", "recovered"]
        cols_to_keep = [col for col in data.columns if col == "date" or col in id_cols or col in data_cols]        
        averages = data[cols_to_keep]
        
        # Get list of unique dates
        dates = averages["date"].drop_duplicates(keep="first")

        # Generate a sequence assigning group numbers to each date, and make it a dataframe
        num_groups = len(dates) // x + 1
        group_nums = np.repeat(range(0, num_groups), x)
        group_nums = group_nums[0: len(dates)]
        groups = pd.DataFrame({"group_num": group_nums}, index=dates)
        
        # Use drop_duplicates keep parameter to get last and first days for each group
        first_days = groups["group_num"].drop_duplicates(keep="first")
        last_days = groups["group_num"].drop_duplicates(keep="last")

        # Swap the index and values for each of those two series, so we can join on the day numbers
        first_days = pd.Series(first_days.index, index=first_days, name="group_first_day")
        last_days = pd.Series(last_days.index, index=last_days, name="group_last_day")

        # Join those series into the group nums dataframe, joining on group num
        groups = groups.join(first_days, on="group_num")
        groups = groups.join(last_days, on="group_num")

        # Drop the group num column from the group dataframe
        groups = groups.drop(columns="group_num")

        # Join the group first and last days dataframe to the selected data and to the original dataframe, joining on date.
        averages = averages.join(groups, on="date")
        data = data.join(groups, on="date")

        # Fill NaNs in the grouping columns, so they don't get messed up in groupby or join operations
        for id_col in id_cols:
            averages[id_col] = averages[id_col].fillna("n/a")
            data[id_col] = data[id_col].fillna("n/a")

        # Group by first day and id cols, and aggregate
        group_cols = id_cols + ["group_first_day"]
        averages = averages.groupby(group_cols).aggregate(np.mean)

        # Rename the averaged columns
        averages = averages.rename(columns=lambda name, x=x: f"{name}_avg{x}days")

        # Join the averaged data into the original dataframe
        data = data.join(averages, on=group_cols) # Join the averages into the original dataframe
        
        # Put the NaNs back in
        for id_col in id_cols:
            data[id_col] = data[id_col].replace(to_replace="n/a", value=np.nan)

        if not keep_unaveraged:
            cols_to_drop = ["date"] + data_cols
            data = data.drop(columns=cols_to_drop)
            data = data.drop_duplicates()

    return data

def calc_daily_change(data, data_type="all"):
    """Get the daily change in the number of cases/deaths/recoveries, instead of cumulative counts.
    
    Parameters:
    data (pandas.DataFrame): The cumulative counts from which to calculate the daily change.
    data_type (str): When your table contains multiple count types (e.g. cases, deaths, recovered), use this parameter to specify which columns you want to calculate the daily change for. Other columns will be left unchanged. Default "all".
    
    Returns:
    pandas.DataFrame: The same table, but with daily change in counts. The column is named "daily_" + data_type
    """
    wide = False
    if "date" not in data.columns:
        wide = True

    if data_type == "all":
        if wide:
            raise ParameterError("You passed data_type='all', but having your wide table format and processing multiple data types is not possible, since a wide format table contains only one data type. Pass a long format table, or process one data type at a time.")

        data_types = ["cases", "deaths"]
        if "recovered" in data.columns:
            data_types.append("recovered")

    elif data_type in ["cases", "deaths", "recovered"]:
        data_types = [data_type]
    else:
        raise ParameterError(f"{data_type} is not a valid data type. Pass 'cases', 'deaths', or 'recovered'.")

    # Search for defined grouping cols (based on data source and region)
    if {"Province/State", "Country/Region"}.issubset(data.columns): # JHU global table
        group_cols = ["Province/State", "Country/Region"]
    elif {"Combined_Key"}.issubset(data.columns): # JHU USA table
        group_cols = ["Combined_Key"]
    elif {"county", "state"}.issubset(data.columns): # NYT USA state and county table
        group_cols = ["county", "state"]
    elif {"state"}.issubset(data.columns): # NYT USA state only table. Note that this column also exists in the state/county table, so we do the check after we've determined it's not the state/county table.
        group_cols = ["state"]
    else:
        raise ParameterError("The dataframe you passed does not contain any of the standard location grouping columns. Must contain one of these sets of columns: \n\n{'Province/State', 'Country/Region'}\n{'Combined_Key'}\n{'county', 'state'}\n{'state'}\n\n" + f"Your dataframe's columns are:\n{data.columns}")

    if wide:
        if not data.columns.map(lambda x: issubclass(type(x), datetime.date)).any():
            raise ParameterError("Invalid table format. Must either have a 'date' column, or have dates as the columns.")

        cols_are_dates_filter = data.columns.map(lambda x: issubclass(type(x), datetime.date)).to_series().astype(bool)
        id_cols = data.columns[~cols_are_dates_filter].tolist()
        date_cols = data.columns[cols_are_dates_filter].tolist()

        new_data = data[id_cols]
        new_data = new_data.assign(**{str(date_cols[0]): data[date_cols[0]]}) # All counts on first day were new
        for i in range(1, len(date_cols)):
            day = date_cols[i]
            prev_day = date_cols[i - 1]
            new_data.insert(loc=len(new_data.columns), column=day, value=data[day] - data[prev_day])

        data = new_data

    else: # It's a long format table
        for iter_data_type in data_types:

            if iter_data_type not in data.columns:
                raise ParameterError(f"There is no '{iter_data_type}' column in the dataframe you passed. Existing columns: \n{data.columns}")

            # Duplicate grouping cols, since they'll be lost when used for grouping
            for group_col in group_cols:
                data = data.assign(**{group_col + "_group": data[group_col]})

            # Add the suffix to the group_cols list, so we group by (and lose) the duplicated columns
            suffix_group_cols = [col + "_group" for col in group_cols]

            # Duplicate the count col so we can keep the cumulative counts if desired
            daily_col = "daily_" + iter_data_type
            data = data.assign(**{daily_col: data[iter_data_type]})

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
    """Create a column where the value for each row is the number of days since the country/region in that row had a particular count of cases, deaths, or recoveries. You can then index by this column to compare how different countries were doing after similar amounts of time from first having infections.

    Parameters:
    data (pandas.DataFrame): The dataframe to do the calculation for.
    data_type (str): The data type you want the days since the minimum count of. If other data types are present in the table, they will also be kept for days that pass the cutoff in this data type.
    min_count (int): The minimum number of cases, deaths, or recovered that you want to start counting from for each country/region.
    group_by (str or list of str): The column(s) that uniquely identify each region for each day.
    
    Returns:
    pandas.DataFrame: The original table, with days since the xth case/death/recovery. Note: This function only outputs data in long format tables, since wide format tables would be messy with this transformation.
    """
    date_col = "date"

    # Allow them to pass either a string for one column, or a list of str for several columns.
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

def _long_to_wide(data, data_type, possible_data_types=["cases", "deaths", "recovered"], sort_by=None):
    """Convert a dataframe from long format to wide format.

    Parameters:
    data (pandas.DataFrame): The dataframe to convert. Must have a column called "date".
    data_type (str): The name of the data type to keep when we pivot. Either "cases", "deaths", or "recovered".
    possible_data_types (list of str, optional): A list of other data_type columns that may exist in the table, which will be dropped. Default is the standard data types ["cases", "deaths", "recovered"]
    sort_by (str, optional): The name of one of the indexing columns to sort the dataframe by before returning it. Default of None causes no extra sorting to be performed.

    Returns:
    pandas.DataFrame: The dataframe in wide format.
    """
    # If there are multiple data type columns, only keep the one specified
    cols_to_drop = [col for col in possible_data_types if col != data_type and col in data.columns]
    data = data.drop(columns=cols_to_drop)

    # Spread the table, a la tidyr
    id_cols = [col for col in data.columns if col != data_type]
    data = data.set_index(id_cols) # Putting these in the index keeps them from being spread
    data = data.unstack(level=0, fill_value=0)
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
