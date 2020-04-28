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
import numpy as np
import os
import warnings
import datetime
from .exceptions import ParameterError

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
