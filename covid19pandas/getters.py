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
from .download import download_github_file
from .exceptions import NoInternetError, FileDoesNotExistError, FileNotUpdatedWarning

def get_cases():
    return _get_table("time_series_covid19_confirmed_global.csv")

def get_deaths():
    return _get_table("time_series_covid19_deaths_global.csv")

def get_recovered():
    return _get_table("time_series_covid19_recovered_global.csv")

# Helper functions

def _get_table(file_name):

    # Construct the url and path for the file
    csse_time_series_url = "https://api.github.com/repos/CSSEGISandData/COVID-19/contents/csse_covid_19_data/csse_covid_19_time_series/"
    path_here = os.path.abspath(os.path.dirname(__file__))
    data_files_path = os.path.join(path_here, "data")

    url = csse_time_series_url + file_name
    path = os.path.join(data_files_path, file_name)

    # Download the latest version of the file
    try:
        download_github_file(url, path)
    except NoInternetError:
        warnings.warn("Insufficient internet to update data files. Data from most recent download will be used.", FileNotUpdatedWarning, stacklevel=3)

    if not os.path.isfile(path):
        raise FileDoesNotExistError("Data file has not been downloaded previously, and current internet connection is not sufficient to download it. Try again when you have a better internet connection.")

    df = pd.read_csv(path)

    return df
