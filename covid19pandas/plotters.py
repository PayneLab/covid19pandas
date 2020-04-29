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
import seaborn as sns
import matplotlib.pyplot as plt

from .download import download_github_file
from .exceptions import ParameterError


# Line plot with possible multiple lines
def line_plot(data, x_col, y_col, group_col, x_lab=None, y_lab=None, title=None, y_logscale=False, dimensions=(15, 8), seaborn_style="darkgrid"):

    # Set plot colors and dimensions
    sns.set_style(seaborn_style)
    plt.figure(figsize=dimensions)

    # Create the plot
    ax = sns.lineplot(x=x_col, 
                y=y_col, 
                data=data,
                hue=group_col)

    # Auto-generate labels if not provided
    if x_lab is None:
        x_lab = x_col

    if y_lab is None:
        y_lab = y_col

    if y_logscale:
        y_lab = y_lab + " (log scale)"

    if title is None:
        title = f"{x_lab} vs. {y_lab}"

    # Set labels
    ax.set(title=title,
          xlabel=x_lab,
          ylabel=y_lab)
        
    # Set y log scale if desired
    if y_logscale:
        ax.set(yscale="log")

    plt.show()

# Line plot with two lines and scaled axes
