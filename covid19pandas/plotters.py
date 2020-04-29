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
def line_plot(data, x_col, y_col, group_col, x_lab=None, y_lab=None, title=None, legend_title=None, y_logscale=False, dimensions=(15, 8), seaborn_style="darkgrid"):

    # Set plot colors and dimensions
    sns.set_style(seaborn_style)
    fig, ax = plt.subplots(figsize=dimensions)

    # Create the plot
    sns.lineplot(x=x_col,
                y=y_col,
                data=data,
                hue=group_col,
                ax=ax)

    # Set y log scale if desired
    if y_logscale:
        ax.set(yscale="log") 

    # If they specified an x axis label, set it. Otherwise will just default to x_col name.
    if x_lab is not None:
        ax.set(xlabel=x_lab)

    # If they wanted the y axis on a log scale, we append that to the y axis label
    if y_logscale:
        if y_lab is None:
            y_lab = y_col 
        y_lab = y_lab + " (log scale)"

    # Set the y axis label if they provided one and/or we added " (log scale)" to the end of it. Otherwise, just defaults to y_col name.
    if y_lab is not None:
        ax.set(ylabel=y_lab)

    # Generate a default title if none is supplied
    if title is None:
        title = f"{x_col} vs {y_col}"

    # If they wanted the y axis on a log scale, note that on the plot title
    if y_logscale:
        title = title + " (y axis log scale)"

    # Set the title
    ax.set(title=title)

    # If they didn't provide a legend title, default to the group_col name.
    if legend_title is None:
        legend_title = group_col

    # Set the legend title.
    # Because Seaborn doesn't use the actual title property for its default legend title 
    # and instead just hacks one of the legend labels, we have to re-create the legend, 
    # excluding that first label, in order to set the title font size.

    handles, labels = ax.get_legend_handles_labels() # Get the handles and labels from the default legend Seaborn creates
    ax.legend(handles=handles[1:], labels=labels[1:], title="yatahey", title_fontsize="15") # Re-make the legend, excluding the first label and its handle that they hacked for the legend title

    return fig, ax

# Line plot with two lines and scaled axes
def line_plot_two_y(data, x_col, y1_col, y2_col, x_lab=None, y1_lab=None, y2_lab=None, title=None, legend_title=None, y_logscale=False, dimensions=(15, 8), seaborn_style="darkgrid", legend_loc="best"):

    # Set plot colors and dimensions
    sns.set_style(seaborn_style)
    fig, ax1 = plt.subplots(figsize=dimensions)

    # Create the plot
    sns.lineplot(x=x_col,
                y=y1_col,
                data=data,
                ax=ax1,
                color="b")
    
    ax2 = ax1.twinx()
    sns.lineplot(x=x_col,
                y=y2_col,
                data=data,
                ax=ax2,
                color="g")
    
    # Set y log scale if desired
    if y_logscale:
        ax1.set(yscale="log") 
        ax2.set(yscale="log") 

    # If they specified an x axis label, set it. Otherwise will just default to x_col name.
    if x_lab is not None:
        ax2.set(xlabel=x_lab)

    # If they wanted the y axis on a log scale, we append that to the y axis label
    if y_logscale:

        # Generate y labels if not provided
        if y1_lab is None:
            y1_lab = y1_col 
        if y2_lab is None:
            y2_lab = y2_col 

        y1_lab = y1_lab + " (log scale)"
        y2_lab = y2_lab + " (log scale)"

    # Set the y axes labels if they provided them and/or we added " (log scale)" to the ends. Otherwise, just defaults to y1_col/y2_col name.
    if y1_lab is not None:
        ax1.set(ylabel=y1_lab)
    if y2_lab is not None:
        ax2.set(ylabel=y2_lab)

    # Generate a default title if none is supplied
    if title is None:
        title = f"{x_col} vs {y1_col} and {y2_col}"

    # If they wanted the y axis on a log scale, note that on the plot title
    if y_logscale:
        title = title + " (y axes log scale)"

    # Set the title
    ax2.set(title=title)

    # Create the legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [y1_col, y2_col]
    leg = ax2.legend(lines, labels, loc=legend_loc)

    # If they specified a legend title, set it. Otherwise it won't have one.
    if legend_title is not None:
        ax2.legend().set_title(legend_title)
    
    return fig, (ax1, ax2)
