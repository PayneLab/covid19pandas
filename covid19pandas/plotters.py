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

def plot_lines(data, x_col, y_col, group_col, x_lab=None, y_lab=None, title=None, legend_title=None, legend_order=None, y_logscale=False, dimensions=(11, 8.5), seaborn_style="darkgrid"):
    """Plot the values in x_col versus the values in y_col, divided into different lines based on the values in group_col.

    Parameters:
    data (pandas.DataFrame): A dataframe with the data to plot. Must be in "long" format.
    x_col (str): The name of the column with the x values in it.
    y_col (str): The name of the column with the y values in it.
    group_col (str): The name of the column with the grouping values in it.
    x_lab (str, optional): Label for the x axis. Default None will use the x_col name.
    y_lab (str, optional): Label for the y axis. Default None will use the y_col name.
    title (str, optional): Title for the plot. Default None will cause one to be automatically generated based on column names.
    legend_title (str, optional): Title for the legend. Default None will use the group_col string.
    legend_order (list of str, optional): A list of the unique values in the group_col column, specifying the order in which you want them to appear on the plot legend. Default None uses order of appearance in the group_col column.
    y_logscale (bool, optional): Whether to use a log scale for the y axis. If True, will automatically note it on the y axis label and plot title. Default False.
    dimensions (2-tuple of int or float, optional): Tuple to be passed to the figsize parameter of the matplotlib.pyplot.subplots function. Default (11, 8.5).
    seaborn_style (string, optional): String to pass to the seaborn.set_style function. Must be "darkgrid", "whitegrid", "dark", "white", or "ticks". Default "darkgrid".

    Returns:
    matplotlib.figure.Figure: The figure object created for the plot.
    matplotlib.axes._subplots.AxesSubplot: The single axes object on the figure.
    """

    # Set plot colors and dimensions
    sns.set_style(seaborn_style)
    fig, ax = plt.subplots(figsize=dimensions)

    # Create the plot
    sns.lineplot(
        x=x_col,
        y=y_col,
        data=data,
        hue=group_col,
        hue_order=legend_order,
        ax=ax)

    # Set y log scale if desired
    if y_logscale:
        ax.set(yscale="log") 

    # If they specified an x axis label, set it. Otherwise it will automatically default to x_col name.
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

    # If we put the y axis on a log scale, note that on the plot title
    if y_logscale:
        title = title + " (y axis log scale)"

    # Set the title
    ax.set(title=title)

    # If they didn't provide a legend title, default to the group_col name.
    if legend_title is None:
        legend_title = group_col

    # Set the legend title.
    # Because Seaborn doesn't use the actual title property for its default legend title and instead just hacks one of the legend 
    # labels, we have to re-create the legend, excluding that first label, in order to set the title font size.

    handles, labels = ax.get_legend_handles_labels() # Get the handles and labels from the default legend Seaborn creates
    ax.legend(handles=handles[1:], labels=labels[1:], title=legend_title, title_fontsize="14") # Re-make the legend, excluding the first label and its handle that they hacked for the legend title

    return fig, ax

def plot_lines_two_y(data, x_col, y1_col, y2_col, x_lab=None, y1_lab=None, y2_lab=None, title=None, legend_title=None, legend_loc="best", y_logscale=False, dimensions=(11, 8.5), seaborn_style="darkgrid"):
    """Plot the values in x_col versus the values in y1_col on the left y axis, and the values in y2_col on the right y axis.

    Parameters:
    data (pandas.DataFrame): A dataframe with the data to plot. Must be in "long" format.
    x_col (str): The name of the column with the x values in it.
    y1_col (str): The name of the column with the y values to plot on the left y axis.
    y2_col (str): The name of the column with the y values to plot on the right y axis.
    x_lab (str, optional): Label for the x axis. Default None will use the x_col name.
    y1_lab (str, optional): Label for the left y axis. Default None will use the y1_col name.
    y2_lab (str, optional): Label for the right y axis. Default None will use the y2_col name.
    title (str, optional): Title for the plot. Default None will cause one to be automatically generated based on column names.
    legend_title (str, optional): Title for the legend. Default None will have no title on the legend.
    legend_loc (str or int or 2-tuple of floats, optional): Legend position specifier, passed directly to the matplotlib.axes.Axes.legend function. See loc parameter details at <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend> for options. Examples are "best", "center right", "upper left", "lower center". Default "best" optimizes position based on position of line on right y axis.
    y_logscale (bool, optional): Whether to use a log scale for the y axis. If True, will automatically note it on the y axis label and plot title. Default False.
    dimensions (2-tuple of int or float, optional): Tuple to be passed to the figsize parameter of the matplotlib.pyplot.subplots function. Default (11, 8.5).
    seaborn_style (string, optional): String to pass to the seaborn.set_style function. Must be "darkgrid", "whitegrid", "dark", "white", or "ticks". Default "darkgrid".

    Returns:
    matplotlib.figure.Figure: The figure object created for the plot.
    2-tuple of matplotlib.axes._subplots.AxesSubplot: The two axes objects on the figure. First corresponds to the left y axis, and second corresponds to the right y axis.
    """

    # Set plot colors and dimensions
    sns.set_style(seaborn_style)
    fig, ax1 = plt.subplots(figsize=dimensions)

    # Create the plot
    sns.lineplot(
        x=x_col,
        y=y1_col,
        data=data,
        ax=ax1,
        color="b")
    
    ax2 = ax1.twinx()
    sns.lineplot(
        x=x_col,
        y=y2_col,
        data=data,
        ax=ax2,
        color="g")
    
    # Set y log scale if desired
    if y_logscale:
        ax1.set(yscale="log") 
        ax2.set(yscale="log") 

    # Generate labels if not provided
    if x_lab is None:
        x_lab = x_col
    if y1_lab is None:
        y1_lab = y1_col 
    if y2_lab is None:
        y2_lab = y2_col 
    if title is None:
        title = f"{x_lab} vs {y1_lab} and {y2_lab}"

    # If they wanted the y axis on a log scale, we append that to the y axis labels and the title
    if y_logscale:
        y1_lab = y1_lab + " (log scale)"
        y2_lab = y2_lab + " (log scale)"
        title = title + " (y axes log scale)"

    # Set the labels
    ax1.set(xlabel=x_lab)
    ax1.set(ylabel=y1_lab)
    ax2.set(ylabel=y2_lab)
    ax2.set(title=title)

    # Create the legend
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [y1_lab, y2_lab]
    leg = ax2.legend(lines, labels, loc=legend_loc, title_fontsize="14")

    # If they specified a legend title, set it. Otherwise it won't have one.
    if legend_title is not None:
        leg.set_title(legend_title)

    return fig, (ax1, ax2)
