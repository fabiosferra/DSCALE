import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import matplotlib
import math
from matplotlib import colors
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import re
from downscaler.utils_pandas import fun_xs
from downscaler.utils import SliceableDict
from typing import Callable, Dict, List, Optional, Union, List, Dict, Any, Tuple
import plotly.express as px

def generate_shades(base_color, num_shades):
    # Convert the base color from RGB to HLS
    base_hls = colorsys.rgb_to_hls(*base_color)

    shades = []
    for i in range(num_shades):
        # Calculate the lightness value
        lightness = i / (num_shades - 1)
        # Create a new color with the same hue and saturation but varying lightness
        new_color_hls = (base_hls[0], lightness, base_hls[2])
        # Convert the new color back to RGB
        new_color_rgb = colorsys.hls_to_rgb(*new_color_hls)
        shades.append(new_color_rgb)

#     return shades
    return [matplotlib.colors.to_hex(v) for  v in shades]

def plot_palette(colors):
    fig, ax = plt.subplots(1, len(colors), figsize=(15, 2),
                           subplot_kw=dict(xticks=[], yticks=[], frame_on=False))

    for sp, color in zip(ax, colors):
        sp.set_facecolor(color)

    plt.show()

def hex_to_rgb(hex_colors):
    return [to_rgb(hex_color) for hex_color in hex_colors]

def create_custom_colormap(hex_colors, name='custom_palette'):
    rgb_colors = hex_to_rgb(hex_colors)
    return LinearSegmentedColormap.from_list(name, rgb_colors, N=len(rgb_colors))

def plot_palette(colors):
    fig, ax = plt.subplots(1, len(colors), figsize=(15, 2),
                           subplot_kw=dict(xticks=[], yticks=[], frame_on=False))

    for sp, color in zip(ax, colors):
        sp.set_facecolor(color)

    plt.show()

def create_my_own_palette(base_color = (1, 0, 0),num_shades = 10):# Base color in RGB (e.g., red)
    
    if isinstance(base_color, str):
        base_color=colors.to_rgb('blue')
    # Generate the palette
    palette = generate_shades(base_color, num_shades)

    # List of hex colors
    hex_colors = palette

    # Create the custom colormap
    return create_custom_colormap(hex_colors)

def hex_to_rgb2(hex_color):
    """
    Convert a hex color string to an RGB tuple.

    Parameters:
    hex_color (str): A hex color string (e.g., '#RRGGBB' or '#RGB').

    Returns:
    tuple: An (R, G, B) tuple with values ranging from 0 to 1.
    """
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip('#')
    
    # Expand shorthand hex notation to full form (e.g., '03F' to '0033FF')
    if len(hex_color) == 3:
        hex_color = ''.join([char * 2 for char in hex_color])
    
    # Convert hex to RGB
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def create_my_own_palette(base_color = (1, 0, 0),num_shades = 10):# Base color in RGB (e.g., red)
    
    if isinstance(base_color, str):
        if '#' in base_color:
            base_color=hex_to_rgb2(base_color)
        else:
            base_color=colors.to_rgb(base_color)
    # Generate the palette
    palette = generate_shades(base_color, num_shades)

    # List of hex colors
    hex_colors = palette

    # Create the custom colormap
    return create_custom_colormap(hex_colors)


def fun_phase_out_date_colormap(data, error_bars=True, columns_that_are_not_models= ['median','index','GDPCAP'], colored_error_bars:bool=False):
    data=data.copy(deep=True)
    # Reset the index to have 'Country' as a column
    data.reset_index(inplace=True)
    data.rename(columns={'index': 'Country'}, inplace=True)

    # Normalize the 'GDPCAP' column to use it as color intensity
    norm = plt.Normalize(data['GDPCAP'].min(), data['GDPCAP'].max())
    gdp_values = data['GDPCAP'].values
    
    # Create the colormap
    cmap = sns.color_palette("mako", as_cmap=True)
    colors = cmap(norm(gdp_values))

    # Generate y positions for countries
    y_positions = range(len(data))

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    points = plt.scatter(data['median'], y_positions, c=data['GDPCAP'], s=300, cmap="mako", edgecolor='black')


    if error_bars:
        # Add columns for the minimum and maximum phase-out dates across models
        models=[x for x in data.columns if x not in columns_that_are_not_models]
        data['min'] = data[models].min(axis=1)
        data['max'] = data[models].max(axis=1)
        
        # Add uncertainty bars (error bars)
        for i, (median, y_pos) in enumerate(zip(data['median'], y_positions)):
            if colored_error_bars:
    	        # use same colors as GDP for errorbars
                plt.errorbar(x=median, y=y_pos, xerr=[[median - data['min'][i]], [data['max'][i] - median]], 
                         fmt='o', color=colors[i], ecolor=colors[i], capsize=3, elinewidth=3 
                         )
            else:
                # Use  'black' with alpha=0.35 for error bars
                plt.errorbar(x=median, y=y_pos, xerr=[[median - data['min'][i]], [data['max'][i] - median]], 
                        fmt='',  color='black', alpha=0.35,capsize=5, 
                         )
            

    # Set y-ticks to country names
    plt.yticks(y_positions, data['Country'])

    # Add a color bar
    plt.colorbar(points, label='GDPCAP')
    return plt.plot()



# def fun_phase_out_date_colormap2(data, error_bars=True):
#     data=data.copy(deep=True)
#     # Example dataframe (you can replace this with your actual dataframe)

#     # Add columns for the minimum and maximum phase-out dates across models
#     data['min'] = data[['GCAM 6.0 NGFS', 'MESSAGEix-GLOBIOM 1.1-M-R12', 'REMIND-MAgPIE 3.2-4.6']].min(axis=1)
#     data['max'] = data[['GCAM 6.0 NGFS', 'MESSAGEix-GLOBIOM 1.1-M-R12', 'REMIND-MAgPIE 3.2-4.6']].max(axis=1)

#     # Reset the index to have 'Country' as a column
#     data.reset_index(inplace=True)
#     data.rename(columns={'index': 'Country'}, inplace=True)

#     # Normalize the 'GDPCAP' column to use it as color intensity
#     gdp_values = data['GDPCAP'].values
#     norm = plt.Normalize(gdp_values.min(), gdp_values.max())

#     # Generate y positions for countries
#     y_positions = np.arange(len(data))

#     # Create the colormap
#     cmap = sns.color_palette("mako", as_cmap=True)

#     # Create the scatter plot
#     plt.figure(figsize=(10, 6))
#     colors = cmap(norm(gdp_values))
#     points = plt.scatter(data['median'], y_positions, c=gdp_values, s=300, cmap=cmap, edgecolor='black')

#     # Add uncertainty bars (error bars) with the same colors
#     for i, (median, y_pos) in enumerate(zip(data['median'], y_positions)):
#         plt.errorbar(x=median, y=y_pos, xerr=[[median - data['min'][i]], [data['max'][i] - median]], 
#                     fmt='o', color=colors[i], ecolor=colors[i], capsize=5, elinewidth=10)

#     # Set y-ticks to country names
#     plt.yticks(y_positions, data['Country'])

#     # Add a color bar
#     cbar = plt.colorbar(points, label='GDPCAP')
#     cbar.set_label('GDPCAP')

#     # Set labels and title
#     plt.xlabel('Median across models')
#     plt.ylabel('Country')
#     plt.title('Phase out dates across countries')


#     return plt.plot()

def create_multi_panel_figure(df_iam:pd.DataFrame, max_cols:int=3, sup_title:Optional[str]=None, y_label:Optional[str]=None, x_label:Optional[str]=None, by:str='VARIABLE', title_legend:str='MODEL', top_adj:float=0.85)->plt.plot:
    """
    Create a multi-panel figure to visualize all the variables present in the dataframe `df_iam`.
    We plot each panel `by` a your selected level (by default 'VARIABLE' -> one panel for each variable)

    Parameters:
    -----------
    df_iam : pandas.DataFrame
        DataFrame containing the data to plot, indexed appropriately.
    max_cols : int, optional
        Maximum number of columns in the figure (default is 3).
    sup_title : str, optional
        Super title for the entire figure (default is None).
    y_label : str, optional
        Label for the y-axis (default is None).
    x_label : str, optional
        Label for the x-axis (default is None).
    by : str, optional
        The level in the DataFrame to group by for each subplot (default is 'VARIABLE').
    title_legend : str, optional
        Title for the legend (default is 'MODEL').
    top_adj : float, optional
        Top adjustment for the layout of the figure (default is 0.85).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object containing the multi-panel plot.
    """
    level=by
    sel_vars=list(df_iam.reset_index()[level].unique())
    
    # Check if we have a resonable amout of data to plot
    sanity_check_panel_plot(df_iam, sel_vars, level=level)

    # Calculate the number of rows and columns based on the number of variables
    n_vars = len(sel_vars)
    n_cols = min(n_vars, max_cols)  # Up to `max_cols` columns (by default 3)
    n_rows = math.ceil(n_vars / n_cols)

    # Create the subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharey=True)

    # Flatten axes array for easier iteration
    axes = axes.flatten()
    if sup_title:
        plt.suptitle(sup_title)
    for i, var in enumerate(sel_vars):
        res = df_iam.xs(var, level=level)
        res.T.plot(ax=axes[i])
        axes[i].set_title(var)
        if x_label:
            axes[i].set_xlabel(x_label)
        if y_label:
            axes[i].set_ylabel(y_label if i % n_cols == 0 else "")
        axes[i].legend(title=title_legend)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.subplots_adjust(top=top_adj)
    return fig

def sanity_check_panel_plot(df_iam:pd.DataFrame, sel_vars:list, level:str):
    """
    Perform a sanity check before creating a multi-panel plot.

    Parameters:
    -----------
    df_iam : pandas.DataFrame
        DataFrame containing the data to plot.
    sel_vars : list
        List of selected variables to plot.
    level : str
        The level in the DataFrame to group by for each subplot.

    Raises:
    -------
    ValueError
        If the user chooses not to proceed with the plotting based on the sanity check prompts.
    """

    txt0='Do you wish to proceed? y/n'
    txt1=f'You are going to create a figure with {len(sel_vars)} panels (one variable for each panel). Here is a sample of the first 20 variables found = {sel_vars[:20]}... '
    if len(sel_vars)>12:
        action = input(f"{txt1} {txt0}")
        if action.lower() not in ["yes", "y"]:
            raise ValueError(f"Simulation aborted by the user (user input={action})")
    lines_by_panel=len(df_iam)//len(sel_vars)
    if lines_by_panel>50:
        lines_sample=list(df_iam.xs(sel_vars[0], level=level,drop_level=False).index)[:10]
        txt2=f'Each panel of your figure will contain approximately ~{lines_by_panel} lines.'
        txt3=f'Here is a sample of what you want to plot in each panel (first 10 lines) {lines_sample}.'
        action = input(f"{txt2} {txt3} {txt0}")
        if action.lower() not in ["yes", "y"]:
            raise ValueError(f"Simulation aborted by the user (user input={action})")
        

def create_parallel_coordinates(df: pd.DataFrame, selcol: str, t: int, default_d=None, palette=px.colors.diverging.Tealrose, countrylist:list=None) -> px:
    """
    Creates a parallel coordinates plot based on selected columns and time from a dataframe.
    
    Parameters:
    df : pd.DataFrame
        The dataframe containing the data to plot.
    selcol : str
        The column to be used for color scaling in the plot.
    t : int
        The year to filter data on for plotting.
    sliceable_dict : Optional[dict]
        A dictionary that can control slicing operations, allowing for customization, by default None.
    countrylist : Optional[List[str]]
        A list of country names to include in the plot (will follow exact order), by default None.
        
    Returns: px.Figure
        The plotly graph object figure for the parallel coordinates plot.
    """
    # Determine slicing based on selected column
    if default_d is None:
        default_d={'METHOD':'wo_smooth_enlong_ENLONG_RATIO', 'CONVERGENCE':'2150'} #,'FUNC':'log-log'}
    
    default_d=SliceableDict(default_d)
    if selcol == 'FUNC':
        d_slice = default_d.slice('METHOD', 'CONVERGENCE')
    elif selcol == 'CONVERGENCE':
        d_slice = default_d.slice('METHOD')
    else:
        d_slice = None


    # Select and reshape data
    default_d=SliceableDict({'METHOD':'wo_smooth_enlong_ENLONG_RATIO', 'CONVERGENCE':'2150'})
    if selcol == 'FUNC':
        d_slice = default_d.slice('METHOD', 'CONVERGENCE')
    elif selcol == 'CONVERGENCE':
        d_slice = default_d.slice('METHOD')
    else:
        d_slice = None

    # Select data using (d_slice) and reshape
    if d_slice is None:
        mydata=df[t].unstack('REGION')
    else:
        mydata=fun_xs(df, d_slice)[t].unstack('REGION')
    mydata=mydata.reset_index()

    # Sort the data by the functional form
    mydata=mydata.sort_values('FUNC', ascending=True)
    # Print the data (index associated to parameters)
    print(mydata.reset_index().set_index('index')[['FUNC','CONVERGENCE','METHOD']])
    mydata=mydata.drop([x for x in df.index.names if x not in [selcol,'REGION']], axis=1)

    # Convert the palette to a list of RGB colors in the format 'rgb(r, g, b)' if the palette is a ListedColormap object
    palette=convert_colormap_to_rgb(palette)
    try:
        palette=colors_to_rgba(palette)
    except:
        pass
    
    # Plot the countries in selected order
    if countrylist is not None:
        mycols=[x for x in mydata.columns if x not in countrylist]+countrylist
        mydata=mydata[mycols]
   
    # Plot the data Line
    fig = px.parallel_coordinates(mydata.reset_index().drop([selcol], axis=1), color='index', labels={'index': f"Param. Index"},
                                color_continuous_scale=palette,
                                color_continuous_midpoint=mydata.reset_index()['index'].mean(),
                                title=f'Indexed final energy results in {t} (2010=1)')
    # Update layout to set font size and family
    fig.update_layout(
      title_font={'size': 30, 'family': 'Times'},  # Updating title font
        font=dict(
         family="Times",  # Overall font family
        size=24,  # Overall font size for labels and tick labels
        # color="RebeccaPurple"  # You can specify the font color here if needed
         )
     )
    
    fig.show()
    return fig


def convert_colormap_to_rgb(palette: Union[matplotlib.colors.ListedColormap, List[str]]) -> Union[List[str], List[str]]:
    """
    Convert a matplotlib ListedColormap object to a list of RGB color strings formatted as 'rgb(r, g, b)'.
    
    Parameters:
    -----------
    palette : Union[matplotlib.colors.ListedColormap, List[str]]
        A `ListedColormap` object from matplotlib or a list of color strings. 
        If it's already a list of color strings, the function will return it unchanged.

    Returns:
    --------
    Union[List[str], List[str]]:
        A list of colors in the format 'rgb(r, g, b)' if a ListedColormap is provided, 
        or the original list if the palette is already a list of color strings.
    """
    
    if isinstance(palette, matplotlib.colors.ListedColormap):
        # Extract RGB colors from the ListedColormap object
        rgb_colors = palette.colors
        
        # Convert to the 'rgb(r, g, b)' format (scaled to 0-255 range)
        rgb_list = ['rgb({:.0f}, {:.0f}, {:.0f})'.format(r * 255, g * 255, b * 255) for r, g, b in rgb_colors]
        return rgb_list
    else:
        print('The palette is not a ListedColormap object. We return the original list of colors.')
        return palette


def rgb_to_hex(rgb_colors: Union[List[Tuple[int, int, int]], List[str]]) -> List[str]:
    """
    Converts a list of RGB color tuples or 'rgb(r, g, b)' formatted strings to their corresponding hex color codes.
    
    Parameters:
    -----------
    rgb_colors : Union[List[Tuple[int, int, int]], List[str]]
        A list of either RGB color tuples (r, g, b) or strings in the format 'rgb(r, g, b)',
        where each tuple or string contains values in the range 0-255.
        
    Returns:
    --------
    List[str]
        A list of hex color codes in the format '#RRGGBB'.
    """
    hex_colors = []
    
    for color in rgb_colors:
        if isinstance(color, tuple) and len(color) == 3:
            # Convert tuple (r, g, b) to hex
            r, g, b = color
            hex_colors.append('#{:02x}{:02x}{:02x}'.format(r, g, b))
        elif isinstance(color, str):
            # Extract the numbers from 'rgb(r, g, b)' formatted string
            match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if match:
                r, g, b = map(int, match.groups())
                hex_colors.append('#{:02x}{:02x}{:02x}'.format(r, g, b))
            else:
                print(f"Invalid 'rgb(r, g, b)' format: {color}")
        else:
            raise ValueError(f"Unsupported color format: {color}")

    return hex_colors



def colors_to_rgba(color_source: Union[List[str], mcolors.Colormap], alpha: float = 1.0, num_colors: int = 256) -> List[str]:
    """
    Convert a list of color names or a colormap object into rgba strings with specified alpha transparency.

    Args:
        color_source (Union[List[str], mcolors.Colormap]): Either a list of color names (e.g., ['blue', 'red', 'black'])
                                                           or a LinearSegmentedColormap to sample colors from.
        alpha (float): Alpha value for transparency (between 0 and 1). Default is 1.0 (no transparency).
        num_colors (int): Number of colors to sample from the colormap if color_source is a colormap. Default is 256.

    Returns:
        List[str]: List of colors in 'rgba(r,g,b,a)' format.
    """
    rgba_colors = []

    if isinstance(color_source, list):
        # If color_source is a list of color names
        for color in color_source:
            rgba = mcolors.to_rgba(color, alpha=alpha)
            rgba_str = f'rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{rgba[3]})'
            rgba_colors.append(rgba_str)

    elif isinstance(color_source, mcolors.Colormap):
        # If color_source is a LinearSegmentedColormap, sample num_colors evenly from it
        for i in np.linspace(0, 1, num_colors):
            rgba = color_source(i)  # Get RGBA values from colormap
            rgba_str = f'rgba({int(rgba[0] * 255)},{int(rgba[1] * 255)},{int(rgba[2] * 255)},{alpha})'
            rgba_colors.append(rgba_str)

    return rgba_colors


def create_color_palette(colors: List[str], steps: int = 256, show_palette: bool = False) -> mcolors.LinearSegmentedColormap:
    """
    Creates a LinearSegmentedColormap from the given list of colors and optionally plots the gradient.

    Args:
        colors (List[str]): A list of hex color codes or valid matplotlib color names.
                            At least two colors are required to create a gradient.
        steps (int): The number of steps in the gradient. Default is 256.
        show_palette (bool): If True, displays the gradient palette as a plot. Default is False.
    
    Returns:
        LinearSegmentedColormap: The generated colormap that can be used for plotting gradients.
    """
    
    # Create a LinearSegmentedColormap between the given colors
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    if show_palette:
        # Generate a gradient color palette using the colormap with the given number of steps
        gradient = np.linspace(0, 1, steps).reshape(1, -1)

        # Plot the gradient
        plt.imshow(gradient, aspect='auto', cmap=cmap)
        plt.axis('off')  # Turn off the axis
        plt.show()

    return cmap


import re
from typing import List

def rgba_to_rgb(rgba_colors: List[str]) -> List[str]:
    """
    Convert a list of rgba colors into rgb format by removing the alpha channel.

    Args:
        rgba_colors (List[str]): A list of colors in 'rgba(r,g,b,a)' format.

    Returns:
        List[str]: A list of colors in 'rgb(r,g,b)' format.
    """
    rgb_colors = []
    
    # Regular expression pattern to match 'rgba(r,g,b,a)'
    pattern = r'rgba\((\d+),(\d+),(\d+),[\d\.]+\)'
    
    for rgba in rgba_colors:
        # Extract the r, g, b values using regex
        match = re.match(pattern, rgba)
        if match:
            r, g, b = match.groups()  # Get r, g, b values as strings
            rgb_str = f'rgb({r},{g},{b})'  # Create the rgb string
            rgb_colors.append(rgb_str)
    
    return rgb_colors
