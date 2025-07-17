import ipywidgets as widgets
import numpy as np
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import Dropdown, HBox, IntSlider, VBox, interact
from plotly.subplots import make_subplots


def rotated_data_year(gsfc, year):
    """
    Get the residual data for a specific year.

    Parameters
    ----------
    year : int
        The year for which to retrieve the residual data.

    Returns
    -------
    xarray.DataArray
        The residual data for the specified year.
    """
    if "year" not in gsfc.ds.dims:
        raise ValueError("The dataset does not contain a 'year' dimension.")

    data = gsfc.ds.sel(time=year).residual
    return np.rot90(data, k=0)


def create_interactive_residual_plot(residuals, year, basin_id=None):
    """
    Create an interactive Plotly heatmap for residual data

    Parameters:
    -----------
    year : int, year to plot
    basin_id : int or None, basin ID to plot (None for all data)
    """
    try:
        x_coords = residuals.ds.x.values
        y_coords = residuals.ds.y.values

        # Get data for the specified basin/year
        data = residuals.get_basin_data(year, basin_id)

        # Create the heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=data,
                x=x_coords,
                y=y_coords,
                colorscale="RdBu",  # Red-Blue colorscale, good for residuals
                zmid=0,  # Center colorscale at 0
                colorbar=dict(
                    title="Ice Mask Residual",
                    # titleside="right"
                ),
                hoverongaps=False,
                hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Residual: %{z:.4f}<extra></extra>",
            )
        )

        # Set title based on basin selection
        if basin_id is not None:
            basin_name = residuals.ds.basin_names.values[basin_id]
            title = (
                f"Residual Ice Mask for {year} - Basin {basin_name} (ID: {basin_id})"
            )
        else:
            title = f"Residual Ice Mask for {year} - All Basins"

        fig.update_layout(
            title=title,
            xaxis_title="X Coordinate (m)",
            yaxis_title="Y Coordinate (m)",
            width=800,
            height=600,
            font=dict(size=12),
        )

        # Make sure aspect ratio is preserved
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    except Exception as e:
        print(f"Error creating plot for year {year}: {e}")
        print(f"Available years in dataset: {residuals.ds.time.values}")
        return None


def create_basin_statistics_plot(basin_stats, year):
    """
    Create an interactive bar plot for basin statistics
    """
    if year not in basin_stats:
        print(f"No data for year {year}")
        return None

    # Prepare data for plotting
    basins = []
    means = []
    stds = []
    counts = []
    rms_values = []

    for basin_name, stats in basin_stats[year].items():
        if stats["count"] > 0:
            basins.append(basin_name)
            means.append(stats["mean"])
            stds.append(stats["std"])
            counts.append(stats["count"])
            rms_values.append(stats["rms"])

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Mean Residual", "Standard Deviation", "Data Count", "RMS"),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Add traces
    fig.add_trace(
        go.Bar(x=basins, y=means, name="Mean", marker_color="lightblue"), row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=basins, y=stds, name="Std Dev", marker_color="lightcoral"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(x=basins, y=counts, name="Count", marker_color="lightgreen"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=basins, y=rms_values, name="RMS", marker_color="lightyellow"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Basin Statistics for Year {year}",
        showlegend=False,
        height=600,
        width=900,
    )

    return fig


def interactive_plot(basin_stats, year, basin_id, plot_type):
    """Enhanced interactive plotting function with multiple plot types"""

    if plot_type == "residual":
        basin_id = None if basin_id == -1 else int(basin_id)
        fig = create_interactive_residual_plot(year, basin_id)
        if fig:
            fig.show()

    elif plot_type == "stats":
        fig = create_basin_statistics_plot(basin_stats, year)
        if fig:
            fig.show()


# Create the interactive widget


# Also create a combined view function
def create_combined_dashboard(basin_stats, year):
    """Create a combined dashboard with both residual map and statistics"""

    # Create residual map
    residual_fig = create_interactive_residual_plot(year, basin_id=None)

    # Create statistics plot
    stats_fig = create_basin_statistics_plot(basin_stats, year)

    if residual_fig:
        residual_fig.show()

    if stats_fig:
        stats_fig.show()
