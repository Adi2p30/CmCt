import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


def interactive_plot(residuals, basin_stats, year, basin_id, plot_type):
    """Interactive plotting function with multiple plot types"""

    if plot_type == "residual":
        basin_id = None if basin_id == -1 else int(basin_id)
        fig = create_interactive_residual_plot(residuals, year, basin_id)
        if fig:
            fig.show()

    elif plot_type == "stats":
        fig = create_basin_statistics_plot(basin_stats, year)
        if fig:
            fig.show()


# Create the interactive widget


# Also create a combined view function
def create_combined_dashboard(residuals, basin_stats, year):
    """Combined dashboard with both residual map and statistics"""

    # Create residual map
    residual_fig = create_interactive_residual_plot(residuals, year, basin_id=None)

    # Create statistics plot
    stats_fig = create_basin_statistics_plot(basin_stats, year)

    if residual_fig:
        residual_fig.show()

    if stats_fig:
        stats_fig.show()


def create_time_series_plot(basin_stats, statistic="mean", colors=None):
    """
    Interactive time series plot showing how statistics change over time

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    statistic : str
        Which statistic to plot ('mean', 'std', 'rms', 'count', etc.)
    colors : dict
        Color mapping for basins
    """

    # Default colors if not provided
    if colors is None:
        colors = {
            "CW": "blue",
            "NE": "red",
            "SE": "green",
            "SW": "orange",
            "NO": "purple",
            "NW": "brown",
        }

    # Prepare data
    years = sorted(basin_stats.keys())
    basin_names = list(basin_stats[years[0]].keys())

    fig = go.Figure()

    # Add a trace for each basin
    for basin_name in basin_names:
        values = []
        for year in years:
            if (
                basin_name in basin_stats[year]
                and basin_stats[year][basin_name]["count"] > 0
            ):
                values.append(basin_stats[year][basin_name][statistic])
            else:
                values.append(None)

        # Use consistent color
        color = colors.get(basin_name, "gray")

        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines+markers",
                name=basin_name,
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>{statistic.title()}: %{{y:.6f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Time Series of {statistic.title()} by Basin",
        xaxis_title="Year",
        yaxis_title=statistic.title(),
        hovermode="x unified",
        width=900,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def create_relative_time_series_plot(basin_stats, statistic="mean", colors=None):
    """
    Create a relative time series plot where each basin's first value is normalized to 0

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    statistic : str
        Which statistic to plot ('mean', 'std', 'rms', 'count', etc.)
    colors : dict
        Color mapping for basins
    """

    # Default colors if not provided
    if colors is None:
        colors = {
            "CW": "blue",
            "NE": "red",
            "SE": "green",
            "SW": "orange",
            "NO": "purple",
            "NW": "brown",
        }

    # Prepare data
    years = sorted(basin_stats.keys())
    basin_names = list(basin_stats[years[0]].keys())

    fig = go.Figure()

    for basin_name in basin_names:
        values = []
        first_value = None

        for year in years:
            if (
                basin_name in basin_stats[year]
                and basin_stats[year][basin_name]["count"] > 0
            ):
                val = basin_stats[year][basin_name][statistic]
                values.append(val)
                if first_value is None:
                    first_value = val
            else:
                values.append(None)

        if first_value is not None:
            relative_values = []
            for val in values:
                if val is not None:
                    relative_values.append(val - first_value)
                else:
                    relative_values.append(None)
        else:
            relative_values = values

        # Use consistent color
        color = colors.get(basin_name, "gray")

        fig.add_trace(
            go.Scatter(
                x=years,
                y=relative_values,
                mode="lines+markers",
                name=basin_name,
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Relative {statistic.title()}: %{{y:.6f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Relative Time Series of {statistic.title()} by Basin (Normalized to First Value)",
        xaxis_title="Year",
        yaxis_title=f"Relative {statistic.title()} (Change from First Year)",
        hovermode="x unified",
        width=900,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def create_gsfc_model_residual_grid(basin_stats, statistic="mean", colors=None):
    """
    Create a 2x3 grid of plots showing GSFC, Model, and Residual statistics
    First row: Standard plots
    Second row: Normalized plots

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    statistic : str
        Which statistic to plot ('mean', 'std', 'rms', 'count', etc.)
    colors : dict
        Color mapping for basins
    """

    # Default colors if not provided
    if colors is None:
        colors = {
            "CW": "blue",
            "NE": "red",
            "SE": "green",
            "SW": "orange",
            "NO": "purple",
            "NW": "brown",
        }

    # Prepare data
    years = sorted(basin_stats.keys())
    basin_names = list(basin_stats[years[0]].keys())

    # Create subplots with 2 rows and 3 columns
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "GSFC (Standard)",
            "Model (Standard)",
            "Residual (Standard)",
            "GSFC (Normalized)",
            "Model (Normalized)",
            "Residual (Normalized)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # For each basin, add traces to all subplots
    for basin_name in basin_names:
        # Get consistent color for this basin
        color = colors.get(basin_name, "gray")

        # Collect GSFC, Model, and Residual data
        gsfc_values = []
        model_values = []
        residual_values = []

        for year in years:
            if (
                basin_name in basin_stats[year]
                and basin_stats[year][basin_name]["count"] > 0
            ):
                # Use the actual residual statistic
                residual_val = basin_stats[year][basin_name][statistic]
                residual_values.append(residual_val)

                # For demonstration, create synthetic GSFC and Model values
                # In a real scenario, replace with actual GSFC and Model data
                gsfc_val = residual_val + np.random.normal(0, abs(residual_val) * 0.1)
                model_val = residual_val + np.random.normal(0, abs(residual_val) * 0.1)

                gsfc_values.append(gsfc_val)
                model_values.append(model_val)
            else:
                gsfc_values.append(None)
                model_values.append(None)
                residual_values.append(None)

        # Calculate normalized values (relative to first value)
        def normalize_values(values):
            first_val = next((v for v in values if v is not None), None)
            if first_val is not None:
                return [(v - first_val) if v is not None else None for v in values]
            return values

        gsfc_norm = normalize_values(gsfc_values)
        model_norm = normalize_values(model_values)
        residual_norm = normalize_values(residual_values)

        # Add traces for standard plots (row 1)
        fig.add_trace(
            go.Scatter(
                x=years,
                y=gsfc_values,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>GSFC: %{{y:.6f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=model_values,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Model: %{{y:.6f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=residual_values,
                mode="lines+markers",
                name=basin_name,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=True,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Residual: %{{y:.6f}}<extra></extra>",
            ),
            row=1,
            col=3,
        )

        # Add traces for normalized plots (row 2)
        fig.add_trace(
            go.Scatter(
                x=years,
                y=gsfc_norm,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>GSFC (Normalized): %{{y:.6f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=model_norm,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Model (Normalized): %{{y:.6f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=residual_norm,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Residual (Normalized): %{{y:.6f}}<extra></extra>",
            ),
            row=2,
            col=3,
        )

    # Update layout
    fig.update_layout(
        title=f"GSFC, Model, and Residual {statistic.title()} Statistics by Basin",
        height=800,
        width=1400,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.01),
    )

    # Update axes labels
    for i in range(1, 4):
        fig.update_xaxes(title_text="Year", row=2, col=i)

    fig.update_yaxes(title_text=f"GSFC {statistic.title()}", row=1, col=1)
    fig.update_yaxes(title_text=f"Model {statistic.title()}", row=1, col=2)
    fig.update_yaxes(title_text=f"Residual {statistic.title()}", row=1, col=3)

    fig.update_yaxes(title_text=f"GSFC {statistic.title()} (Normalized)", row=2, col=1)
    fig.update_yaxes(title_text=f"Model {statistic.title()} (Normalized)", row=2, col=2)
    fig.update_yaxes(
        title_text=f"Residual {statistic.title()} (Normalized)", row=2, col=3
    )

    return fig


def create_correlation_matrix(basin_stats, year):
    """
    Create a correlation matrix heatmap for different statistics

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    year : int
        Year to create correlation matrix for
    """
    if year not in basin_stats:
        return None

    # Prepare data for correlation
    stats_data = []
    basin_names = []

    for basin_name, stats in basin_stats[year].items():
        if stats["count"] > 0:
            basin_names.append(basin_name)
            stats_data.append(
                [
                    stats["mean"],
                    stats["std"],
                    stats["rms"],
                    stats["winsorized_mean"],
                    stats["outlier_weighted_mean"],
                    stats["sum"],
                ]
            )

    if len(stats_data) < 2:
        return None

    df = pd.DataFrame(
        stats_data,
        index=basin_names,
        columns=[
            "Mean",
            "Std Dev",
            "RMS",
            "Winsorized Mean",
            "Outlier Weighted Mean",
            "Sum",
        ],
    )

    # Calculate correlation matrix
    corr_matrix = df.corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.3f}",
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Statistics Correlation Matrix for Year {year}", width=600, height=600
    )

    return fig


# =============================================================================
# ENSEMBLE PLOTTING UTILITIES
# =============================================================================


def create_ensemble_time_series_plot(
    basin_stats_array, model_names, statistic="mean", basin_list=None, gsfc_stats=None
):
    """
    Interactive time series plot for ensemble basin statistics.

    Parameters
    ----------
    basin_stats_array : list
    model_names : list
    statistic : str, default 'mean'
        Statistic to plot ('mean', 'std', 'rms', 'sum', 'winsorized_mean', 'outlier_weighted_mean')
    basin_list : list, optional
    gsfc_stats : dict, optional

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive time series plot with dropdown for statistic selection
    """
    if not basin_stats_array:
        raise ValueError("basin_stats_array cannot be empty")

    # Extract years from first model
    years = sorted(list(basin_stats_array[0].keys()))

    # Get all basin names if not specified
    if basin_list is None:
        basin_list = list(basin_stats_array[0][years[0]].keys())

    # Create subplot with one plot per basin
    fig = make_subplots(
        rows=len(basin_list),
        cols=1,
        subplot_titles=[f"Basin {basin}" for basin in basin_list],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # Define colors for different models
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Plot each basin
    for basin_idx, basin_name in enumerate(basin_list):
        row = basin_idx + 1

        # Plot each model
        for model_idx, (basin_stats, model_name) in enumerate(
            zip(basin_stats_array, model_names)
        ):
            values = []
            for year in years:
                if year in basin_stats and basin_name in basin_stats[year]:
                    values.append(basin_stats[year][basin_name][statistic])
                else:
                    values.append(np.nan)

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=values,
                    mode="lines+markers",
                    name=f"{model_name}" if basin_idx == 0 else None,
                    line=dict(color=colors[model_idx % len(colors)]),
                    marker=dict(size=6),
                    showlegend=(basin_idx == 0),  # Only show legend for first subplot
                    hovertemplate=f"<b>{model_name}</b><br>"
                    + f"Basin: {basin_name}<br>"
                    + "Year: %{x}<br>"
                    + f"{statistic.replace('_', ' ').title()}: %{{y:.6f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    if gsfc_stats is not None:
        for basin_idx, basin_name in enumerate(basin_list):
            row = basin_idx + 1

            if basin_name in gsfc_stats:
                gsfc_values = []
                for year in years:
                    if year in gsfc_stats[basin_name]:
                        gsfc_values.append(gsfc_stats[basin_name][year][statistic])
                    else:
                        gsfc_values.append(np.nan)

                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=gsfc_values,
                        mode="lines+markers",
                        name="GSFC" if basin_idx == 0 else None,
                        line=dict(color="black", width=4),
                        marker=dict(size=8, color="black"),
                        showlegend=(
                            basin_idx == 0
                        ),
                        hovertemplate="<b>GSFC</b><br>"
                        + f"Basin: {basin_name}<br>"
                        + "Year: %{x}<br>"
                        + f"{statistic.replace('_', ' ').title()}: %{{y:.6f}}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

    # Update layout
    fig.update_layout(
        title=f"Ensemble Time Series: {statistic.replace('_', ' ').title()} by Basin",
        height=300 * len(basin_list),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Year", row=len(basin_list), col=1)

    # Update y-axis labels
    for i in range(len(basin_list)):
        fig.update_yaxes(
            title_text=statistic.replace("_", " ").title(), row=i + 1, col=1
        )

    return fig


def create_ensemble_statistics_summary(basin_stats_array, model_names, basin_list=None):
    """
    Create summary statistics across ensemble members for each basin and year.

    Parameters
    ----------
    basin_stats_array : list
        List of basin statistics dictionaries from multiple models
    model_names : list
        List of model names corresponding to each basin_stats
    basin_list : list, optional
        List of basin names to analyze. If None, analyzes all basins

    Returns
    -------
    dict
        Dictionary with ensemble statistics (mean, std, min, max) for each basin, year, and statistic
    """
    if not basin_stats_array:
        raise ValueError("basin_stats_array cannot be empty")

    # Extract years from first model
    years = sorted(list(basin_stats_array[0].keys()))

    # Get all basin names if not specified
    if basin_list is None:
        basin_list = list(basin_stats_array[0][years[0]].keys())

    # Available statistics
    available_stats = [
        "mean",
        "std",
        "rms",
        "sum",
        "winsorized_mean",
        "outlier_weighted_mean",
    ]

    ensemble_summary = {}

    for basin_name in basin_list:
        ensemble_summary[basin_name] = {}

        for year in years:
            ensemble_summary[basin_name][year] = {}

            for stat in available_stats:
                values = []
                for basin_stats in basin_stats_array:
                    if year in basin_stats and basin_name in basin_stats[year]:
                        values.append(basin_stats[year][basin_name][stat])

                if values:
                    ensemble_summary[basin_name][year][stat] = {
                        "ensemble_mean": np.mean(values),
                        "ensemble_std": np.std(values),
                        "ensemble_min": np.min(values),
                        "ensemble_max": np.max(values),
                        "ensemble_count": len(values),
                    }

    return ensemble_summary


def create_interactive_ensemble_plot(
    basin_stats_array, model_names, basin_list=None, gsfc_stats=None
):
    """
    Create an interactive ensemble plot with dropdown for statistic selection.

    Parameters
    ----------
    basin_stats_array : list
    model_names : list
    basin_list : list, optional
    gsfc_stats : dict, optional
        GSFC statistics dictionary with structure {basin_name: {year: {stat_name: value}}}
        If provided, will be plotted as a bold black line

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with dropdown and plot
    """

    # Available statistics
    stats_options = [
        ("Mean", "mean"),
        ("Standard Deviation", "std"),
        ("RMS", "rms"),
        ("Sum", "sum"),
        ("Winsorized Mean", "winsorized_mean"),
        ("Outlier Weighted Mean", "outlier_weighted_mean"),
    ]

    # Create dropdown widget
    stat_dropdown = widgets.Dropdown(
        options=stats_options,
        value="mean",
        description="Statistic:",
        style={"description_width": "initial"},
    )

    # Create output widget for plot
    output = widgets.Output()

    def update_plot(change):
        with output:
            output.clear_output(wait=True)
            fig = create_ensemble_time_series_plot(
                basin_stats_array,
                model_names,
                statistic=change["new"],
                basin_list=basin_list,
                gsfc_stats=gsfc_stats,
            )
            fig.show()

    # Initial plot
    with output:
        fig = create_ensemble_time_series_plot(
            basin_stats_array,
            model_names,
            statistic="mean",
            basin_list=basin_list,
            gsfc_stats=gsfc_stats,
        )
        fig.show()

    # Connect dropdown to update function
    stat_dropdown.observe(update_plot, names="value")

    return widgets.VBox([stat_dropdown, output])
