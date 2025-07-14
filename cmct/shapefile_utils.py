import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.interpolate import interp1d

# Made using AI
def shapefile_to_xy(shapefile_path, target_crs='EPSG:3413', interpolate_points=True, num_points=100):
    """
    Convert shapefile to projected coordinates and return GeoDataFrame with interpolated x,y points
    
    Parameters:
    shapefile_path (str): Path to input shapefile
    target_crs (str): Target CRS for projection (default: WGS84)
    interpolate_points (bool): Whether to interpolate points along boundaries for polygons/lines
    num_points (int): Number of points to interpolate along each boundary
    
    Returns:
    GeoDataFrame with interpolated x,y coordinates
    """
        
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Check if CRS is missing and set default if needed
    if gdf.crs is None:
        print("Warning: No CRS found. Assuming WGS84 (EPSG:4326)")
        gdf = gdf.set_crs('EPSG:4326')
    
    # Convert to target projection
    gdf_projected = gdf.to_crs(target_crs)
    
    # Initialize lists to store all interpolated points
    all_x = []
    all_y = []
    feature_ids = []
    
    # Process each geometry
    for idx, geom in enumerate(gdf_projected.geometry):
        if geom.geom_type == 'Point':
            all_x.append(geom.x)
            all_y.append(geom.y)
            feature_ids.append(idx)
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            
            # Extract boundary coordinates
            if geom.geom_type == 'Polygon':
                boundaries = [geom.exterior]
                boundaries.extend(geom.interiors)
            
            else:
                boundaries = []
                for poly in geom.geoms:
                    boundaries.append(poly.exterior)
                    boundaries.extend(poly.interiors)
            
            # Interpolate points along each boundary
            for boundary in boundaries:
                coords = list(boundary.coords)
                if len(coords) > 1 and interpolate_points:
                    
                    # Create interpolation along the boundary
                    x_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    
                    # Calculate cumulative distance for parameterization
                    distances = [0]
                    for i in range(1, len(coords)):
                        dist = np.sqrt((x_coords[i] - x_coords[i-1])**2 + 
                                     (y_coords[i] - y_coords[i-1])**2)
                        distances.append(distances[-1] + dist)
                    
                    # Normalize distances
                    total_distance = distances[-1]
                    if total_distance > 0:
                        distances = [d / total_distance for d in distances]
                        
                        # Interpolate points
                        t_new = np.linspace(0, 1, num_points)
                        f_x = interp1d(distances, x_coords, kind='linear')
                        f_y = interp1d(distances, y_coords, kind='linear')
                        
                        x_interp = f_x(t_new)
                        y_interp = f_y(t_new)
                        
                        all_x.extend(x_interp)
                        all_y.extend(y_interp)
                        feature_ids.extend([idx] * len(x_interp))
                        
                    else:
                        # If boundary has no length, just add the first point
                        all_x.append(x_coords[0])
                        all_y.append(y_coords[0])
                        feature_ids.append(idx)
                        
                else:
                    # Just add the original coordinates
                    for coord in coords[:-1]:  # Exclude last point to avoid duplication
                        all_x.append(coord[0])
                        all_y.append(coord[1])
                        feature_ids.append(idx)
        
        
        elif geom.geom_type in ['LineString', 'MultiLineString']:
            # Handle LineString geometries
            if geom.geom_type == 'LineString':
                lines = [geom]
            else:
                lines = geom.geoms
            
            for line in lines:
                coords = list(line.coords)
                if len(coords) > 1 and interpolate_points:
                    x_coords = [c[0] for c in coords]
                    y_coords = [c[1] for c in coords]
                    
                    # Calculate cumulative distance
                    distances = [0]
                    for i in range(1, len(coords)):
                        dist = np.sqrt((x_coords[i] - x_coords[i-1])**2 + 
                                     (y_coords[i] - y_coords[i-1])**2)
                        distances.append(distances[-1] + dist)
                    
                    total_distance = distances[-1]
                    if total_distance > 0:
                        distances = [d / total_distance for d in distances]
                        
                        t_new = np.linspace(0, 1, num_points)
                        f_x = interp1d(distances, x_coords, kind='linear')
                        f_y = interp1d(distances, y_coords, kind='linear')
                        
                        x_interp = f_x(t_new)
                        y_interp = f_y(t_new)
                        
                        all_x.extend(x_interp)
                        all_y.extend(y_interp)
                        feature_ids.extend([idx] * len(x_interp))
                    
                    else:
                        all_x.append(x_coords[0])
                        all_y.append(y_coords[0])
                        feature_ids.append(idx)
                else:
                    for coord in coords:
                        all_x.append(coord[0])
                        all_y.append(coord[1])
                        feature_ids.append(idx)
    
    # Create a new GeoDataFrame with interpolated points
    interpolated_gdf = gpd.GeoDataFrame({
        'x': all_x,
        'y': all_y,
        'feature_id': feature_ids,
        'geometry': [Point(x, y) for x, y in zip(all_x, all_y)]
    }, crs=target_crs)
    
    interpolated_gdf.reset_index(drop=True, inplace=True)
    
    print(f"Conversion complete. CRS: {target_crs}")
    print(f"Number of interpolated points: {len(interpolated_gdf)}")
    
    return interpolated_gdf

def simple_plot(gdf, title="Shapefile Points"):
    """
    Simple plotting function for the interpolated points
    
    Parameters:
    gdf: GeoDataFrame with x,y coordinates
    title: Title for the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.scatter(gdf['x'], gdf['y'], s=10, alpha=0.6)
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, alpha=0.3)
    plt.show()


# BASINS  FILE AND POLYGON ANALYSIS
def load_basin_polygons(shapefile_path):
    """
    Load basin polygons from a shapefile and return a dictionary of polygons.
    
    Parameters:
    shapefile_path (str): Path to the shapefile containing basin polygons.
    
    Returns:
    dict: A dictionary where keys are basin names and values are Shapely Polygon objects.
    """
    gdf = gpd.read_file(shapefile_path)
    basin_polygons = {}

    for idx, row in gdf.iterrows():
        basin_name = row['SUBREGION1']  # Column name in GRE_Basins_IMBIE2_v1.3 shapefile
        basin_polygons[basin_name] = row['geometry']

    return basin_polygons


def point_in_polygon(point, basins_polygons):
    """
    Check which basin polygon contains the given point.

    Parameters:
    point (tuple): A tuple (x, y) representing the point coordinates.
    basins_polygons (dict): A dictionary of basin polygons.

    Returns:
    str: The key of the basin polygon that contains the point, or None if not found.
    """
    for key, polygon in basins_polygons.items():
        if polygon.contains(Point(point)):
            return key
    return None


#####CALVING SPECIFIC USAGE#######

def get_nonzero_indices(data):
    
    nonzero_indices = np.nonzero(data)
    if len(nonzero_indices[0]) > 0 and len(nonzero_indices[1]) > 0:
        min_row, max_row = nonzero_indices[0].min(), nonzero_indices[0].max()
        min_col, max_col = nonzero_indices[1].min(), nonzero_indices[1].max()
        return [min_col, max_col, min_row, max_row]
    else:
        # Fallback to original shape if no non-zero values found
        return [0, data.shape[1], 0, data.shape[0]]

def scaling_shape_to_target(data, target_shape):
    """
    Convert the coordinates of a GeoDataFrame or numpy array to match target dimensions.
    
    Parameters:
    data: GeoDataFrame with x,y coordinates or numpy array with shape (n, 2)
    target_shape: List or array [min_x, max_x, min_y, max_y] defining target bounds
    
    Returns:
    For GeoDataFrame: Returns a new GeoDataFrame with transformed coordinates
    For numpy array: Returns transformed numpy array
    """
    
    if isinstance(data, gpd.GeoDataFrame):
        # Extract coordinates from GeoDataFrame
        coords = np.array(list(zip(data['x'], data['y'])))
        
        # Calculate the scale factors for x and y dimensions
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        # Avoid division by zero
        if x_range == 0 or y_range == 0:
            print("Warning: Zero range in coordinates, returning original data")
            return data
            
        x_scale = (target_shape[1] - target_shape[0]) / x_range
        y_scale = (target_shape[3] - target_shape[2]) / y_range

        # Apply the transformation
        transformed_x = (coords[:, 0] - coords[:, 0].min()) * x_scale + target_shape[0]
        transformed_y = (coords[:, 1] - coords[:, 1].min()) * y_scale + target_shape[2]
        
        # Create new GeoDataFrame with transformed coordinates
        transformed_gdf = data.copy()
        transformed_gdf['x'] = transformed_x
        transformed_gdf['y'] = transformed_y
        transformed_gdf['geometry'] = [Point(x, y) for x, y in zip(transformed_x, transformed_y)]
        
        return transformed_gdf
        
    elif isinstance(data, np.ndarray):
        # Handle numpy array directly
        if len(data.shape) != 2 or data.shape[1] != 2:
            raise ValueError("Numpy array must have shape (n, 2) representing (x, y) coordinates")
        
        coords = data.copy()  # Make a copy to avoid modifying original
        
        # Calculate the scale factors for x and y dimensions
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        
        # Avoid division by zero
        if x_range == 0 or y_range == 0:
            print("Warning: Zero range in coordinates, returning original data")
            return coords
            
        x_scale = (target_shape[1] - target_shape[0]) / x_range
        y_scale = (target_shape[3] - target_shape[2]) / y_range

        # Apply the transformation
        coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) * x_scale + target_shape[0]
        coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) * y_scale + target_shape[2]

        return coords
    
    else:
        raise TypeError("Data must be either a GeoDataFrame or numpy array")

def plot_residuals_with_basins(residuals_data, basin_polygons_dict, year, figsize=(15, 12)):
    """
    Create an overlay plot showing residuals with basin boundaries.
    
    Parameters:
    -----------
    residuals_data : object
        Residuals dataset containing the gridded data
    basin_polygons_dict : dict
        Dictionary of standardized basin polygons
    year : int
        Year to plot
    figsize : tuple
        Figure size for the plot
    """
    
    # Get residual data for the specified year
    residual_year_data = residuals_data.ds.sel(year=year)
    rotated_data = rotated_data_year(residuals_data, year)
    
    # Get coordinate information from the dataset
    x_coords = residuals_data.ds.x.values
    y_coords = residuals_data.ds.y.values
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the residuals as background
    im = ax.imshow(rotated_data, 
                   extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
                   cmap='RdBu_r', 
                   origin='lower', 
                   aspect='equal',
                   alpha=0.8,
                   vmin=-1, vmax=1)  # Adjust vmin/vmax based on your data range
    
    # Add colorbar for residuals
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Residual Ice Mask Value', fontsize=12)
    
    # Overlay basin boundaries
    for basin_name, polygon in basin_polygons_dict.items():
        if basin_name == 'unassigned':
            continue
            
        if hasattr(polygon, 'exterior'):
            # Extract boundary coordinates
            x_boundary, y_boundary = polygon.exterior.xy
            
            # Get color for this basin
            basin_color = colors.get(basin_name, 'black')
            
            # Plot basin boundary
            ax.plot(x_boundary, y_boundary, 
                   color=basin_color, 
                   linewidth=2, 
                   label=f'{basin_name} Basin',
                   alpha=0.9)
            
            # Add basin label at centroid
            centroid = polygon.centroid
            ax.annotate(basin_name, 
                       xy=(centroid.x, centroid.y),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=10, 
                       fontweight='bold',
                       color=basin_color,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                alpha=0.7,
                                edgecolor=basin_color))
    
    # Set labels and title
    ax.set_xlabel('X Coordinate (m)', fontsize=12)
    ax.set_ylabel('Y Coordinate (m)', fontsize=12)
    ax.set_title(f'Residuals with Basin Boundaries - Year {year}', fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Add coordinate information as text
    coord_text = f"Data bounds: X=[{x_coords.min():.0f}, {x_coords.max():.0f}], Y=[{y_coords.min():.0f}, {y_coords.max():.0f}]"
    ax.text(0.02, 0.98, coord_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.show()
    
    # Print basin coverage statistics
    print(f"\nBasin Coverage Analysis for Year {year}:")
    print("=" * 50)
    
    total_data_points = np.sum(~np.isnan(rotated_data))
    print(f"Total non-NaN data points: {total_data_points}")
    
    if basin_dataset and year in basin_dataset:
        for basin_name in basin_polygons_dict.keys():
            if basin_name in basin_dataset[year]:
                basin_points = len(basin_dataset[year][basin_name]['data_value'])
                coverage_pct = (basin_points / total_data_points) * 100 if total_data_points > 0 else 0
                print(f"{basin_name:>12}: {basin_points:>6} points ({coverage_pct:>5.1f}% coverage)")

##################################


# Example usage:
# gdf_xy = shapefile_to_xy('/Users/aditya_pachpande/Documents/GitHub/CmCt/data/calving/GRE_Basins_IMBIE2_v1.3.shp')
# simple_plot(gdf_xy, "My Shapefile Points")  