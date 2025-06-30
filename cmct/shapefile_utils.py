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


# BASINS EXP FILE AND POLYGON ANALYSIS

def analyze_exp_files(filepath):
    with open(filepath, 'r') as file:
        xy_data = file.readlines()[5:]
    # Start from line 6
    xy = {"x": [], "y": []}
    for i in range(len(xy_data)):
        xy_data[i] = xy_data[i].strip().split()
        xy_data[i] = [float(x) for x in xy_data[i]]
        xy["x"].append(xy_data[i][0])
        xy["y"].append(xy_data[i][1])
    
    # Convert to numpy arrays after all data is collected
    xy["x"] = np.array(xy["x"])
    xy["y"] = np.array(xy["y"])
    return xy


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


##################################


# Example usage:
# gdf_xy = shapefile_to_xy('/Users/aditya_pachpande/Documents/GitHub/CmCt/data/calving/GRE_Basins_IMBIE2_v1.3.shp')
# simple_plot(gdf_xy, "My Shapefile Points")  