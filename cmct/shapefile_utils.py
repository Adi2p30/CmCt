import geopandas as gpd
import os
import numpy as np
from shapely.geometry import Point
from scipy.interpolate import interp1d

@Made_using_AI
def shapefile_to_xy(shapefile_path, target_crs='EPSG:3857', interpolate_points=True, num_points=100):
    """
    Convert shapefile to projected coordinates and return GeoDataFrame with interpolated x,y points
    
    Parameters:
    shapefile_path (str): Path to input shapefile
    target_crs (str): Target CRS for projection (default: Web Mercator)
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

# Example usage:
gdf_xy = shapefile_to_xy('/Users/aditya_pachpande/Documents/GitHub/CmCt/data/calving/GRE_Basins_IMBIE2_v1.3.shp')
simple_plot(gdf_xy, "My Shapefile Points")  