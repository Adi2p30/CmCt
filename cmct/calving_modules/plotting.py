import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import json 

def plot_scored_data(data, cmap='viridis', title=None, save_path=None):
    """
    Plot scored data using matplotlib.
    
    Parameters:
        data (xarray.DataArray): The scored data to plot.
        cmap (str): Colormap to use for the plot.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image.
    """
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, cmap=cmap)
    plt.colorbar(im, label='Score')
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

