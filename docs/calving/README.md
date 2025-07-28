# CMCT Calving Module

## Overview

The **Calving Module** is part of the Cryospheric Model Comparison Tool (CmCt), an advanced ice sheet and glacier model validation tool that facilitates direct comparisons between satellite observational data and ice sheet/glacier models. The Calving module specifically focuses on **Ice Area Extent** evaluation, comparing modeled ice extent against satellite observations.

This module enables researchers to evaluate ice extent in ice sheet models against satellite observations, providing crucial insights for reducing sea level projection uncertainty, building model credibility, and supporting climate policy decisions.

## Key Features

- **Flexible Interpolation**: Multiple interpolation methods with smart error correction using trimming
- **Flexible Basin Allotment**: Dynamic basin assignment for regional analysis  
- **Parallel Processing**: Optimized performance with JIT compilation and vectorization
- **CUDA Support**: GPU acceleration for large-scale computations
- **Ensemble Processing**: Support for multiple model comparison and analysis
- **Memory Management**: Intelligent memory monitoring and cleanup

### 🔧 **Core Capabilities**
- Ice extent comparison between models and observations
- Basin-based aggregation and analysis
- Multiple interpolation methods (nearest, linear, cubic, Lanczos 3D)
- Time series analysis and statistical metrics
- Visualization and reporting tools

## Installation

### Prerequisites

```bash
# Required Python packages
numpy
geopandas  
xarray
pandas
matplotlib
shapely
cftime
psutil
ipywidgets
glob
json
logging
datetime
pathlib
warnings
```

### Setup

1. **Clone the CmCt repository:**
```bash
git clone https://github.com/NASA-Cryospheric-Sciences-Laboratory/CmCt.git
cd CmCt
```

2. **Set up Python environment:**
```bash
# Navigate to the main CmCt directory
cmct_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

# Add to Python path
sys.path.insert(0, cmct_dir)
```

3. **Import the calving module:**
```python
from cmct.calving import *
from cmct.calving_modules.interpolation import *
from cmct.calving_modules.json_to_netcdf import *
from cmct.shapefile_utils import *
```

## Quick Start

### Basic Usage

```python
import os, sys
import numpy as np
import xarray as xr
import glob

# Set up paths
cmct_dir = '/path/to/CmCt'
sys.path.insert(0, cmct_dir)

# Import calving modules
from cmct.calving import *
from cmct.shapefile_utils import *

# Configuration
loc = 'GIS'  # 'GIS' (Greenland) or 'AIS' (Antarctica)
obs_filename = cmct_dir + '/data/calving/observed_icemask_ismip_annual.nc'
basin_aggregation = True
basin_shapes = cmct_dir + '/data/ne_10m_coastline/ne_10m_coastline.shp'

# Model files (single or ensemble)
model_files = glob.glob(cmct_dir + '/test/calving/ensemble/*.nc')
model_names = [os.path.splitext(os.path.basename(f))[0] for f in model_files]

# Time range
start_year = 2008
end_year = 2012

# Basin list
basin_list = ["NW", "NE", "SE", "SW", "NO", "CW", "unassigned"]

# Load basins
basins = load_basins_exp(cmct_dir, basin_list)

# Run comparison (implementation details in main workflow)
```

## Data Requirements

### Input Model Data

**Format**: NetCDF files (.nc)

**Requirements**:
- Ice extent/mask data (typically `sftgif` variable)
- 2-D latitude and longitude variables with proper attributes
- Time dimension for temporal analysis
- Projected coordinate system (preferably ISMIP6 polar stereographic)

**Grid Requirements**:
- Rectangular X-Y grid in projected space
- Compatible with ISMIP6 standard projections:
  - Greenland: EPSG:3413
  - Antarctica: EPSG:3031

### Observational Data

The module uses satellite-derived ice mask observations:
- **File**: `observed_icemask_ismip_annual.nc`
- **Location**: `data/calving/` directory
- **Format**: Annual time series of observed ice extent

### Basin Shapefiles

For regional analysis, basin shapefiles are required:
- **Greenland**: Calving basins shapefile
- **Antarctica**: Antarctic basin definitions
- **Format**: ESRI Shapefile (.shp) with basin polygons

## Configuration Options

### Ice Sheet Selection
```python
loc = 'GIS'  # Greenland Ice Sheet
loc = 'AIS'  # Antarctic Ice Sheet
```

### Interpolation Methods
```python
interpolation_method = 'nearest'  # Fast, preserves discrete values
interpolation_method = 'linear'   # Smooth interpolation
interpolation_method = 'cubic'    # Higher-order smoothness
```

### Basin Configuration
```python
# Enable basin aggregation
basin_aggregation = True

# Select specific basins
basin_list = ["NW", "NE", "SE", "SW", "NO", "CW"]  # Greenland basins
basin_list = "all"  # All available basins
```

### Processing Options
```python
# Memory management
chunk_size = 2  # Process 2 years at a time
memory_threshold = 8000  # MB threshold for cleanup

# Accuracy calculation
accuracy_calculation_method = 'mean'  # or 'RMS'
```

## Workflow

### 1. Data Preparation

```python
# Load and validate input files
obs_data = load_observation_data(obs_filename)
model_data = load_model_data(model_files)
basins = load_basins_exp(cmct_dir, basin_list)
```

### 2. Preprocessing

```python
# Time range selection
time_subset = select_time_range(model_data, start_year, end_year)

# Spatial preprocessing
model_regrid = regrid_to_observation_grid(model_data, obs_data)
```

### 3. Comparison Analysis

```python
# Interpolation and comparison
comparison_results = compare_ice_extent(
    model_regrid, 
    obs_data, 
    interpolation_method=interpolation_method
)

# Basin aggregation
basin_results = aggregate_by_basin(comparison_results, basins)
```

### 4. Statistical Analysis

```python
# Calculate metrics
statistics = calculate_comparison_metrics(
    basin_results,
    method=accuracy_calculation_method
)

# Time series analysis
time_series = extract_time_series(basin_results, basin_list)
```

## Output Products

### Statistical Metrics

The module generates comprehensive statistics including:

- **Accuracy Metrics**: Mean differences, RMS errors
- **Spatial Statistics**: Basin-wise comparison results
- **Temporal Analysis**: Time series of ice extent changes
- **Agreement Indices**: Spatial correlation and pattern matching

### Visualization Products

- **Time Series Plots**: Basin-wise ice extent evolution
- **Spatial Maps**: Difference maps between model and observations
- **Statistical Summaries**: Box plots and distribution analysis
- **Ensemble Plots**: Multi-model comparison visualization

### Output Files

```
output/
├── ensemble_results/
│   ├── basin_time_series.nc      # Time series data
│   ├── spatial_differences.nc    # Gridded difference maps
│   ├── statistics_summary.csv    # Statistical metrics
│   └── visualization_plots.png   # Generated plots
```

## Basin Definitions

### Greenland Basins

| Basin Code | Description | Region |
|------------|-------------|---------|
| NW | Northwest | Northwest Greenland |
| NE | Northeast | Northeast Greenland |
| SE | Southeast | Southeast Greenland |
| SW | Southwest | Southwest Greenland |
| NO | North | Central North Greenland |
| CW | Central West | Central West Greenland |

### Basin Analysis Features

- **Flexible Assignment**: Point-polygon assignment algorithms
- **Unassigned Handling**: Grid cells outside basin boundaries
- **Nested Analysis**: Sub-basin and regional aggregation
- **Temporal Tracking**: Basin-wise temporal evolution

## Advanced Features

### Ensemble Processing

```python
# Multiple model comparison
ensemble_results = process_ensemble(
    model_files=model_files,
    model_names=model_names,
    obs_filename=obs_filename,
    basin_list=basin_list
)

# Ensemble statistics
ensemble_stats = calculate_ensemble_statistics(ensemble_results)
```

### Memory Optimization

```python
# Memory monitoring
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Automatic cleanup
def clear_memory():
    gc.collect()
    print(f"Memory after cleanup: {get_memory_usage():.1f} MB")
```

### Parallel Processing

The module implements several optimization strategies:

- **JIT Compilation**: Just-in-time compilation for numerical operations
- **Vectorization**: NumPy vectorized operations for efficiency  
- **Chunk Processing**: Memory-efficient processing of large datasets
- **CUDA Support**: GPU acceleration for compatible operations

## Performance Considerations

### Memory Management

- **Chunked Processing**: Processes data in temporal chunks to manage memory
- **Garbage Collection**: Automatic memory cleanup between operations
- **Memory Monitoring**: Real-time memory usage tracking
- **Threshold-based Cleanup**: Automatic cleanup when memory exceeds limits

### Computational Optimization

- **Smart Interpolation**: Efficient algorithms for spatial interpolation
- **Vectorized Operations**: NumPy-based vectorization for speed
- **Selective Processing**: Process only necessary spatial/temporal subsets
- **Caching**: Intermediate result caching for repeated operations

## Troubleshooting

### Common Issues

1. **Memory Errors**
   ```python
   # Reduce chunk size
   chunk_size = 1  # Process one year at a time
   
   # Lower memory threshold
   memory_threshold = 4000  # MB
   ```

2. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install geopandas shapely xarray
   ```

3. **File Path Issues**
   ```python
   # Verify paths exist
   assert os.path.exists(obs_filename), f"Observation file not found: {obs_filename}"
   assert len(model_files) > 0, "No model files found"
   ```

4. **Projection Issues**
   ```python
   # Ensure proper coordinate system
   # Check that model data uses projected coordinates
   # Verify basin shapefiles match model projection
   ```

### Performance Optimization

1. **Reduce Spatial Resolution**: Coarsen grids for faster processing
2. **Limit Time Range**: Process shorter time periods
3. **Selective Basins**: Analyze only required basins
4. **Parallel Processing**: Utilize multiple cores when available

## Examples

### Single Model Comparison

```python
# Single model analysis
model_file = 'sftgif_model_hist.nc'
obs_file = 'observed_icemask_ismip_annual.nc'

result = compare_single_model(
    model_file=model_file,
    obs_file=obs_file,
    start_year=2008,
    end_year=2012,
    basins=['NW', 'NE', 'SE']
)
```

### Ensemble Analysis

```python
# Multi-model ensemble comparison  
ensemble_files = ['model_A.nc', 'model_B.nc', 'model_C.nc']

ensemble_analysis = compare_ensemble(
    model_files=ensemble_files,
    obs_file=obs_file,
    basin_aggregation=True,
    output_dir='./results/'
)
```

### Basin-Specific Analysis

```python
# Focus on specific basin
nw_analysis = analyze_basin(
    model_files=model_files,
    obs_file=obs_file,
    basin='NW',
    temporal_resolution='annual'
)
```

## Contributing

Contributions to the CMCT Calving module are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Add tests** for new functionality
3. **Update documentation** for any changes
4. **Follow coding standards** and existing patterns
5. **Submit pull requests** with clear descriptions

## Support and Contact

For questions and support:

- **GitHub Issues**: [CMCT Issues](https://github.com/Earth-Information-System/CmCt/issues)
- **Documentation**: See main CMCT documentation
- **Research Paper**: [AGU Publications](https://agupubs.onlinelibrary.wiley.com/)

## License

This software is part of the NASA Cryospheric Model Comparison Tool and follows the same licensing terms as the main CMCT project.

## Citation

When using the CMCT Calving module, please cite:

```
Pachpande, A., & Felikson, D. (2024). Evaluating Ice extent in ice sheet models 
against satellite observations in the CMCT. Purdue University & 
NASA Goddard Space Flight Center.
```

---

**Note**: This module represents ongoing research and development. Features and APIs may evolve as the tool continues to be developed and refined based on community feedback and scientific requirements.