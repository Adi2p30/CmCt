# CmCt Calving Tool Documentation

## Overview

The **Cryosphere model Comparison tool (CmCt) Calving module** is a specialized tool designed to compare ice sheet model outputs with satellite-derived calving observations. This tool enables researchers to perform quantitative assessments of ice sheet model performance by comparing modeled ice mass changes with observed calving data from satellite missions.

## Why Use the Calving Tool?

### Scientific Purpose
- **Model Validation**: Compare ice sheet model predictions against real satellite observations
- **Performance Assessment**: Quantify model-observation discrepancies to improve model accuracy
- **Spatial Analysis**: Perform basin-specific analysis of ice cover changes
- **Temporal Trends**: Analyze ice extent changes over multiple years

### Key Benefits
- **Direct Comparison**: Brings model and satellite data to the same spatial resolution
- **Quantitative Metrics**: Provides statistical measures of model performance
- **Basin-Level Analysis**: Enables detailed regional analysis of ice dynamics
- **Multi-Year Processing**: Supports time series analysis across multiple years

### Data Processing Workflow

1. **Data Loading**: Reads both satellite observations and model outputs from netCDF files
2. **Projection Alignment**: Ensures both datasets use the same polar stereographic projection
3. **Temporal Matching**: Aligns time periods between model and observation data
4. **Spatial Interpolation**: Resamples data to match grid resolutions
5. **Comparison Analysis**: Calculates residuals and statistical metrics
6. **Basin Aggregation**: Performs regional analysis using predefined basin boundaries

### Comparison Methodology

The tool calculates residuals by subtracting modeled ice mask values from observed ice mask values:
```
Residual = Observed_Ice_Mask - Model_Ice_Masks
```

Statistical metrics include:
- **Mean Residual**: Average difference between model and observations
- **RMS Residual**: Root Mean Square of residuals
- **Basin Statistics**: Regional aggregated metrics

## Data Requirements

### Input Data Formats

#### Model Data
- **File Format**: netCDF (.nc)
- **Grid Type**: Rectangular X-Y grid in ISMIP6 polar stereographic projection
- **Required Variables**:
  - `ice_mask`: Ice mask values (annual mean percentage ice cover)
  - `time`: Time dimension in years
  - `x`: Cartesian x-coordinates in meters
  - `y`: Cartesian y-coordinates in meters

#### Satellite Observation Data
- **File Format**: netCDF (.nc)
- **Resolution**: Typically 5x higher resolution than model data
- **Data Range**: Ice cover percentage (0-100%)
- **Coordinate System**: Same projection as model data

### Data Specifications

| Component | Model Data | Satellite Data |
|-----------|------------|----------------|
| **Grid Resolution** | 337 × 577 | 1680 × 2880 |
| **Ice Mask Format** | Fraction (0-1) | Percentage (0-100) |
| **Time Coverage** | 2000-2020 | 2000-2020 |
| **Coordinate System** | ISMIP6 Polar Stereographic | ISMIP6 Polar Stereographic |

### Important Notes
- **Projection Requirement**: Both datasets must use ISMIP6 standard polar stereographic projection
- **Data Conversion**: Satellite data percentages are automatically converted to fractions (divided by 100)
- **Grid Alignment**: Interpolation handles resolution differences between datasets

### Output Data Structure

The output dataset contains:
- **residual**: Model-observation differences
- **model_ice_mask**: Model ice mask values
- **observed_ice_mask**: Satellite observation values
- **time**: Time dimension (years)
- **x, y**: Spatial coordinates
- **basin_id**: Basin classification (if enabled)

## Analysis Capabilities

### Statistical Metrics

For each comparison, the tool calculates:
- **Valid Points**: Number of grid cells with valid data
- **Mean Residual**: Average difference between model and observations
- **RMS Residual**: Root Mean Square of residuals
- **Sum Residual**: Total residual across all valid points
- **Standard Deviation**: Variability of residuals

### Basin-Specific Analysis

When basin aggregation is enabled:
- **Regional Statistics**: Separate metrics for each basin
- **Spatial Patterns**: Identification of regions with systematic biases
- **Temporal Trends**: Evolution of regional differences over time

### Visualization Options

- **Residual Maps**: Spatial distribution of model-observation differences
- **Time Series Plots**: Temporal evolution of metrics
- **Basin Comparisons**: Side-by-side regional analysis
- **Interactive Plots**: Time slider for multi-year visualization

#### File Not Found Errors
```python
# Check file existence
if not os.path.exists(obs_filename):
    raise FileNotFoundError(f"Observation file not found: {obs_filename}")
```

#### Projection Mismatches
- Verify both datasets use the same coordinate system
- Check coordinate ranges and units
- Ensure proper grid alignment

#### Memory Issues
- Reduce time range for processing
- Use data chunking for large datasets
- Implement garbage collection

#### Data Quality Problems
- Check for missing values (NaN)
- Validate coordinate ranges
- Verify time dimension consistency

### Error Messages

| Error Type | Possible Cause | Solution |
|------------|----------------|----------|
| `FileNotFoundError` | Missing input files | Check file paths |
| `KeyError` | Missing variables | Verify netCDF structure |
| `ValueError` | Data type mismatch | Check data formats |
| `MemoryError` | Large datasets | Reduce data size or use chunking |

## Advanced Features

### Custom Interpolation Methods
The tool supports multiple interpolation approaches:
- **Nearest Neighbor**: Fastest, preserves sharp boundaries
- **Linear**: Smooth interpolation, good for gradual changes
- **Cubic**: Highest quality, computationally intensive

### Basin Definition
Users can define custom basin boundaries:
- **Shapefile Import**: Load custom basin definitions
- **Polygon Creation**: Define basins programmatically
- **Multi-Level Basins**: Hierarchical basin structures

### Batch Processing
For large-scale analysis:
- **Multi-Year Processing**: Automated time series analysis
- **Ensemble Analysis**: Compare multiple model runs
- **Parameter Sweeps**: Test different configuration options


*Last updated: [2025-7-07]*
*Version: [1.0.0]*