# Shared utilities
import numpy as np
import cftime 
import datetime

def check_datarange(*args, **kwargs):
    """
    Flexible function to check if date range is available in datasets.
    Handles both legacy and new calling patterns.
    """
    # Legacy calling pattern: check_datarange(time_var, start_date_cftime, end_date_cftime)
    if len(args) == 3 and hasattr(args[0], 'to_index'):
        time_var, start_date_cftime, end_date_cftime = args
        calendar_type = time_var.to_index().calendar
           
        # Get the minimum and maximum values directly from the time variable
        min_time = time_var.values.min()
        max_time = time_var.values.max()
        
        # Check if the selected start and end dates are within the range
        if min_time <= start_date_cftime <= max_time and min_time <= end_date_cftime <= max_time:
            print(f"The selected dates {start_date_cftime} and {end_date_cftime} are within the range of the model data.")
        else:
            raise ValueError(f"Error: The selected dates {start_date_cftime} or {end_date_cftime} are out of range. Model data time range is from {min_time} to {max_time}.")
    
    # New calling pattern: check_datarange(gsfc_time, model_time, start_year, end_year)
    elif len(args) == 4:
        gsfc_time, model_time, start_date, end_date = args
        
        # Use the enhanced date range checking function
        try:
            import pandas as pd
            HAS_PANDAS = True
        except ImportError:
            HAS_PANDAS = False
        
        def extract_year(time_val):
            """Extract year from various time formats"""
            if isinstance(time_val, (int, float)):
                # Assume it's already a year (possibly decimal year)
                return float(time_val)
            elif hasattr(time_val, 'year'):
                # cftime, datetime, or similar objects with year attribute
                return float(time_val.year)
            elif isinstance(time_val, np.datetime64):
                # numpy datetime64
                if HAS_PANDAS:
                    return float(pd.to_datetime(time_val).year)
                else:
                    # Fallback without pandas
                    return float(str(time_val)[:4])
            elif isinstance(time_val, str):
                # Try to parse string dates
                if HAS_PANDAS:
                    try:
                        dt = pd.to_datetime(time_val)
                        return float(dt.year)
                    except:
                        pass
                # If pandas parsing fails or not available, try to extract year from string
                import re
                year_match = re.search(r'\d{4}', str(time_val))
                if year_match:
                    return float(year_match.group())
            
            # Fallback: try to convert to float directly
            try:
                return float(time_val)
            except:
                raise ValueError(f"Cannot extract year from time value: {time_val} (type: {type(time_val)})")
        
        # Convert time arrays to years
        try:
            gsfc_years = [extract_year(t) for t in gsfc_time]
            model_years = [extract_year(t) for t in model_time]
        except Exception as e:
            print(f"Error processing time values: {e}")
            print(f"GSFC time sample: {gsfc_time[:3] if len(gsfc_time) > 3 else gsfc_time}")
            print(f"Model time sample: {model_time[:3] if len(model_time) > 3 else model_time}")
            raise
        
        # Sort and get min/max years
        gsfc_years.sort()
        gsfc_time_min = gsfc_years[0]
        gsfc_time_max = gsfc_years[-1]
        
        model_years.sort()
        model_time_min = model_years[0]
        model_time_max = model_years[-1]

        minimum_time = max(gsfc_time_min, model_time_min)
        maximum_time = min(gsfc_time_max, model_time_max)
        
        print(f"GSFC data range: {gsfc_time_min:.1f} to {gsfc_time_max:.1f}")
        print(f"Model data range: {model_time_min:.1f} to {model_time_max:.1f}")
        print(f"Overlapping range: {minimum_time:.1f} to {maximum_time:.1f}")
        print(f"Requested range: {start_date} to {end_date}")

        if not (minimum_time <= start_date <= end_date <= maximum_time):
            raise ValueError(f"Date range {start_date} to {end_date} is outside the available overlapping data range: {minimum_time:.1f} to {maximum_time:.1f}.")
        else: 
            print(f"✓ The selected dates {start_date} to {end_date} are within the overlapping data range.")
    
    else:
        raise ValueError(f"Invalid arguments for check_datarange. Expected 3 or 4 arguments, got {len(args)}")


def days_in_year(date):
    if date.calendar == '365_day' or date.calendar == 'noleap':
        diy = 365
    elif date.calendar == '366_day' or date.calendar == 'all_leap':
        diy = 366
    elif date.calendar == '360_day':
        diy = 360
    else:
        if cftime.is_leap_year(date.year, date.calendar):
            diy = 366
        else:
            diy = 365

    return diy