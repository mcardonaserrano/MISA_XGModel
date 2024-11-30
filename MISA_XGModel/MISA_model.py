import requests
import os
import numpy as np
from xgboost import XGBRegressor
import joblib
import xarray as xr
import pandas as pd
import importlib.resources as pkg_resources
from MISA_XGModel import data  # Import the package where your data files are located

# Define file names (these files should be in the MISA_XGModel/data directory)
MODEL_FILE_NAME = "xgboost_optimized_model.json"
SCALER_FILE_NAME = "scaler_large.pkl"
MASTER_GEO_DS_FILE_NAME = "master_geo_ds.nc"

# Bounds for clamping
input_bounds = {
    'lat': (np.float64(37.5), np.float64(49.9)),
    'lon': (np.float64(-85.7), np.float64(-76.1)),
    'alt': (np.float64(94.6), np.float64(500)),
    'slt': (np.float64(0.0), np.float64(24))
}


def clamp(value, min_value, max_value):
    """Clamp a value to the specified range."""
    return max(min_value, min(max_value, value))


def load_resource(file_name):
    """
    Load a file from the package resources.

    Parameters:
        file_name (str): The name of the file to load.

    Returns:
        str: Path to the temporary extracted file.
    """
    with pkg_resources.path(data, file_name) as resource_path:
        return str(resource_path)


# Load resources
MODEL_PATH = load_resource(MODEL_FILE_NAME)
SCALER_PATH = load_resource(SCALER_FILE_NAME)
MASTER_GEO_DS_PATH = load_resource(MASTER_GEO_DS_FILE_NAME)

# Load the model, scaler, and geophysical dataset
optimized_xgb = XGBRegressor()
optimized_xgb.load_model(MODEL_PATH)
scaler_large = joblib.load(SCALER_PATH)
master_geo_ds = xr.open_dataset(MASTER_GEO_DS_PATH)


def predict_ne(lat, lon, doy, alt, slt, year, master_geo_ds=master_geo_ds, model=optimized_xgb, scaler=scaler_large):
    """
    Predict the electron density (Ne) using geophysical indices and model features,
    clamping input values to the range of training data.

    Parameters:
        lat (float): Latitude of the input location (clamped to training data range).
        lon (float): Longitude of the input location (clamped to training data range).
        doy (int): Day of the year (1-365, clamped to training data range).
        alt (float): Altitude in kilometers (clamped to training data range).
        slt (float): Solar local time in hours (0-24, clamped to training data range).
        year (int): Target year for geophysical indices lookup.
        master_geo_ds (xarray.Dataset): Dataset containing geophysical indices.
        model (XGBRegressor): Trained XGBoost model for predictions.
        scaler (StandardScaler): Scaler used for feature normalization.

    Returns:
        float: Predicted electron density (Ne) in the original scale.
    """
    # Clamp inputs
    lat = clamp(lat, *input_bounds["lat"])
    lon = clamp(lon, *input_bounds["lon"])
    alt = clamp(alt, *input_bounds["alt"])
    slt = clamp(slt, *input_bounds["slt"])

    # Ensure `dates` coordinate is in datetime format
    dates_as_datetime = pd.to_datetime(master_geo_ds["dates"].values)

    # Filter by year and DOY
    year_mask = dates_as_datetime.year == year
    if not np.any(year_mask):
        raise ValueError(f"No data available in `master_geo_ds` for year {year}.")
    filtered_data = master_geo_ds.sel(dates=year_mask)
    dates_doy = pd.to_datetime(filtered_data["dates"].values).dayofyear
    doy_mask = dates_doy == doy
    if not np.any(doy_mask):
        raise ValueError(f"No data available in `master_geo_ds` for DOY {doy} in year {year}.")
    matched_dates = filtered_data.sel(dates=doy_mask)

    # Extract geophysical indices
    geo_indices = matched_dates.isel(dates=0)
    hp30 = geo_indices["hp30"].values.item()
    ap30 = geo_indices["ap30"].values.item()
    f107 = geo_indices["f107"].values.item()
    kp = geo_indices["kp"].values.item()
    fism2 = geo_indices["fism2"].values.item()

    # Predict using the query_model function
    return query_model(lat, lon, doy, alt, slt, hp30, ap30, f107, kp, fism2, model=model, scaler=scaler)


def query_model(lat, lon, doy, alt, slt, hp30, ap30, f107, kp, fism2, model=optimized_xgb, scaler=scaler_large):
    """
    Predicts Ne using the model and precomputed geophysical indices,
    clamping input values to the range of training data.

    Parameters:
        lat, lon, doy, alt, slt, hp30, ap30, f107, kp, fism2: Model inputs.

    Returns:
        float: Predicted electron density (Ne) in the original scale.
    """
    # Clamp inputs
    lat = clamp(lat, *input_bounds["lat"])
    lon = clamp(lon, *input_bounds["lon"])
    alt = clamp(alt, *input_bounds["alt"])
    slt = clamp(slt, *input_bounds["slt"])

    # Compute trigonometric features for SLT and DOY
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)
    slt_sin = np.sin(2 * np.pi * slt / 24)
    slt_cos = np.cos(2 * np.pi * slt / 24)

    # Prepare input features
    input_features = np.array([[
        lat, lon, alt, slt, doy, hp30, ap30, f107, kp, fism2,
        slt_sin, slt_cos, doy_sin, doy_cos, alt * f107, lat * fism2,
        hp30 / (ap30 + 1e-6), f107 * kp, alt ** 2, f107 ** 2,
        slt ** 3, doy ** 3, np.log1p(f107), np.log1p(ap30)
    ]])

    # Scale features
    input_features_scaled = scaler.transform(input_features)

    # Predict using the model
    prediction_log = model.predict(input_features_scaled)
    return np.expm1(prediction_log)[0]  # Transform back from log scale