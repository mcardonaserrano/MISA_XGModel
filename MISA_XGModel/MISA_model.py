import requests
import os
import numpy as np
from xgboost import XGBRegressor
import xarray as xr
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

# Dropbox URLs for required files
MODEL_URL = "https://www.dropbox.com/scl/fi/buerwbp580l98c5egbmvg/xgboost_optimized_model.json?rlkey=0mxboow2r44j7pz3xx199inko&dl=1"
SCALER_URL = "https://www.dropbox.com/scl/fi/6oas604oh4xcupzc8at2b/scaler_large.json?rlkey=6huwm1baf1vc7cf9r4of4xxds&st=2s3wu078&dl=1"
GEO_DS_URL = "https://www.dropbox.com/scl/fi/m9loyzdo0r6j5mvb5lku4/master_geo_ds.nc?rlkey=6s7br1o4wh7vbiuctmrtlhe7v&st=x4mpl3e3&dl=1"

# Paths to save downloaded files
MODEL_PATH = "data/xgboost_optimized_model.json"
SCALER_PATH = "data/scaler_large.json"
MASTER_GEO_DS_PATH = "data/master_geo_ds.nc"

# Bounds for clamping
input_bounds = {
    'lat': (np.float64(37.5), np.float64(49.9)),
    'lon': (np.float64(-85.7), np.float64(-76.1)),
    'alt': (np.float64(94.6), np.float64(500)),
    'slt': (np.float64(0.0), np.float64(24))
}


# Utility to download files from Dropbox
def download_file(url, save_path):
    """Download a file from a URL if it doesn't exist locally."""
    if not os.path.exists(save_path):
        print(f"Downloading file from {url}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks
                    f.write(chunk)
            print(f"File downloaded and saved to {save_path}.")
        else:
            raise RuntimeError(f"Failed to download file. HTTP Status Code: {response.status_code}")

# Utility to load scaler
def load_scaler(json_path):
    """Load a StandardScaler from a JSON file."""
    with open(json_path, "r") as f:
        scaler_params = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params["mean"])
    scaler.scale_ = np.array(scaler_params["scale"])
    scaler.var_ = np.array(scaler_params["var"])
    scaler.n_features_in_ = len(scaler.mean_)  # Ensure compatibility with sklearn's expectations
    return scaler

# Download required files
download_file(MODEL_URL, MODEL_PATH)
download_file(SCALER_URL, SCALER_PATH)
download_file(GEO_DS_URL, MASTER_GEO_DS_PATH)


# Load the model, scaler, and geophysical dataset
optimized_xgb = XGBRegressor()
optimized_xgb.load_model(MODEL_PATH)
scaler_large = load_scaler(SCALER_PATH)
master_geo_ds = xr.open_dataset(MASTER_GEO_DS_PATH)

def predict_ne(lat, lon, doy, alt, slt, year, master_geo_ds=master_geo_ds, model=optimized_xgb, scaler=scaler_large):
    """
    Predict the electron density (Ne) using geophysical indices and model features,
    clamping input values to the range of training data.

    Parameters:
        lat, lon, doy, alt, slt: Scalars or arrays representing the inputs.
        year: Target year to lookup geophysical indices in the dataset.
        master_geo_ds: xarray.Dataset containing geophysical indices.
        model: Pre-trained XGBoost model.
        scaler: Scaler used for feature normalization.

    Returns:
        np.ndarray or float: Predicted electron density (Ne) in the original scale.
    """
    # Ensure inputs are arrays for consistent processing
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    doy = np.atleast_1d(doy)
    alt = np.atleast_1d(alt)
    slt = np.atleast_1d(slt)

    # Check if all input arrays are the same length
    lengths = [len(lat), len(lon), len(doy), len(alt), len(slt)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Received lengths: {lengths}"
        )

    # Clamp inputs
    lat = np.clip(lat, *input_bounds["lat"])
    lon = np.clip(lon, *input_bounds["lon"])
    alt = np.clip(alt, *input_bounds["alt"])
    slt = np.clip(slt, *input_bounds["slt"])
    doy = doy % 365  # Wrap DOY to [0, 364]

    # Prepare predictions
    predictions = []
    for i in range(len(lat)):
        # Filter dataset by year
        dates_as_datetime = pd.to_datetime(master_geo_ds["dates"].values)
        year_mask = dates_as_datetime.year == year
        if not np.any(year_mask):
            raise ValueError(f"No data available in `master_geo_ds` for year {year}.")
        filtered_data = master_geo_ds.sel(dates=year_mask)

        # Match DOY
        dates_doy = pd.to_datetime(filtered_data["dates"].values).dayofyear
        doy_mask = dates_doy == doy[i]
        if not np.any(doy_mask):
            raise ValueError(f"No data available in `master_geo_ds` for DOY {doy[i]} in year {year}.")
        matched_dates = filtered_data.sel(dates=doy_mask)

        # Find the SLT closest to the input SLT
        slt_diff = np.abs(matched_dates["slt"].values - slt[i])
        closest_slt_idx = slt_diff.argmin()
        geo_indices = matched_dates.isel(dates=closest_slt_idx)

        hp30 = geo_indices["hp30"].values.item()
        ap30 = geo_indices["ap30"].values.item()
        f107 = geo_indices["f107"].values.item()
        kp = geo_indices["kp"].values.item()
        fism2 = geo_indices["fism2"].values.item()

        # Predict Ne using query_model
        prediction = query_model(
            lat[i], lon[i], doy[i], alt[i], slt[i],
            hp30, ap30, f107, kp, fism2,
            model=model, scaler=scaler
        )
        predictions.append(prediction)

    # Return predictions as an array
    return np.squeeze(predictions)  # Squeeze to handle scalar results

def query_model(lat, lon, doy, alt, slt, hp30, ap30, f107, kp, fism2, model=optimized_xgb, scaler=scaler_large):
    """
    Predicts Ne using the model and precomputed geophysical indices,
    clamping input values to the range of training data.

    Parameters:
        lat, lon, doy, alt, slt, hp30, ap30, f107, kp, fism2: Model inputs. These can be scalars or arrays.

    Returns:
        np.ndarray or float: Predicted electron density (Ne) in the original scale.
    """
    # Ensure inputs are arrays for consistent processing
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    doy = np.atleast_1d(doy)
    alt = np.atleast_1d(alt)
    slt = np.atleast_1d(slt)
    hp30 = np.atleast_1d(hp30)
    ap30 = np.atleast_1d(ap30)
    f107 = np.atleast_1d(f107)
    kp = np.atleast_1d(kp)
    fism2 = np.atleast_1d(fism2)

    # Check if all input arrays are the same length
    lengths = [len(lat), len(lon), len(doy), len(alt), len(slt), len(hp30), len(ap30), len(f107), len(kp), len(fism2)]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All input arrays must have the same length. Received lengths: {lengths}"
        )

    # Clamp inputs
    lat = np.clip(lat, *input_bounds["lat"])
    lon = np.clip(lon, *input_bounds["lon"])
    alt = np.clip(alt, *input_bounds["alt"])
    slt = np.clip(slt, *input_bounds["slt"])
    doy = doy % 365  # Wrap DOY to [0, 364]

    # Prepare predictions
    predictions = []
    for i in range(len(lat)):
        # Compute trigonometric features for SLT and DOY
        doy_sin = np.sin(2 * np.pi * doy[i] / 365)
        doy_cos = np.cos(2 * np.pi * doy[i] / 365)
        slt_sin = np.sin(2 * np.pi * slt[i] / 24)
        slt_cos = np.cos(2 * np.pi * slt[i] / 24)

        # Prepare input features
        input_features = np.array([[
            lat[i], lon[i], alt[i], slt[i], doy[i], hp30[i], ap30[i], f107[i], kp[i], fism2[i],
            slt_sin, slt_cos, doy_sin, doy_cos,
            alt[i] * f107[i],  # Engineered feature: alt_f107
            lat[i] * fism2[i],  # Engineered feature: lat_fism2
            hp30[i] / (ap30[i] + 1e-6),  # Engineered feature: hp30_ap30
            f107[i] * kp[i],  # Engineered feature: f107_kp
            alt[i] ** 2,  # Engineered feature: alt_squared
            f107[i] ** 2,  # Engineered feature: f107_squared
            slt[i] ** 3,  # Engineered feature: slt_cubed
            doy[i] ** 3,  # Engineered feature: doy_cubed
            np.log1p(f107[i]),  # Engineered feature: log_f107
            np.log1p(ap30[i])  # Engineered feature: log_ap30
        ]])

        # Scale features
        input_features_scaled = scaler.transform(input_features)

        # Predict using the model
        prediction_log = model.predict(input_features_scaled)
        predictions.append(np.expm1(prediction_log)[0])  # Transform back from log scale

    # Return predictions as an array
    return np.squeeze(predictions)  # Squeeze to handle scalar results