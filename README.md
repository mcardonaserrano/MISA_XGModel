# **MISA_XGModel**  
*A Python library for predicting electron density using XGBoost.*

---
## **Overview**  

This library provides tools to predict electron density (`Ne`) using a machine-learning model based on XGBoost. It supports querying predictions for specific latitude, longitude, day of year (DOY), altitude, and solar local time (SLT), while leveraging pre-computed geophysical indices.  

---

## **Features**  
- Predict electron density (`Ne`) for specific geospatial and temporal conditions.
- Clamp input values to the bounds of the training dataset for robustness.
- Supports querying geophysical indices from provided datasets.
- Modular design for integrating geophysical models in Python applications.

---

## **Installation**  

You can install this package via **pip**:

```bash
pip install MISA_XGModel
```

---

## **Usage**  

### **Quickstart**

Here’s how you can use the library to predict electron density:

```python
from MISA_XGModel import predict_ne, query_model

# Predict Ne with dataset lookup for geophysical indices
predicted_ne = predict_ne(
    lat=42.0, lon=-71.0, doy=99, alt=150.0, slt=12.0, year=2024
)
print(f"Predicted Ne: {predicted_ne:.2e}")

# Predict Ne with precomputed geophysical indices
predicted_ne = query_model(
    lat=42.0, lon=-71.0, doy=99, alt=150.0, slt=12.0,
    hp30=2, ap30=7, f107=209, kp=2.3, fism2=0.0007678
)
print(f"Predicted Ne: {predicted_ne:.2e}")
```
---

## **Inputs and Parameters**

### **Clamping Input Values**

Input parameters (`lat`, `lon`, `doy`, `alt`, `slt`) are clamped to the boundaries of the training data. These boundaries are defined as follows:

| Parameter | Min Value | Max Value |
|-----------|-----------|-----------|
| `lat`     | 37.5      | 49.9      |
| `lon`     | -85.7     | -76.1     |
| `alt`     | 94.6 km   | 500 km    |
| `slt`     | 0.0 hrs   | 24 hrs    |

---

### **`predict_ne` Function**

| Parameter   | Type    | Description                                                                 |
|-------------|---------|-----------------------------------------------------------------------------|
| `lat`       | `float` | Latitude of the input location (degrees). Clamped to `[37.5, 49.9]`.        |
| `lon`       | `float` | Longitude of the input location (degrees). Clamped to `[-85.7, -76.1]`.     |
| `doy`       | `int`   | Day of year (1-365).                                                       |
| `alt`       | `float` | Altitude in kilometers. Clamped to `[94.6, 500]`.                          |
| `slt`       | `float` | Solar local time (0-24 hours). Clamped to `[0.0, 24]`.                     |
| `year`      | `int`   | Year to lookup geophysical indices in the dataset.                         |
| `master_geo_ds` | `xarray.Dataset` | Dataset containing geophysical indices (default: `master_geo_ds`). |
| `model`     | `XGBRegressor` | Pre-trained XGBoost model for predictions (default: `optimized_xgb`).    |
| `scaler`    | `StandardScaler` | Scaler used for feature normalization (default: `scaler_large`).      |

This function also retrieves geophysical indices (`hp30`, `ap30`, `f107`, `kp`, `fism2`) from the provided dataset for the specified year and day of year (`doy`).

---

### **`query_model` Function**

| Parameter   | Type    | Description                                                                 |
|-------------|---------|-----------------------------------------------------------------------------|
| `lat`       | `float` | Latitude of the input location (degrees). Clamped to `[37.5, 49.9]`.        |
| `lon`       | `float` | Longitude of the input location (degrees). Clamped to `[-85.7, -76.1]`.     |
| `doy`       | `int`   | Day of year (1-365).                                                       |
| `alt`       | `float` | Altitude in kilometers. Clamped to `[94.6, 500]`.                          |
| `slt`       | `float` | Solar local time (0-24 hours). Clamped to `[0.0, 24]`.                     |
| `hp30`      | `float` | Precomputed geophysical index.                                              |
| `ap30`      | `float` | Precomputed geophysical index.                                              |
| `f107`      | `float` | Precomputed geophysical index.                                              |
| `kp`        | `float` | Precomputed geophysical index.                                              |
| `fism2`     | `float` | Precomputed geophysical index.                                              |
| `model`     | `XGBRegressor` | Pre-trained XGBoost model for predictions (default: `optimized_xgb`).    |
| `scaler`    | `StandardScaler` | Scaler used for feature normalization (default: `scaler_large`).      |

`predict_ne` requires geophysical indices as input, allowing for greater control over predictions in scenarios where geophysical data is already available.

---

## **Requirements**

- Python 3.7+
- **Dependencies**:
  - `xgboost`
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `xarray`
  - `joblib`
  - `netcdf4`

---

## **License**  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Author**  

- **Mateo Cardona Serrano (them)**  
- [GitHub Profile](https://github.com/mcardonaserrano)  
- [Email](mailto:mcardonaserrano@berkeley.edu)  

---

## **Acknowledgments**  

- Thank you to Sevag Derghazarian for his continuous support and consultation on this project.  

---