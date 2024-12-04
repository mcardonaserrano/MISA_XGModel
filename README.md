# **MISA_XGModel**  
*A Python library for predicting electron density using XGBoost.*

---
## **Overview**  

This library provides tools to predict electron density (`Ne`) using a machine-learning model based on XGBoost. It supports querying predictions for specific latitude, longitude, day of year (DOY), altitude, and solar local time (SLT), while leveraging pre-computed geophysical indices. The library also supports efficient vectorized predictions for multiple inputs.

---

## **Features**  
- Predict electron density (`Ne`) for specific geospatial and temporal conditions.
- Efficient batch predictions using vectorized input arrays for higher performance.
- Clamp or wrap input values to the bounds of the training dataset for robustness.
- Supports querying geophysical indices from provided datasets.
- Modular design for integrating geophysical models in Python applications.

---

## **Installation**  

You can install this package via **pip**:

```bash
pip install MISA_XGModel
```



The model downloads missing dependencies on first import (e.g., the XGBoost model weights, scaler, and geophysical dataset):
```bash
Downloading model from https://www.dropbox.com/...
Model downloaded and saved to data/xgboost_optimized_model.json.
Downloading scaler from https://www.dropbox.com/...
Scaler downloaded and saved to data/scaler_large.json.
Downloading dataset from https://www.dropbox.com/...
Dataset downloaded and saved to data/master_geo_ds.nc.
```

If the files already exist locally, the library will skip the download step and use the existing files.

---

## **Usage**  

### **Quickstart**

Here’s how you can use the library to predict electron density:

#### **Single Prediction**

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

#### **Batch Predictions**

To predict multiple inputs efficiently, you can pass arrays to `predict_ne` or `query_model`:

```python
import numpy as np

# Vectorized inputs
lats = np.array([42.0, 41.5, 40.0])
lons = np.array([-71.0, -72.0, -73.0])
doys = np.array([99, 100, 101])
alts = np.array([150.0, 200.0, 250.0])
slts = np.array([12.0, 14.0, 16.0])
year = 2024

# Batch predict Ne with dataset lookup for geophysical indices
predicted_ne = predict_ne(
    lat=lats, lon=lons, doy=doys, alt=alts, slt=slts, year=year
)
print(f"Predicted Ne: {predicted_ne}")

# Batch predict Ne with precomputed geophysical indices
predicted_ne = query_model(
    lat=lats, lon=lons, doy=doys, alt=alts, slt=slts,
    hp30=np.array([2, 3, 4]),
    ap30=np.array([7, 8, 9]),
    f107=np.array([209, 210, 211]),
    kp=np.array([2.3, 2.5, 2.7]),
    fism2=np.array([0.0007678, 0.00078, 0.00079])
)
print(f"Predicted Ne: {predicted_ne}")
```

---

## **Inputs and Parameters**

### **Clamping and Wrapping Input Values**

Input parameters (`lat`, `lon`, `alt`, `slt`) are clamped to the boundaries of the training data, while `doy` is wrapped to stay within `[0, 364]`. These boundaries are defined as follows:

| Parameter | Min Value | Max Value | Notes                                   |
|-----------|-----------|-----------|-----------------------------------------|
| `lat`     | 37.5      | 49.9      | Clamped to range                        |
| `lon`     | -85.7     | -76.1     | Clamped to range                        |
| `alt`     | 94.6 km   | 500 km    | Clamped to range                        |
| `slt`     | 0.0 hrs   | 24 hrs    | Clamped to range                        |
| `doy`     | 0         | 364       | Wrapped (e.g., `365 → 0`, `-1 → 364`)   |
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