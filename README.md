
# Stock Price Prediction Using RNN and LSTM

## **Project Overview**

This project demonstrates **time-series forecasting of stock prices** using **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM) networks**.
The model predicts future stock prices based on historical data for different time windows (1-day, 5-day, 10-day). It uses **hyperparameter tuning** with `GridSearchCV` and evaluates model performance using **Mean Squared Error (MSE)**.

---

## **Dataset**

* **Source:** TSLA stock data (`TSLA.csv`)
* **Columns:** `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
* **Used Column:** `Adj Close` (renamed to `Close`)
* **Data Cleaning:**

  * Checked for missing values (`None found`)
  * Removed duplicates (`None found`)
  * Converted `Date` to datetime and set as index

---

## **Libraries and Dependencies**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout

from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

---

## **Data Preprocessing**

1. Scaled data to range `[0, 1]` using `MinMaxScaler`.
2. Created sequences of data for **1-day, 5-day, and 10-day windows**.
3. Split data into **training (80%)** and **testing (20%)** sets.
4. Reshaped sequences for **RNN/LSTM input** with shape `(samples, timesteps, features)`.

---

## **Model Architecture**

### **1. SimpleRNN**

* Single RNN layer with customizable units
* Dropout layer for regularization
* Dense output layer
* Loss function: Mean Squared Error
* Optimizer: Adam

### **2. LSTM**

* Single LSTM layer with customizable units
* Dropout layer for regularization
* Dense output layer
* Loss function: Mean Squared Error
* Optimizer: Adam

---

## **Hyperparameter Tuning**

* **GridSearchCV parameters:**

  ```python
  param_grid = {
      "model__units": [50, 100],
      "model__dropout_rate": [0.2, 0.4],
      "optimizer__learning_rate": [1e-4, 1e-3],
      "fit__batch_size": [32, 64],
      "fit__epochs": [20, 40]
  }
  ```
* EarlyStopping and ModelCheckpoint callbacks used to **prevent overfitting** and **save the best model**.

---

## **Training & Evaluation**

1. Trained RNN and LSTM models for **1-day, 5-day, 10-day windows**.
2. Predicted stock prices on test sets.
3. Inverse-transformed scaled predictions back to **original stock price scale**.
4. Evaluated models using **Mean Squared Error (MSE)**.
5. Plotted **Actual vs Predicted prices**.

---

## **Sample Results**

| Model | Window | MSE (approx.) |
| ----- | ------ | ------------- |
| RNN   | 1-day  | 161.78        |
| RNN   | 5-day  | 246.71        |
| RNN   | 10-day | 249.83        |
| LSTM  | 1-day  | 147.41        |
| LSTM  | 5-day  | 342.89        |
| LSTM  | 10-day | 513.87        |

**Observation:**

* LSTM performs better for short-term (1-day) predictions.
* Longer window predictions show higher error due to increased volatility in stock prices.

---

## **How to Run**

1. Clone the repository.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow scikeras
```

3. Place the `TSLA.csv` file in the appropriate directory.
4. Run the Python script:

```bash
python stock_prediction_rnn_lstm.py
```

---

## **Visualization**

* Plots show **Actual vs Predicted** stock prices for each window and model.
* Helps to **visually assess model performance**.

---

## **Future Work**

* Include more features like `Volume`, `Open`, `High`, `Low` for **multivariate forecasting**.
* Experiment with **stacked LSTM or GRU models**.
* Use **Transformer-based time-series models** for potentially better accuracy.
* Incorporate **sentiment analysis from news or social media** for improved prediction.


