import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings


# years = sorted(df["Year"].unique())



# if "year" not in st.session_state:
#     st.session_state.year = 2017

import streamlit as st

def domestic_page(df,state ,year = 2017):
    data_year = df[df["Year"] == year]
    result = eval(
    data_year['State of Production'].values[0],
    {"array": np.array, "object": object, "__builtins__": {}}
    )
    # Handle empty or missing export/import data
    export_import_str = data_year['Yearly_Export_Import(dollar)'].values[0]
    export_import = np.fromstring(export_import_str.strip('[]'), sep=' ')
    if export_import.size == 2:
        total_export, total_import = export_import[0], export_import[1]
    else:
        total_export, total_import = 0, 0

    # Handle monthly production
    prod_str = data_year["Monthly_Production"].values[0]
    total_production = np.fromstring(prod_str.strip('[]'), sep=' ')
    # fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Import, Export {year}", f"Monthly Production {year}"))
    fig1 = go.Figure()
    fig2 = go.Figure()

    fig1.add_bar(
        x=["Import", "Export"], y=[total_import, total_export], name="Trade"
    )

    fig2.add_scatter(
            x=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
            y=total_production,
            mode="markers+text",
            name="Production",
            fill='tozeroy',
            fillcolor="rgba(100, 200, 100, 0.3)"
    )
    fig1.update_layout(title_text=f"Import & Export ({year})")
    fig2.update_layout(title_text=f"Monthly Production ({year})")
    # fig.update_yaxes(title_text="Quantity", row=1, col=1)
    # fig.update_yaxes(title_text="Quantity", row=1, col=2)
    # fig.update_layout(height=450, showlegend=False)
    fig1.update_yaxes(title_text = "Amount in Million Dollar")
    fig2.update_yaxes(title_text = "Indian Rupees")
    fig1.update_layout(height=450, showlegend=False)
    fig2.update_layout(height=450, showlegend=False)

     # ---- State-wise Production ----

    fig3 = go.Figure()
    monthly_state_production = result[state]
        
    fig3.add_scatter(
        x=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        y=monthly_state_production,
        mode="markers+text",
        name="State Production",
        fill='tozeroy',
        # fillcolor="rgba(100, 200, 100, 0.3)"
    )
    fig3.update_layout(
        title=f"Monthly Production in {state} for {year}",
        yaxis_title="Indian Rupees",
        height=450,
    )
    

    return fig1,fig2,fig3





#### Forecasting Models ####

warnings.filterwarnings('ignore')

def train_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit
def arima_residuals(series, arima_model):
    fitted = arima_model.fittedvalues
    residuals = series[len(series)-len(fitted):] - fitted
    return residuals

def create_lstm_data(data, lookback=5):
    X, y = [], []
    for i in range(len(data)-lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
def hybrid_arima_lstm(series, steps=2, lookback=5):
    warnings.filterwarnings('ignore')
    series = np.asarray(series).astype(float)

    # ---- ARIMA ----
    arima_model = train_arima(series)
    arima_forecast = arima_model.forecast(steps=steps)

    # ---- Residuals ----
    residuals = arima_residuals(series, arima_model)

    # ---- Scale residuals ----
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(residuals.reshape(-1,1))

    # ---- LSTM data ----
    X, y = create_lstm_data(residuals_scaled, lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # ---- Train LSTM ----
    lstm = build_lstm((lookback,1))
    lstm.fit(X, y, epochs=50, batch_size=8, verbose=0)

    # ---- Predict residuals ----
    last_seq = residuals_scaled[-lookback:].reshape((1, lookback, 1))
    lstm_residual_pred = []

    for _ in range(steps):
        pred = lstm.predict(last_seq, verbose=0)
        lstm_residual_pred.append(pred[0,0])
        last_seq = np.append(last_seq[:,1:,:], pred.reshape(1,1,1), axis=1)

    lstm_residual_pred = scaler.inverse_transform(
        np.array(lstm_residual_pred).reshape(-1,1)
    ).flatten()

    # ---- Final Hybrid Forecast ----
    hybrid_forecast = arima_forecast + lstm_residual_pred

    return hybrid_forecast,arima_forecast

def hybrid(series,step = 2):
    export_series = series['Export'].values
    import_series = series['Import'].values

# Example: forecast for the first mineral (index 0)
    export_pred = hybrid_arima_lstm(export_series[0], steps=step)
    import_pred = hybrid_arima_lstm(import_series[0], steps=step)
# print("Next 2-year Export Forecast:", export_pred)
# print("Next 2-year Import Forecast:", import_pred)
    return export_pred,import_pred
# hybrid(processed_data[processed_data["Mineral"]=='Antimony'],step=2)

