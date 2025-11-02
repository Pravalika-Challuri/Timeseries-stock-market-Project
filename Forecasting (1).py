# ===============================================
# 📈Forecasting_v3.py — LSTM Stock Forecast App
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Streamlit Configuration
# ----------------------------
st.set_page_config(page_title="Stock Price LSTM Forecast App", layout="wide")
st.markdown("## Forecasting Next 30 Days using LSTM 📈")
# ----------------------------
# Sidebar Upload & Controls
# ----------------------------
st.sidebar.header("📂 Upload File & Choose Model Settings")

uploaded_file = st.sidebar.file_uploader("Upload your Stock price data (CSV or Excel)", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.checkbox("Use sample data", value=False)

if uploaded_file is None and not use_sample:
    st.sidebar.info("Upload your dataset or select 'Use sample data'.")
    st.stop()

# ----------------------------
# Helper Functions
# ----------------------------
def load_table(uploaded_file):
    """Read CSV/XLSX and return DataFrame with Date index and Close column."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None

    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime", "timestamp"]), df.columns[0])
    close_col = next((c for c in df.columns if "close" in c.lower()), None)

    if close_col is None:
        st.error("❌ Could not find a 'Close' column in your data.")
        return None

    df = df[[date_col, close_col]].rename(columns={date_col: "Date", close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna(subset=["Close"])


def create_sequences(values, lookback):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def build_lstm(lookback, units=64, layers=1, dropout=0.2, lr=0.001):
    model = Sequential()
    for i in range(layers):
        return_seq = (i < layers - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq, input_shape=(lookback, 1)))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

# ----------------------------
# Load Dataset
# ----------------------------
if use_sample:
    dates = pd.bdate_range(start="2015-01-01", end="2024-01-12")
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, len(dates))) + 1000
    df = pd.DataFrame({"Close": prices}, index=dates)
else:
    df = load_table(uploaded_file)

if df is None:
    st.stop()

dataset_name = uploaded_file.name if uploaded_file else "Sample Dataset"
st.markdown(f"**Forecasting Dataset:** `{dataset_name}`")
st.markdown("---")

# ----------------------------
# Data Preview + Raw Plot
# ----------------------------
st.subheader("📂 Data Overview")

col1, col2 = st.columns([1, 3])
with col1:
    st.write("**Preview of Uploaded Data**")
    st.dataframe(df.sample(n=10,random_state=42).style.format({"Close": "{:.2f}"}))

with col2:
    st.write("**Closing Price Chart (Raw Data)**")
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close Price", line=dict(color="royalblue")))
    fig_raw.update_layout(title=dict(
            text="Closing Price Over Time",
            x=0.4,  # slightly left
            xanchor="center",
            yanchor="top",
            font=dict(size=14, color="black", family="Arial Black")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=12),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=14, color="black")),
            tickfont=dict(color="black"),
            gridcolor="lightgray",
            zerolinecolor="gray",
        ),
        yaxis=dict(
            title=dict(text="Close Price", font=dict(size=14, color="black")),
            tickfont=dict(color="black"),
            gridcolor="lightgray",
            zerolinecolor="gray",
        ),
        hovermode="x unified",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="black", size=12)
        ),
        margin=dict(l=70, r=30, t=70, b=50))
    st.plotly_chart(fig_raw, use_container_width=True)

st.markdown("---")

st.write("### Model Training And Testing ")
# ----------------------------
# Restrict to last 8 years (6y train / 2y test)
# ----------------------------
latest_date = df.index.max()
start_8yr = latest_date - pd.DateOffset(years=8)
df_recent = df.loc[df.index >= start_8yr].copy()
if len(df_recent) < 500:
    st.warning("⚠️ Not enough data for 8 years — using all available data instead.")
    df_recent = df.copy()

split_date = latest_date - pd.DateOffset(years=2)
train_df = df_recent.loc[df_recent.index < split_date]
test_df = df_recent.loc[df_recent.index >= split_date]

st.success(f"Data restricted to last 8 years: {df_recent.index.min().date()} → {df_recent.index.max().date()}")
st.info(f"Train: {train_df.index.min().date()} → {train_df.index.max().date()} ({len(train_df)} rows)-First 6 years")
st.info(f"Test: {test_df.index.min().date()} → {test_df.index.max().date()} ({len(test_df)} rows)-Last 2 years")

# ----------------------------
# Hyperparameters (Sidebar)
# ----------------------------
st.sidebar.subheader("⚙️ Choose Model Hyperparameters")
lookback = st.sidebar.number_input("Lookback (timesteps)", value=45, min_value=5, max_value=200)
units = st.sidebar.selectbox("Units", [32, 64, 128], index=1)
dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, 0.05)
lr = float(st.sidebar.text_input("Learning rate", "0.001"))
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32], index=1)
epochs = st.sidebar.number_input("Epochs", value=50, min_value=5, max_value=500)

train_button = st.sidebar.button("Train Model", use_container_width=True)
st.sidebar.markdown("<h4 style='text-align:center;'>— OR —</h4>", unsafe_allow_html=True)
uploaded_model = st.sidebar.file_uploader("Upload pre-trained model (.h5)", type=["h5"])

# ----------------------------
# Scaling
# ----------------------------
scaler = MinMaxScaler((0, 1))
scaler.fit(train_df["Close"].values.reshape(-1, 1))
train_scaled = scaler.transform(train_df["Close"].values.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_df["Close"].values.reshape(-1, 1)).flatten()
combined = pd.concat([train_df["Close"], test_df["Close"]])
combined_scaled = scaler.transform(combined.values.reshape(-1, 1)).flatten()

# ----------------------------
# Train or Load Model
# ----------------------------
model = None
if uploaded_model is not None:
    model = load_model(uploaded_model)
    st.sidebar.success("✅ Loaded pre-trained model.")
elif train_button:
    with st.spinner("⏳ Training model... please wait."):
        X_train, y_train = create_sequences(train_scaled, lookback)
        model = build_lstm(lookback, units, layers=1, dropout=dropout, lr=lr)
        es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=0, callbacks=[es])
    st.success("✅ Training complete!")

if model is None:
    st.warning("⚠️ Please train or upload a model before forecasting.")
    st.stop()

# ----------------------------
# Predict on Test Set
# ----------------------------
X_combined, y_combined = create_sequences(combined_scaled, lookback)
target_pos = np.arange(lookback, len(combined_scaled))
target_dates = combined.index[target_pos]
mask = np.isin(target_dates, test_df.index)

X_test = X_combined[mask]
y_test_scaled = y_combined[mask]
y_pred_scaled = model.predict(X_test).flatten()
y_true = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
test_dates = target_dates[mask]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.write("#### 📊 Test Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R²", f"{r2:.3f}")

st.markdown("---")

# ----------------------------
# Forecast next 30 business days
# ----------------------------
last_date = test_df.index.max()
future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=30)
window = combined_scaled[-lookback:].tolist()
future_scaled = []

for _ in range(len(future_dates)):
    x = np.array(window).reshape(1, lookback, 1)
    pred = float(model.predict(x).flatten()[0])
    future_scaled.append(pred)
    window.append(pred)
    window.pop(0)

future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
future_series = pd.Series(future, index=future_dates)
first_10 = future_series.head(10).rename("Forecast_Close").reset_index().rename(columns={"index": "Date"})

# ----------------------------
# Interactive Forecast Plot
# ----------------------------
colA, colB = st.columns([3, 1])
with colA:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df["Close"], mode="lines", name="Train", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df["Close"], mode="lines", name="Test", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=test_dates, y=y_pred, mode="lines+markers", name="Predicted", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=future_series.index, y=future_series, mode="lines+markers", name="30-Day Forecast", line=dict(color="orange")))

    ymin, ymax = float(min(df["Close"].min(), y_pred.min(), future_series.min())), float(max(df["Close"].max(), y_pred.max(), future_series.max()))
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=ymin, y1=ymax, line=dict(color="black", dash="dash"))
    fig.add_annotation(x=last_date, y=ymax, text="Test End", showarrow=False, yshift=10, font=dict(color="black"))

    fig.update_layout(
        title=dict(
            text="Next 30-Day(B) Forecast With Train / Test / Predicted.",
            x=0.4,  # slightly left
            xanchor="center",
            yanchor="top",
            font=dict(size=14, color="black", family="Arial Black")
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=12),
        xaxis=dict(
            title=dict(text="Date", font=dict(size=14, color="black")),
            tickfont=dict(color="black"),
            gridcolor="lightgray",
            zerolinecolor="gray",
        ),
        yaxis=dict(
            title=dict(text="Close Price", font=dict(size=14, color="black")),
            tickfont=dict(color="black"),
            gridcolor="lightgray",
            zerolinecolor="gray",
        ),
        hovermode="x unified",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(color="black", size=12)
        ),
        margin=dict(l=70, r=30, t=70, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)

with colB:
    st.write("##### Next 10 Days(B) Forecast")
    st.dataframe(first_10.set_index("Date").style.format("{:.2f}"))

# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Internship — Zidio development internship Program (Aug–Sep 2025)</p>", unsafe_allow_html=True)
