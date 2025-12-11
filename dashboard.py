#!/usr/bin/env python3
"""
Interactive Streamlit Dashboard — single file (dashboard.py)

Features:
- Load CSV (default: /mnt/data/stocks.csv) or upload
- Data validation + preview
- Interactive per-ticker time series (candlestick or line)
- Moving averages (7/21/50), volatility (21-day)
- Correlation heatmap across tickers
- Simple ML: next-day Close predictions (Linear Regression, RandomForest)
- Export sample predictions / metrics

Run:
    streamlit run dashboard.py -- --csv /mnt/data/stocks.csv

Dependencies:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
"""

import argparse
import sys
import io
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Page config
st.set_page_config(page_title="Stock Market Dashboard", layout="wide", initial_sidebar_state="expanded")
sns.set_style("whitegrid")

# Default path to the uploaded CSV (provided earlier)
DEFAULT_CSV_PATH = "/mnt/data/stocks.csv"

# -------------------------
# Utility / Loading
# -------------------------
@st.cache_data
def try_load_csv_from_path(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

@st.cache_data
def load_dataframe(buffer) -> pd.DataFrame:
    # buffer: file path or file-like
    if isinstance(buffer, str):
        df = pd.read_csv(buffer)
    else:
        # file-like (UploadedFile)
        buffer.seek(0)
        df = pd.read_csv(buffer)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def validate_and_prepare(df: pd.DataFrame) -> (bool, str, pd.DataFrame):
    """Check for required columns, convert Date to datetime, fill/rename typical columns."""
    required = ["Date", "Close"]
    # Ticker sometimes absent (single ticker file); create Ticker if missing
    if "Ticker" not in df.columns:
        # if file has Symbol or symbol
        for alt in ("Symbol", "symbol", "TickerSymbol"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Ticker"})
                break
        else:
            # create single ticker column using filename placeholder
            df["Ticker"] = "TICKER"
    # ensure required columns exist
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required column(s): {missing}", df

    # parse Date
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        return False, f"Could not parse 'Date' column to datetime: {e}", df

    # If Adj Close not present, create from Close (safe)
    if "Adj Close" not in df.columns and "AdjClose" not in df.columns:
        df["Adj Close"] = df["Close"]

    # safe numeric conversions
    for col in ["Close", "Open", "High", "Low", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return True, "OK", df

# -------------------------
# Indicators (cached)
# -------------------------
@st.cache_data
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # moving averages by ticker
    for w in (7, 21, 50):
        out[f"ma_{w}"] = out.groupby("Ticker")["Close"].transform(lambda s: s.rolling(window=w, min_periods=1).mean())
    # returns & vol
    out["ret"] = out.groupby("Ticker")["Close"].pct_change()
    out["vol_21"] = out.groupby("Ticker")["ret"].transform(lambda s: s.rolling(window=21, min_periods=1).std())
    return out

# -------------------------
# ML feature engineering
# -------------------------
@st.cache_data
def create_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy().sort_values(["Ticker", "Date"])
    for lag in (1, 2, 3, 5):
        df2[f"lag_{lag}"] = df2.groupby("Ticker")["Close"].shift(lag)
    df2["day_of_week"] = df2["Date"].dt.weekday
    df2["target_next_close"] = df2.groupby("Ticker")["Close"].shift(-1)
    df2 = df2.dropna(subset=["target_next_close", "lag_1"])
    return df2

# -------------------------
# ML training (safe)
# -------------------------
def train_and_evaluate(df_feat: pd.DataFrame, sample_size:int=10000):
    features = ["lag_1","lag_2","lag_3","lag_5","ma_7","ma_21","vol_21","Volume","day_of_week"]
    # drop rows missing features
    dff = df_feat.dropna(subset=features + ["target_next_close"])
    if len(dff) < 20:
        return None, "Not enough data to train models (need >=20 rows after feature construction)."

    # optionally limit sample size for speed
    if sample_size and len(dff) > sample_size:
        dff = dff.tail(sample_size).copy()

    X = dff[features].values
    y = dff["target_next_close"].values

    # time-aware split: last 20% as test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # models
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    def metrics(y_t, y_p):
        return {
            "mse": float(mean_squared_error(y_t, y_p)),
            "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "r2": float(r2_score(y_t, y_p))
        }

    results = {
        "lr_metrics": metrics(y_test, pred_lr),
        "rf_metrics": metrics(y_test, pred_rf),
        "y_test": y_test,
        "pred_lr": pred_lr,
        "pred_rf": pred_rf,
        "model_lr": lr,
        "model_rf": rf
    }
    return results, "OK"

# -------------------------
# Plot helpers
# -------------------------
def plot_line_with_ma(df_ticker: pd.DataFrame, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ticker["Date"], y=df_ticker["Close"], mode="lines", name="Close", line=dict(color="black")))
    for ma in ("ma_7","ma_21","ma_50"):
        if ma in df_ticker.columns:
            fig.add_trace(go.Scatter(x=df_ticker["Date"], y=df_ticker[ma], mode="lines", name=ma))
    fig.update_layout(title=f"{ticker} — Close with Moving Averages", xaxis_title="Date", yaxis_title="Price", height=400)
    return fig

def plot_candlestick(df_ticker: pd.DataFrame, ticker: str):
    # require OHLC
    if not set(["Open","High","Low","Close"]).issubset(df_ticker.columns):
        # fallback to line chart if OHLC missing
        return plot_line_with_ma(df_ticker, ticker)
    fig = go.Figure(data=[go.Candlestick(
        x=df_ticker["Date"],
        open=df_ticker["Open"],
        high=df_ticker["High"],
        low=df_ticker["Low"],
        close=df_ticker["Close"],
        name=ticker
    )])
    # overlay MA7, MA21 if present
    for ma in ("ma_7","ma_21"):
        if ma in df_ticker.columns:
            fig.add_trace(go.Scatter(x=df_ticker["Date"], y=df_ticker[ma], mode="lines", name=ma))
    fig.update_layout(title=f"{ticker} — Candlestick", xaxis_title="Date", height=500)
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    # pivot Adj Close or Close
    valcol = "Adj Close" if "Adj Close" in df.columns else "Close"
    pivot = df.pivot_table(index="Date", columns="Ticker", values=valcol)
    corr = pivot.corr().fillna(0)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", origin="lower")
    fig.update_layout(title="Correlation between Tickers", height=480)
    return fig, corr

# -------------------------
# Main App
# -------------------------
def app_main(default_csv_path: Optional[str] = None):
    st.title("📈 Stock Market Dashboard (Interactive)")

    # Sidebar controls
    st.sidebar.header("Data / Controls")

    # Load either default path, uploaded file, or file via CLI arg
    cli_csv = None
    # parse known args (if run with -- --csv <path>)
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--csv", required=False, help="Path to CSV file")
        args, unknown = parser.parse_known_args()
        if args.csv:
            cli_csv = args.csv
    except Exception:
        cli_csv = None

    csv_path_used = None
    # show default / CLI info
    st.sidebar.write("Data source options:")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional, overrides default)", type=["csv"])
    use_default = st.sidebar.checkbox(f"Use default dataset at `{default_csv_path}`", value=True if default_csv_path else False)
    if uploaded is not None:
        try:
            df = load_dataframe(uploaded)
            csv_path_used = "<uploaded>"
        except Exception as e:
            st.sidebar.error(f"Could not read uploaded CSV: {e}")
            return
    else:
        # attempt CLI CSV first, then provided default path
        if cli_csv:
            df_try = try_load_csv_from_path(cli_csv)
            if df_try is not None:
                df = load_dataframe(cli_csv)
                csv_path_used = cli_csv
            else:
                st.sidebar.warning(f"CLI CSV path `{cli_csv}` not found or couldn't be read.")
                df = None
        elif use_default and default_csv_path:
            df_try = try_load_csv_from_path(default_csv_path)
            if df_try is not None:
                df = load_dataframe(default_csv_path)
                csv_path_used = default_csv_path
            else:
                df = None
        else:
            df = None

    if df is None:
        st.info("No CSV loaded. Please upload a CSV or enable using the default dataset (or pass --csv).")
        return

    st.sidebar.markdown(f"**Loaded:** `{csv_path_used}`")
    valid, msg, df = validate_and_prepare(df)
    if not valid:
        st.error(msg)
        return

    # show basic stats
    tickers = list(df["Ticker"].unique())
    st.sidebar.markdown(f"Detected tickers: **{', '.join(tickers[:10])}**{'...' if len(tickers)>10 else ''}")

    # date range
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    if start_date > end_date:
        st.sidebar.error("Start date must be ≤ end date.")
        return

    # indicator toggles
    st.sidebar.subheader("Indicators & Options")
    show_candlestick = st.sidebar.checkbox("Candlestick (if OHLC present)", value=False)
    show_ma = st.sidebar.checkbox("Show Moving Averages (7/21/50)", value=True)
    show_volatility = st.sidebar.checkbox("Show 21-day Volatility", value=True)
    run_ml = st.sidebar.checkbox("Run ML predictions (add'l compute)", value=False)
    selected_ticker = st.sidebar.selectbox("Choose ticker to display", tickers)

    # subset by date
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    if df.empty:
        st.warning("No data in the chosen date range.")
        return

    # add indicators
    df_ind = add_indicators(df)

    # Layout: top: summary + preview
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader("Data preview")
        st.dataframe(df_ind.head(200))
    with col2:
        st.subheader("Summary")
        st.metric("Rows", f"{len(df_ind):,}")
        st.metric("Tickers", len(tickers))
        st.metric("Date range", f"{df_ind['Date'].min().date()} → {df_ind['Date'].max().date()}")

    # Per-ticker display
    st.markdown("---")
    st.subheader(f"{selected_ticker} — Price chart")
    df_t = df_ind[df_ind["Ticker"] == selected_ticker].copy()
    if df_t.empty:
        st.warning("Ticker has no data in the selected range.")
    else:
        # Choose view
        chart_type = st.radio("Chart type", ("Line + MA", "Candlestick (if available)"), index=0 if not show_candlestick else 1, horizontal=True)
        if chart_type == "Candlestick (if available)" and set(["Open","High","Low","Close"]).issubset(df_t.columns):
            fig = plot_candlestick(df_t, selected_ticker)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = plot_line_with_ma(df_t, selected_ticker)
            st.plotly_chart(fig, use_container_width=True)
        # show volatility optionally
        if show_volatility:
            st.subheader("Volatility (21-day rolling)")
            fig_vol = px.line(df_t, x="Date", y="vol_21", title=f"{selected_ticker} — 21-day volatility")
            st.plotly_chart(fig_vol, use_container_width=True)

    # Correlation heatmap (all tickers)
    st.markdown("---")
    st.subheader("Correlation across tickers")
    fig_corr, corr_df = plot_correlation_heatmap(df_ind)
    st.plotly_chart(fig_corr, use_container_width=True)

    # ML predictions
    st.markdown("---")
    if run_ml:
        st.subheader("ML: Next-day Close prediction (baseline)")

        df_feat = create_ml_features(df_ind)
        results, status = train_and_evaluate(df_feat)
        if results is None:
            st.warning(status)
        else:
            st.write("Linear Regression metrics:")
            st.json(results["lr_metrics"])
            st.write("Random Forest metrics:")
            st.json(results["rf_metrics"])

            # show sample of actual vs predicted (last N)
            y_test = results["y_test"]
            pred_lr = results["pred_lr"]
            pred_rf = results["pred_rf"]
            n_plot = min(200, len(y_test))
            idx = np.arange(len(y_test))[-n_plot:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test[idx], mode="lines+markers", name="Actual", line=dict(color="black")))
            fig.add_trace(go.Scatter(y=pred_lr[idx], mode="lines+markers", name="LR Pred", line=dict(color="blue")))
            fig.add_trace(go.Scatter(y=pred_rf[idx], mode="lines+markers", name="RF Pred", line=dict(color="orange")))
            fig.update_layout(title="Actual vs Predicted (test set tail)", height=420)
            st.plotly_chart(fig, use_container_width=True)

            # allow user to download metrics & sample predictions
            st.download_button("Download metrics JSON", data=str({
                "lr_metrics": results["lr_metrics"],
                "rf_metrics": results["rf_metrics"]
            }), file_name="metrics.json", mime="application/json")

            # prepare CSV of predictions
            pred_df = pd.DataFrame({
                "y_true": y_test,
                "pred_lr": pred_lr,
                "pred_rf": pred_rf
            })
            csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions (CSV)", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    st.markdown("---")
    st.caption("Built with Streamlit — interactive analysis, safe default dataset: `/mnt/data/stocks.csv`")

# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":
    # We allow passing --csv via CLI; otherwise use default
    default = DEFAULT_CSV_PATH if DEFAULT_CSV_PATH else None
    app_main(default_csv_path=default)
