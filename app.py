import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.set_page_config(
    page_title="Proton Beam Anomaly Detector",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Real-Time Proton Beam Anomaly Detection")
st.markdown("*Simulating ML-based dose verification for proton therapy*")
st.divider()

@st.cache_resource
def load_and_train():
    X     = np.load('bragg_signals_cleaned.npy')
    y     = np.load('bragg_labels.npy')
    df    = pd.read_csv('bragg_features.csv')
    depth = np.linspace(0, 30, 300)

    feature_cols = [c for c in df.columns if c != 'label']
    X_feat  = df[feature_cols].values
    y_labels = df['label'].values

    # ── Random Forest ─────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_feat, y_labels)

    # ── Isolation Forest ──────────────────────────────────────
    iso = IsolationForest(n_estimators=100,
                          contamination='auto', random_state=42)
    iso.fit(X_feat[y_labels == 'normal'])

    # ── LSTM Autoencoder ──────────────────────────────────────
    lstm_model = keras.models.load_model('lstm_autoencoder.keras')

    # Calculate threshold from normal signals
    X_normal   = X[y == 'normal']
    X_norm_3d  = X_normal[:, ::3].reshape(-1, 100, 1)
    recon      = lstm_model.predict(X_norm_3d, verbose=0)
    errors     = np.mean(np.power(X_norm_3d - recon, 2), axis=(1, 2))
    threshold  = np.percentile(errors, 95)

    return X, y, depth, rf, iso, lstm_model, threshold, feature_cols

X, y, depth, rf_model, iso_model, lstm_model, threshold, feature_cols = load_and_train()

# ── Signal generator ──────────────────────────────────────────
def generate_bragg_curve(peak_position=15.0, peak_width=0.8,
                          amplitude=1.0, noise_level=0.02):
    entrance = 0.3 * np.exp(-0.05 * depth)
    peak     = amplitude * np.exp(
        -((depth - peak_position)**2) / (2 * peak_width**2))
    noise    = np.random.normal(0, noise_level, size=len(depth))
    signal   = np.clip(entrance + peak + noise, 0, None)
    sig_min, sig_max = signal.min(), signal.max()
    return (signal - sig_min) / (sig_max - sig_min + 1e-9)

def extract_features(signal):
    f = {}
    peak_idx = np.argmax(signal)
    f['peak_position'] = depth[peak_idx]
    f['peak_height']   = signal[peak_idx]
    half_max = f['peak_height'] / 2
    above    = signal >= half_max
    if above.sum() > 1:
        l = np.argmax(above)
        r = len(above) - np.argmax(above[::-1]) - 1
        f['peak_width_fwhm'] = depth[r] - depth[l]
    else:
        f['peak_width_fwhm'] = 0.0
    f['total_area']    = np.trapz(signal, depth)
    f['entrance_area'] = np.trapz(signal[:peak_idx], depth[:peak_idx]) if peak_idx > 0 else 0
    f['tail_area']     = np.trapz(signal[peak_idx:], depth[peak_idx:])
    f['entrance_to_peak_ratio'] = f['entrance_area'] / (f['total_area'] + 1e-9)
    f['pre_peak_slope']  = (signal[peak_idx] - signal[max(0, peak_idx-5)]) / 5
    f['post_peak_slope'] = (signal[min(len(signal)-1, peak_idx+5)] - signal[peak_idx]) / 5
    f['signal_mean'] = signal.mean()
    f['signal_std']  = signal.std()
    return np.array([f[c] for c in feature_cols])

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("🎛️ Signal Controls")
signal_type = st.sidebar.selectbox(
    "Signal Type",
    ["Normal", "Range Shift", "Amplitude Drop",
     "Scatter Artifact", "Missing Peak"]
)
add_noise = st.sidebar.slider("Noise Level", 0.01, 0.1, 0.02)
st.sidebar.divider()
st.sidebar.button("▶ Generate New Signal", type="primary")

# ── Generate signal ───────────────────────────────────────────
np.random.seed(None)

if signal_type == "Normal":
    signal = generate_bragg_curve(
        peak_position=np.random.uniform(14, 16), noise_level=add_noise)
elif signal_type == "Range Shift":
    signal = generate_bragg_curve(
        peak_position=np.random.uniform(18, 22), noise_level=add_noise)
elif signal_type == "Amplitude Drop":
    signal = generate_bragg_curve(amplitude=0.3, noise_level=add_noise)
elif signal_type == "Scatter Artifact":
    signal = generate_bragg_curve(peak_width=3.0, noise_level=add_noise)
else:
    entrance = 0.3 * np.exp(-0.05 * depth)
    signal   = np.clip(
        entrance + np.random.normal(0, add_noise, len(depth)), 0, None)
    signal   = (signal - signal.min()) / (signal.max() - signal.min() + 1e-9)

# ── Run all three models ──────────────────────────────────────
features    = extract_features(signal).reshape(1, -1)
rf_pred     = rf_model.predict(features)[0]
rf_conf     = rf_model.predict_proba(features).max() * 100
iso_pred    = "Normal" if iso_model.predict(features)[0] == 1 else "⚠️ ANOMALY"
iso_score   = iso_model.decision_function(features)[0]

signal_100  = signal[::3].reshape(1, 100, 1)
lstm_recon  = lstm_model.predict(signal_100, verbose=0)
lstm_error  = np.mean((signal_100 - lstm_recon) ** 2)
lstm_pred   = "⚠️ ANOMALY" if lstm_error > threshold else "Normal"

# ── Layout ────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Signal")
    fig, ax = plt.subplots(figsize=(9, 4))
    color   = 'steelblue' if rf_pred == 'normal' else 'tomato'
    ax.plot(depth, signal, color=color, linewidth=2)
    ax.axvline(x=15, color='gray', linestyle='--',
               alpha=0.5, label='Expected peak (15cm)')
    ax.fill_between(depth, signal, alpha=0.15, color=color)
    ax.set_xlabel('Depth in tissue (cm)')
    ax.set_ylabel('Normalized Energy')
    ax.set_title(f'Signal Type Selected: {signal_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("🤖 Model Predictions")

    # Random Forest
    st.markdown("**Random Forest** *(supervised)*")
    if rf_pred == 'normal':
        st.success(f"✅ Normal  ({rf_conf:.1f}% confidence)")
    else:
        st.error(f"🚨 {rf_pred.replace('_',' ').title()}  ({rf_conf:.1f}%)")

    st.divider()

    # Isolation Forest
    st.markdown("**Isolation Forest** *(unsupervised)*")
    if "ANOMALY" in iso_pred:
        st.error(f"{iso_pred}  (score: {iso_score:.3f})")
    else:
        st.success(f"✅ {iso_pred}  (score: {iso_score:.3f})")

    st.divider()

    # LSTM Autoencoder
    st.markdown("**LSTM Autoencoder** *(deep learning)*")
    if "ANOMALY" in lstm_pred:
        st.error(f"🚨 {lstm_pred}  (error: {lstm_error:.5f})")
    else:
        st.success(f"✅ {lstm_pred}  (error: {lstm_error:.5f})")

    st.divider()

    # Signal features
    st.markdown("**Signal Features**")
    feats = extract_features(signal)
    st.metric("Peak Position", f"{feats[0]:.2f} cm")
    st.metric("Peak Width",    f"{feats[2]:.2f} cm")
    st.metric("Entrance Ratio",f"{feats[6]:.3f}")

st.divider()
st.caption("Proton Therapy ML Research | TU Chemnitz")
