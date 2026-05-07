# ⚡ Real-Time Proton Beam Anomaly Detection

> End-to-end machine learning system for detecting anomalies in proton therapy
> dose-deposition signals combining supervised, unsupervised, and deep learning
> approaches with a live interactive dashboard.

---

## 🎯 Project Overview

In proton therapy, a beam of protons is directed at a tumor with extreme precision.
The beam's energy deposits in a characteristic shape called a **Bragg Peak**.
Any deviation a shifted peak, reduced amplitude, or scattered beam can mean
the tumor receives insufficient dose or healthy tissue is damaged.

This project builds a real-time ML monitoring pipeline that:
- Generates 6,000 physics-grounded synthetic Bragg peak signals
- Detects 4 clinically relevant beam delivery anomalies
- Compares three ML approaches: supervised, unsupervised, and deep learning
- Visualizes predictions live in an interactive Streamlit dashboard

---

## 🚨 Anomaly Types Detected

| Anomaly | Description | Clinical Risk |
|---|---|---|
| **Range Shift** | Peak displaced deeper than planned | Healthy tissue damage |
| **Amplitude Drop** | Beam energy too low | Insufficient tumor dose |
| **Scatter Artifact** | Beam spreading in tissue | Loss of beam precision |
| **Missing Peak** | Complete beam delivery failure | No therapeutic dose |

---

## 🖥️ Live Dashboard

The Streamlit dashboard simulates real-time beam monitoring.
Select signal type and noise level from the sidebar, all three
models respond instantly.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 ML Pipeline
Phase 1  →  Data Generation         (6,000 physics-grounded Bragg signals)
Phase 2  →  Cleaning & Normalization (clip negatives, min-max per signal)
Phase 3  →  Feature Engineering     (11 domain-specific signal features)
Phase 4  →  Model Training          (Random Forest + Isolation Forest)
Phase 4b →  LSTM Autoencoder        (sequence reconstruction approach)
Phase 5  →  Streamlit Dashboard     (live 3-model prediction interface)
---

## 🤖 Three-Model Architecture

### 1. Random Forest *(Supervised)*
- Trained on all 5 labeled signal classes
- Identifies exact anomaly type with class probabilities
- Test accuracy: 100% on synthetic test set

### 2. Isolation Forest *(Unsupervised)*
- Trained on normal signals only — no anomaly labels needed
- Assigns anomaly score per signal (negative = anomalous)
- Simulates real deployment scenario where labeled anomalies
  are unavailable
- Anomaly recall: 100%

### 3. LSTM Autoencoder *(Deep Learning)*
- Sequence-to-sequence encoder-decoder trained on normal signals
- Learns to reconstruct normal Bragg peak patterns
- Anomalous signals produce high reconstruction error
- Threshold set at 95th percentile of normal signal errors
- Anomaly recall: 100%
- Framework: TensorFlow 2.16

---

## 📊 Results Summary

| Model | Type | Anomaly Recall | Identifies Type? |
|---|---|---|---|
| Random Forest | Supervised | 100% | ✅ Yes |
| Isolation Forest | Unsupervised | 100% | ❌ Binary only |
| LSTM Autoencoder | Deep Learning | 100% | ❌ Binary only |

> *Evaluated on synthetic test set. Results reflect clean signal
> separation in generated data. Real clinical signals would
> present additional noise and overlap challenges.*

---

## 🔬 Feature Engineering

11 domain-specific features extracted from each raw signal:

| Feature | Description |
|---|---|
| `peak_position` | Depth at maximum energy (cm) |
| `peak_height` | Normalized peak amplitude |
| `peak_width_fwhm` | Full width at half maximum (cm) |
| `total_area` | Total energy delivered |
| `entrance_area` | Energy before peak region |
| `tail_area` | Energy after peak region |
| `entrance_to_peak_ratio` | Energy distribution ratio |
| `pre_peak_slope` | Signal rise rate before peak |
| `post_peak_slope` | Signal fall rate after peak |
| `signal_mean` | Overall signal average |
| `signal_std` | Signal variability |

**Top 3 features by Random Forest importance:**
1. `peak_width_fwhm` — beam sharpness
2. `peak_position` — beam range in tissue
3. `entrance_to_peak_ratio` — energy distribution

---

## 📁 Project Structure
bragg-anomaly-detection/
│
├── app.py                           # Streamlit live dashboard
│
├── phase1_data_generation.ipynb     # Signal generation
├── phase2_cleaning.ipynb            # Preprocessing & normalization
├── phase3_features.ipynb            # Feature engineering
├── phase4_models.ipynb              # Random Forest & Isolation Forest
├── phase4b_lstm_autoencoder.ipynb   # LSTM Autoencoder (TensorFlow)
│
├── bragg_signals_cleaned.npy        # Processed signal dataset
├── bragg_features.csv               # Engineered feature matrix
├── bragg_labels.npy                 # Signal class labels
├── lstm_autoencoder.keras           # Saved LSTM model weights
├── project_summary.png              # Summary visualization
│
└── requirements.txt

---

## 🛠️ Technical Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| TensorFlow 2.16 | LSTM Autoencoder |
| Scikit-learn | Random Forest, Isolation Forest |
| NumPy / SciPy | Signal generation & processing |
| Pandas | Feature dataset management |
| Matplotlib | Visualization |
| Streamlit | Live dashboard |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/maheshvannavada/bragg-anomaly-detection.git
cd bragg-anomaly-detection

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

---

## 📚 Domain Background

The **Bragg peak** is the fundamental physics signature of proton therapy —
energy deposition rises sharply at the end of the proton range, enabling
precise tumor targeting while sparing surrounding healthy tissue.

Signal modelling is based on a simplified Bethe-Bloch energy loss
approximation, producing four detector-realistic anomaly patterns
corresponding to real clinical beam delivery failure modes.


