# NETGUARD AI

> **Multi-Signal Self-Tuning Thresholds for Streaming Anomaly Detection**  
> *Academic Machine Learning Research & Enterprise SOC Platform*

---

## 🛡️ Project Overview

**NETGUARD AI** is a production-grade cybersecurity intrusion detection and analytics platform developed for high-throughput streaming environments. It addresses the fundamental flaw of static and naive adaptive thresholds in security operations centers (SOCs):

1. **Static Thresholds Fail Under Drift:** Static thresholds ($\tau_0 = 32.7688$) cannot adapt to legitimate temporal network load fluctuations.
2. **Rolling MAD Suffers Runaway:** Unbounded rolling median + MAD thresholds ($\tau_t = \text{median}(W) + k \cdot \text{MAD}(W)$) absorb attack traffic during sustained volumetric bursts, exponentially increasing the threshold and blinding the SOC ($F_1$ score plunges to **0.4079**).
3. **The Proposed Solution (Stabilized Adaptive Threshold):**
   - **Dual-Signal Score Fusion:** Combines autoencoder reconstruction error ($e_r$) with latent bottleneck distance ($d_z$).
   - **Bounded Clamping:** Restricts dynamic threshold drift strictly within $[\tau_0 \times 0.8, \tau_0 \times 1.2]$.
   - **Exponential Smoothing:** $\tau_{t} = \alpha \cdot \tau_t^* + (1 - \alpha) \cdot \tau_{t-1}$ with $\alpha = 0.05$.
   - **Attack Exclusion:** Flagged anomaly scores are quarantined from the rolling history window.
   - Retains **0.6891 $F_1$-score** (99.5% of baseline) while reducing threshold trajectory variance from **142.30 down to 2.41**.

---

## 🎨 User Interface & Design Philosophy

The frontend is built with **React 18 + TypeScript + Vite + Tailwind CSS + Recharts**, completely decoupled from the backend and faithfully designed after the modern enterprise SOC theme:
- **Warm Eggshell/Cream Canvas (`#F5F3EC`):** Minimalist, high legibility, scientific visualization background.
- **Matte Dark Sidebar (`#131316`):** Enterprise SOC navigation with pink toggle accents and live autoencoder status indicator.
- **4 Distinct Pastel Hero Cards:**
  1. **Butter Yellow (`#FEF3C7`):** Total Ingested Flows & Sub-batch Volume Sparklines.
  2. **Rose Pink (`#FCE7F3`):** Streaming Score Curve with Recharts line graph and peak anomaly marker.
  3. **Sage Olive Green (`#DCFCE7`):** Traffic Condition breakdown (Normal: 88.3%, Warning: 8.4%, Critical: 3.3%).
  4. **Periwinkle Blue (`#DBEAFE`):** 5 Orthogonal Signal Families & 16-dimensional Latent Telemetry.
- **Interactive SOC Timeline & Calendar Rail:** Live streaming event feed with circular pastel status badges and date selector.

---

## 🚀 8 Application Pages

| # | Route | Page | Key Capabilities |
|---|---|---|---|
| 1 | `/` | **Overview** | 4 pastel hero KPI cards, live score curve with peak anomaly dot, flow list with avatar pills, detailed vector inspection card, SOC timeline. |
| 2 | `/dataset-explorer` | **Dataset Explorer** | Drag-and-drop CSV upload, missing/inf value audit, class imbalance bar chart, feature schema table with variance and summary stats, raw data preview. |
| 3 | `/traffic-analysis` | **Traffic Analysis** | 5 orthogonal signal groups tabs, 25-bin distribution histogram, Tukey outlier fences ($Q_1 - 1.5\text{IQR}$ to $Q_3 + 1.5\text{IQR}$), Pearson correlation matrix. |
| 4 | `/anomaly-detection` | **Anomaly Detection** | PyTorch Deep Autoencoder (78→128→64→16→64→128→78) inference runner, $\alpha$ dual-signal slider, dynamic threshold line, audited flows table, CSV export. |
| 5 | `/threshold-comparison`| **Threshold Comparison** | Live streaming simulation comparing Fixed vs Rolling MAD vs Stabilized Bounded, adversarial burst injection toggle, runaway visualization, variance bar chart. |
| 6 | `/model-performance` | **Model Performance** | 10-model official benchmark leaderboard, horizontal F1 bar chart, interactive 2×2 confusion matrix, Deep Autoencoder layer diagram. |
| 7 | `/detection-history` | **Detection History** | Full searchable and filterable incident history log with severity tags, ground-truth labels, and one-click CSV export. |
| 8 | `/about` | **About Project** | Academic research presentation, mathematical formulations, system architecture, and team contributor roster. |

---

## 📊 Experimental Results (Review 2 Benchmark)

| Rank | Model Architecture | Supervision | $F_1$-Score | Precision | Recall | Latency | Runaway Risk |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **XGBoost Classifier** | Supervised | 0.9970 | 0.9981 | 0.9959 | 1.2 ms | N/A (Static) |
| 2 | **Random Forest** | Supervised | 0.9947 | 0.9960 | 0.9934 | 3.8 ms | N/A (Static) |
| 3 | **Decision Tree (CART)** | Supervised | 0.9906 | 0.9912 | 0.9900 | 0.4 ms | N/A (Static) |
| 4 | **Logistic Regression** | Linear Supervised | 0.8334 | 0.8410 | 0.8260 | 0.2 ms | N/A (Static) |
| 5 | **Standard Autoencoder** | Unsupervised | 0.6937 | 0.7102 | 0.6780 | 2.1 ms | N/A (Static) |
| 6 | **Multi-Signal AE (Fixed $\tau$)** | Unsupervised | 0.6924 | 0.7090 | 0.6765 | 2.4 ms | Low |
| 7 | **Multi-Signal AE (Stabilized)** | **Streaming Unsupervised (Ours)** | **0.6891** | **0.7050** | **0.6740** | **2.5 ms** | **Mitigated (Stable)** |
| 8 | **Isolation Forest** | Unsupervised | 0.5997 | 0.6210 | 0.5800 | 4.5 ms | N/A (Static) |
| 9 | **Multi-Signal AE (Original Adaptive)**| Streaming Adaptive | 0.4079 | 0.4320 | 0.3860 | 2.5 ms | **CRITICAL (Runaway)** |
| 10 | **One-Class SVM** | Kernel Unsupervised | 0.3850 | 0.4120 | 0.3610 | 18.2 ms | N/A (Static) |

---

## ⚡ Execution Instructions

### Single-Command Start (Recommended)
Launch both backend and frontend concurrently:
```bash
./run.sh
```
- **React Frontend:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Backend:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Starting Backend Independently
```bash
export PYTHONPATH="/Users/safalkora/Library/Python/3.9/lib/python/site-packages:$PWD"
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Starting Frontend Independently
```bash
cd frontend
export PATH="$HOME/.local/node/bin:$PATH"
npm run dev -- --port 5173
```

---

## 👥 Project Team
- **Pravar** — Full-Stack Architecture, Design Systems & React UI
- **Ishan** — Machine Learning, PyTorch Autoencoder Modeling
- **Attharv** — Streaming Analytics & Threshold Stabilization Formulations
- **Samruddha** — Preprocessing Pipeline, Collinearity Audit & Evaluation Benchmarks
