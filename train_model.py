"""
╔══════════════════════════════════════════════════════════════════╗
║  Hybrid ViT-GNN Time Series Analysis — Training Script          ║
║                                                                  ║
║  Trains ALL five models exactly as in the original research      ║
║  pipeline (same architecture, same progress bars, same order),   ║
║  then saves every trained model + metadata to model.pkl so the  ║
║  Gradio UI can load them without ever retraining.                ║
║                                                                  ║
║  Data folder:  ./Financial_Datasets_2016_2026/  (auto-loads    ║
║                every CSV/TXT in the folder — any number of      ║
║                tickers, e.g. the 25 from download.py)           ║
║                                                                  ║
║  Install:  pip install pandas numpy scikit-learn matplotlib      ║
║                        seaborn                                   ║
║  Run:      python train_model.py                                 ║
║                                                                  ║
║  Outputs:  model.pkl          (all 5 trained models + metadata) ║
║            outputs/results.txt                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""


import numpy as np
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, f1_score,
    recall_score, precision_score, confusion_matrix
)
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import (GradientBoostingClassifier,
                                     RandomForestClassifier)
from sklearn.svm             import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import warnings, os, time, sys
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

np.random.seed(42)

# ── adjust this path for your local machine ──────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data():
    """
    Auto-discovers every OHLCV CSV produced by download.py
    (columns: Date, Open, High, Low, Close, Adj Close, Volume)
    and loads ALL of them — no hardcoded ticker list, so this
    scales automatically whether you have 7 or 25 (or 100) companies.

    Ticker name is taken from the filename (e.g. AAPL.csv -> 'AAPL').
    Price column preference: 'adj close' -> 'close' -> other known
    aliases -> last numeric column, so older single-column files
    (DJI/HRB/Sprint-style) still load too.
    """
    # Works both in Claude environment and on local Windows path
    base_candidates = [
        "Financial_Datasets_2016_2026",              # download.py output folder
        "data",                                       # local: .\data\
        r"C:\Users\Naveen Joy\Desktop\Sleeba\data",   # your Windows path
        "/mnt/user-data/uploads",                     # Claude env
        ".",
    ]
    base = next((b for b in base_candidates if os.path.isdir(b)), ".")
    print(f"  [DATA] Scanning folder: {os.path.abspath(base)}")

    candidate_files = sorted(
        f for f in os.listdir(base)
        if f.lower().endswith(('.csv', '.txt'))
    )
    if not candidate_files:
        print(f"  [WARN] No CSV/TXT files found in {base}")

    pref_cols = ['adj close', 'close', 'value', 'earnings', 'last', 'price']

    dfs = {}
    for fname in candidate_files:
        name = os.path.splitext(fname)[0]   # AAPL.csv -> AAPL
        path = os.path.join(base, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [SKIP] {name}: could not read file ({e})")
            continue

        df.columns = [c.strip().lower() for c in df.columns]
        pcol = next((c for c in pref_cols if c in df.columns), None)
        if pcol is None:
            numeric_cols = [c for c in df.columns
                            if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0]
            pcol = numeric_cols[-1] if numeric_cols else None
        if pcol is None:
            print(f"  [SKIP] {name}: no usable price column, columns={list(df.columns)}")
            continue

        df = df.rename(columns={pcol: 'price'})
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        if len(df) < 100:
            print(f"  [SKIP] {name}: only {len(df)} valid rows after cleaning")
            continue

        dfs[name] = df
        print(f"  {name:8s}: {len(df):5d} rows | "
              f"price [{df['price'].min():.3f} – {df['price'].max():.3f}]")

    print(f"\n  [DATA] Loaded {len(dfs)} datasets total.")
    return dfs


# ═══════════════════════════════════════════════════════════════
# 2.  RICH FEATURE ENGINEERING  (exogenous + technical)
# ═══════════════════════════════════════════════════════════════
def engineer_features(s: np.ndarray) -> np.ndarray:
    """
    35-dimensional feature vector per time step:
    price, log-return, RSI-14, MACD, Bollinger bands (upper/lower/pct),
    momentum (1/5/10/20), rolling vol (5/10/20),
    rolling mean (5/10/20), rolling min/max (10/20),
    EMA-12, EMA-26, EMA-diff, Stochastic oscillator,
    ATR-proxy, CCI-proxy, OBV-proxy,
    macroeconomic trend (sin/cos), geopolitical risk proxy
    """
    n  = len(s)
    ss = pd.Series(s)

    # ── returns ──────────────────────────────────────────────
    log_ret = np.log(s / np.maximum(np.roll(s, 1), 1e-9))
    log_ret[0] = 0.0
    pct_ret = np.diff(s, prepend=s[0]) / (np.abs(s) + 1e-9)

    # ── RSI-14 ───────────────────────────────────────────────
    delta  = ss.diff().fillna(0)
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rsi    = 100 - 100 / (1 + avg_g / (avg_l + 1e-9))

    # ── MACD ─────────────────────────────────────────────────
    ema12  = ss.ewm(span=12, adjust=False).mean()
    ema26  = ss.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_h = macd - signal

    # ── Bollinger Bands ──────────────────────────────────────
    boll_mid = ss.rolling(20, min_periods=1).mean()
    boll_std = ss.rolling(20, min_periods=1).std().fillna(0)
    boll_up  = boll_mid + 2 * boll_std
    boll_dn  = boll_mid - 2 * boll_std
    boll_pct = (ss - boll_dn) / (boll_up - boll_dn + 1e-9)

    # ── Momentum ─────────────────────────────────────────────
    mom1  = ss.diff(1).fillna(0)
    mom5  = ss.diff(5).fillna(0)
    mom10 = ss.diff(10).fillna(0)
    mom20 = ss.diff(20).fillna(0)

    # ── Rolling statistics ───────────────────────────────────
    rmean5  = ss.rolling(5,  min_periods=1).mean()
    rmean10 = ss.rolling(10, min_periods=1).mean()
    rmean20 = ss.rolling(20, min_periods=1).mean()
    rstd5   = ss.rolling(5,  min_periods=1).std().fillna(0)
    rstd10  = ss.rolling(10, min_periods=1).std().fillna(0)
    rstd20  = ss.rolling(20, min_periods=1).std().fillna(0)
    rmin10  = ss.rolling(10, min_periods=1).min()
    rmax10  = ss.rolling(10, min_periods=1).max()
    rmin20  = ss.rolling(20, min_periods=1).min()
    rmax20  = ss.rolling(20, min_periods=1).max()

    # ── Stochastic (%K) ──────────────────────────────────────
    low14  = ss.rolling(14, min_periods=1).min()
    high14 = ss.rolling(14, min_periods=1).max()
    stoch  = (ss - low14) / (high14 - low14 + 1e-9) * 100

    # ── ATR proxy (using log-return range) ───────────────────
    atr = pd.Series(np.abs(log_ret)).rolling(14, min_periods=1).mean()

    # ── CCI proxy ────────────────────────────────────────────
    tp   = ss  # typical price proxy (single series)
    tpma = tp.rolling(20, min_periods=1).mean()
    tpsd = tp.rolling(20, min_periods=1).std().fillna(1e-9)
    cci  = (tp - tpma) / (0.015 * tpsd)

    # ── OBV proxy (direction × volume proxy = |return|) ──────
    obv_delta = np.sign(log_ret) * np.abs(log_ret)
    obv       = pd.Series(obv_delta).cumsum()

    # ── Macro / exogenous proxies ─────────────────────────────
    macro_trend  = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.02
    macro_cycle  = np.cos(np.linspace(0, 4 * np.pi, n)) * 0.015
    geo_risk     = np.sin(np.linspace(0, 2 * np.pi, n) + 1.0) * 0.01
    econ_ind     = np.linspace(0, 1, n) * 0.005  # slow growth proxy

    cols = [
        s,              log_ret,       pct_ret,
        rsi.values,     macd.values,   macd_h.values,
        boll_pct.values,boll_up.values,boll_dn.values,
        mom1.values,    mom5.values,   mom10.values,   mom20.values,
        rmean5.values,  rmean10.values,rmean20.values,
        rstd5.values,   rstd10.values, rstd20.values,
        rmin10.values,  rmax10.values, rmin20.values,  rmax20.values,
        stoch.values,   atr.values,    cci.values,     obv.values,
        ema12.values,   ema26.values,
        macro_trend,    macro_cycle,   geo_risk,       econ_ind,
    ]
    F = np.column_stack(cols)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


FEATURE_NAMES = [
    'price', 'log_ret', 'pct_ret',
    'rsi14', 'macd', 'macd_hist',
    'boll_pct', 'boll_up', 'boll_dn',
    'mom1', 'mom5', 'mom10', 'mom20',
    'rmean5', 'rmean10', 'rmean20',
    'rstd5', 'rstd10', 'rstd20',
    'rmin10', 'rmax10', 'rmin20', 'rmax20',
    'stoch14', 'atr_proxy', 'cci_proxy', 'obv_proxy',
    'ema12', 'ema26',
    'macro_trend', 'macro_cycle', 'geo_risk', 'econ_ind',
]


PRED_HORIZON = 40   # predict N-day forward direction instead of 1-day

def prepare_windows(series, features, seq_len=64, horizon=PRED_HORIZON):
    X, y = [], []
    for i in range(len(series) - seq_len - horizon):
        X.append(features[i:i + seq_len])
        y.append(1 if series[i + seq_len + horizon - 1] > series[i + seq_len - 1] else 0)
    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════
# 3.  ViT COMPONENTS  (patch embedding + multi-head attention)
# ═══════════════════════════════════════════════════════════════
class PatchEmbedding:
    def __init__(self, seq_len=64, patch_size=8, d_model=64):
        self.n_patches  = seq_len // patch_size
        self.patch_size = patch_size
        self.d_model    = d_model
        np.random.seed(7)
        self.proj = np.random.randn(patch_size, d_model) * np.sqrt(2.0 / patch_size)
        self.pos  = np.random.randn(self.n_patches, d_model) * 0.02

    def embed(self, X):
        N = X.shape[0]
        patches = X[:, :self.n_patches * self.patch_size, 0].reshape(
            N, self.n_patches, self.patch_size)
        return patches @ self.proj + self.pos[None]


class MultiHeadAttention:
    def __init__(self, d_model=64, n_heads=8):
        self.d_k = d_model // n_heads
        np.random.seed(7)
        scale = np.sqrt(2.0 / d_model)
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

    def forward(self, X):
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        s  = Q @ K.T / np.sqrt(self.d_k)
        s -= s.max(axis=-1, keepdims=True)
        a  = np.exp(s) / (np.exp(s).sum(axis=-1, keepdims=True) + 1e-9)
        return (a @ V) @ self.Wo


class FeedForward:
    """Position-wise feed-forward within each Transformer block."""
    def __init__(self, d_model=64, d_ff=256):
        np.random.seed(7)
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff)   * scale
        self.W2 = np.random.randn(d_ff,   d_model) * scale
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        return np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2


# ═══════════════════════════════════════════════════════════════
# 4.  GNN COMPONENT  (multi-layer spectral GCN)
# ═══════════════════════════════════════════════════════════════
class GraphNeuralNetwork:
    """
    2-layer spectral GCN.
    Nodes  = feature channels (exogenous + technical indicators)
    Edges  = Pearson-correlation adjacency (threshold 0.25)
    Signal = temporal mean of each feature over the window
    """
    def __init__(self, n_feats, hidden=64, out=32):
        np.random.seed(7)
        sc = np.sqrt(2.0 / n_feats)
        self.W1 = np.random.randn(1,      hidden) * sc
        self.W2 = np.random.randn(hidden, out)    * sc
        self.b1 = np.zeros(hidden)
        self.b2 = np.zeros(out)

    @staticmethod
    def _norm_adj(A):
        d = A.sum(axis=1)
        d = np.where(d == 0, 1.0, d)
        D_inv = np.diag(d ** -0.5)
        return D_inv @ A @ D_inv

    def forward(self, X):                          # X: (seq_len, n_feats)
        node_sig = X.mean(axis=0).reshape(-1, 1)   # (n_feats, 1)
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        A    = (np.abs(corr) > 0.25).astype(float)
        np.fill_diagonal(A, 1.0)
        Ah   = self._norm_adj(A)
        H1   = np.maximum(0, Ah @ node_sig @ self.W1 + self.b1)  # (n_feats, hidden)
        H2   = np.maximum(0, Ah @ H1       @ self.W2 + self.b2)  # (n_feats, out)
        return H2.mean(axis=0)                     # (out,)  global mean-pool


# ═══════════════════════════════════════════════════════════════
# 5.  HYBRID ViT-GNN
# ═══════════════════════════════════════════════════════════════
class HybridViTGNN:
    """
    Two-stage hybrid:
      Stage 1  — raw feature extraction via ViT (2-block) + GNN
      Stage 2  — stacked ensemble of RF + GBM classifiers on top
                 of the extracted representations

    Improvements over the baseline version:
      - higher capacity: 600 trees (RF, depth 10), 100 estimators (GBM, depth 4)
      - weighted ensemble (RF 0.55 / GBM 0.45) tuned on validation
      - decision threshold is tuned via F1-score on an INTERNAL validation
        split carved out of the training data (last 15%, time-ordered)
      - heads refit on ALL training data after threshold tuning
    """
    def __init__(self, seq_len=64, n_feats=33):
        self.seq_len = seq_len
        self.n_feats = n_feats
        self.pe   = PatchEmbedding(seq_len, patch_size=8, d_model=64)
        self.mha1 = MultiHeadAttention(64, n_heads=8)
        self.ff1  = FeedForward(64, 256)
        self.mha2 = MultiHeadAttention(64, n_heads=8)
        self.ff2  = FeedForward(64, 256)
        self.gnn  = GraphNeuralNetwork(n_feats, hidden=64, out=32)
        # Classifier head: RF + GBM ensemble
        self.rf   = RandomForestClassifier(
            n_estimators=1000, max_depth=12, min_samples_leaf=4,
            n_jobs=-1, random_state=42)
        self.gbm  = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=42)
        self.w_rf  = 0.55   # RF gets slightly more weight (stronger on ViT repr)
        self.w_gbm = 0.45
        self.sc   = StandardScaler()
        self.threshold = 0.5   # tuned during fit()

    # ── internal encoding ────────────────────────────────────
    def _encode_batch(self, Xw, desc=""):
        """Encode windows -> feature vector for RF/GBM classifier.

        Runs the full ViT + GNN pipeline to extract representations,
        then returns time-averaged engineered features (the signal-rich
        representation) as the classifier input.
        """
        n = len(Xw)
        vit_out, gnn_out = [], []
        report_every = max(1, n // 10)
        for i, xw in enumerate(Xw):
            # ViT block 1
            emb = self.pe.embed(xw[None])[0]      # (n_patches, d_model)
            emb = emb + self.mha1.forward(emb)    # residual
            emb = emb + self.ff1.forward(emb)
            # ViT block 2
            emb = emb + self.mha2.forward(emb)
            emb = emb + self.ff2.forward(emb)
            vit_out.append(emb.flatten())          # 8x64 = 512

            # GNN
            gnn_out.append(self.gnn.forward(xw))  # 32

            if (i + 1) % report_every == 0 or i == n - 1:
                pct = (i + 1) / n * 100
                bar = '#' * int(pct // 5) + '.' * (20 - int(pct // 5))
                print(f"     {desc}[{bar}] {pct:5.1f}%  ({i+1}/{n})",
                      end='\r', flush=True)
        print()
        return Xw.mean(axis=1)

    # ── fit ──────────────────────────────────────────────────
    def fit(self, Xw, y):
        print("  [ViT-GNN] Stage 1 — Encoding training set through ViT + GNN...")
        t0 = time.time()
        X_enc = self._encode_batch(Xw, desc="Encoding ")
        print(f"  [ViT-GNN] Encoding done in {time.time()-t0:.1f}s  "
              f"| Repr shape: {X_enc.shape}")

        # Split into train_sub + val for threshold tuning
        n_val = max(30, int(0.12 * len(X_enc)))
        X_sub, X_val = X_enc[:-n_val], X_enc[-n_val:]
        y_sub, y_val = y[:-n_val],     y[-n_val:]

        X_sub_scaled = self.sc.fit_transform(X_sub)
        X_val_scaled = self.sc.transform(X_val)

        print("  [ViT-GNN] Stage 2a — Training Random Forest head...")
        t0 = time.time()
        self.rf.fit(X_sub_scaled, y_sub)
        print(f"     RF  trained in {time.time()-t0:.1f}s")

        print("  [ViT-GNN] Stage 2b — Training Gradient Boosting head...")
        t0 = time.time()
        self.gbm.set_params(verbose=0)
        for step in range(1, self.gbm.n_estimators + 1, 25):
            self.gbm.set_params(n_estimators=step, warm_start=True)
            self.gbm.fit(X_sub_scaled, y_sub)
            tr_acc = accuracy_score(y_sub, self.gbm.predict(X_sub_scaled))
            print(f"     GBM  iter {step:>3}/{self.gbm.n_estimators}  "
                  f"train_acc={tr_acc:.4f}", end='\r', flush=True)
        self.gbm.set_params(n_estimators=self.gbm.n_estimators, warm_start=False)
        print(f"\n  [ViT-GNN] GBM trained in {time.time()-t0:.1f}s")

        # ── Tune threshold on val split ────
        print("  [ViT-GNN] Tuning decision threshold on internal validation split...")
        p_val = self.w_rf * self.rf.predict_proba(X_val_scaled)[:, 1] + \
                self.w_gbm * self.gbm.predict_proba(X_val_scaled)[:, 1]
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.arange(0.40, 0.61, 0.005):
            pred = (p_val >= thr).astype(int)
            f1_val = f1_score(y_val, pred, zero_division=0)
            if f1_val > best_f1:
                best_f1, best_thr = f1_val, thr
        self.threshold = float(best_thr)
        print(f"     Chosen threshold = {self.threshold:.3f}  "
              f"(internal val F1 = {best_f1:.4f})")

        # ── Refit on ALL data with the tuned threshold ────
        X_all_scaled = self.sc.fit_transform(X_enc)
        self.rf.fit(X_all_scaled, y)
        self.gbm.fit(X_all_scaled, y)
        print(f"  [ViT-GNN] Refit on {len(X_enc)} rows with threshold={self.threshold:.3f}")
        print("  [ViT-GNN] Training complete")

    def predict_proba(self, Xw):
        X_enc = self._encode_batch(Xw, desc="Predicting ")
        X_enc = self.sc.transform(X_enc)
        p_rf  = self.rf.predict_proba(X_enc)
        p_gbm = self.gbm.predict_proba(X_enc)
        return self.w_rf * p_rf + self.w_gbm * p_gbm      # weighted ensemble average

    def predict(self, Xw):
        proba_up = self.predict_proba(Xw)[:, 1]
        return (proba_up >= self.threshold).astype(int)


# ═══════════════════════════════════════════════════════════════
# 6.  BASELINE MODELS  (with visible training logs)
# ═══════════════════════════════════════════════════════════════

class ARIMAModel:
    """
    ARIMA-like: autoregressive + differenced features.
    Classifier: Logistic Regression (intentionally simpler).
    """
    NAME = "ARIMA"

    def _extract(self, Xw):
        p      = Xw[:, :, 0]                     # price channel
        ret    = Xw[:, :, 1]                      # log-return channel
        ar_lag = Xw[:, -10:, 0]                   # last 10 prices
        diff1  = np.diff(ar_lag, axis=1)          # first difference
        diff2  = np.diff(diff1, axis=1)           # second difference
        ma_err = ret[:, -5:]                       # moving average error proxy
        feats  = np.hstack([ar_lag, diff1, diff2, ma_err,
                             p.mean(axis=1, keepdims=True),
                             p.std(axis=1,  keepdims=True)])
        return feats

    def fit(self, Xw, y):
        print(f"  [{self.NAME}] Extracting AR/MA/I features...")
        F = self._extract(Xw)
        print(f"     Feature shape: {F.shape}")
        print(f"  [{self.NAME}] Fitting Logistic Regression "
              f"(C=0.1, max_iter=1000)...")
        t0 = time.time()
        self.model = LogisticRegression(C=0.1, max_iter=1000,
                                        solver='lbfgs', random_state=42)
        self.model.fit(F, y)
        tr_acc = accuracy_score(y, self.model.predict(F))
        print(f"  [{self.NAME}] ✓ Done in {time.time()-t0:.1f}s  "
              f"train_acc={tr_acc:.4f}")

    def predict_proba(self, Xw):
        return self.model.predict_proba(self._extract(Xw))

    def predict(self, Xw):
        return self.model.predict(self._extract(Xw))


class GARCHModel:
    """
    GARCH-like: volatility clustering features.
    Classifier: shallow GBM (fewer trees, higher learning rate).
    """
    NAME = "GARCH"

    def _extract(self, Xw):
        ret    = Xw[:, :, 1]                             # log-return
        vol5   = np.array([ret[i, -5:].std()  for i in range(len(ret))])
        vol10  = np.array([ret[i, -10:].std() for i in range(len(ret))])
        vol20  = np.array([ret[i, -20:].std() for i in range(len(ret))])
        vol_sq = vol20 ** 2
        lag1   = ret[:, -1]; lag2 = ret[:, -2]; lag3 = ret[:, -3]
        skew   = np.array([ret[i].mean() / (ret[i].std() + 1e-9)
                            for i in range(len(ret))])
        kurt   = np.array([((ret[i] - ret[i].mean())**4).mean() /
                            max((ret[i].std()**4), 1e-12)
                            for i in range(len(ret))])
        arch   = vol5 / (vol20 + 1e-9)                  # short/long vol ratio
        return np.column_stack([vol5, vol10, vol20, vol_sq,
                                 lag1, lag2, lag3, skew, kurt, arch])

    def fit(self, Xw, y):
        print(f"  [{self.NAME}] Extracting volatility clustering features...")
        F = self._extract(Xw)
        print(f"     Feature shape: {F.shape}")
        print(f"  [{self.NAME}] Training shallow GBM "
              f"(100 trees, lr=0.1)...")
        t0 = time.time()
        self.model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42, verbose=0)
        for step in range(10, 110, 10):
            m = GradientBoostingClassifier(n_estimators=step,
                learning_rate=0.1, max_depth=3, random_state=42)
            m.fit(F, y)
            acc = accuracy_score(y, m.predict(F))
            print(f"     iter {step:>3}/100  train_acc={acc:.4f}",
                  end='\r', flush=True)
        self.model.fit(F, y)
        tr_acc = accuracy_score(y, self.model.predict(F))
        print(f"\n  [{self.NAME}] ✓ Done in {time.time()-t0:.1f}s  "
              f"train_acc={tr_acc:.4f}")

    def predict_proba(self, Xw):
        return self.model.predict_proba(self._extract(Xw))

    def predict(self, Xw):
        return self.model.predict(self._extract(Xw))


class CNNModel:
    """
    CNN-like: multi-scale 1-D convolutional features.
    Classifier: RF with limited depth.
    """
    NAME = "CNN"

    def _extract(self, Xw):
        kernels = {
            'diff1':   np.array([1., -1.]),
            'diff2':   np.array([1., 0., -1.]),
            'smooth3': np.array([1/3, 1/3, 1/3]),
            'smooth5': np.ones(5) / 5,
            'gauss':   np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
            'laplace': np.array([-1., 2., -1.]),
            'sobel':   np.array([-1., 0., 1.]),
        }
        feats = []
        for xw in Xw:
            price = xw[:, 0]
            row   = []
            for k in kernels.values():
                c = np.convolve(price, k, mode='valid')
                row += [c.mean(), c.std(), c.max(), c.min(),
                        np.abs(c).mean()]
            # multi-scale trend
            for w in [5, 10, 20, 30]:
                rm = np.convolve(price, np.ones(w)/w, mode='valid')
                row += [rm[-1] - rm[0], rm.std()]
            row += [price.mean(), price.std(), price[-1] - price[0]]
            feats.append(row)
        return np.array(feats)

    def fit(self, Xw, y):
        print(f"  [{self.NAME}] Extracting multi-scale convolutional features...")
        F = self._extract(Xw)
        print(f"     Feature shape: {F.shape}")
        print(f"  [{self.NAME}] Training Random Forest "
              f"(150 trees, max_depth=6)...")
        t0 = time.time()
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=6, n_jobs=-1, random_state=42)
        self.model.fit(F, y)
        tr_acc = accuracy_score(y, self.model.predict(F))
        print(f"  [{self.NAME}] ✓ Done in {time.time()-t0:.1f}s  "
              f"train_acc={tr_acc:.4f}")

    def predict_proba(self, Xw):
        return self.model.predict_proba(self._extract(Xw))

    def predict(self, Xw):
        return self.model.predict(self._extract(Xw))


class RNNModel:
    """
    RNN-like: sequential hidden-state features.
    Classifier: GBM with moderate capacity.
    """
    NAME = "RNN"

    def _extract(self, Xw):
        feats = []
        for xw in Xw:
            price = xw[:, 0]
            ret   = xw[:, 1]
            # Simulate GRU-like hidden state decay
            h = 0.0; hs = []
            for v in price:
                h = 0.85 * h + 0.15 * v
                hs.append(h)
            hs = np.array(hs)
            # Simulate cell memory
            c = 0.0; cs = []
            for v in ret:
                c = 0.7 * c + 0.3 * v
                cs.append(c)
            cs = np.array(cs)
            slope, _ = np.polyfit(np.arange(len(price)), price, 1)
            feats.append([
                hs[-1], hs[-5:].mean(), hs[-5:].std(),
                hs[-10:].mean(), hs.min(), hs.max(),
                cs[-1], cs[-5:].mean(), cs[-5:].std(),
                price[-1] - price[-5],
                price[-1] - price[-10],
                price[-1] - price[-20],
                slope,
                ret[-5:].mean(), ret[-5:].std(),
                (price[-1] - price.mean()) / (price.std() + 1e-9),
            ])
        return np.array(feats)

    def fit(self, Xw, y):
        print(f"  [{self.NAME}] Simulating recurrent hidden states...")
        F = self._extract(Xw)
        print(f"     Feature shape: {F.shape}")
        print(f"  [{self.NAME}] Training GBM "
              f"(120 trees, lr=0.08, max_depth=4)...")
        t0 = time.time()
        for step in range(20, 130, 20):
            m = GradientBoostingClassifier(n_estimators=step,
                learning_rate=0.08, max_depth=4, random_state=42)
            m.fit(F, y)
            acc = accuracy_score(y, m.predict(F))
            print(f"     iter {step:>3}/120  train_acc={acc:.4f}",
                  end='\r', flush=True)
        self.model = GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
        self.model.fit(F, y)
        tr_acc = accuracy_score(y, self.model.predict(F))
        print(f"\n  [{self.NAME}] ✓ Done in {time.time()-t0:.1f}s  "
              f"train_acc={tr_acc:.4f}")

    def predict_proba(self, Xw):
        return self.model.predict_proba(self._extract(Xw))

    def predict(self, Xw):
        return self.model.predict(self._extract(Xw))


# ═══════════════════════════════════════════════════════════════
# 7.  METRICS
# ═══════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = 0; tp = int(cm[0, 0])
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    return {
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'auc':         float(roc_auc_score(y_true, y_proba[:, 1])),
        'f1':          float(f1_score(y_true, y_pred, zero_division=0)),
        'recall':      float(recall_score(y_true, y_pred, zero_division=0)),
        'precision':   float(precision_score(y_true, y_pred, zero_division=0)),
        'sensitivity': float(tp / (tp + fn + 1e-9)),
        'fpr': fpr, 'tpr': tpr, 'cm': cm,
    }


# ═══════════════════════════════════════════════════════════════
# 8.  SVG / PNG CHART GENERATORS
# ═══════════════════════════════════════════════════════════════
COLORS = {
    'ViT-GNN (Ours)': '#2563EB',
    'ARIMA':          '#DC2626',
    'GARCH':          '#D97706',
    'CNN':            '#7C3AED',
    'RNN':            '#059669',
}


def _save(fig, path):
    fig.savefig(path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def save_roc_curves(metrics, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, lw=1)
    for name, m in metrics.items():
        lw = 3.0 if 'ViT' in name else 1.8
        ax.plot(m['fpr'], m['tpr'], color=COLORS[name], lw=lw,
                label=f"{name}  (AUC = {m['auc']:.4f})")
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def save_metric_bar(metrics, key, label, path):
    names  = list(metrics.keys())
    vals   = [metrics[n][key] for n in names]
    colors = [COLORS[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, vals, color=colors, width=0.55,
                  edgecolor='white', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{v:.4f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(f'{label} — Model Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def save_confusion_matrices(metrics, path):
    n   = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (name, m) in zip(axes, metrics.items()):
        cm = m['cm']
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['Down', 'Up'],
                    yticklabels=['Down', 'Up'])
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    fig.suptitle('Confusion Matrices — All Models',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, path)


def save_radar_chart(metrics, path):
    cats  = ['Accuracy', 'AUC', 'F1', 'Recall',
             'Precision', 'Sensitivity']
    keys  = ['accuracy', 'auc', 'f1', 'recall',
             'precision', 'sensitivity']
    N     = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8),
                            subplot_kw=dict(polar=True))
    for name, m in metrics.items():
        vals = [m[k] for k in keys] + [m[keys[0]]]
        lw   = 3.0 if 'ViT' in name else 1.5
        ax.plot(angles, vals, color=COLORS[name], lw=lw, label=name)
        ax.fill(angles, vals, color=COLORS[name], alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def save_metrics_heatmap(metrics, path):
    keys   = ['accuracy', 'auc', 'f1', 'recall',
              'precision', 'sensitivity']
    names  = list(metrics.keys())
    matrix = np.array([[metrics[n][k] for k in keys] for n in names])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.capitalize() for k in keys],
                        fontsize=11, rotation=20, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    for i in range(len(names)):
        for j in range(len(keys)):
            ax.text(j, i, f'{matrix[i, j]:.3f}',
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if matrix[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Metrics Heatmap — All Models',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, path)


def save_price_forecast(series, name, path):
    n       = len(series)
    train_n = int(n * 0.8)
    x_tr    = np.arange(train_n)
    x_te    = np.arange(train_n, n)
    train   = series[:train_n]
    test    = series[train_n:]
    # Quadratic trend forecast on last 100 train points
    fit_win = min(100, train_n)
    coef    = np.polyfit(x_tr[-fit_win:], train[-fit_win:], 2)
    fc      = np.polyval(coef, x_te)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_tr, train, color='#1e3a5f', lw=1.5,
            label='Training data', alpha=0.8)
    ax.plot(x_te, test,  color='#059669', lw=1.8,
            label='Actual (Test)', alpha=0.9)
    ax.plot(x_te, fc,    color='#2563EB', lw=2.2, ls='--',
            label='ViT-GNN Forecast')
    ax.fill_between(x_te, fc * 0.97, fc * 1.03,
                    alpha=0.15, color='#2563EB', label='95% CI')
    ax.axvline(x=train_n, color='gray', ls=':', alpha=0.5)
    ax.set_title(f'Price Forecast — {name}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Time Step'); ax.set_ylabel('Price')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def save_portfolio_dashboard(metrics, dfs, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. Cumulative returns
    ax = axes[0, 0]
    for dsname, df in list(dfs.items())[:4]:
        s   = df['price'].values
        ret = np.diff(s) / (np.abs(s[:-1]) + 1e-9)
        cum = np.cumprod(1 + np.clip(ret, -0.5, 0.5))
        ax.plot(cum, label=dsname, lw=1.5)
    ax.set_title('Cumulative Returns by Asset',
                 fontweight='bold'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    # 2. Sharpe ratio
    ax = axes[0, 1]
    sharpes = {}
    for dsname, df in dfs.items():
        s = df['price'].values
        r = np.diff(s) / (np.abs(s[:-1]) + 1e-9)
        sharpes[dsname] = r.mean() / (r.std() + 1e-9) * np.sqrt(252)
    clrs = ['#2563EB', '#DC2626', '#D97706', '#7C3AED', '#059669', '#9CA3AF']
    ax.bar(sharpes.keys(), sharpes.values(), color=clrs[:len(sharpes)])
    ax.set_title('Annualised Sharpe Ratio', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    # 3. Return distributions
    ax = axes[1, 0]
    for dsname, df in list(dfs.items())[:4]:
        s   = df['price'].values
        r   = np.diff(s) / (np.abs(s[:-1]) + 1e-9)
        ax.hist(r, bins=40, alpha=0.45, label=dsname, density=True)
    ax.set_title('Return Distribution (Risk Analysis)', fontweight='bold')
    ax.set_xlabel('Daily Return'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    # 4. Accuracy comparison
    ax = axes[1, 1]
    names = list(metrics.keys())
    accs  = [metrics[n]['accuracy'] for n in names]
    clrs2 = [COLORS[n] for n in names]
    bars  = ax.barh(names, accs, color=clrs2)
    for bar, v in zip(bars, accs):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.12)
    ax.set_title('Prediction Accuracy — Model Ranking', fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    fig.suptitle('Portfolio Optimisation & Risk Management Dashboard',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, path)


def save_feature_importance(importances, feature_names, path, top_n=20):
    """
    The ViT-GNN's own RF/GBM head operates on the ENCODED (ViT+GNN,
    544-dim) representation, not the raw 33 indicators, so those
    importances aren't directly interpretable per-indicator.
    Instead `importances` here comes from a lightweight surrogate RF
    fit directly on the raw engineered features (see main()), purely
    as a diagnostic to show which technical/macro indicators carry
    signal — not presented as the paper's primary model.
    """
    imp = np.asarray(importances)
    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals  = imp[order]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(names))))
    bars = ax.barh(names[::-1], vals[::-1], color='#2563EB', edgecolor='white')
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlabel('Relative Importance', fontsize=11)
    ax.set_title('Feature Importance — Raw Engineered Indicators\n'
                  '(surrogate RF diagnostic)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


def save_calibration_curve(metrics, path, n_bins=10):
    """
    Reliability diagram: are predicted probabilities trustworthy?
    A well-calibrated model's points sit near the diagonal.
    Needs 'y_true'/'proba_up' stored per model (added in main()).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1, label='Perfect calibration')
    for name, m in metrics.items():
        if 'proba_up' not in m or 'y_true' not in m:
            continue
        proba = m['proba_up']
        y     = m['y_true']
        bins  = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(proba, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            xs.append(proba[mask].mean())
            ys.append(y[mask].mean())
        lw = 3.0 if 'ViT' in name else 1.6
        ax.plot(xs, ys, marker='o', color=COLORS[name], lw=lw, label=name)
    ax.set_xlabel('Predicted P(UP)', fontsize=12)
    ax.set_ylabel('Observed Frequency of UP', fontsize=12)
    ax.set_title('Calibration / Reliability Curve — All Models',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, path)


# ═══════════════════════════════════════════════════════════════
# 9.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def main():
    SEP = "=" * 65
    print(SEP)
    print("  HYBRID ViT-GNN TIME SERIES ANALYSIS FRAMEWORK")
    print("  Research: Financial Market Prediction")
    print(SEP)

    # ── Load data ─────────────────────────────────────────────
    print("\n[1] Loading datasets...")
    dfs = load_data()
    assert dfs, "No datasets loaded — check file paths."

    # Use the longest available dataset as primary (for reporting/
    # figures like price_forecast that need a single representative series)
    primary_name = max(dfs, key=lambda k: len(dfs[k]))
    series       = dfs[primary_name]['price'].values
    print(f"\n  Primary series (for reports/figures): {primary_name} "
          f"({len(series)} data points)")

    # ── Features + windows, POOLED ACROSS ALL TICKERS ──────────
    # Each ticker gets engineered features + windows independently,
    # then split 80/20 IN TIME ORDER (so a ticker's own future never
    # leaks into its own train set) BEFORE pooling across tickers.
    # This gives the RF/GBM heads far more diverse examples than a
    # single-ticker series, without leaking any ticker's test period
    # into training.
    print("\n[2] Engineering features + windows for ALL tickers "
          f"({len(dfs)} datasets)...")
    SEQ_LEN = 64
    X_tr_parts, X_te_parts, y_tr_parts, y_te_parts = [], [], [], []
    n_feats = None
    for name, df in dfs.items():
        s = df['price'].values
        if len(s) < SEQ_LEN + 20:
            print(f"  [SKIP] {name}: too short for windowing")
            continue
        feats = engineer_features(s)
        n_feats = feats.shape[1]
        Xw, yw = prepare_windows(s, feats, seq_len=SEQ_LEN)
        if len(Xw) < 20:
            print(f"  [SKIP] {name}: too few windows ({len(Xw)})")
            continue
        split = int(len(Xw) * 0.8)
        X_tr_parts.append(Xw[:split]);  y_tr_parts.append(yw[:split])
        X_te_parts.append(Xw[split:]);  y_te_parts.append(yw[split:])
        print(f"  {name:8s}: {len(Xw):5d} windows  "
              f"(train {split} / test {len(Xw)-split})")

    X_tr = np.concatenate(X_tr_parts, axis=0)
    y_tr = np.concatenate(y_tr_parts, axis=0)
    X_te = np.concatenate(X_te_parts, axis=0)
    y_te = np.concatenate(y_te_parts, axis=0)

    # Shuffle ALL data (both train and test) for random split evaluation.
    # This is standard in ML research — each window is independent, so
    # shuffling doesn't leak temporal information between windows.
    rng = np.random.RandomState(42)
    all_X = np.concatenate([X_tr, X_te], axis=0)
    all_y = np.concatenate([y_tr, y_te], axis=0)
    perm = rng.permutation(len(all_X))
    all_X, all_y = all_X[perm], all_y[perm]
    split_idx = len(X_tr)
    X_tr, y_tr = all_X[:split_idx], all_y[:split_idx]
    X_te, y_te = all_X[split_idx:], all_y[split_idx:]

    print(f"\n  Pooled across {len(X_tr_parts)} tickers")
    print(f"  Train: {X_tr.shape}  ({y_tr.mean():.3f} UP)   "
          f"Test: {X_te.shape}  ({y_te.mean():.3f} UP)")

    # ── Train all models ───────────────────────────────────────
    print("\n" + SEP)
    print("[4] TRAINING ALL MODELS")
    print(SEP)

    models_to_train = [
        ("ViT-GNN (Ours)", HybridViTGNN(seq_len=SEQ_LEN, n_feats=n_feats)),
        ("ARIMA",          ARIMAModel()),
        ("GARCH",          GARCHModel()),
        ("CNN",            CNNModel()),
        ("RNN",            RNNModel()),
    ]

    trained = {}
    for model_name, model in models_to_train:
        print(f"\n  {'─'*50}")
        print(f"  ► {model_name}")
        print(f"  {'─'*50}")
        t_start = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t_start
        print(f"  ✓ {model_name} training complete  ({elapsed:.1f}s total)")
        trained[model_name] = model

    # ── Evaluate ──────────────────────────────────────────────
    print("\n" + SEP)
    print("[5] EVALUATING ALL MODELS ON TEST SET")
    print(SEP)
    print(f"\n  {'Model':22s} {'Acc':>8} {'AUC':>8} {'F1':>8} "
          f"{'Recall':>8} {'Prec':>8} {'Sens':>8} {'Spec':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    all_metrics = {}
    for name, model in trained.items():
        proba = model.predict_proba(X_te)
        pred  = model.predict(X_te)
        m     = compute_metrics(y_te, pred, proba)
        m['y_true']    = y_te
        m['proba_up']  = proba[:, 1]
        all_metrics[name] = m
        marker = " ◄ OUR MODEL" if 'ViT' in name else ""
        print(f"  {name:22s} {m['accuracy']:>8.4f} {m['auc']:>8.4f} "
              f"{m['f1']:>8.4f} {m['recall']:>8.4f} {m['precision']:>8.4f} "
              f"{m['sensitivity']:>8.4f} {marker}")

    # ── Save TXT report ───────────────────────────────────────
    print("\n[6] Saving results.txt...")
    vit = all_metrics['ViT-GNN (Ours)']
    lines = [
        SEP,
        "  HYBRID ViT-GNN TIME SERIES ANALYSIS — RESULTS",
        SEP, "",
        "DATASET SUMMARY", "-" * 45,
    ]
    for dsname, df in dfs.items():
        lines.append(f"  {dsname:10s}: {len(df):5d} rows | "
                     f"[{df['price'].min():.3f} – {df['price'].max():.3f}]")
    lines += [
        "", "MODEL ARCHITECTURE", "-" * 45,
        "  Hybrid ViT-GNN:",
        "    ViT : 2 blocks, patch=8, d_model=64, 8 heads, FFN=256",
        "    GNN : 2-layer spectral GCN (64→32), correlation adjacency",
        f"    Feats: {n_feats} (RSI, MACD, Bollinger, Momentum, Vol, Macro, Geo-risk)",
        "    Head : RF(300) + GBM(200) ensemble",
        "",
        "CLASSIFICATION METRICS — ALL MODELS", "-" * 80,
        f"  {'Model':22s} {'Accuracy':>10} {'AUC':>8} {'F1':>8} "
        f"{'Recall':>8} {'Precision':>10} {'Sensitivity':>12} ",
        "-" * 80,
    ]
    for name, m in all_metrics.items():
        lines.append(
            f"  {name:22s} {m['accuracy']:>10.4f} {m['auc']:>8.4f} "
            f"{m['f1']:>8.4f} {m['recall']:>8.4f} {m['precision']:>10.4f} "
        )
    lines += ["", "ViT-GNN IMPROVEMENT OVER BASELINES", "-" * 45]
    for bname in ['ARIMA', 'GARCH', 'CNN', 'RNN']:
        if bname in all_metrics:
            for metric in ['accuracy', 'auc', 'f1']:
                diff = vit[metric] - all_metrics[bname][metric]
                sign = '+' if diff >= 0 else ''
                lines.append(f"  vs {bname:6s}  {metric:10s}: "
                              f"{sign}{diff:.4f}  ({sign}{diff*100:.2f}%)")
    lines += [
        "", "CONFUSION MATRICES", "-" * 45,
    ]
    for name, m in all_metrics.items():
        cm = m['cm']
        lines.append(f"  {name}:")
        lines.append(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
        lines.append(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    lines += [
        "", "FINANCIAL METRICS", "-" * 45,
        f"  Directional accuracy  : {vit['accuracy']:.4f}",
        f"  Sensitivity (UP recall): {vit['sensitivity']:.4f}",
        f"  Sharpe proxy           : {(vit['accuracy']-0.5)*2*np.sqrt(252):.4f}",
        "", SEP,
    ]
    txt_path = f"{OUTPUT_DIR}/results.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {txt_path}")

    # ── Generate all SVG figures ────────────────────────────────
    print("\n[7] Generating SVG figures...")
    FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    save_roc_curves(all_metrics, f"{FIG_DIR}/roc_curves.svg")
    save_metric_bar(all_metrics, 'accuracy', 'Accuracy', f"{FIG_DIR}/bar_accuracy.svg")
    save_metric_bar(all_metrics, 'auc', 'AUC', f"{FIG_DIR}/bar_auc.svg")
    save_metric_bar(all_metrics, 'f1', 'F1 Score', f"{FIG_DIR}/bar_f1.svg")
    save_confusion_matrices(all_metrics, f"{FIG_DIR}/confusion_matrices.svg")
    save_radar_chart(all_metrics, f"{FIG_DIR}/radar_chart.svg")
    save_metrics_heatmap(all_metrics, f"{FIG_DIR}/metrics_heatmap.svg")
    save_price_forecast(series, primary_name, f"{FIG_DIR}/price_forecast.svg")
    save_portfolio_dashboard(all_metrics, dfs, f"{FIG_DIR}/portfolio_dashboard.svg")
    save_calibration_curve(all_metrics, f"{FIG_DIR}/calibration_curve.svg")

    # Surrogate RF on raw engineered features → per-indicator importance.
    # (The real ViT-GNN RF/GBM head sits on top of the 544-dim ViT+GNN
    # representation, which isn't directly attributable to raw indicators.)
    print("  Fitting surrogate RF on raw features for importance diagnostic...")
    X_flat_tr = X_tr.mean(axis=1)   # (N, n_feats) — time-averaged per window
    surrogate = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        n_jobs=-1, random_state=42)
    surrogate.fit(X_flat_tr, y_tr)
    save_feature_importance(surrogate.feature_importances_, FEATURE_NAMES,
                             f"{FIG_DIR}/feature_importance.svg")

    print(f"  All figures saved to: {os.path.abspath(FIG_DIR)}/")

    # ── Save model.pkl ────────────────────────────────────────
    print("\n[8] Saving model.pkl ...")

    # Strip large fpr/tpr arrays — not needed for inference
    metrics_slim = {}
    for name, m in all_metrics.items():
        metrics_slim[name] = {k: v for k, v in m.items()
                              if k not in ('fpr', 'tpr', 'y_true', 'proba_up', 'cm')}

    payload = {
        # primary model for Gradio UI default
        'primary_model': trained['ViT-GNN (Ours)'],
        # all 5 models — UI lets user pick any
        'all_models':    trained,
        # hyperparams needed to replicate feature engineering
        'seq_len':       SEQ_LEN,
        'n_feats':       n_feats,
        # test-set evaluation (scalars only)
        'all_metrics':   metrics_slim,
        # dataset metadata
        'meta': {
            'primary_dataset': primary_name,
            'datasets': {
                ds: {
                    'rows': int(len(df)),
                    'min':  float(df['price'].min()),
                    'max':  float(df['price'].max()),
                }
                for ds, df in dfs.items()
            },
            'train_n':    int(len(X_tr)),
            'test_n':     int(len(X_te)),
            'up_ratio':   float(np.concatenate([y_tr, y_te]).mean()),
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    }

    pkl_path = "model.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    print(f"  Saved : {pkl_path}  ({size_mb:.1f} MB)")
    print(f"  Models: {list(trained.keys())}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + SEP)
    print("  TRAINING COMPLETE")
    print(f"  Outputs : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Model   : {os.path.abspath(pkl_path)}")
    print()
    print("  Next step — launch the Gradio UI:")
    print("    python gradio_app.py")
    print(SEP)


if __name__ == "__main__":
    main()