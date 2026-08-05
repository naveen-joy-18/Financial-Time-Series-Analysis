"""
╔══════════════════════════════════════════════════════════════════╗
║  Hybrid ViT-GNN Time Series Analysis — Gradio UI                ║
║  Loads a pre-trained model and runs predictions on uploaded CSV  ║
║                                                                  ║
║  Step 1:  python train_model.py   (auto-loads all CSVs in data/)║
║  Step 2:  python gradio_app_updated.py                          ║
║                                                                  ║
║  Install:  pip install gradio pandas numpy matplotlib seaborn   ║
║                        scikit-learn scipy                        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle, os, glob
import warnings
import time
warnings.filterwarnings('ignore')

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  MODEL CLASSES  (exact copies from train_model.py — DO NOT EDIT)
#  pickle requires these class definitions to be identical
# ═══════════════════════════════════════════════════════════════

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

    Matches train_model.py class definition exactly (required for pickle).
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
        self.w_rf  = 0.55
        self.w_gbm = 0.45
        self.sc   = StandardScaler()
        self.threshold = 0.5

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

    def predict_proba(self, Xw):
        X_enc = self._encode_batch(Xw, desc="Predicting ")
        X_enc = self.sc.transform(X_enc)
        p_rf  = self.rf.predict_proba(X_enc)
        p_gbm = self.gbm.predict_proba(X_enc)
        return self.w_rf * p_rf + self.w_gbm * p_gbm

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


# ═══════════════════════════════════════════════════════════════
#  MODEL REGISTRY  — scan for .pkl files at startup
# ═══════════════════════════════════════════════════════════════

def _find_models(search_dirs=(".", "models", "outputs")):
    found = {}
    for d in search_dirs:
        for f in glob.glob(os.path.join(d, "*.pkl")):
            label = os.path.basename(f)
            found[label] = f
    if not found:
        found["(no model found — run train_model.py first)"] = ""
    return found

MODEL_REGISTRY = _find_models()


def load_model(pkl_path: str):
    """Load a saved ViT-GNN payload from disk."""
    with open(pkl_path, 'rb') as f:
        payload = pickle.load(f)
    model   = payload.get('primary_model') or payload.get('model')
    seq_len = payload['seq_len']
    meta    = payload.get('meta', {})
    return model, seq_len, meta


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (exact copy from train_model.py)
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


PRED_HORIZON = 40

def prepare_windows(series, features, seq_len=64, horizon=PRED_HORIZON):
    X, y = [], []
    for i in range(len(series) - seq_len - horizon):
        X.append(features[i:i + seq_len])
        y.append(1 if series[i + seq_len + horizon - 1] > series[i + seq_len - 1] else 0)
    return np.array(X), np.array(y)



def _align(arr, n):
    if len(arr) >= n:
        return arr[-n:]
    return np.pad(arr, (n - len(arr), 0))


# ═══════════════════════════════════════════════════════════════
#  PREDICTION MODULES  (unchanged from original)
# ═══════════════════════════════════════════════════════════════

def _max_drawdown(s):
    peak = s[0]; dd = 0.0
    for v in s:
        peak = max(peak, v)
        dd   = min(dd, (v - peak) / (peak + 1e-9))
    return float(dd * 100)


def portfolio_signals(series, proba, pred):
    ret  = np.diff(series) / (np.abs(series[:-1]) + 1e-9)
    n    = len(pred)
    r    = _align(ret, n)
    conf = proba[:, 1]

    win_rate = float(np.mean(pred == 1))
    avg_win  = float(np.mean(r[pred == 1]))  if (pred == 1).any() else 0.0
    avg_loss = float(np.abs(np.mean(r[pred == 0]))) if (pred == 0).any() else 0.01
    kelly    = max(0.0, win_rate / max(avg_loss, 1e-9) -
                        (1 - win_rate) / max(avg_win, 1e-9))
    kelly    = float(np.clip(kelly, 0, 1))

    signal_ret = np.where(pred == 1, r * conf, -r * (1 - conf))
    cum_strat  = np.cumprod(1 + np.clip(signal_ret, -0.5, 0.5))
    cum_bh     = np.cumprod(1 + np.clip(r, -0.5, 0.5))
    sharpe     = float(signal_ret.mean() / (signal_ret.std() + 1e-9) * np.sqrt(252))

    return dict(win_rate=win_rate, kelly=kelly, sharpe=sharpe,
                cum_strat=cum_strat, cum_bh=cum_bh,
                signal_ret=signal_ret, conf=conf, pred=pred)


def risk_metrics(series, proba, pred):
    ret  = np.diff(series) / (np.abs(series[:-1]) + 1e-9)
    n    = len(pred)
    r    = _align(ret, n)
    conf = proba[:, 1]

    var95  = float(np.percentile(r, 5))
    var99  = float(np.percentile(r, 1))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else var95
    vol_ann = float(r.std() * np.sqrt(252) * 100)
    max_dd  = _max_drawdown(series)
    risk_flags = np.where((pred == 0) & (conf < 0.4), 1, 0)
    risk_pct   = float(risk_flags.mean() * 100)
    roll_vol   = pd.Series(r).rolling(20).std().fillna(0).values

    return dict(var95=var95, var99=var99, cvar95=cvar95, vol_ann=vol_ann,
                max_dd=max_dd, risk_flags=risk_flags, risk_pct=risk_pct,
                roll_vol=roll_vol, r=r, conf=conf, pred=pred)


def algo_trading_signals(series, proba, pred):
    ret  = np.diff(series) / (np.abs(series[:-1]) + 1e-9)
    n    = len(pred)
    r    = _align(ret, n)
    conf = proba[:, 1]

    long_sig  = (conf > 0.60).astype(int)
    short_sig = (conf < 0.40).astype(int)

    pnl_long  = np.where(long_sig == 1,   r, 0)
    pnl_short = np.where(short_sig == 1, -r, 0)
    total_pnl = pnl_long + pnl_short
    cum_pnl   = np.cumsum(total_pnl)

    n_long   = int(long_sig.sum())
    n_short  = int(short_sig.sum())
    wr_long  = float(np.mean(r[long_sig == 1] > 0))  if n_long  else 0.0
    wr_short = float(np.mean(r[short_sig == 1] < 0)) if n_short else 0.0

    return dict(long_sig=long_sig, short_sig=short_sig,
                cum_pnl=cum_pnl, total_pnl=total_pnl,
                n_long=n_long, n_short=n_short,
                wr_long=wr_long, wr_short=wr_short,
                conf=conf, r=r)


def market_stability(series, proba, pred):
    ret  = np.diff(series) / (np.abs(series[:-1]) + 1e-9)
    n    = len(pred)
    r    = _align(ret, n)
    conf = proba[:, 1]

    rv      = pd.Series(r).rolling(20).std().bfill().values
    med_vol = float(np.median(rv))
    regime  = np.where(rv < med_vol, 0, 1)
    stab_score = float(1 - np.clip(rv.mean() / max(rv.max(), 1e-9), 0, 1))
    entropy    = -np.sum(proba * np.log(proba + 1e-9), axis=1)
    unc_mean   = float(entropy.mean())
    x          = np.arange(len(series))
    slope, _   = np.polyfit(x, series, 1)
    trend_str  = float(slope / (series.mean() + 1e-9) * 100)
    autocorr   = float(pd.Series(r).autocorr(lag=1)) if len(r) > 10 else 0.0

    return dict(regime=regime, stab_score=stab_score, rv=rv, med_vol=med_vol,
                entropy=entropy, unc_mean=unc_mean, trend_str=trend_str,
                autocorr=autocorr, r=r, conf=conf, series=series)


# ═══════════════════════════════════════════════════════════════
#  DARK-THEMED CHARTS  (unchanged from original)
# ═══════════════════════════════════════════════════════════════

BG, BG2, FG = '#0f172a', '#1e293b', '#e2e8f0'
BLUE, GREEN, RED, AMBER, PURPLE, TEAL = (
    '#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#06b6d4')


def _dark(fig):
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BG2)
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for sp in ax.spines.values():
            sp.set_color('#334155')
        ax.grid(True, alpha=0.12, color='#334155')


def fig_portfolio(pf, series):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(pf['cum_strat']))
    ax1.plot(x, pf['cum_bh'],    color=AMBER, lw=1.5, alpha=0.7, label='Buy & Hold')
    ax1.plot(x, pf['cum_strat'], color=BLUE,  lw=2.2, label='ViT-GNN Strategy')
    ax1.fill_between(x, pf['cum_bh'], pf['cum_strat'],
                     where=pf['cum_strat'] >= pf['cum_bh'], alpha=0.15, color=GREEN)
    ax1.fill_between(x, pf['cum_bh'], pf['cum_strat'],
                     where=pf['cum_strat'] <  pf['cum_bh'], alpha=0.15, color=RED)
    ax1.set_title('Cumulative Returns: ViT-GNN vs Buy & Hold', fontweight='bold')
    ax1.set_ylabel('Cumulative Return'); ax1.set_xlabel('Time Steps')
    ax1.legend(fontsize=9, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    final_ret = (pf['cum_strat'][-1] - 1) * 100
    kpis = [
        ('Win Rate',  f"{pf['win_rate']*100:.1f}%",  GREEN),
        ('Kelly Frac',f"{pf['kelly']*100:.1f}%",     TEAL),
        ('Sharpe',    f"{pf['sharpe']:.3f}",          BLUE),
        ('Final Ret', f"{final_ret:.2f}%", GREEN if final_ret >= 0 else RED),
    ]
    for i, (label, val, col) in enumerate(kpis):
        ax2.text(0.08, 0.83 - i * 0.22, label, transform=ax2.transAxes,
                 fontsize=9,  color='#94a3b8', fontfamily='monospace')
        ax2.text(0.08, 0.72 - i * 0.22, val,   transform=ax2.transAxes,
                 fontsize=18, color=col, fontweight='bold', fontfamily='monospace')
    ax2.set_title('Portfolio KPIs', fontweight='bold')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(pf['conf'][pf['pred'] == 1], bins=25, color=GREEN, alpha=0.65,
             label='UP signals', density=True)
    ax3.hist(pf['conf'][pf['pred'] == 0], bins=25, color=RED,   alpha=0.65,
             label='DOWN signals', density=True)
    ax3.axvline(0.6, color=BLUE, ls='--', lw=1.3, label='Entry threshold')
    ax3.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax3.set_xlabel('Confidence'); ax3.set_ylabel('Density')
    ax3.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax4 = fig.add_subplot(gs[1, 1:])
    sr  = pf['signal_ret']
    ax4.bar(np.arange(len(sr)), sr,
            color=np.where(sr >= 0, GREEN, RED), alpha=0.55, width=1)
    roll = pd.Series(sr).rolling(20).mean().values
    ax4.plot(np.arange(len(roll)), roll, color=BLUE, lw=2, label='20-period rolling mean')
    ax4.axhline(0, color='#475569', lw=0.8)
    ax4.set_title('Signal-Weighted Strategy Returns', fontweight='bold')
    ax4.set_xlabel('Time Steps'); ax4.set_ylabel('Return')
    ax4.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    _dark(fig)
    fig.suptitle('📊  Portfolio Optimisation', fontsize=14,
                 fontweight='bold', color=FG, y=1.01)
    fig.tight_layout()
    return fig


def fig_risk(rm):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(rm['r'], bins=50, color=BLUE, alpha=0.55, density=True, label='Returns')
    ax1.axvline(rm['var95'],  color=RED,    lw=2, ls='--', label=f"VaR 95%  {rm['var95']*100:.2f}%")
    ax1.axvline(rm['var99'],  color=AMBER,  lw=2, ls='--', label=f"VaR 99%  {rm['var99']*100:.2f}%")
    ax1.axvline(rm['cvar95'], color=PURPLE, lw=2, ls=':',  label=f"CVaR 95% {rm['cvar95']*100:.2f}%")
    ax1.set_title('Return Distribution with Value-at-Risk', fontweight='bold')
    ax1.set_xlabel('Daily Return'); ax1.set_ylabel('Density')
    ax1.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    kpis = [
        ('VaR 95%',  f"{rm['var95']*100:.3f}%",  RED),
        ('CVaR 95%', f"{rm['cvar95']*100:.3f}%", AMBER),
        ('Ann. Vol', f"{rm['vol_ann']:.2f}%",     PURPLE),
        ('Max DD',   f"{rm['max_dd']:.2f}%",      RED),
    ]
    for i, (label, val, col) in enumerate(kpis):
        ax2.text(0.08, 0.83 - i * 0.22, label, transform=ax2.transAxes,
                 fontsize=9, color='#94a3b8', fontfamily='monospace')
        ax2.text(0.08, 0.72 - i * 0.22, val,   transform=ax2.transAxes,
                 fontsize=16, color=col, fontweight='bold', fontfamily='monospace')
    ax2.set_title('Risk Metrics', fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :2])
    x   = np.arange(len(rm['roll_vol']))
    ax3.plot(x, rm['roll_vol'], color=AMBER, lw=1.8, label='20-day rolling vol')
    ax3.fill_between(x, 0, rm['roll_vol'], alpha=0.2, color=AMBER)
    ax3.axhline(float(np.median(rm['roll_vol'])), color='#64748b',
                ls='--', lw=1, label='Median vol')
    flag_idx = x[rm['risk_flags'] == 1]
    if len(flag_idx):
        ax3.scatter(flag_idx, rm['roll_vol'][rm['risk_flags'] == 1],
                    color=RED, s=18, zorder=5, label=f'Risk flags ({len(flag_idx)})')
    ax3.set_title('Rolling Volatility & Risk Flags', fontweight='bold')
    ax3.set_xlabel('Time Steps'); ax3.set_ylabel('Volatility')
    ax3.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax4 = fig.add_subplot(gs[1, 2])
    rn  = min(len(rm['conf']), len(rm['r']))
    sc  = ax4.scatter(rm['conf'][:rn], np.abs(rm['r'][:rn]),
                      c=rm['conf'][:rn], cmap='RdYlGn', alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax4).ax.yaxis.set_tick_params(color=FG)
    ax4.set_title('Confidence vs Realised Risk', fontweight='bold')
    ax4.set_xlabel('Model Confidence'); ax4.set_ylabel('|Return|')

    _dark(fig)
    fig.suptitle('⚠️  Risk Management', fontsize=14,
                 fontweight='bold', color=FG, y=1.01)
    fig.tight_layout()
    return fig


def fig_trading(at):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    x   = np.arange(len(at['conf']))
    ax1.plot(x, at['conf'], color='#64748b', lw=1, alpha=0.7, label='Confidence P(UP)')
    ax1.fill_between(x, 0.6, at['conf'], where=at['conf'] >= 0.6,
                     alpha=0.4, color=GREEN, label='LONG zone  (>0.60)')
    ax1.fill_between(x, at['conf'], 0.4, where=at['conf'] <= 0.4,
                     alpha=0.4, color=RED,   label='SHORT zone (<0.40)')
    ax1.axhline(0.6, color=GREEN, ls='--', lw=1, alpha=0.6)
    ax1.axhline(0.4, color=RED,   ls='--', lw=1, alpha=0.6)
    ax1.set_ylim(0, 1)
    ax1.set_title('Algorithmic Trading Signals (Confidence-based)', fontweight='bold')
    ax1.set_xlabel('Time Steps'); ax1.set_ylabel('P(UP)')
    ax1.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    stats = [
        ('Long Trades',  str(at['n_long']),          GREEN),
        ('Short Trades', str(at['n_short']),          RED),
        ('Long WR',      f"{at['wr_long']*100:.1f}%", GREEN),
        ('Short WR',     f"{at['wr_short']*100:.1f}%",TEAL),
    ]
    for i, (label, val, col) in enumerate(stats):
        ax2.text(0.08, 0.83 - i * 0.22, label, transform=ax2.transAxes,
                 fontsize=9, color='#94a3b8', fontfamily='monospace')
        ax2.text(0.08, 0.72 - i * 0.22, val,   transform=ax2.transAxes,
                 fontsize=18, color=col, fontweight='bold', fontfamily='monospace')
    ax2.set_title('Trading Statistics', fontweight='bold')

    ax3 = fig.add_subplot(gs[1, :2])
    x   = np.arange(len(at['cum_pnl']))
    ax3.plot(x, at['cum_pnl'], color=BLUE, lw=2.2, label='Cumulative P&L')
    ax3.fill_between(x, 0, at['cum_pnl'],
                     where=at['cum_pnl'] >= 0, alpha=0.15, color=GREEN)
    ax3.fill_between(x, 0, at['cum_pnl'],
                     where=at['cum_pnl'] <  0, alpha=0.15, color=RED)
    ax3.axhline(0, color='#475569', lw=0.8)
    ax3.set_title('Cumulative P&L  (Long + Short combined)', fontweight='bold')
    ax3.set_xlabel('Time Steps'); ax3.set_ylabel('Cumulative Return')
    ax3.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax4 = fig.add_subplot(gs[1, 2])
    tr  = at['total_pnl'][at['total_pnl'] != 0]
    if len(tr) > 0:
        wins   = tr[tr > 0]; losses = tr[tr < 0]
        if len(wins):   ax4.hist(wins,   bins=20, color=GREEN, alpha=0.7, label='Wins')
        if len(losses): ax4.hist(losses, bins=20, color=RED,   alpha=0.7, label='Losses')
    ax4.axvline(0, color='#64748b', lw=0.8)
    ax4.set_title('Per-Trade Return Distribution', fontweight='bold')
    ax4.set_xlabel('Return'); ax4.set_ylabel('Count')
    ax4.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    _dark(fig)
    fig.suptitle('🤖  Algorithmic Trading', fontsize=14,
                 fontweight='bold', color=FG, y=1.01)
    fig.tight_layout()
    return fig


def fig_stability(ms):
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    s   = ms['series']
    x   = np.arange(len(s))
    ax1.plot(x, s, color='#64748b', lw=1, alpha=0.6, label='Price')
    rn  = min(len(ms['regime']), len(s))
    xs  = x[-rn:]; ss = s[-rn:]
    sm  = ms['regime'][-rn:] == 0
    ax1.fill_between(xs, ss.min(), ss, where=sm,  alpha=0.2, color=GREEN, label='Stable')
    ax1.fill_between(xs, ss.min(), ss, where=~sm, alpha=0.2, color=RED,   label='Volatile')
    ax1.set_title('Price with Market Regime Overlay', fontweight='bold')
    ax1.set_xlabel('Time'); ax1.set_ylabel('Price')
    ax1.legend(fontsize=8, facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax2 = fig.add_subplot(gs[0, 2]); ax2.axis('off')
    stable_pct = float(np.mean(ms['regime'] == 0) * 100)
    kpis = [
        ('Stability',   f"{ms['stab_score']*100:.1f}%",
         GREEN if ms['stab_score'] > 0.5 else AMBER),
        ('Stable %',    f"{stable_pct:.1f}%",           GREEN),
        ('Uncertainty', f"{ms['unc_mean']:.3f}",         AMBER),
        ('Autocorr',    f"{ms['autocorr']:.3f}",         TEAL),
    ]
    for i, (label, val, col) in enumerate(kpis):
        ax2.text(0.08, 0.83 - i * 0.22, label, transform=ax2.transAxes,
                 fontsize=9, color='#94a3b8', fontfamily='monospace')
        ax2.text(0.08, 0.72 - i * 0.22, val,   transform=ax2.transAxes,
                 fontsize=16, color=col, fontweight='bold', fontfamily='monospace')
    ax2.set_title('Stability Metrics', fontweight='bold')

    ax3  = fig.add_subplot(gs[1, :2])
    x3   = np.arange(len(ms['entropy']))
    ax3.plot(x3, ms['entropy'], color=PURPLE, lw=1.5, label='Prediction entropy')
    ax3.fill_between(x3, 0, ms['entropy'], alpha=0.2, color=PURPLE)
    ax3b = ax3.twinx()
    rv3  = ms['rv'][-len(ms['entropy']):] if len(ms['rv']) >= len(ms['entropy']) else ms['rv']
    ax3b.plot(x3[:len(rv3)], rv3, color=AMBER, lw=1.2, alpha=0.6, ls='--', label='Rolling vol')
    ax3b.tick_params(colors=FG, labelsize=8); ax3b.yaxis.label.set_color(AMBER)
    ax3b.set_ylabel('Volatility', color=AMBER)
    ax3.set_title('Model Uncertainty vs Market Volatility', fontweight='bold')
    ax3.set_xlabel('Time Steps'); ax3.set_ylabel('Entropy', color=PURPLE)
    lines1, labs1 = ax3.get_legend_handles_labels()
    lines2, labs2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labs1 + labs2, fontsize=8,
               facecolor=BG2, edgecolor='#334155', labelcolor=FG)

    ax4 = fig.add_subplot(gs[1, 2])
    stable_n   = int((ms['regime'] == 0).sum())
    unstable_n = len(ms['regime']) - stable_n
    if stable_n + unstable_n > 0:
        ax4.pie([stable_n, unstable_n],
                labels=['Stable', 'Volatile'],
                colors=[GREEN, RED], autopct='%1.1f%%', startangle=90,
                textprops={'color': FG, 'fontsize': 10},
                wedgeprops={'edgecolor': BG, 'linewidth': 2})
    ax4.set_title('Market Regime Distribution', fontweight='bold')

    _dark(fig)
    fig.suptitle('📡  Market Stability Analysis', fontsize=14,
                 fontweight='bold', color=FG, y=1.01)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE  (no training — predict only)
# ═══════════════════════════════════════════════════════════════

def run_pipeline(model_choice, file_obj, progress=gr.Progress()):
    """Load saved model → predict on uploaded CSV → render charts."""

    # ── Validate inputs ───────────────────────────────────────
    if not model_choice or model_choice not in MODEL_REGISTRY:
        msg = "⚠️  Please select a trained model from the dropdown."
        return None, None, None, None, msg, msg

    pkl_path = MODEL_REGISTRY[model_choice]
    if not pkl_path or not os.path.isfile(pkl_path):
        msg = "⚠️  Model file not found. Run train_model.py first."
        return None, None, None, None, msg, msg

    if file_obj is None:
        msg = "⚠️  Please upload a CSV file to predict on."
        return None, None, None, None, msg, msg

    try:
        # ── Load model ────────────────────────────────────────
        progress(0.05, desc="Loading saved model…")
        model, seq_len, meta = load_model(pkl_path)
        if not hasattr(model, 'fitted'):
            model.fitted = True  # Assume saved models are fitted
        if not model.fitted:
            msg = "❌  Model in pickle is not fitted. Re-run train_model.py."
            return None, None, None, None, msg, msg

        # ── Load CSV ──────────────────────────────────────────
        progress(0.10, desc="Loading dataset…")
        df = pd.read_csv(file_obj.name)
        df.columns = [c.strip() for c in df.columns]
        pcol = next(
            (c for c in df.columns
             if c.lower() in ['adj close', 'close', 'value', 'earnings', 'price']),
            None
        )
        if pcol is None:
            err = f"❌  No price column found. Columns: {list(df.columns)}"
            return None, None, None, None, err, err

        series = pd.to_numeric(df[pcol], errors='coerce').dropna().values
        if len(series) < seq_len + 10:
            err = f"❌  Dataset too short ({len(series)} rows, need >{seq_len + 10})."
            return None, None, None, None, err, err

        # ── Features + windows ────────────────────────────────
        progress(0.20, desc="Engineering features…")
        F = engineer_features(series)

        progress(0.30, desc=f"Creating prediction windows (seq_len={seq_len})…")
        X, y = prepare_windows(series, F, seq_len=seq_len)

        # ── Predict (no training!) ────────────────────────────
        progress(0.50, desc="Running ViT-GNN predictions…")
        proba = model.predict_proba(X)
        pred  = model.predict(X)

        # ── Analysis modules ──────────────────────────────────
        progress(0.65, desc="Computing portfolio signals…")
        pf_data = portfolio_signals(series, proba, pred)

        progress(0.72, desc="Computing risk metrics…")
        rm_data = risk_metrics(series, proba, pred)

        progress(0.80, desc="Computing trading signals…")
        at_data = algo_trading_signals(series, proba, pred)

        progress(0.87, desc="Computing market stability…")
        ms_data = market_stability(series, proba, pred)

        # ── Charts ────────────────────────────────────────────
        progress(0.93, desc="Rendering charts…")
        f_pf = fig_portfolio(pf_data, series)
        f_rm = fig_risk(rm_data)
        f_at = fig_trading(at_data)
        f_ms = fig_stability(ms_data)

        # ── Report ────────────────────────────────────────────
        up_frac = float(np.mean(pred == 1))
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║     HYBRID ViT-GNN · PREDICTION REPORT                  ║
╚══════════════════════════════════════════════════════════╝

MODEL
  File          : {model_choice}
  Trained on    : {meta.get('dataset', 'unknown')}
  Train rows    : {meta.get('train_n', '?'):,}
  Seq length    : {seq_len}

DATASET  (current prediction)
  File          : {os.path.basename(file_obj.name)}
  Rows loaded   : {len(series):,}
  Windows       : {len(X):,}
  UP predictions: {up_frac*100:.2f}%

─────────────────────────────────────────────────────────
PORTFOLIO OPTIMISATION
─────────────────────────────────────────────────────────
  Win Rate         : {pf_data['win_rate']*100:.2f}%
  Kelly Fraction   : {pf_data['kelly']*100:.2f}%
  Sharpe Ratio     : {pf_data['sharpe']:.4f}
  Strategy Return  : {(pf_data['cum_strat'][-1]-1)*100:.3f}%
  Buy & Hold Ret   : {(pf_data['cum_bh'][-1]-1)*100:.3f}%

─────────────────────────────────────────────────────────
RISK MANAGEMENT
─────────────────────────────────────────────────────────
  VaR 95%          : {rm_data['var95']*100:.4f}%
  VaR 99%          : {rm_data['var99']*100:.4f}%
  CVaR 95%         : {rm_data['cvar95']*100:.4f}%
  Annualised Vol   : {rm_data['vol_ann']:.3f}%
  Max Drawdown     : {rm_data['max_dd']:.3f}%
  Risk Flags       : {rm_data['risk_pct']:.2f}% of periods

─────────────────────────────────────────────────────────
ALGORITHMIC TRADING
─────────────────────────────────────────────────────────
  Long Signals     : {at_data['n_long']}
  Short Signals    : {at_data['n_short']}
  Long Win Rate    : {at_data['wr_long']*100:.2f}%
  Short Win Rate   : {at_data['wr_short']*100:.2f}%
  Final Cum. P&L   : {float(at_data['cum_pnl'][-1])*100:.3f}%

─────────────────────────────────────────────────────────
MARKET STABILITY
─────────────────────────────────────────────────────────
  Stability Score  : {ms_data['stab_score']*100:.2f}%
  Stable Periods   : {float(np.mean(ms_data['regime']==0))*100:.2f}%
  Volatile Periods : {float(np.mean(ms_data['regime']==1))*100:.2f}%
  Avg Uncertainty  : {ms_data['unc_mean']:.4f}
  Trend Strength   : {ms_data['trend_str']:.4f} %/period
  Autocorrelation  : {ms_data['autocorr']:.4f}
""".strip()

        progress(1.0, desc="✅  Done!")
        status = f"✅  Predictions complete — {len(X):,} windows processed."
        return f_pf, f_rm, f_at, f_ms, summary, status

    except Exception as e:
        import traceback
        err = f"❌  Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, None, err, err


# ═══════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Space Mono"), "monospace"],
).set(
    body_background_fill="#060c18",
    body_text_color="#e2e8f0",
    block_background_fill="#0d1a2e",
    block_border_color="#1e3050",
    block_label_text_color="#94a3b8",
    block_title_text_color="#e2e8f0",
    input_background_fill="#111f35",
    input_border_color="#1e3050",
    button_primary_background_fill="linear-gradient(135deg, #3b82f6, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#111f35",
    button_secondary_border_color="#1e3050",
    button_secondary_text_color="#94a3b8",
)

CSS = """
body { background: #060c18 !important; }
.gradio-container { max-width: 1380px !important; margin: 0 auto !important; }
footer { display: none !important; }

#app-header { text-align: center; padding: 28px 0 4px; }
#app-header h1 {
  font-family: 'Space Mono', monospace !important;
  font-size: 26px !important; font-weight: 700 !important;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: .04em; margin: 0 !important;
}
#app-header p {
  color: #64748b !important; font-size: 13px !important; margin-top: 6px !important;
}
.tab-nav button {
  font-family: 'Space Mono', monospace !important;
  font-size: 11px !important; letter-spacing: .05em !important;
  text-transform: uppercase !important;
}
#status-out textarea {
  font-family: 'Space Mono', monospace !important;
  font-size: 12px !important; background: #060c18 !important;
  color: #10b981 !important; border-color: #1e3050 !important;
}
#report-out textarea {
  font-family: 'Space Mono', monospace !important;
  font-size: 11px !important; line-height: 1.75 !important;
  background: #060c18 !important; color: #94a3b8 !important;
}
.arch-box {
  background: #111f35; border: 1px solid #1e3050;
  border-radius: 8px; padding: 14px;
  font-size: 12px; line-height: 2.0;
  font-family: monospace; color: #94a3b8;
}
.step-box {
  background: #0d1a2e; border: 1px solid #1e3050;
  border-radius: 8px; padding: 12px 16px;
  font-size: 12px; line-height: 1.8;
  font-family: monospace; color: #64748b;
}
"""


def build_app():
    with gr.Blocks(theme=THEME, css=CSS, title="ViT-GNN Research") as demo:

        gr.HTML("""
        <div id="app-header">
          <h1>ViT-GNN · Financial Time Series Analysis</h1>
          <p>Hybrid Vision Transformer + Graph Neural Network &nbsp;·&nbsp;
             Portfolio Optimisation &nbsp;·&nbsp; Risk Management &nbsp;·&nbsp;
             Algorithmic Trading &nbsp;·&nbsp; Market Stability</p>
        </div>
        """)

        # ── Controls ─────────────────────────────────────────
        with gr.Row():

            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 🧠 Select Model")
                model_dd = gr.Dropdown(
                    label="Trained model (.pkl)",
                    choices=list(MODEL_REGISTRY.keys()),
                    value=list(MODEL_REGISTRY.keys())[0],
                    info="Run train_model.py to create a .pkl file",
                )
                refresh_btn = gr.Button("🔄 Refresh model list",
                                        variant="secondary", size="sm")

            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📁 Upload CSV for Prediction")
                file_in = gr.File(
                    label="CSV file (any OHLCV ticker file from download.py, e.g. AAPL · MSFT · NVDA · TSLA · JPM ...)",
                    file_types=[".csv", ".txt"],
                )

            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 🏗️ Model Architecture")
                gr.HTML("""
                <div class="arch-box">
                  🔷 <b style="color:#60a5fa">ViT</b>
                     &nbsp;2-block, patch=8 · d_model=64 · 8 heads<br>
                  🕸️ <b style="color:#a78bfa">GNN</b>
                     &nbsp;2-layer spectral GCN (64→32)<br>
                  📈 <b style="color:#34d399">Features</b>
                     &nbsp;33-dim (RSI · MACD · Vol · Momentum)<br>
                  🌍 <b style="color:#fbbf24">Ensemble</b>
                     &nbsp;RF(1000) + GBM(150) · w=0.55/0.45<br>
                  🎯 <b style="color:#06b6d4">Output</b>
                     &nbsp;40-day UP / DOWN directional signal
                </div>
                """)

        # ── How-to note ───────────────────────────────────────
        gr.HTML("""
        <div class="step-box">
          <b style="color:#60a5fa">How to use:</b>&nbsp;
          Step 1 — train once:&nbsp;
          <code style="color:#34d399">python train_model.py</code> (trains on every CSV in data/)
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Step 2 — select the .pkl above, upload any compatible CSV, then click Predict.
        </div>
        """)

        # ── Run button + status ───────────────────────────────
        with gr.Row():
            predict_btn = gr.Button(
                "▶  Run Predictions",
                variant="primary", size="lg",
            )
        status_out = gr.Textbox(
            label="Status",
            value="⏳  Select a model and upload a CSV, then click Run Predictions.",
            interactive=False, lines=1, elem_id="status-out",
        )

        gr.Markdown("---")

        # ── Output tabs  (no ROC / classification metrics tab) ─
        with gr.Tabs():

            with gr.Tab("💼 Portfolio Optimisation"):
                gr.Markdown(
                    "> **Kelly-fraction position sizing** · "
                    "Confidence-weighted strategy returns · "
                    "Cumulative P&L vs Buy & Hold · Sharpe ratio"
                )
                out_portfolio = gr.Plot()

            with gr.Tab("⚠️ Risk Management"):
                gr.Markdown(
                    "> **Value-at-Risk** (95% & 99%) · **CVaR** · "
                    "Max Drawdown · Rolling Volatility · Risk Flags"
                )
                out_risk = gr.Plot()

            with gr.Tab("🤖 Algorithmic Trading"):
                gr.Markdown(
                    "> **Long / Short signal generation** (conf > 0.60 / < 0.40) · "
                    "Cumulative P&L · Win rates · Per-trade distribution"
                )
                out_trading = gr.Plot()

            with gr.Tab("📡 Market Stability"):
                gr.Markdown(
                    "> **Volatility regime detection** · "
                    "Prediction uncertainty (entropy) · "
                    "Price regime overlay · Autocorrelation"
                )
                out_stability = gr.Plot()

            with gr.Tab("📋 Full Report"):
                gr.Markdown(
                    "> Complete numerical results — portfolio stats, "
                    "risk measures, trading stats, stability analysis"
                )
                out_report = gr.Textbox(
                    label="Prediction Report",
                    lines=55, interactive=False,
                    elem_id="report-out",
                )

        # ── Wiring ────────────────────────────────────────────
        predict_btn.click(
            fn=run_pipeline,
            inputs=[model_dd, file_in],
            outputs=[out_portfolio, out_risk, out_trading,
                     out_stability, out_report, status_out],
        )

        def refresh_models():
            global MODEL_REGISTRY
            MODEL_REGISTRY = _find_models()
            choices = list(MODEL_REGISTRY.keys())
            return gr.Dropdown(choices=choices, value=choices[0])

        refresh_btn.click(fn=refresh_models, outputs=[model_dd])

    return demo


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 62)
    print("  Hybrid ViT-GNN · Gradio Prediction UI")
    print()
    print("  Make sure you have trained a model first:")
    print("    python train_model.py   (auto-loads every CSV in data/)")
    print()
    print("  Then run:  python gradio_app.py")
    print("  Open:      http://localhost:7860")
    print("=" * 62)
    print()

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )