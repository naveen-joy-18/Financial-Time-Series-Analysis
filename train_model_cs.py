"""
=======================================================================
 HYBRID ViT-GNN -- LEAKAGE-CONTROLLED RE-EVALUATION (corrected)
=======================================================================

This version fixes, in order of severity, the defects found in the audit
of the previous script:

 1. EMBARGO WAS TOO SHORT (real residual leakage).  With a horizon H and
    window length L, the last training window's *label* reads price at row
    i+L-1+H.  An embargo of only H therefore leaves the first test
    window's input overlapping rows that a training label was computed
    from.  The embargo is now L+H, which is the exact requirement
    (Section 2.4.1 of the manuscript).

 2. VALIDATION SPLIT WAS NOT CHRONOLOGICAL PER TICKER.  The old code took
    the pooled tail of the training array, which is only the most recent
    windows of the alphabetically last few tickers.  Validation is now the
    trailing VAL_FRAC of *each* ticker's training windows (with its own
    embargo), so all 75 tickers contribute.

 3. TEST-SET LEAKAGE THROUGH THE SCALER.  The old ablation code called
    StandardScaler().fit_transform() separately on train / val / TEST,
    i.e. it fitted the test scaler on test data.  Features are now scaled
    once, per ticker, with a scaler fitted only on that ticker's
    sub-training rows.

 4. INCONSISTENT THRESHOLD PROTOCOL.  The four classical baselines were
    thresholded at a hard-coded 0.5 while the deep model and ablations
    used a validation-tuned threshold.  Every model now uses the same
    validation-tuned threshold.

 5. MACD SIGNAL LINE WAS A DUPLICATE OF THE MACD LINE, and the histogram
    was MACD - EMA9(EMA12) rather than MACD - EMA9(MACD).

 6. THE "OBV PROXY" WAS ALGEBRAICALLY cumsum(log price): sign(r)*|r| == r.
    The CSV files do contain Volume, so a genuine OBV is now used.

 7. CCI USED A STANDARD DEVIATION although the text specifies a mean
    absolute deviation.  Code now matches the text.

 8. BASELINES WERE WEAKER THAN THE TEXT CLAIMS (no Laplacian/Sobel
    kernels, no skewness/kurtosis/ARCH ratio).  They now have them, so the
    deep model is not flattered by a straw-man baseline.

 9. CONFUSING STATISTICS.  Cohen's d on 5 per-seed accuracies produced
    values such as 276.  It is removed.  Effect size is now the McNemar
    effect r = (b-c)/(b+c), and inference is reported both iid and with
    dates as clusters, because the 35,400 pooled test windows are 75
    cross-sectionally correlated assets observed on ~472 common dates and
    are therefore NOT 35,400 independent observations.

10. THE DEEP MODEL WAS REPORTED ONLY AS A SEED ENSEMBLE (accuracy std
    exactly 0.0000), which is not comparable with single-run baselines.
    Per-seed mean +/- std is now the headline row and the ensemble is
    reported as a separate, clearly labelled row.

11. Per-ticker accuracies in the previous manuscript Table 6 were
    hard-coded placeholders.  They are now computed.

Run:  python train_model_cs.py --horizon 90 --seeds 5
"""
import os, json, time, sys, warnings, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, \
                             HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             recall_score, precision_score, confusion_matrix,
                             matthews_corrcoef, log_loss, brier_score_loss,
                             cohen_kappa_score, balanced_accuracy_score,
                             average_precision_score)
from scipy import stats
from itertools import combinations

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ----------------------------------------------------------------- config
OUTPUT_DIR   = "outputs_cs"
SEQ_LEN      = 64
D_MODEL      = 64
N_HEADS      = 4
N_BLOCKS     = 2
FF_DIM       = 128
GNN_HIDDEN   = 64
GNN_OUT      = 32
CORR_THRESH  = 0.25
N_BOOT       = 2000
TEST_FRAC    = 0.20
VAL_FRAC     = 0.12
EPOCHS       = 20
BATCH        = 512
LR           = 2e-3
WEIGHT_DECAY = 1e-4
MOM_LOOKS    = [5, 21, 60, 120, 250, 500]
FACTOR_PATH  = True
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

BASE_FEATURES = ['logprice', 'log_ret', 'pct_ret', 'rsi14', 'macd', 'macd_signal',
                 'macd_hist', 'boll_pct', 'boll_width', 'boll_pos', 'mom1', 'mom5',
                 'mom10', 'mom20', 'rmean5', 'rmean10', 'rmean20', 'rstd5',
                 'rstd10', 'rstd20', 'rmin10', 'rmax10', 'rmin20', 'rmax20',
                 'stoch14', 'atr_proxy', 'cci_proxy', 'obv', 'vol_z20', 'ema12_26']
N_BASE = len(BASE_FEATURES)
MOM_CHANS = [f'relmom_{L}' for L in MOM_LOOKS]
FEATURE_NAMES = BASE_FEATURES + MOM_CHANS
N_FEATURES = len(FEATURE_NAMES)


# ------------------------------------------------------------------ data
def load_data(path="data"):
    """Read every CSV in `path`, returning {ticker: (price, volume, dates)}."""
    files = sorted(f for f in os.listdir(path)
                   if f.lower().endswith(('.csv', '.txt')))
    pref_cols = ['adj close', 'close', 'value', 'last', 'price']
    out = {}
    for fname in files:
        name = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(path, fname))
        df.columns = [c.strip().lower() for c in df.columns]
        pcol = next((c for c in pref_cols if c in df.columns), None)
        if pcol is None:
            numeric = [c for c in df.columns
                       if pd.to_numeric(df[c], errors='coerce').notna().sum() > 0]
            pcol = numeric[-1] if numeric else None
        if pcol is None:
            continue
        df = df.rename(columns={pcol: 'price'})
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        vcol = next((c for c in ['volume', 'vol'] if c in df.columns), None)
        df['volume'] = (pd.to_numeric(df[vcol], errors='coerce')
                        if vcol else np.nan)
        dcol = next((c for c in ['date', 'datetime', 'timestamp'] if c in df.columns), None)
        df['date'] = pd.to_datetime(df[dcol], errors='coerce') if dcol else pd.NaT
        df = df.dropna(subset=['price']).reset_index(drop=True)
        if len(df) >= 100:
            out[name] = df[['price', 'volume', 'date']].copy()
    return out


# -------------------------------------------------------------- features
def engineer_features(price, volume):
    """30 strictly-causal per-asset price/volume channels.

    Every indicator uses only information available at t (backward-looking
    rolling / exponential windows and cumulative sums from the start of the
    series).  Nothing here peeks forward.
    """
    s = pd.Series(np.asarray(price, float))
    logp = np.log(s.where(s > 0))
    log_ret = logp.diff().fillna(0.0)
    pct_ret = s.pct_change().fillna(0.0)

    # --- RSI-14, Wilder smoothing (alpha = 1/14  <=>  com = 13) ---------
    d = s.diff().fillna(0.0)
    avg_g = d.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_l = (-d.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi = (100.0 - 100.0 / (1.0 + avg_g / (avg_l + 1e-9))).values

    # --- MACD: line, TRUE 9-period EMA of the MACD line, histogram ------
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()     # EMA9 of the MACD line
    macd_h = macd - macd_sig

    # --- Bollinger: %B, band width, position of the 20-MA ---------------
    bmid = s.rolling(20, min_periods=1).mean()
    bstd = s.rolling(20, min_periods=1).std().fillna(0.0)
    bup, bdn = bmid + 2 * bstd, bmid - 2 * bstd
    boll_pct = ((s - bdn) / (bup - bdn + 1e-9)).values
    boll_width = (bup - bdn).values
    boll_pos = ((s - bmid) / (bstd + 1e-9)).values

    # --- multi-horizon momentum -----------------------------------------
    mom = {h: s.diff(h).fillna(0.0).values for h in (1, 5, 10, 20)}

    # --- rolling level / volatility / range -----------------------------
    rmean = {w: s.rolling(w, min_periods=1).mean().values for w in (5, 10, 20)}
    rstd = {w: s.rolling(w, min_periods=1).std().fillna(0.0).values for w in (5, 10, 20)}
    rmin = {w: s.rolling(w, min_periods=1).min().values for w in (10, 20)}
    rmax = {w: s.rolling(w, min_periods=1).max().values for w in (10, 20)}

    # --- stochastic %K ---------------------------------------------------
    lo14 = s.rolling(14, min_periods=1).min()
    hi14 = s.rolling(14, min_periods=1).max()
    stoch = ((s - lo14) / (hi14 - lo14 + 1e-9) * 100.0).values

    # --- ATR proxy: 14-period mean |log return| -------------------------
    atr = log_ret.abs().rolling(14, min_periods=1).mean().values

    # --- CCI with the mean absolute deviation the text specifies -------
    ma20 = s.rolling(20, min_periods=1).mean()
    mad20 = (s - ma20).abs().rolling(20, min_periods=1).mean().replace(0, 1e-9)
    cci = ((s - ma20) / (0.015 * mad20)).values

    # --- On-Balance Volume: genuine signed volume ------------------------
    if volume is not None and np.isfinite(np.asarray(volume, float)).sum() > 0:
        vol = pd.Series(np.nan_to_num(np.asarray(volume, float)))
        obv = (np.sign(s.diff().fillna(0.0)) * vol).cumsum().values
    else:                                   # documented fallback
        obv = np.sign(log_ret).cumsum().values

    # --- volume shock (z-score of volume over 20 days) -------------------
    if volume is not None and np.isfinite(np.asarray(volume, float)).sum() > 0:
        vmu = vol.rolling(20, min_periods=1).mean()
        vsd = vol.rolling(20, min_periods=1).std().fillna(1.0)
        volz = ((vol - vmu) / (vsd + 1e-9)).values
    else:
        volz = np.zeros(len(s))

    # --- normalised EMA spread (scale-free, unlike raw EMA12/EMA26) ------
    emaspr = ((ema12 - ema26) / (s + 1e-9)).values

    cols = [logp.values, log_ret.values, pct_ret.values, rsi,
            macd.values, macd_sig.values, macd_h.values,
            boll_pct, boll_width, boll_pos,
            mom[1], mom[5], mom[10], mom[20],
            rmean[5], rmean[10], rmean[20],
            rstd[5], rstd[10], rstd[20],
            rmin[10], rmax[10], rmin[20], rmax[20],
            stoch, atr, cci, obv, volz, emaspr]
    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def cross_sectional_momentum(prices, lookbacks):
    """(T,K) prices -> (T,K,len(lookbacks)) strictly-causal trailing
    relative strength against the equal-weight log index."""
    T, K = prices.shape
    logp = np.log(np.maximum(prices, 1e-9))
    r = np.vstack([np.zeros((1, K)), np.diff(logp, axis=0)])
    rel = r - r.mean(axis=1, keepdims=True)     # demeaned cross-section
    cum = np.cumsum(rel, axis=0)
    out = np.empty((T, K, len(lookbacks)))
    for li, L in enumerate(lookbacks):
        lag = np.zeros_like(cum)
        lag[L:] = cum[L:] - cum[:-L]
        out[:, :, li] = lag
    return out


def mom_agg(X, n_mom=len(MOM_CHANS)):
    """last / mean / std of the trailing CS-momentum channels."""
    m = X[:, :, N_BASE:N_BASE + n_mom]
    return np.hstack([m[:, -1, :], m.mean(axis=1), m.std(axis=1)])


# ------------------------------------------------------- window building
def build_ticker(name, feats, dates):
    """Return dict of windowed train/val/test arrays for ONE ticker.

    The scaler is fitted on the sub-training rows only, so neither the
    validation nor the test rows influence the standardisation.
    """
    T = feats.shape[0]
    n_win = T - SEQ_LEN - HORIZON                 # 2514 - 64 - 90 = 2360
    split = int(n_win * (1 - TEST_FRAC))          # 1888
    embargo = SEQ_LEN + HORIZON                   # 154  (was 90 -> leaked)
    n_train_win = split - embargo                 # 1734
    n_val_raw = int(round(VAL_FRAC * n_train_win))
    val_embargo = SEQ_LEN
    n_sub = n_train_win - n_val_raw - val_embargo

    end_idx = np.arange(n_win)
    lab_lo = end_idx + SEQ_LEN - 1
    lab_hi = lab_lo + HORIZON
    d = np.asarray(dates)

    rows = dict(
        win=end_idx,
        d_win=d[lab_lo],
        y=(feats[lab_hi, 0] > feats[lab_lo, 0]).astype(np.int8),
    )
    part = {}
    is_test = end_idx >= split
    # The validation band must be bounded ABOVE as well as below.  Without the
    # upper bound the rule `(~is_test) & (end_idx >= n_sub + val_embargo)`
    # swallowed the 154-window train/test embargo band, so the validation
    # labels read prices up to row (split - 1 + SEQ_LEN - 1 + HORIZON) = 2040
    # while the first test window's input starts at row 1888: the decision
    # threshold was tuned on windows whose outcome lies inside the test
    # period.  That is exactly the leak this paper is about.
    is_val = ((~is_test) & (end_idx >= n_sub + val_embargo)
              & (end_idx < n_train_win))
    is_sub = end_idx < n_sub
    part['test'] = np.where(is_test)[0]
    part['val'] = np.where(is_val)[0]
    part['sub'] = np.where(is_sub)[0]
    # The two embargo bands are DISJOINT from each other and from the three
    # partitions: val_embargo sits immediately after the sub-training block and
    # before the validation block, embargo sits immediately after validation and
    # before test.  Their total is (SEQ_LEN) + (SEQ_LEN + HORIZON) = 64 + 154,
    # which is what makes the tiling identity below hold.  n_sub already
    # subtracts val_embargo, so the sum below counts each embargo exactly once.
    assert part['sub'].size + part['val'].size + part['test'].size \
        + val_embargo + embargo == n_win, (
            f'partition does not tile the sample: {part["sub"].size} + '
            f'{part["val"].size} + {part["test"].size} + {val_embargo} + '
            f'{embargo} != {n_win}')
    # and the three partitions must be disjoint and in order
    assert part['sub'].max() < part['val'].min() < part['test'].min(), \
        'partitions are not disjoint or not in chronological order'
    assert part['val'].max() - part['val'].min() + 1 == part['val'].size, \
        'validation block is not contiguous'

    # scaler fitted on sub-training rows only
    fit_rows = np.arange(0, n_sub + SEQ_LEN - 1)
    sc = StandardScaler().fit(feats[fit_rows])
    Z = sc.transform(feats).astype(np.float32)

    starts = end_idx
    Xw = np.lib.stride_tricks.sliding_window_view(Z, (SEQ_LEN, Z.shape[1]))[:, 0]
    Xw = np.ascontiguousarray(Xw, dtype=np.float32)
    return dict(name=name, Xw=Xw, y=rows['y'], d=rows['d_win'], parts=part,
                n_win=n_win, split=split, embargo=embargo,
                n_train_win=n_train_win, n_sub=n_sub, n_val_raw=n_val_raw,
                val_embargo=val_embargo)


# ------------------------------------------------------------ PyTorch ViT
class ViTEncoder(nn.Module):
    def __init__(self, n_feats=N_FEATURES, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = SEQ_LEN // patch_size
        self.proj = nn.Linear(patch_size * n_feats, D_MODEL)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, D_MODEL) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(D_MODEL, N_HEADS, FF_DIM, 0.1, 'gelu',
                                       batch_first=True) for _ in range(N_BLOCKS)])
        self.emb_dim = D_MODEL

    def forward(self, X):
        N, T, C = X.shape
        p = X[:, :self.n_patches * self.patch_size, :].reshape(
            N, self.n_patches, self.patch_size * C)
        z = self.proj(p) + self.pos
        z = torch.cat([self.cls.expand(N, -1, -1), z], dim=1)
        for b in self.blocks:
            z = b(z)
        return z[:, 0]


class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(1, GNN_HIDDEN)
        self.W2 = nn.Linear(GNN_HIDDEN, GNN_OUT)

    @staticmethod
    def _norm_adj(A):
        d = A.sum(-1)
        d = torch.where(d == 0, torch.ones_like(d), d)
        Dm = d.pow(-0.5).diag_embed()
        return Dm @ A @ Dm

    def forward(self, X):
        N, T, nf = X.shape
        sig = X.mean(dim=1).unsqueeze(-1)
        xc = X - X.mean(dim=1, keepdim=True)
        cov = torch.einsum('bti,btj->bij', xc, xc) / (T - 1 + 1e-6)
        s = cov.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
        corr = cov / (s.unsqueeze(-1) * s.unsqueeze(-2) + 1e-8)
        A = (corr.abs() > CORR_THRESH).float()
        A = A + torch.eye(nf, device=A.device)          # self loops
        Ah = self._norm_adj(A)
        H = torch.relu(torch.einsum('bik,bkj->bij', Ah, sig) @ self.W1.weight.T
                       + self.W1.bias)
        H = torch.relu(torch.einsum('bik,bkj->bij', Ah, H) @ self.W2.weight.T
                       + self.W2.bias)
        return H.mean(dim=1)


def head_stats(X):
    """Window summaries for the fusion head: per-channel mean/std/min/max/last
    plus the momentum aggregates."""
    m = X.mean(axis=1); s = X.std(axis=1)
    mn = X.min(axis=1); mx = X.max(axis=1); last = X[:, -1, :]
    return np.hstack([m, s, mn, mx, last, mom_agg(X)]).astype(np.float32)


def factor_features(X):
    """Reduced linear factor family shared with the ARIMA baseline."""
    p = X[:, :, 0]
    return np.hstack([p[:, -10:], np.diff(p[:, -10:], axis=1),
                      p.mean(1, keepdims=True), p.std(1, keepdims=True),
                      mom_agg(X)]).astype(np.float32)


class FusedViTGNN(nn.Module):
    def __init__(self, n_feats=N_FEATURES, factor_path=FACTOR_PATH):
        super().__init__()
        self.factor_path = factor_path
        self.vit = ViTEncoder(n_feats)
        self.gnn = GNNEncoder()
        n_sum = 5 * n_feats + 3 * len(MOM_CHANS)
        fused = D_MODEL + GNN_OUT + n_sum
        self.head = nn.Sequential(nn.Linear(fused, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))
        self.n_sum = n_sum
        if factor_path:
            n_fac = 10 + 9 + 2 + 3 * len(MOM_CHANS)
            self.factor_reg = nn.Linear(n_fac, 1)
            self.register_buffer('fac_shift', torch.zeros(n_fac))
            self.register_buffer('fac_scale', torch.ones(n_fac))

    def forward(self, X, stat=None, fac=None):
        h = torch.cat([self.vit(X), self.gnn(X)], dim=-1)
        if stat is not None:
            h = torch.cat([h, stat], dim=-1)
        logit = self.head(h).squeeze(-1)
        if self.factor_path and fac is not None:
            facz = (fac - self.fac_shift) / (self.fac_scale + 1e-8)
            logit = logit + self.factor_reg(facz).squeeze(-1)
        return logit


def train_torch(Xtr, ytr, Xva, yva, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    pos_ratio = ytr.sum() / (len(ytr) - ytr.sum() + 1e-6)
    model = FusedViTGNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_w = torch.tensor(min(max(pos_ratio, 0.5), 10.0),
                         dtype=torch.float32, device=DEVICE)
    stat_tr = torch.tensor(head_stats(Xtr), device=DEVICE)
    stat_va = torch.tensor(head_stats(Xva), device=DEVICE)
    fac_tr = torch.tensor(factor_features(Xtr), device=DEVICE)
    fac_va = torch.tensor(factor_features(Xva), device=DEVICE)
    if FACTOR_PATH:
        model.fac_shift.copy_(fac_tr.mean(0))
        model.fac_scale.copy_(fac_tr.std(0) + 1e-8)
    Xv = torch.tensor(Xva, device=DEVICE)
    N = len(Xtr); best_auc, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        idx = np.random.permutation(N)
        for st in range(0, N, BATCH):
            bidx = idx[st:st + BATCH]
            Xb = torch.tensor(Xtr[bidx], device=DEVICE)
            yb = torch.tensor(ytr[bidx], dtype=torch.float32, device=DEVICE)
            logit = model(Xb, stat_tr[bidx], fac_tr[bidx])
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv, stat_va, fac_va)).cpu().numpy()
            auc = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else -1.0
            if auc > best_auc:                      # manuscript: val AUC
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, Xw, bs=2048):
    stat = torch.tensor(head_stats(Xw), device=DEVICE)
    fac = torch.tensor(factor_features(Xw), device=DEVICE)
    model.eval()
    out = []
    for st in range(0, len(Xw), bs):
        Xb = torch.tensor(Xw[st:st + bs], device=DEVICE)
        out.append(torch.sigmoid(model(Xb, stat[st:st + bs],
                                       fac[st:st + bs])).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------- metrics
def compute_metrics(y, pred, proba_up):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    two = len(np.unique(y)) > 1
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'auc': float(roc_auc_score(y, proba_up)) if two else float('nan'),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'mcc': float(matthews_corrcoef(y, pred)),
        'cohen_kappa': float(cohen_kappa_score(y, pred)),
        'log_loss': float(log_loss(y, np.clip(proba_up, 1e-12, 1 - 1e-12))) if two else float('nan'),
        'brier': float(brier_score_loss(y, proba_up)),
        'average_precision': float(average_precision_score(y, proba_up)) if two else float('nan'),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'specificity': float(tn / (tn + fp + 1e-12)),
        'cm': cm.tolist(), 'y_true': y, 'pred': pred, 'proba_up': proba_up,
        'quantile_metrics': quantile_edge(y, proba_up),
    }


def quantile_edge(y, proba_up):
    p = np.asarray(proba_up)
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    out = {}
    for nm, q in [('top10', 0.10), ('top20', 0.20), ('top33', 1.0 / 3.0)]:
        k = max(1, int(round(len(ys) * q)))
        out[f'hitrate_{nm}'] = float(ys[:k].mean())
    step = max(1, int(round(len(ys) * 0.10)))
    out['long_short'] = float(ys[:step].mean() - ys[-step:].mean())
    out['rank_ic'] = float(stats.spearmanr(p, y).statistic)
    out['base_rate'] = float(np.mean(y))
    return out


def cross_sectional_ic(dates, y, proba_up):
    """Rank IC computed WITHIN each date, then averaged over dates.

    The pooled rank IC mixes the cross-sectional signal with the
    time-series direction signal; because the paper's hypothesis is
    cross-sectional momentum, the date-by-date statistic is the
    appropriate one and is reported with a date-level t statistic.
    """
    d = pd.factorize(pd.to_datetime(dates))[0]
    ics = []
    for g in np.unique(d):
        m = d == g
        if len(np.unique(y[m])) > 1:
            ics.append(stats.spearmanr(proba_up[m], y[m]).statistic)
    ics = np.asarray(ics, float)
    ics = ics[np.isfinite(ics)]
    if len(ics) < 3:
        return {'ic_mean': float('nan'), 'ic_t': float('nan'),
                'ic_p': float('nan'), 'ic_frac_pos': float('nan'),
                'n_dates': int(len(ics))}
    t, p = stats.ttest_1samp(ics, 0.0)
    return {'ic_mean': float(ics.mean()), 'ic_t': float(t), 'ic_p': float(p),
            'ic_frac_pos': float((ics > 0).mean()), 'n_dates': int(len(ics))}


def cluster_bootstrap_ci(y, pred, proba_up, dates, n=N_BOOT, seed=0):
    """95% CI for a SINGLE prediction set, resampling DATES (clusters).

    The 35,400 pooled test windows are 75 assets on ~472 common dates, so
    an iid bootstrap over windows understates the true interval.
    """
    rng = np.random.RandomState(seed)
    d = pd.factorize(pd.to_datetime(dates))[0]
    ud = np.unique(d)
    idx_by_d = [np.where(d == g)[0] for g in ud]
    res = {'accuracy': [], 'auc': [], 'mcc': []}
    for _ in range(n):
        pick = rng.randint(0, len(ud), len(ud))
        ii = np.concatenate([idx_by_d[k] for k in pick])
        yy, pp, qq = y[ii], pred[ii], proba_up[ii]
        res['accuracy'].append(accuracy_score(yy, pp))
        res['mcc'].append(matthews_corrcoef(yy, pp))
        res['auc'].append(roc_auc_score(yy, qq) if len(np.unique(yy)) > 1 else np.nan)
    out = {}
    for k, v in res.items():
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        out[k] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def _acc_mcc_from_cm(C):
    """C[..., 4] ordered (tn, fp, fn, tp) -> accuracy, MCC.  Both are ratios of
    per-window indicator sums, so they are exactly decomposable by cluster."""
    tn, fp, fn, tp = C[..., 0], C[..., 1], C[..., 2], C[..., 3]
    n = tn + fp + fn + tp
    acc = (tp + tn) / n
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return acc, mcc


def cluster_bootstrap_multiseed(y, preds, probas, dates, n=N_BOOT,
                                 n_auc=400, seed=0):
    """95% CI for the MULTI-SEED MEAN metric, resampling DATES as clusters.

    The headline row of every results table is the mean over seeds, so the
    interval must be built from the same quantity: each bootstrap replicate
    resamples the ~472 test dates with replacement, recomputes the metric for
    EVERY seed on the resampled dates, and averages over seeds.  (An interval
    for the last seed alone would not in general contain the reported mean.)
    """
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    d = pd.factorize(pd.to_datetime(dates))[0]
    ud = np.unique(d)
    D, S = len(ud), len(preds)

    y1 = y.astype(np.int8)
    cm = np.zeros((D, S, 4))
    for s, p in enumerate(preds):
        # Column order MUST match _acc_mcc_from_cm, which reads (tn, fp, fn, tp):
        #   tn = pred==0 and y==0 -> correct and y==0 -> ok*(1 - y)
        #   fp = pred==1 and y==0 -> wrong   and y==0 -> (1 - ok)*(1 - y)
        #   fn = pred==0 and y==1 -> wrong   and y==1 -> (1 - ok)*y
        #   tp = pred==1 and y==1 -> correct and y==1 -> ok*y
        # The previous order was (fp, fn, tn, tp), which is a permutation that
        # leaves the total intact but corrupts the numerator and the MCC
        # denominator, so every clustered CI was silently wrong.
        ok = (p == y).astype(np.int8)
        for k, col in enumerate([ok * (1 - y1), (1 - ok) * (1 - y1),
                                 (1 - ok) * y1, ok * y1]):
            cm[:, s, k] = np.bincount(d, weights=col, minlength=D)

    accs, mccs = [], []
    CH = 128
    for i in range(0, n, CH):
        b = min(CH, n - i)
        pick = rng.randint(0, D, size=(b, D))
        C = cm[pick].sum(axis=1)                       # (b, S, 4)
        a, m = _acc_mcc_from_cm(C)
        accs.append(a.mean(axis=1)); mccs.append(m.mean(axis=1))
    accs = np.concatenate(accs); mccs = np.concatenate(mccs)

    # AUC is not cluster-decomposable -> recompute directly on fewer draws
    idx_by_d = [np.where(d == g)[0] for g in ud]
    aucs = []
    for _ in range(int(n_auc)):
        pick = rng.randint(0, D, D)
        ii = np.concatenate([idx_by_d[k] for k in pick])
        yy = y[ii]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(np.mean([roc_auc_score(yy, probas[s][ii]) for s in range(S)]))
    aucs = np.asarray(aucs, float); aucs = aucs[np.isfinite(aucs)]

    def pc(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {'accuracy': pc(accs), 'mcc': pc(mccs),
            'auc': pc(aucs) if len(aucs) else [float('nan')] * 2,
            'n_seeds': int(S), 'n_bootstrap': int(n),
            'n_bootstrap_auc': int(len(aucs)),
            'cluster': 'date', 'n_date_clusters': int(D)}


def cross_sectional_ic_multiseed(dates, y, probas):
    """Rank IC computed WITHIN each date for every seed, then averaged.

    The headline is the mean of the per-seed date-mean ICs (what a single run
    would report); the t statistic is computed on the seed-averaged per-date
    ICs, which is the quantity the interval/significance refers to.
    """
    d = pd.factorize(pd.to_datetime(dates))[0]
    ud = np.unique(d)
    per_seed = np.full((len(probas), len(ud)), np.nan)
    for s, p in enumerate(probas):
        for j, g in enumerate(ud):
            m = d == g
            if len(np.unique(y[m])) > 1:
                per_seed[s, j] = stats.spearmanr(p[m], y[m]).statistic
    per_seed_mean = np.nanmean(per_seed, axis=1)
    shared = np.isfinite(per_seed).all(axis=0)
    ics = per_seed[:, shared].mean(axis=0)
    out = {'ic_mean': float(np.nanmean(per_seed_mean)),
           'ic_mean_per_seed': [float(v) for v in per_seed_mean],
           'n_dates': int(shared.sum())}
    if len(ics) >= 3 and np.std(ics) > 0:
        t, pv = stats.ttest_1samp(ics, 0.0)
        out.update({'ic_t': float(t), 'ic_p': float(pv),
                    'ic_frac_pos': float((ics > 0).mean()),
                    'ic_std_over_dates': float(np.std(ics, ddof=1))})
    else:
        out.update({'ic_t': float('nan'), 'ic_p': float('nan'),
                    'ic_frac_pos': float('nan'),
                    'ic_std_over_dates': float('nan')})
    return out


def mcnemar(y, p1, p2):
    b = int(np.sum((p1 == y) & (p2 != y)))
    c = int(np.sum((p1 != y) & (p2 == y)))
    if b + c == 0:
        return 1.0, b, c, 0.0
    z = (abs(b - c) - 1.0) / np.sqrt(b + c)
    return float(2 * (1 - stats.norm.cdf(z))), b, c, float((b - c) / (b + c))


def cluster_perm_p(y, m1, m2, dates, n=2000, seed=0):
    """Two-sided p-value for m1 - m2 > 0 with DATES as clusters.

    Uses a cluster wild-bootstrap (sign flips applied to whole dates),
    which is valid when observations are correlated within date.
    """
    rng = np.random.RandomState(seed)
    d = pd.factorize(pd.to_datetime(dates))[0]
    ud = np.unique(d)
    deltas = []
    for g in ud:
        m = d == g
        deltas.append(np.mean(m1[m] - m2[m]))
    deltas = np.asarray(deltas, float)
    t_obs = deltas.mean() / (deltas.std(ddof=1) / np.sqrt(len(deltas)) + 1e-12)
    signs = rng.choice([-1.0, 1.0], size=(n, len(deltas)))
    t_null = (signs * deltas).mean(1) / (deltas.std(ddof=1) / np.sqrt(len(deltas)) + 1e-12)
    p = float(np.mean(np.abs(t_null) >= abs(t_obs)))
    return max(p, 1.0 / n), float(t_obs)


def cluster_perm_p_multiseed(y, preds_a, preds_b, dates, n=2000, seed=0):
    """As cluster_perm_p but on the difference of the SEED-AVERAGED per-window
    correctness, so the test refers to the same quantity as the table row."""
    d = pd.factorize(pd.to_datetime(dates))[0]
    ud = np.unique(d)
    ca = np.mean([(p == y).astype(float) for p in preds_a], axis=0)
    cb = np.mean([(p == y).astype(float) for p in preds_b], axis=0)
    deltas = np.array([(ca - cb)[d == g].mean() for g in ud], float)
    if deltas.std(ddof=1) == 0:
        return 1.0, float('nan')
    rng = np.random.RandomState(seed)
    se = deltas.std(ddof=1) / np.sqrt(len(deltas))
    t_obs = deltas.mean() / se
    signs = rng.choice([-1.0, 1.0], size=(n, len(deltas)))
    t_null = (signs * deltas).mean(1) / se
    return max(float(np.mean(np.abs(t_null) >= abs(t_obs))), 1.0 / n), float(t_obs)


def paired_ttest(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a - b) == 0:
        return float('nan'), float('nan')
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def threshold_on_validation(p_val, y_val, n_grid=201):
    """Identical protocol for EVERY model (manuscript Section 2.8).

    The threshold is the cut that maximises balanced accuracy on the
    validation partition.  The grid is a quantile grid of the VALIDATION
    scores themselves, not a fixed probability interval: a fixed interval
    such as [0.30, 0.70] silently excludes the only useful cut whenever a
    model's scores are compressed into a narrow range or pushed towards one
    end by the class weighting, and would then report a model as
    uninformative purely because of an arbitrary constant.  Searching the
    validation score range is the same rule for every model and has no free
    parameter.
    """
    p = np.asarray(p_val, float)
    qs = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_grid)))
    best, best_thr = -1.0, 0.5
    for th in qs:
        b = balanced_accuracy_score(y_val, (p >= th).astype(int))
        if b > best:
            best, best_thr = b, float(th)
    return best_thr


# ------------------------------------------------------------- baselines
_KERNELS = {
    'diff':   np.array([1., -1.]),
    'smooth': np.ones(3) / 3,
    'gauss':  np.array([.1, .2, .4, .2, .1]),
    'lap':    np.array([1., -2., 1.]),
    'sobelx': np.array([-1., 0., 1.]),
}


def _conv_valid(A, k):
    """Vectorised 'valid' 1-D convolution along axis 1. A: (N, L)."""
    w = A.shape[1] - k.shape[0] + 1
    sw = np.lib.stride_tricks.sliding_window_view(A, k.shape[0], axis=1)
    return np.einsum('ijk,k->ij', sw, k)[:, :w]


def stat_block(X, n_ch=N_FEATURES):
    """Per-channel mean/std/min/max/last for the first n_ch channels."""
    Z = X[:, :, :n_ch]
    return np.hstack([Z.mean(1), Z.std(1), Z.min(1), Z.max(1), Z[:, -1, :]])


class ARIMAModel:
    NAME = "ARIMA"
    def fit_predict(self, Xtr, ytr, Xte, seed=0):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=0.1, max_iter=2000, solver='lbfgs',
                               random_state=seed)
        m.fit(factor_features(Xtr), ytr)
        return m.predict_proba(factor_features(Xte))[:, 1]


class GARCHModel:
    NAME = "GARCH"
    @staticmethod
    def ex(X):
        r = X[:, :, 1]
        def roll(w):
            """std of the LAST w log-returns of each window (window-end value)."""
            sw = np.lib.stride_tricks.sliding_window_view(r, w, axis=1)
            return sw.std(axis=2)[:, -1]
        v5, v10, v20 = roll(5), roll(10), roll(20)
        mu = r.mean(1)
        sd = r.std(1) + 1e-9
        sk = ((r - mu[:, None]) ** 3).mean(1) / sd ** 3
        ku = ((r - mu[:, None]) ** 4).mean(1) / sd ** 4
        arch = v5 / (v20 + 1e-9)               # short/long volatility ratio
        return np.column_stack([v5, v10, v20, v20 ** 2, v20 ** 3, r[:, -1], r[:, -2],
                                r[:, -3], sk, ku, arch, mom_agg(X)]).astype(np.float32)
    def fit_predict(self, Xtr, ytr, Xte, seed=0):
        m = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=seed)
        m.fit(self.ex(Xtr), ytr)
        return m.predict_proba(self.ex(Xte))[:, 1]


class CNNModel:
    NAME = "CNN"
    @staticmethod
    def ex(X):
        p = X[:, :, 0]
        feats = []
        for k in _KERNELS.values():
            c = _conv_valid(p, k)
            feats += [c.mean(1), c.std(1), c.max(1), np.abs(c).mean(1)]
        t = np.arange(p.shape[1], dtype=np.float32)
        tc = t - t.mean(); den = (tc ** 2).sum()
        slope = ((p - p.mean(1, keepdims=True)) * tc).sum(1) / den
        feats += [p.mean(1), p.std(1), p[:, -1] - p[:, 0], slope]
        return np.hstack([np.column_stack(feats), mom_agg(X)]).astype(np.float32)
    def fit_predict(self, Xtr, ytr, Xte, seed=0):
        m = RandomForestClassifier(n_estimators=150, max_depth=6, n_jobs=-1,
                                   random_state=seed)
        m.fit(self.ex(Xtr), ytr)
        return m.predict_proba(self.ex(Xte))[:, 1]


class RNNModel:
    NAME = "RNN"
    @staticmethod
    def ex(X):
        from scipy.signal import lfilter
        p = X[:, :, 0]
        h = lfilter([0.15], [1.0, -0.85], p, axis=1)      # h_t = .85 h_{t-1} + .15 v_t
        t = np.arange(p.shape[1], dtype=np.float32)
        tc = t - t.mean(); den = (tc ** 2).sum()
        slope = ((p - p.mean(1, keepdims=True)) * tc).sum(1) / den
        F = np.column_stack([h[:, -1], h[:, -5:].mean(1), h[:, -10:].mean(1),
                             h.max(1), h.min(1), h.std(1),
                             p[:, -1] - p[:, -5], p[:, -1] - p[:, -20],
                             p[:, -1] / (p[:, -20] + 1e-9) - 1.0, slope])
        return np.hstack([F, mom_agg(X)]).astype(np.float32)
    def fit_predict(self, Xtr, ytr, Xte, seed=0):
        m = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08,
                                       max_depth=4, random_state=seed)
        m.fit(self.ex(Xtr), ytr)
        return m.predict_proba(self.ex(Xte))[:, 1]


# ------------------------------------------------------------------ main
def _jsonable(o):
    """Serialise numpy containers as JSON arrays.

    `default=str` (used previously) silently turned every stored array into a
    quoted python-repr STRING, which made the result file unusable for
    downstream verification of the confusion matrix, the McNemar counts and
    the significance tests.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    if o is pd.NaT:
        return None
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _sanitize(o):
    """Recursively convert a result tree into strict-JSON-safe data.

    Non-finite floats become null rather than the non-standard NaN/Infinity
    tokens, and numpy arrays become real JSON lists.  A plain `default=str`
    destroys an array into a string such as "[0 1 1 ... 1 1 1]", which cannot
    be used to recompute any metric, and `allow_nan` defaults to emitting the
    non-standard NaN token for a float nan, which strict JSON parsers reject.
    """
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, (list, tuple, set)):
        return [_sanitize(v) for v in o]
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if o is None or o is pd.NaT:
        return None
    return str(o)


def _date_mean_ic(x, y, d):
    """Mean within-date Spearman rank IC between score x and binary label y,
    where d is a per-row date label.  Returns (mean, t, n_dates)."""
    ics = []
    for g in np.unique(d):
        m = d == g
        if m.sum() > 2 and len(np.unique(y[m])) > 1:
            r = stats.spearmanr(x[m], y[m]).statistic
            if np.isfinite(r):
                ics.append(r)
    ics = np.asarray(ics, float)
    if len(ics) < 3:
        return None, None, int(len(ics))
    t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)) + 1e-12))
    return float(ics.mean()), t, int(len(ics))


def momentum_regime_report(packs, names):
    """Is the cross-sectional momentum signal stable across the sample?

    Reports, for each of the six cross-sectional momentum channels and each
    protocol partition, the pooled AUC of the window-mean channel score for
    the up/down label and the mean within-date rank IC with its t statistic;
    and, for the 120-day channel, the same rank IC by calendar year of the
    window reference date.

    This is the diagnostic that separates "the model carries no signal" from
    "the signal changed sign between the training and the test period".
    """
    nc = len(MOM_CHANS)
    out = {'channels': list(MOM_CHANS),
           'statistic': 'window-mean channel score; rank IC computed within '
                        'each date, then averaged over dates',
           'pooled_auc': {}, 'per_period': {}}

    for pname in ('sub', 'val', 'test'):
        X, Y, D = [], [], []
        for pk in packs:
            idx = pk['parts'][pname]
            X.append(pk['Xw'][idx][:, :, N_BASE:].mean(axis=1))
            Y.append(pk['y'][idx])
            D.append(pd.to_datetime(pk['d'][idx]))
        X = np.concatenate(X); Y = np.concatenate(Y); D = pd.to_datetime(
            np.concatenate(D))
        two = len(np.unique(Y)) > 1
        out['pooled_auc'][pname] = {
            MOM_CHANS[c]: (float(roc_auc_score(Y, X[:, c])) if two else None)
            for c in range(nc)}
        dd = pd.factorize(D)[0]
        ic = {}
        for c in range(nc):
            mu, t, nd = _date_mean_ic(X[:, c], Y, dd)
            ic[MOM_CHANS[c]] = {'ic_mean': mu, 'ic_t': t, 'n_dates': nd}
        out['per_period'][pname] = ic

    # ---- the 120-day channel, by calendar year -------------------------
    c = MOM_CHANS.index('relmom_120') if 'relmom_120' in MOM_CHANS else 0
    years = sorted({str(v)[:4] for pk in packs for v in pk['d']})
    per_year = {}
    for yy in years:
        X, Y, D = [], [], []
        for pk in packs:
            idx = np.concatenate([pk['parts'][p] for p in
                                  ('sub', 'val', 'test')])
            d = pd.to_datetime(pk['d'][idx])
            m = np.array([str(v)[:4] == yy for v in d])
            idx = idx[m]
            X.append(pk['Xw'][idx][:, :, N_BASE + c].mean(axis=1))
            Y.append(pk['y'][idx]); D.append(pd.to_datetime(pk['d'][idx]))
        X = np.concatenate(X); Y = np.concatenate(Y)
        mu, t, nd = _date_mean_ic(X, Y, pd.factorize(pd.to_datetime(
            np.concatenate(D)))[0])
        per_year[yy] = {'ic_mean': mu, 'ic_t': t, 'n_dates': nd,
                        'n_windows': int(len(Y)),
                        'up_rate': float(Y.mean()) if len(Y) else None}
    out['per_year_relmom_120'] = per_year
    out['year_channel'] = MOM_CHANS[c]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=90)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--lookbacks', type=str, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--tickers', type=int, default=None)
    ap.add_argument('--tail', type=int, default=None)
    ap.add_argument('--datadir', type=str, default="data")
    ap.add_argument('--no-factor-path', action='store_true')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    global MOM_LOOKS, MOM_CHANS, FEATURE_NAMES, N_FEATURES, EPOCHS, \
        FACTOR_PATH, HORIZON
    if args.lookbacks:
        MOM_LOOKS = [int(x) for x in args.lookbacks.split(',')]
    MOM_CHANS = [f'relmom_{L}' for L in MOM_LOOKS]
    FEATURE_NAMES = BASE_FEATURES + MOM_CHANS
    N_FEATURES = len(FEATURE_NAMES)
    if args.epochs:
        EPOCHS = args.epochs
    FACTOR_PATH = not args.no_factor_path
    HORIZON = args.horizon

    print("=" * 78)
    print(f" CORRECTED leakage-controlled ViT-GNN   horizon={HORIZON}  "
          f"seeds={args.seeds}  n_features={N_FEATURES}")
    print("=" * 78)
    t0 = time.time()

    dfs = load_data(args.datadir)
    names = sorted(dfs.keys())
    if args.tickers:
        names = names[:args.tickers]
    print(f"  {len(names)} tickers | {len(dfs[names[0]])} rows each")
    print(f"  device={DEVICE} | epochs={EPOCHS} | lookbacks={MOM_LOOKS}")

    prices = np.column_stack([dfs[n]['price'].values for n in names])
    if args.tail:
        prices = prices[-args.tail:]
    cs_mom = cross_sectional_momentum(prices, MOM_LOOKS)

    packs, meta = [], []
    for ti, nm in enumerate(names):
        s = dfs[nm]['price'].values
        v = dfs[nm]['volume'].values
        d = dfs[nm]['date'].values
        if args.tail:
            s, v = s[-args.tail:], v[-args.tail:]
        f = engineer_features(s, v)
        f = np.column_stack([f, cs_mom[:len(f), ti, :]])
        pk = build_ticker(nm, f, d)
        pk['ti'] = ti
        packs.append(pk)

    def gather(part):
        X = np.concatenate([p['Xw'][p['parts'][part]] for p in packs])
        y = np.concatenate([p['y'][p['parts'][part]] for p in packs])
        dt = np.concatenate([p['d'][p['parts'][part]] for p in packs])
        tk = np.concatenate([np.full(len(p['parts'][part]), p['ti'], np.int16)
                             for p in packs])
        return X, y, dt, tk

    X_sub, y_sub, _, _ = gather('sub')
    X_val, y_val, _, _ = gather('val')
    X_te, y_te, d_te, t_te = gather('test')
    print(f"  windows/ticker: total {packs[0]['n_win']} = sub {packs[0]['n_sub']}"
          f" + valEmb {packs[0]['val_embargo']} + val {packs[0]['n_val_raw']}"
          f" + emb {packs[0]['embargo']} + test {packs[0]['n_win']-packs[0]['split']}")
    print(f"  pooled  sub={len(y_sub)} (up={y_sub.mean():.4f})  "
          f"val={len(y_val)} (up={y_val.mean():.4f})  "
          f"test={len(y_te)} (up={y_te.mean():.4f})")
    print(f"  embargo = SEQ_LEN + HORIZON = {SEQ_LEN + HORIZON} windows")

    # scaler for the ablation heads: fit on sub-training rows only
    sc = StandardScaler().fit(stat_block(X_sub))
    zs = lambda A: sc.transform(stat_block(A)).astype(np.float32)
    Z_sub, Z_val, Z_te = zs(X_sub), zs(X_val), zs(X_te)
    # features-only ablation: 30 price-derived channels, no CS momentum, no last
    n4 = 4 * N_BASE
    Zf_sub, Zf_val, Zf_te = Z_sub[:, :n4], Z_val[:, :n4], Z_te[:, :n4]

    runs = {}                       # name -> list of per-seed metric dicts
    ens = {'te': [], 'val': []}

    for sd in range(args.seeds):
        print(f"  ---- seed {sd} ----", flush=True)
        ts = time.time()
        model = train_torch(X_sub, y_sub, X_val, y_val, seed=sd)
        p_te = predict_torch(model, X_te)
        p_val = predict_torch(model, X_val)
        ens['te'].append(p_te); ens['val'].append(p_val)
        thr = threshold_on_validation(p_val, y_val)
        m = compute_metrics(y_te, (p_te >= thr).astype(int), p_te)
        m['thr'] = float(thr)
        runs.setdefault('ViT-GNN (ours, trained+CS)', []).append(m)
        print(f"    ViT-GNN  thr={thr:.2f}  ({time.time()-ts:.0f}s)", flush=True)

        ts = time.time()
        for nm, (Ztr, Zvl, Zte_) in [
                ('Abl-WindowStats+CS', (Z_sub, Z_val, Z_te)),
                ('Abl-FeaturesOnly', (Zf_sub, Zf_val, Zf_te))]:
            clf = HistGradientBoostingClassifier(max_iter=700, learning_rate=0.04,
                                                 max_depth=4, class_weight='balanced',
                                                 random_state=sd * 7 + 1)
            clf.fit(Ztr, y_sub)
            thr = threshold_on_validation(clf.predict_proba(Zvl)[:, 1], y_val)
            p = clf.predict_proba(Zte_)[:, 1]
            m = compute_metrics(y_te, (p >= thr).astype(int), p)
            m['thr'] = float(thr)
            runs.setdefault(nm, []).append(m)
        print(f"    ablations ({time.time()-ts:.0f}s)", flush=True)

        ts = time.time()
        for nm, cls in [('ARIMA', ARIMAModel), ('GARCH', GARCHModel),
                        ('CNN', CNNModel), ('RNN', RNNModel)]:
            bm = cls()
            p_val = bm.fit_predict(X_sub, y_sub, X_val, seed=sd)
            p_te = bm.fit_predict(X_sub, y_sub, X_te, seed=sd)
            thr = threshold_on_validation(p_val, y_val)
            m = compute_metrics(y_te, (p_te >= thr).astype(int), p_te)
            m['thr'] = float(thr)
            runs.setdefault(nm, []).append(m)
        print(f"    baselines ({time.time()-ts:.0f}s)", flush=True)

    # the deep ensemble is reported as its own, clearly-labelled row
    if args.seeds > 1:
        ete = np.mean(ens['te'], 0); eval_ = np.mean(ens['val'], 0)
        thr = threshold_on_validation(eval_, y_val)
        m = compute_metrics(y_te, (ete >= thr).astype(int), ete)
        m['thr'] = float(thr)
        ens_name = f'ViT-GNN ({args.seeds}-seed ensemble)'
        runs[ens_name] = [m]

    # -------------------------------------------------------- aggregate
    METRICS = ['accuracy', 'auc', 'f1', 'mcc', 'balanced_accuracy',
               'average_precision', 'cohen_kappa', 'log_loss', 'brier',
               'recall', 'precision', 'specificity']
    agg = {}
    for name, ms in runs.items():
        a = {f'{k}_mean': float(np.mean([m[k] for m in ms])) for k in METRICS}
        a.update({f'{k}_std': float(np.std([m[k] for m in ms], ddof=0))
                  for k in METRICS})
        a['acu_all'] = [float(m['accuracy']) for m in ms]
        a['auc_all'] = [float(m['auc']) for m in ms]
        a['mcc_all'] = [float(m['mcc']) for m in ms]
        a['thr_all'] = [float(m.get('thr', np.nan)) for m in ms]
        a['thr'] = float(ms[0].get('thr', np.nan))
        a['n_runs'] = int(len(ms))
        last = ms[-1]
        # stored arrays are the LAST seed's predictions (for reproducibility of
        # the confusion matrix and the single-run cluster tests)
        a['stored_arrays_from_run'] = int(len(ms) - 1)
        a['y_true'] = last['y_true']; a['pred'] = last['pred']
        a['proba_up'] = last['proba_up']; a['cm'] = last['cm']
        a['cm_per_seed'] = [m['cm'] for m in ms]
        # the interval is built for the SEED-AVERAGED metric, i.e. the same
        # quantity that a['accuracy_mean'] reports
        a['ci95_cluster'] = cluster_bootstrap_multiseed(
            ms[0]['y_true'], [m['pred'] for m in ms],
            [m['proba_up'] for m in ms], d_te)
        a['ci95_cluster_last_seed'] = cluster_bootstrap_ci(
            last['y_true'], last['pred'], last['proba_up'], d_te, n=400)
        a['quantile'] = {k: float(np.mean([m['quantile_metrics'][k] for m in ms]))
                         for k in ms[0]['quantile_metrics']}
        a['quantile_per_seed'] = {k: [float(m['quantile_metrics'][k]) for m in ms]
                                  for k in ms[0]['quantile_metrics']}
        a['xs_ic'] = cross_sectional_ic_multiseed(
            d_te, ms[0]['y_true'], [m['proba_up'] for m in ms])
        agg[name] = a

    # per-ticker performance of the fused deep model, averaged over seeds
    fused = 'ViT-GNN (ours, trained+CS)'
    ms_f = runs[fused]
    y_f = ms_f[0]['y_true']
    p_f = [m['proba_up'] for m in ms_f]
    r_f = [m['pred'] for m in ms_f]
    per_ticker = []
    for ti, nm in enumerate(names):
        m = t_te == ti
        yy = y_f[m]
        accs = [float(accuracy_score(yy, rr[m])) for rr in r_f]
        aucs = [float(roc_auc_score(yy, pp[m])) if len(np.unique(yy)) > 1
                else float('nan') for pp in p_f]
        mccs = [float(matthews_corrcoef(yy, rr[m])) for rr in r_f]
        per_ticker.append({
            'ticker': nm, 'n': int(m.sum()),
            'accuracy': float(np.mean(accs)),
            'accuracy_per_seed': accs,
            'accuracy_std': float(np.std(accs)),
            'auc': float(np.nanmean(aucs)) if np.isfinite(aucs).any() else None,
            'mcc': float(np.mean(mccs)),
            'up_rate': float(yy.mean()),
        })

    # ------------------------------------------------------------ pairwise
    order = ['ViT-GNN (ours, trained+CS)',
             f'ViT-GNN ({args.seeds}-seed ensemble)',
             'Abl-WindowStats+CS', 'Abl-FeaturesOnly', 'ARIMA', 'GARCH',
             'CNN', 'RNN']
    order = [o for o in order if o in agg]
    pairwise = {}
    for a_, b_ in combinations(order, 2):
        ya = runs[a_][0]['y_true']
        per_seed = []
        for sa, sb in zip(runs[a_], runs[b_]):
            p_i, nb, nc, r = mcnemar(ya, sa['pred'], sb['pred'])
            per_seed.append({'mcnemar_p': p_i, 'n_b': nb, 'n_c': nc,
                             'mcnemar_r': r})
        ps = [q['mcnemar_p'] for q in per_seed]
        cl_p, cl_t = cluster_perm_p_multiseed(
            ya, [m['pred'] for m in runs[a_]], [m['pred'] for m in runs[b_]], d_te)
        tt, tp = paired_ttest(agg[a_]['auc_all'], agg[b_]['auc_all'])
        pairwise[f'{a_} | {b_}'] = {
            'delta_acc': agg[a_]['accuracy_mean'] - agg[b_]['accuracy_mean'],
            'delta_mcc': agg[a_]['mcc_mean'] - agg[b_]['mcc_mean'],
            'n_seeds_a': len(runs[a_]), 'n_seeds_b': len(runs[b_]),
            'per_seed': per_seed,
            'mcnemar_p': float(np.median(ps)),
            'mcnemar_p_min': float(np.min(ps)),
            'mcnemar_p_max': float(np.max(ps)),
            'mcnemar_p_all': [float(v) for v in ps],
            'n_seeds_p_lt_05': int(sum(v < 0.05 for v in ps)),
            'mcnemar_r_mean': float(np.mean([q['mcnemar_r'] for q in per_seed])),
            'n_b': int(np.mean([q['n_b'] for q in per_seed])),
            'n_c': int(np.mean([q['n_c'] for q in per_seed])),
            'cluster_p': cl_p, 'cluster_t': cl_t,
            'paired_t_auc': tt, 'paired_t_auc_p': tp,
            'paired_t_auc_note': ('undefined: the baseline has a single run, so '
                                  'a paired t test over seeds does not exist'
                                  if len(runs[b_]) < 2 or len(runs[a_]) < 2
                                  else 'two-sided paired t over per-seed test AUC'),
        }

    # -------------------------------------------------------------- report
    print("\n" + "=" * 78)
    for name in order:
        a = agg[name]; q = a['quantile']; c = a['xs_ic']
        print(f"  {name:30s} acc={a['accuracy_mean']:.4f}+/-{a['accuracy_std']:.4f} "
              f"auc={a['auc_mean']:.4f} mcc={a['mcc_mean']:+.4f} "
              f"bacc={a['balanced_accuracy_mean']:.4f} "
              f"accCI=[{a['ci95_cluster']['accuracy'][0]:.3f},"
              f"{a['ci95_cluster']['accuracy'][1]:.3f}] "
              f"top10={q['hitrate_top10']:.4f} ic={q['rank_ic']:.4f} "
              f"xsIC={c['ic_mean']:+.4f}(t={c['ic_t']:+.1f})")
    print("  PAIRWISE (iid McNemar median | date-cluster p | effect r | n seeds p<.05):")
    for k, v in pairwise.items():
        print(f"    {k:56s} {v['mcnemar_p']:.2e} | {v['cluster_p']:.3f} "
              f"| r={v['mcnemar_r_mean']:+.3f} | {v['n_seeds_p_lt_05']}")

    out = {
        'horizon': HORIZON, 'n_seeds': args.seeds, 'epochs': EPOCHS,
        'seq_len': SEQ_LEN, 'embargo': SEQ_LEN + HORIZON,
        'embargo_rule': 'SEQ_LEN + HORIZON',
        'n_features': N_FEATURES, 'features': FEATURE_NAMES,
        'n_tickers': len(names), 'tickers': names,
        'n_sub': int(len(y_sub)), 'n_val': int(len(y_val)), 'n_test': int(len(y_te)),
        'up_sub': float(y_sub.mean()), 'up_val': float(y_val.mean()),
        'up_test': float(y_te.mean()),
        'windows_per_ticker': packs[0]['n_win'],
        'windows_per_ticker_sub': packs[0]['n_sub'],
        'windows_per_ticker_val': packs[0]['n_val_raw'],
        'windows_per_ticker_embargo': packs[0]['embargo'],
        'windows_per_ticker_val_embargo': packs[0]['val_embargo'],
        'windows_per_ticker_test': packs[0]['n_win'] - packs[0]['split'],
        'test_dates': sorted({str(x)[:10] for x in d_te}),
        'models': agg, 'pairwise': pairwise, 'per_ticker': per_ticker,
        'momentum_regime': momentum_regime_report(packs, names),
    }
    fn = args.out or os.path.join(OUTPUT_DIR, f'results_h{HORIZON}.json')
    with open(fn, 'w') as f:
        json.dump(_sanitize(out), f, indent=1, allow_nan=False)
    print(f"\n  wrote {fn}  ({time.time()-t0:.0f}s total)")

    mr = out['momentum_regime']
    print("\n  MOMENTUM REGIME (window-mean cross-sectional momentum, "
          "within-date rank IC):")
    for pname in ('sub', 'val', 'test'):
        i120 = mr['per_period'][pname]['relmom_120']
        print(f"    {pname:5s} relmom_120  IC={i120['ic_mean']:+.4f} "
              f"t={i120['ic_t']:+7.2f}  (pooled AUC of the same channel = "
              f"{mr['pooled_auc'][pname]['relmom_120']:.4f})")
    print("    by calendar year:")
    for yy, v in mr['per_year_relmom_120'].items():
        print(f"      {yy}  IC={v['ic_mean']:+.4f}  t={v['ic_t']:+7.2f}  "
              f"n_dates={v['n_dates']:4d}  up_rate={v['up_rate']:.3f}")


HORIZON = 90

if __name__ == "__main__":
    main()
