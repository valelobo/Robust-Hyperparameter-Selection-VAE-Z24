# -*- coding: utf-8 -*-
"""
======================================================================
Z24 BRIDGE - SHALLOW VAE DAMAGE DETECTION - EXTENDED EXPERIMENT (v2)
======================================================================

All experiments below are PRE-DECLARED and ALL are reported. Nothing is
selected by looking at outer-test results.

  E0  Exploratory PSD plot (normal vs damage, descriptive only)
  E1  Baseline under the identical outer folds + threshold protocol:
        z-distance to the normal mean (log-PSD)
        (RMS and PCA baselines removed in v3; old checkpoints containing them are ignored)
  E2  Noise floor: one FIXED VAE config evaluated repeatedly with different
      seeds/splits -> how large is objective noise compared with the spread
      between Optuna trials? (old 1x3-fold protocol vs new repeated protocol)
  E3  Nested-CV VAE, arm "vae_psd": input = log10 Welch PSD, nperseg searched
  E4  Nested-CV VAE, arm "vae_raw": input = raw 1000-point signal
  E5  "vae_nested": per outer fold, the arm (psd/raw) with the better INNER
      objective -> legitimate nested choice of representation

Protocol per outer fold (10-fold stratified, SEED=23):
  outer TEST (10%) ............ untouched until final scoring
  outer TRAIN (90%)
    Optuna objective ........... repeated stratified K-fold (INNER_SPLITS x
                                 INNER_REPEATS); VAE fit on inner-train NORMALS
                                 only; objective = mean over splits of
                                 0.5*(AUC + F1@Youden) on inner validation
    Selection .................. one-standard-error rule (simplest model, by
                                 parameter count, within 1 SE of the best)
    Final model / threshold .... cross-fitting with FINAL_CV_SPLITS models:
                                 each model trains on the normals of 4/5 of
                                 outer-train; out-of-fold scores (~58 normals +
                                 ~58 damaged) -> ROC -> Youden threshold (FIXED)
    Outer TEST ................. score = mean of the FINAL_CV_SPLITS models;
                                 AUC (threshold-free) + metrics @ fixed threshold

Architecture: input -> Dense(N) -> (z_mean, z_log_var) -> Dense(N) -> input
Full-batch, no dropout, no L2. Loss = MSE + beta*KL.
Score = mean((x - decoder(z_mean))^2).

Savepoints: every Optuna trial (SQLite), every finished fold / baseline /
noise-floor repetition (pickle). Re-running the script resumes where it
stopped.
  REPORT_ONLY=True : no training; rebuild tables, figures AND plot data from checkpoints
  REPLOT_ONLY=True : only redraw figures from <OUTPUT_DIR>/plot_data/ (edit style, rerun)
Figures: MSSP palette, PNG (600 dpi) + PDF with embedded fonts.
Plot data: <OUTPUT_DIR>/plot_data/<figure>.pkl + <figure>/<table>.csv + meta.json
======================================================================
"""

# ======================================================================
# 0. RUN MODE / ENVIRONMENT
# ======================================================================
import os
import sys

QUICK_TEST = False    # True = ~5-10 min smoke test of the whole pipeline (separate folder)
REPORT_ONLY = False   # True = no training; rebuild tables/figures/plot data from checkpoints
REPLOT_ONLY = True   # True = only redraw figures from saved plot data (no data file, no checkpoints)
SHOW_PLOTS = False    # keep False for unattended runs (plt.show() would block)

SEED = 23
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import gc
import glob
import json
import pickle
import random
import time
import traceback
import warnings

import matplotlib
if not SHOW_PLOTS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from scipy import stats
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
except Exception:
    pass
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================================
# 1. CONFIGURATION
# ======================================================================
DATA_FILE = "Z24_R3V_Feb_to_Aug.npz"           # None = auto-detect file with X, y in script folder
FS = 100.0                 # sampling frequency (Hz). ONLY used for axis labels.
                           # CHECK THIS for your extraction; it does not affect results.

N_OUTER = 10
INNER_SPLITS = 5           # Optuna objective: repeated stratified K-fold ...
INNER_REPEATS = 2          # ... = 10 VAE fits per trial (was 3)
N_TRIALS = 150             # per outer fold and per arm
STUDY_TIMEOUT_S = None     # optional safety cap per study (seconds); None = off
TPE_STARTUP = 20           # random trials before TPE starts modelling
ONE_SE_RULE = True

EPOCHS = 800
LEARNING_RATE = 1e-3
HIDDEN_ACTIVATION = "relu"
LOGVAR_CLIP = 10.0
SCALER = "standard"        # fitted on training normals only
LOG_EVERY = 10             # loss-curve resolution for recorded models

FINAL_CV_SPLITS = 5        # cross-fitting models for threshold + test ensemble

NEURON_OPTIONS = [16, 32, 64, 128, 256, 512]
LATENT_OPTIONS = [2, 4, 8, 16, 32, 64, 128]
BETA_RANGE = (1e-4, 1e-1)
PSD_NPERSEG_OPTIONS = [256, 512, 1000]   # -> 129, 257, 501 frequency bins
PSD_BASELINE_NPERSEG = 512               # fixed nperseg for PSD baselines / EDA

ARMS = ["vae_psd", "vae_raw"]            # order of execution
NOISE_FLOOR_FOLD = 1
NOISE_FLOOR_REPS = 20
NOISE_FLOOR_CONFIGS = {
    "vae_psd": dict(neurons=128, latent=8, beta=1e-3, feature="psd512"),
    "vae_raw": dict(neurons=128, latent=8, beta=1e-3, feature="raw"),
}
BOOTSTRAP_N = 2000

# Figures (MSSP style)
FIG_DPI = 600
FIG_FORMATS = ("png", "pdf")
SUPTITLES = True          # figure-level titles as in the reference; False for caption-only figures
MAIN_METHOD = "vae_psd"   # headline method of the main multipanel figure
SECOND_METHOD = "vae_raw" # second confusion matrix of the main figure

EXPECTED_SHAPE = (1170, 1000)
OUTPUT_DIR = "z24_vae_overnight"
RUN_FOLDS = list(range(1, N_OUTER + 1))

if QUICK_TEST:
    N_TRIALS, EPOCHS, TPE_STARTUP = 4, 60, 2
    INNER_SPLITS, INNER_REPEATS, FINAL_CV_SPLITS = 3, 1, 3
    NOISE_FLOOR_REPS, BOOTSTRAP_N = 3, 200
    RUN_FOLDS = [1, 2]
    OUTPUT_DIR = OUTPUT_DIR + "_QUICKTEST"

CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TAB_DIR = os.path.join(OUTPUT_DIR, "tables")
PLOT_DATA_DIR = os.path.join(OUTPUT_DIR, "plot_data")

ARCH_CHOICES = [f"N{n}_L{l}" for n in NEURON_OPTIONS for l in LATENT_OPTIONS if l < n]

# RMS and PCA baselines removed on request; only the z-distance baseline is computed/reported.
BASELINES = ["bl_z_psd"]
METHOD_ORDER = ["vae_psd", "vae_raw", "vae_nested"] + BASELINES
METHOD_LABEL = {
    "vae_psd": "VAE (log-PSD)",
    "vae_raw": "VAE (raw)",
    "vae_nested": "VAE (arm chosen in inner CV)",
    "bl_z_psd": "z-distance (log-PSD)",
}
METHOD_SHORT = {"vae_psd": "VAE\nPSD", "vae_raw": "VAE\nraw",
                "vae_nested": "VAE\nnested", "bl_z_psd": "z-dist.\nPSD"}
METRICS = ["AUC", "Accuracy", "BalancedAcc", "Precision", "Recall",
           "Specificity", "F1", "MCC"]
ARM_ID = {"vae_psd": 1, "vae_raw": 2}


def parse_arch(code):
    a, b = code.split("_")
    return int(a[1:]), int(b[1:])


def vae_param_count(d, n, l):
    return (d * n + n) + 2 * (n * l + l) + (l * n + n) + (n * d + d)


# ======================================================================
# 2. INFRASTRUCTURE: seeds, logging, checkpoints
# ======================================================================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
set_global_seed(SEED)


class Tee:
    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.__stdout__

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def ckpt_file(name):
    return os.path.join(CKPT_DIR, name + ".pkl")


def save_ckpt(name, obj):
    path = ckpt_file(name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written file


def load_ckpt(name):
    path = ckpt_file(name)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return None


def free_tf():
    tf.keras.backend.clear_session()
    gc.collect()


def get_storage():
    """Optuna storage persisted to disk (every trial is a savepoint)."""
    try:
        url = "sqlite:///" + os.path.abspath(os.path.join(CKPT_DIR, "optuna.db"))
        return optuna.storages.RDBStorage(url, engine_kwargs={"connect_args": {"timeout": 120}})
    except Exception as exc:
        print(f"[warn] SQLite storage unavailable ({exc}); trying journal file.")
    try:
        from optuna.storages import JournalStorage
        try:
            from optuna.storages.journal import JournalFileBackend as Backend
        except ImportError:
            from optuna.storages import JournalFileStorage as Backend
        return JournalStorage(Backend(os.path.join(CKPT_DIR, "optuna_journal.log")))
    except Exception as exc:
        print(f"[warn] No persistent Optuna storage ({exc}). Trials will NOT be resumable.")
        return None


def fmt_time(s):
    s = int(s)
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


# ======================================================================
# 3. DATA
# ======================================================================
def _script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def _read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    if ext == ".mat":
        try:
            from scipy.io import loadmat
            d = loadmat(path)
            return {k: v for k, v in d.items() if not k.startswith("__")}
        except NotImplementedError:
            import h5py
            with h5py.File(path, "r") as f:
                return {k: np.array(f[k]) for k in f.keys()}
    if ext in (".pkl", ".pickle", ".joblib"):
        if ext == ".joblib":
            import joblib
            obj = joblib.load(path)
        else:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return {"X": obj[0], "y": obj[1]}
        return None
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]) for k in f.keys()}
    return None


def load_dataset():
    if DATA_FILE is not None:
        p = DATA_FILE if os.path.isabs(DATA_FILE) else os.path.join(_script_dir(), DATA_FILE)
        candidates = [p]
    else:
        exts = ("*.npz", "*.mat", "*.pkl", "*.pickle", "*.joblib", "*.h5", "*.hdf5")
        candidates = sorted(p for e in exts for p in glob.glob(os.path.join(_script_dir(), e)))
    for path in candidates:
        try:
            d = _read_file(path)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")
            continue
        if d is None:
            continue
        X = d.get("X", d.get("x"))
        y = d.get("y", d.get("Y"))
        if X is None or y is None:
            continue
        X, y = np.asarray(X), np.asarray(y)
        if X.shape == EXPECTED_SHAPE[::-1]:
            X = X.T
        y = y.reshape(-1)
        if X.shape != EXPECTED_SHAPE or y.shape != (EXPECTED_SHAPE[0],):
            print(f"  [skip] {os.path.basename(path)}: X{X.shape}, y{y.shape}")
            continue
        if not np.all(np.isin(np.unique(y), [0, 1])):
            raise ValueError(f"Labels must be 0/1, found {np.unique(y)}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN/Inf.")
        y = y.astype(int)
        print(f"Loaded: {os.path.basename(path)} | X{X.shape} {X.dtype} | "
              f"normal={np.sum(y == 0)} damaged={np.sum(y == 1)}")
        return X, y
    raise FileNotFoundError(f"No valid X/y file found in {_script_dir()}. Set DATA_FILE.")


def log_psd(X, nperseg):
    """Per-sample log10 Welch PSD. Sample-wise transform: no information is
    shared between samples, so it cannot leak across folds."""
    f, P = welch(X, fs=FS, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                 detrend="constant", scaling="density", axis=1)
    return f, np.log10(np.maximum(P, np.finfo(float).tiny))


def build_features(X):
    feats, freqs = {"raw": X}, {}
    for nps in sorted(set(PSD_NPERSEG_OPTIONS + [PSD_BASELINE_NPERSEG])):
        f, L = log_psd(X, nps)
        feats[f"psd{nps}"], freqs[f"psd{nps}"] = L, f
    return feats, freqs


# ======================================================================
# 4. MODELS / DETECTORS
# ======================================================================
class ShallowVAE(tf.keras.Model):
    def __init__(self, input_dim, neurons, latent_dim):
        super().__init__()
        self.enc_hidden = tf.keras.layers.Dense(neurons, activation=HIDDEN_ACTIVATION)
        self.z_mean_layer = tf.keras.layers.Dense(latent_dim)
        self.z_log_var_layer = tf.keras.layers.Dense(latent_dim)
        self.dec_hidden = tf.keras.layers.Dense(neurons, activation=HIDDEN_ACTIVATION)
        self.dec_out = tf.keras.layers.Dense(input_dim, activation="linear")

    def encode(self, x):
        h = self.enc_hidden(x)
        return (self.z_mean_layer(h),
                tf.clip_by_value(self.z_log_var_layer(h), -LOGVAR_CLIP, LOGVAR_CLIP))

    def decode(self, z):
        return self.dec_out(self.dec_hidden(z))

    def call(self, x):
        return self.decode(self.encode(x)[0])


def train_vae(Xs, neurons, latent_dim, beta, seed, record_history=False):
    set_global_seed(seed)
    x = tf.constant(Xs, dtype=tf.float32)
    model = ShallowVAE(Xs.shape[1], neurons, latent_dim)
    model(x)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    beta_t = tf.constant(beta, dtype=tf.float32)

    def step():
        with tf.GradientTape() as tape:
            mu, lv = model.encode(x)
            z = mu + tf.exp(0.5 * lv) * tf.random.normal(tf.shape(mu))
            xr = model.decode(z)
            rec = tf.reduce_mean(tf.reduce_mean(tf.square(x - xr), axis=1))
            kl = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + lv - tf.square(mu) - tf.exp(lv), axis=1))
            loss = rec + beta_t * kl
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, rec, kl

    hist = {"epoch": [], "loss": [], "rec": [], "kl": []}

    def log(ep, out):
        hist["epoch"].append(ep)
        for k, v in zip(("loss", "rec", "kl"), out):
            hist[k].append(float(v))

    out = step()
    if record_history:
        log(1, out)
    g = tf.function(step)
    for ep in range(2, EPOCHS + 1):
        out = g()
        if record_history and (ep % LOG_EVERY == 0 or ep == EPOCHS):
            log(ep, out)
    final = {"loss": float(out[0]), "rec": float(out[1]), "kl": float(out[2])}
    return model, final, hist


def make_scaler():
    return StandardScaler() if SCALER == "standard" else MinMaxScaler()


class VAEDetector:
    is_vae = True

    def __init__(self, neurons, latent, beta, seed, record_history=False):
        self.neurons, self.latent, self.beta = neurons, latent, beta
        self.seed, self.record = seed, record_history

    def fit(self, Xn):
        self.scaler = make_scaler().fit(Xn)
        self.model, self.final, self.history = train_vae(
            self.scaler.transform(Xn), self.neurons, self.latent, self.beta,
            self.seed, self.record)
        self.finite = bool(np.all(np.isfinite(list(self.final.values()))))
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        mu, _ = self.model.encode(tf.constant(Xs, dtype=tf.float32))
        xr = self.model.decode(mu).numpy()
        return np.mean((Xs - xr) ** 2, axis=1)

    def latent_means(self, X):
        mu, _ = self.model.encode(tf.constant(self.scaler.transform(X), dtype=tf.float32))
        return mu.numpy()


class ZDistDetector:
    is_vae = False

    def fit(self, Xn):
        self.mu = Xn.mean(0)
        self.sd = Xn.std(0, ddof=1)
        self.sd[self.sd == 0] = 1.0
        return self

    def score(self, X):
        return np.mean(((X - self.mu) / self.sd) ** 2, axis=1)


BASELINE_SPEC = {
    "bl_z_psd": (f"psd{PSD_BASELINE_NPERSEG}", ZDistDetector),
}


# ======================================================================
# 5. THRESHOLD / METRICS
# ======================================================================
def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    i = int(np.argmax(j))
    t = thr[i]
    if not np.isfinite(t):
        t = float(np.max(s)) + 1e-12
    return float(t), float(j[i])


def classification_metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "Accuracy": accuracy_score(y, pred), "BalancedAcc": 0.5 * (rec + spec),
        "Precision": precision_score(y, pred, zero_division=0), "Recall": rec,
        "Specificity": spec, "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "cm": cm,
    }


def split_objective(y, s):
    auc = roc_auc_score(y, s)
    thr, _ = youden_threshold(y, s)
    f1 = f1_score(y, (s >= thr).astype(int), zero_division=0)
    return 0.5 * (auc + f1), auc, f1


# ======================================================================
# 6. INNER OBJECTIVE (used by Optuna and by the noise-floor experiment)
# ======================================================================
def inner_objective(cfg, Xf, y, splits, seed_base):
    vals, aucs, f1s = [], [], []
    for i, (tr, va) in enumerate(splits):
        det = VAEDetector(cfg["neurons"], cfg["latent"], cfg["beta"], seed=seed_base + i)
        det.fit(Xf[tr][y[tr] == 0])
        s = det.score(Xf[va]) if det.finite else None
        del det
        free_tf()
        if s is None or not np.all(np.isfinite(s)):
            return None
        v, a, f = split_objective(y[va], s)
        vals.append(v); aucs.append(a); f1s.append(f)
    vals = np.array(vals)
    n = len(vals)
    return {"mean": float(vals.mean()),
            "se": float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "vals": [float(v) for v in vals],
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)) if n > 1 else 0.0}


def inner_splits_for(y_otr, fold, n_splits=None, n_repeats=None, rs=None):
    n_splits = n_splits or INNER_SPLITS
    n_repeats = n_repeats or INNER_REPEATS
    rs = SEED + fold if rs is None else rs
    if n_repeats == 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    else:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=rs)
    return list(cv.split(np.zeros(len(y_otr)), y_otr))


# ======================================================================
# 7. CROSS-FIT EVALUATION (identical for VAEs and baselines)
# ======================================================================
def crossfit_evaluate(make_det, X_otr, y_otr, X_te, y_te, split_seed, extract=False):
    cv = StratifiedKFold(n_splits=FINAL_CV_SPLITS, shuffle=True, random_state=split_seed)
    oof = np.full(len(y_otr), np.nan)
    te_all, extra = [], {}
    for k, (tr, va) in enumerate(cv.split(X_otr, y_otr)):
        det = make_det(k)
        Xn = X_otr[tr][y_otr[tr] == 0]
        det.fit(Xn)
        oof[va] = det.score(X_otr[va])
        te_all.append(det.score(X_te))
        if extract and k == 0 and det.is_vae:
            Zfit = det.latent_means(Xn)
            pca = PCA(n_components=2, random_state=SEED).fit(Zfit)
            va_n, dmg = va[y_otr[va] == 0], np.where(y_otr == 1)[0]
            extra["latent_2d"] = {
                "fit_normal": pca.transform(Zfit),
                "val_normal": pca.transform(det.latent_means(X_otr[va_n])),
                "train_damage": pca.transform(det.latent_means(X_otr[dmg])),
                "test_normal": pca.transform(det.latent_means(X_te[y_te == 0])),
                "test_damage": pca.transform(det.latent_means(X_te[y_te == 1])),
                "evr": pca.explained_variance_ratio_}
            extra["loss_history"] = det.history
            extra["final_loss"] = det.final
        is_vae = det.is_vae
        del det
        if is_vae:
            free_tf()
    te_all = np.vstack(te_all)
    s_te = te_all.mean(0)
    nonfinite = not (np.all(np.isfinite(oof)) and np.all(np.isfinite(s_te)))
    if nonfinite:
        big = np.nanmax(np.where(np.isfinite(oof), oof, np.nan)) * 10 + 1.0
        oof = np.where(np.isfinite(oof), oof, big)
        s_te = np.where(np.isfinite(s_te), s_te, big)

    thr, j = youden_threshold(y_otr, oof)
    oof_pred = (oof >= thr).astype(int)
    oof_m = classification_metrics(y_otr, oof_pred)
    pred = (s_te >= thr).astype(int)
    m = classification_metrics(y_te, pred)
    fpr, tpr, _ = roc_curve(y_te, s_te)
    res = {"AUC": float(roc_auc_score(y_te, s_te)), "threshold": thr,
           "oof_auc": float(roc_auc_score(y_otr, oof)), "oof_J": j,
           "oof_recall": oof_m["Recall"], "oof_specificity": oof_m["Specificity"],
           "fpr": fpr, "tpr": tpr, "scores_test": s_te, "scores_test_models": te_all,
           "y_test": y_te, "pred": pred, "nonfinite": nonfinite}
    res.update(m)
    res.update(extra)
    return res


# ======================================================================
# 8. EXPERIMENTS
# ======================================================================
def run_baselines(fold, feats, y, tr_idx, te_idx):
    name = f"baselines_fold{fold:02d}"
    out = load_ckpt(name) or {}          # old checkpoints may hold extra methods: kept, ignored
    missing = [m for m in BASELINES if m not in out]
    for meth in missing:
        feat, cls = BASELINE_SPEC[meth]
        X = feats[feat]
        r = crossfit_evaluate(lambda k, c=cls: c(), X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                              split_seed=SEED + 500 + fold)
        r.update({"fold": fold, "method": meth, "te_idx": te_idx})
        out[meth] = r
    if missing:
        save_ckpt(name, out)
    return {m: out[m] for m in BASELINES}


def run_noise_floor(arm, feats, y, outer_splits):
    name = f"noise_floor_{arm}"
    rows = load_ckpt(name) or []
    cfg = NOISE_FLOOR_CONFIGS[arm]
    tr_idx, _ = outer_splits[NOISE_FLOOR_FOLD - 1]
    Xf, y_otr = feats[cfg["feature"]][tr_idx], y[tr_idx]
    for r in range(len(rows), NOISE_FLOOR_REPS):
        t0 = time.time()
        sA = inner_splits_for(y_otr, 0, n_splits=3, n_repeats=1, rs=SEED + 7000 + r)
        sB = inner_splits_for(y_otr, 0, rs=SEED + 8000 + r)
        a = inner_objective(cfg, Xf, y_otr, sA, seed_base=SEED + 60000 + 100 * r)
        b = inner_objective(cfg, Xf, y_otr, sB, seed_base=SEED + 70000 + 100 * r)
        rows.append({"rep": r, "old_1x3": a["mean"] if a else np.nan,
                     "new_repeated": b["mean"] if b else np.nan})
        save_ckpt(name, rows)
        print(f"  noise floor {arm} rep {r + 1}/{NOISE_FLOOR_REPS}: "
              f"1x3={rows[-1]['old_1x3']:.4f}  {INNER_REPEATS}x{INNER_SPLITS}="
              f"{rows[-1]['new_repeated']:.4f}  ({time.time() - t0:.0f}s)")
    return rows


def suggest_config(trial, arm):
    n, l = parse_arch(trial.suggest_categorical("arch", ARCH_CHOICES))
    beta = trial.suggest_float("beta", BETA_RANGE[0], BETA_RANGE[1], log=True)
    feat = "raw"
    if arm == "vae_psd":
        feat = f"psd{trial.suggest_categorical('nperseg', PSD_NPERSEG_OPTIONS)}"
    return {"neurons": n, "latent": l, "beta": beta, "feature": feat}


def cfg_from_params(params, arm):
    n, l = parse_arch(params["arch"])
    feat = f"psd{params['nperseg']}" if arm == "vae_psd" else "raw"
    return {"neurons": n, "latent": l, "beta": params["beta"], "feature": feat}


def make_objective(arm, X_otr_feats, y_otr, splits, seed_base):
    def objective(trial):
        cfg = suggest_config(trial, arm)
        Xf = X_otr_feats[cfg["feature"]]
        trial.set_user_attr("n_params", int(vae_param_count(Xf.shape[1], cfg["neurons"], cfg["latent"])))
        if cfg["latent"] >= Xf.shape[1]:
            raise optuna.TrialPruned()
        res = inner_objective(cfg, Xf, y_otr, splits, seed_base)
        if res is None:
            trial.set_user_attr("failed", True)
            return 0.0
        trial.set_user_attr("obj_se", res["se"])
        trial.set_user_attr("obj_vals", res["vals"])
        for k in ("auc_mean", "auc_std", "f1_mean", "f1_std"):
            trial.set_user_attr(k, res[k])
        return res["mean"]
    return objective


def completed_trials(study):
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def select_trial(study):
    ok = [t for t in completed_trials(study) if not t.user_attrs.get("failed", False)]
    best = max(ok, key=lambda t: t.value)
    if not ONE_SE_RULE:
        return best, best
    cut = best.value - best.user_attrs.get("obj_se", 0.0)
    cands = [t for t in ok if t.value >= cut]
    chosen = min(cands, key=lambda t: (t.user_attrs["n_params"], -t.value))
    return chosen, best


def run_vae_fold(arm, fold, feats, y, tr_idx, te_idx, storage):
    name = f"fold_{arm}_{fold:02d}"
    done = load_ckpt(name)
    if done is not None:
        print(f"  [resume] {name} already finished - loaded from checkpoint.")
        return done
    t0 = time.time()
    y_otr, y_te = y[tr_idx], y[te_idx]
    feat_keys = ["raw"] if arm == "vae_raw" else [f"psd{n}" for n in PSD_NPERSEG_OPTIONS]
    X_otr_feats = {k: feats[k][tr_idx] for k in feat_keys}
    splits = inner_splits_for(y_otr, fold)
    seed_base = SEED + 100000 * ARM_ID[arm] + 1000 * fold

    sampler = optuna.samplers.TPESampler(seed=SEED + 10 * fold + ARM_ID[arm],
                                         n_startup_trials=TPE_STARTUP, multivariate=True)
    study = optuna.create_study(study_name=f"{OUTPUT_DIR}_{arm}_fold{fold:02d}", storage=storage,
                                load_if_exists=True, direction="maximize", sampler=sampler)
    n_done = len(completed_trials(study))
    remaining = N_TRIALS - n_done
    if n_done:
        print(f"  [resume] study has {n_done} completed trials; running {max(remaining, 0)} more.")
    t_study = time.time()

    def cb(st, tr):
        c = len(completed_trials(st))
        if c == 1 or c % 10 == 0 or c == N_TRIALS:
            el = time.time() - t_study
            per = el / max(c - n_done, 1)
            print(f"    [{arm} f{fold:02d}] trial {c:3d}/{N_TRIALS} | value={tr.value if tr.value is not None else float('nan'):.4f}"
                  f" | best={st.best_value:.4f} | {per:.1f}s/trial | ETA study {fmt_time(per * (N_TRIALS - c))}")

    if remaining > 0:
        study.optimize(make_objective(arm, X_otr_feats, y_otr, splits, seed_base),
                       n_trials=remaining, timeout=STUDY_TIMEOUT_S, callbacks=[cb],
                       gc_after_trial=True, show_progress_bar=False)

    chosen, best = select_trial(study)
    cfg = cfg_from_params(chosen.params, arm)
    Xf = feats[cfg["feature"]]
    fseed = seed_base + 900
    res = crossfit_evaluate(
        lambda k: VAEDetector(cfg["neurons"], cfg["latent"], cfg["beta"], seed=fseed + k,
                              record_history=(k == 0)),
        Xf[tr_idx], y_otr, Xf[te_idx], y_te, split_seed=SEED + 500 + fold, extract=True)

    trials = []
    for t in completed_trials(study):
        n, l = parse_arch(t.params["arch"])
        trials.append({"number": t.number, "value": t.value, "failed": t.user_attrs.get("failed", False),
                       "se": t.user_attrs.get("obj_se", np.nan), "neurons": n, "latent": l,
                       "beta": t.params["beta"], "nperseg": t.params.get("nperseg", np.nan),
                       "n_params": t.user_attrs.get("n_params", np.nan),
                       "auc_mean": t.user_attrs.get("auc_mean", np.nan),
                       "f1_mean": t.user_attrs.get("f1_mean", np.nan)})
    res.update({
        "fold": fold, "method": arm, "te_idx": te_idx, "neurons": cfg["neurons"],
        "latent": cfg["latent"], "beta": cfg["beta"], "feature": cfg["feature"],
        "n_params": chosen.user_attrs["n_params"],
        "chosen_trial": chosen.number, "chosen_value": chosen.value,
        "chosen_se": chosen.user_attrs.get("obj_se", np.nan),
        "best_trial": best.number, "best_value": best.value,
        "best_se": best.user_attrs.get("obj_se", np.nan),
        "inner_auc_mean": chosen.user_attrs.get("auc_mean", np.nan),
        "inner_auc_std": chosen.user_attrs.get("auc_std", np.nan),
        "inner_f1_mean": chosen.user_attrs.get("f1_mean", np.nan),
        "inner_f1_std": chosen.user_attrs.get("f1_std", np.nan),
        "trials": trials, "time_s": time.time() - t0})
    save_ckpt(name, res)
    print_vae_fold(res)
    return res


def print_vae_fold(r):
    line = "-" * 72
    print(line)
    print(f"{METHOD_LABEL[r['method']]} | FOLD {r['fold']:2d} ({fmt_time(r['time_s'])})")
    print(line)
    print(f"  Chosen config       : N={r['neurons']}  latent={r['latent']}  beta={r['beta']:.3e}"
          f"  feature={r['feature']}  params={r['n_params']:,}")
    print(f"  Optuna best         : trial {r['best_trial']}  value={r['best_value']:.4f} ± {r['best_se']:.4f} (SE)")
    print(f"  Chosen (1-SE rule)  : trial {r['chosen_trial']}  value={r['chosen_value']:.4f}")
    print(f"    inner AUC / F1    : {r['inner_auc_mean']:.4f} ± {r['inner_auc_std']:.4f} / "
          f"{r['inner_f1_mean']:.4f} ± {r['inner_f1_std']:.4f}")
    print(f"  OOF calibration     : AUC={r['oof_auc']:.4f}  J={r['oof_J']:.4f}  "
          f"threshold={r['threshold']:.6f}")
    print(f"  External AUC        : {r['AUC']:.4f}")
    print(f"  Acc / BalAcc        : {r['Accuracy']:.4f} / {r['BalancedAcc']:.4f}")
    print(f"  Precision / Recall  : {r['Precision']:.4f} / {r['Recall']:.4f}")
    print(f"  Specificity / F1    : {r['Specificity']:.4f} / {r['F1']:.4f}   MCC={r['MCC']:.4f}")
    print(f"  Confusion matrix    : TN={r['TN']} FP={r['FP']} | FN={r['FN']} TP={r['TP']}")


def build_nested(results):
    out = {}
    for fold in RUN_FOLDS:
        a, b = results["vae_psd"].get(fold), results["vae_raw"].get(fold)
        if a is None or b is None:
            continue
        pick = a if a["chosen_value"] >= b["chosen_value"] else b
        r = dict(pick)
        r["method"], r["selected_arm"] = "vae_nested", pick["method"]
        out[fold] = r
    return out


# ======================================================================
# 9. STATISTICS
# ======================================================================
def nb_corrected(values, ratio):
    """Nadeau-Bengio corrected resampled t (accounts for overlapping training
    sets between CV folds). Returns mean, CI low, CI high, p (H0: mean=0)."""
    v = np.asarray(values, float)
    k = len(v)
    m = float(v.mean())
    if k < 2:
        return m, np.nan, np.nan, np.nan
    se = np.sqrt((1.0 / k + ratio) * v.var(ddof=1))
    tc = stats.t.ppf(0.975, k - 1)
    p = 2 * stats.t.sf(abs(m) / se, k - 1) if se > 0 else (0.0 if m != 0 else 1.0)
    return m, m - tc * se, m + tc * se, float(p)


def wilson_ci(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959964
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def norm_scores(r):
    return np.log10(np.maximum(r["scores_test"], 1e-300) / max(r["threshold"], 1e-300))


def pooled(rs):
    y = np.concatenate([r["y_test"] for r in rs])
    p = np.concatenate([r["pred"] for r in rs])
    s = np.concatenate([norm_scores(r) for r in rs])
    idx = np.concatenate([r["te_idx"] for r in rs])
    return y, p, s, idx


def bootstrap_pooled(y, s, p, seed=SEED):
    rng = np.random.default_rng(seed)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    out = {"AUC": [], "Accuracy": [], "BalancedAcc": [], "F1": []}
    for _ in range(BOOTSTRAP_N):
        i = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))])
        m = classification_metrics(y[i], p[i])
        out["AUC"].append(roc_auc_score(y[i], s[i]))
        for k in ("Accuracy", "BalancedAcc", "F1"):
            out[k].append(m[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def mcnemar_exact(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def safe_wilcoxon(d):
    d = np.asarray(d, float)
    if np.allclose(d, 0) or len(d) < 2:
        return np.nan
    try:
        return float(stats.wilcoxon(d).pvalue)
    except Exception:
        return np.nan


# ======================================================================
# 10. FIGURES - MSSP STYLE (Mechanical Systems & Signal Processing palette)
#     Every figure = prep_*() -> data dict (saved to plot_data/) -> p_*() draws it.
#     REPLOT_ONLY=True redraws everything from plot_data/ without any computation.
#     This section only draws figures: it never changes any result.
# ======================================================================
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

# ----------------------------------------------------------------------
# Figure options (plotting only - no effect on any computation)
# ----------------------------------------------------------------------
PLOT_METHODS = ["vae_psd", "vae_raw", "vae_nested", "bl_z_psd"]  # delete an entry to hide it in every figure
PANEL_LABELS = "letter"   # "letter": (a), (b), ... | "full": letter + description | "none"
FIG_WIDTH = 7.0           # inches = full text width of a portrait page
_PAPER_DIR = os.path.normpath(os.path.join(_script_dir(), "..", "Paper"))
PAPER_FIG_DIR = os.path.join(_PAPER_DIR, "figures") if os.path.isdir(_PAPER_DIR) else FIG_DIR
# Figure-level titles are disabled on purpose (journal style: the caption is the title).

MSSP_DARK = "#7070B8"
MSSP_MAIN = "#8888C6"
MSSP_MID = "#A0A0D0"
MSSP_LIGHT = "#B8B0D8"
MSSP_PALE = "#E5E2EE"
MSSP_BG = "#F1EFF7"
BLACK = "#303030"
GRAY = "#666666"
LIGHT_GRAY = "#BDBDBD"
FRAME = "#000000"

COLOR_NORMAL = MSSP_DARK
COLOR_DAMAGE = MSSP_LIGHT

METHOD_COLOR = {"vae_psd": MSSP_DARK, "vae_raw": MSSP_LIGHT, "vae_nested": MSSP_MID, "bl_z_psd": "#9E9E9E"}
METHOD_LS = {"vae_psd": "-", "vae_raw": "-", "vae_nested": "--", "bl_z_psd": ":"}
METHOD_MARKER = {"vae_psd": "o", "vae_raw": "s", "vae_nested": "D", "bl_z_psd": "^"}

CM_CMAP = LinearSegmentedColormap.from_list("mssp_cm", [MSSP_BG, MSSP_LIGHT, MSSP_DARK])
FOLD_CMAP = LinearSegmentedColormap.from_list("mssp_folds", [MSSP_LIGHT, MSSP_MAIN, MSSP_DARK, "#3F3F78"])
LETTERS = "abcdefghijklmnopqrstuvwxyz"
_PANELS = []          # (letter, description) of the figure being drawn
PANEL_INDEX = {}      # figure name -> list of (letter, description), written for captions


def apply_mssp_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "axes.labelcolor": BLACK, "text.color": BLACK,
        "axes.edgecolor": FRAME, "axes.linewidth": 0.8,
        "axes.spines.top": True, "axes.spines.right": True,       # box frame on all 4 sides
        "xtick.color": FRAME, "ytick.color": FRAME,
        "xtick.labelcolor": BLACK, "ytick.labelcolor": BLACK,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.grid": True, "grid.color": MSSP_PALE, "grid.linewidth": 0.7, "grid.alpha": 1.0,
        "lines.linewidth": 2.0, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType fonts (journal requirement)
    })


def fold_color(fold, n=None):
    n = n or N_OUTER
    return FOLD_CMAP((fold - 1) / max(n - 1, 1))


def panel(ax, i, text):
    """Panel label. The description is always recorded in panel_index.txt for the captions."""
    _PANELS.append((f"({LETTERS[i]})", text.replace("\n", " ")))
    if PANEL_LABELS == "none":
        return
    label = f"({LETTERS[i]})" if PANEL_LABELS == "letter" else f"({LETTERS[i]}) {text}"
    ax.set_title(label, loc="left", fontweight="bold", fontsize=11)


def single_title(ax, text):
    """Single-panel figures carry no title; the text only goes to panel_index.txt."""
    _PANELS.append(("", text))


def _keep(methods):
    return [m for m in methods if m in PLOT_METHODS]


def _keep_labels(labels):
    hidden = {METHOD_LABEL[m] for m in METHOD_LABEL if m not in PLOT_METHODS}
    return [l for l in labels if l not in hidden]


def make_grid(n, ncols=5, w=3.6, h=3.3):
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes, ncols


def portrait_grid(n, ncols=2, row_h=2.6, width=None):
    """n panels in rows of `ncols` (portrait page). An incomplete last row is centred.
    Returns fig, axes, and for each axis whether it is in the first column / last row."""
    width = width or FIG_WIDTH
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(width, row_h * nrows))
    gs = fig.add_gridspec(nrows, 2 * ncols)
    axes, first_col, last_row = [], [], []
    for i in range(n):
        r, c = divmod(i, ncols)
        in_row = min(ncols, n - r * ncols)
        start = (ncols - in_row) + 2 * c
        axes.append(fig.add_subplot(gs[r, start:start + 2]))
        first_col.append(c == 0)
        last_row.append(r == nrows - 1 or (i + ncols) >= n)
    return fig, axes, first_col, last_row


def top_legend(ax, handles=None, labels=None, ncol=None, **kw):
    """Legend placed above the axes (right-aligned): it never covers the data."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                     ncol=ncol or len(handles), borderaxespad=0.2, handlelength=1.6,
                     columnspacing=1.0, **kw)


def bottom_legend(fig, handles, labels, ncol=None):
    """Figure-level legend below all panels (for grids)."""
    ncol = ncol or len(handles)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, frameon=False)
    rows = int(np.ceil(len(handles) / ncol))
    return (0.08 + 0.19 * rows) / fig.get_size_inches()[1]   # fraction of figure height reserved


OPTION_TICKS = {"neurons": NEURON_OPTIONS, "latent": LATENT_OPTIONS, "nperseg": PSD_NPERSEG_OPTIONS}


def value_ticks(ax, axis, values):
    """Ticks at the actual option values (e.g. 16, 32, ... or 256, 512, 1000) on a log axis."""
    vals = sorted({float(v) for v in values})
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(vals))
    a.set_major_formatter(FixedFormatter([f"{v:g}" for v in vals]))
    a.set_minor_locator(NullLocator())


def finish(fig, title, name, rect_bottom=0.0):
    """Box frame on every visible axis, no figure title, save into Paper/figures."""
    for ax in fig.axes:
        if not ax.axison:
            continue
        ax.set_axisbelow(True)
        for s in ("left", "bottom", "top", "right"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_color(FRAME)
            ax.spines[s].set_linewidth(0.8)
    fig.tight_layout(rect=[0, rect_bottom, 1, 1])
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    for ext in FIG_FORMATS:
        fig.savefig(os.path.join(PAPER_FIG_DIR, f"{name}.{ext}"), dpi=FIG_DPI, bbox_inches="tight")
    PANEL_INDEX[name] = list(_PANELS)
    _PANELS.clear()
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def write_panel_index():
    """Paper/figures/panel_index.txt: what each (a), (b), ... shows - ready for captions."""
    if not PANEL_INDEX:
        return
    path = os.path.join(PAPER_FIG_DIR, "panel_index.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for name in sorted(PANEL_INDEX):
            fh.write(f"{name}\n")
            for letter, text in PANEL_INDEX[name]:
                fh.write(f"   {letter} {text}\n" if letter else f"   {text}\n")
            fh.write("\n")
    print(f"Panel index for captions: {path}")


def mssp_boxplot(ax, data, colors, labels, widths=0.6, show_fliers=True):
    box = ax.boxplot(
        data, patch_artist=True, widths=widths, showfliers=show_fliers,
        medianprops={"color": BLACK, "linewidth": 1.5},
        whiskerprops={"color": GRAY, "linewidth": 1.0},
        capprops={"color": GRAY, "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": MSSP_MID,
                    "markeredgecolor": "none", "alpha": 0.5})
    for b, c in zip(box["boxes"], colors):
        b.set_facecolor(c)
        b.set_alpha(0.80)
        b.set_edgecolor(GRAY)
        b.set_linewidth(0.8)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(labels)
    return box


def jitter_points(ax, data, seed=SEED, color=BLACK, s=9, alpha=0.5, width=0.12):
    rng = np.random.default_rng(seed)
    for i, d in enumerate(data, 1):
        d = np.asarray(d)
        ax.scatter(i + rng.uniform(-width, width, len(d)), d, s=s, color=color,
                   alpha=alpha, edgecolor="none", zorder=3)


def draw_cm(ax, cm, show_ylabel=True, show_yticklabels=True, show_xlabel=True, fontsize=10):
    cm = np.asarray(cm)
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rn, cmap=CM_CMAP, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({rn[i, j]:.0%})", ha="center", va="center",
                    fontsize=fontsize, color=BLACK)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Damage"])
    ax.set_yticklabels(["Normal", "Damage"] if show_yticklabels else ["", ""])
    if show_xlabel:
        ax.set_xlabel("Predicted")
    if show_ylabel:
        ax.set_ylabel("Actual")
    ax.grid(False)


# Full latent groups (kept: they define what prep_* saves in plot_data/)
LAT_STYLE = {   # normal = circles/diamonds (dark), damage = triangles (light)
    "fit_normal": dict(color=COLOR_NORMAL, alpha=0.55, s=16, marker="o", edgecolor="white",
                       linewidth=0.3, label="Normal (training)"),
    "val_normal": dict(facecolors="none", edgecolors=COLOR_NORMAL, s=28, marker="o",
                       linewidths=1.0, label="Normal (out-of-fold)"),
    "train_damage": dict(color=COLOR_DAMAGE, alpha=0.90, s=28, marker="^", edgecolor=MSSP_DARK,
                         linewidth=0.4, label="Damage (outer-train, never trained on)"),
    "test_normal": dict(color=COLOR_NORMAL, s=48, marker="D", edgecolor=BLACK, linewidth=0.8,
                        label="Normal (outer test)"),
    "test_damage": dict(color=COLOR_DAMAGE, s=70, marker="^", edgecolor=BLACK, linewidth=0.8,
                        label="Damage (outer test)"),
}
# What is actually drawn in the latent plots: outer-test samples only
LAT_PLOT_STYLE = {
    "test_normal": dict(color=COLOR_NORMAL, s=26, marker="o", edgecolor="white", linewidth=0.4,
                        alpha=0.9, label="Normal"),
    "test_damage": dict(color=COLOR_DAMAGE, s=32, marker="^", edgecolor=MSSP_DARK, linewidth=0.5,
                        alpha=0.95, label="Damage"),
}
LOG_RATIO_LABEL = r"$\log_{10}$(score / fold threshold)"


# ----------------------------------------------------------------------
# plot-data I/O
# ----------------------------------------------------------------------
def save_plot_data(name, data):
    """<plot_data>/<name>.pkl (everything, used by REPLOT_ONLY) and
    <plot_data>/<name>/<table>.csv (each table, for editing in any software)."""
    with open(os.path.join(PLOT_DATA_DIR, name + ".pkl"), "wb") as fh:
        pickle.dump(data, fh)
    sub = os.path.join(PLOT_DATA_DIR, name)
    os.makedirs(sub, exist_ok=True)
    meta = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(os.path.join(sub, f"{k}.csv"), index=False)
        else:
            meta[k] = v
    with open(os.path.join(sub, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


def emit(name, data):
    try:
        save_plot_data(name, data)
        PLOTTERS[data["kind"]](data, name)
    except Exception:
        print(f"[plot error] {name}:\n{traceback.format_exc()}")


def replot_all():
    files = sorted(glob.glob(os.path.join(PLOT_DATA_DIR, "*.pkl")))
    if not files:
        print(f"No plot data found in {PLOT_DATA_DIR}.")
        return
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            PLOTTERS[d["kind"]](d, name)
            print(f"  redrawn {name}")
        except Exception:
            print(f"[plot error] {name}:\n{traceback.format_exc()}")
    write_panel_index()
    print(f"Figures saved in: {PAPER_FIG_DIR}")


def safe_prep(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        print(f"[prep error] {fn.__name__}:\n{traceback.format_exc()}")
        return None


def _rs(results, m):
    return [results[m][f] for f in sorted(results[m])]


def _mean_roc(rs):
    g = np.linspace(0, 1, 201)
    T = []
    for r in rs:
        t = np.interp(g, r["fpr"], r["tpr"])
        t[0] = 0.0
        T.append(t)
    T = np.array(T)
    m = T.mean(0)
    m[-1] = 1.0
    return g, m, T.std(0)


# ----------------------------------------------------------------------
# PREP functions (results -> tidy data)
# ----------------------------------------------------------------------
def prep_eda(feats, freqs, y):
    key = f"psd{PSD_BASELINE_NPERSEG}"
    L, f = feats[key], freqs[key]
    df = pd.DataFrame({
        "freq_hz": f,
        "normal_mean_db": 10 * L[y == 0].mean(0), "normal_sd_db": 10 * L[y == 0].std(0),
        "damage_mean_db": 10 * L[y == 1].mean(0), "damage_sd_db": 10 * L[y == 1].std(0),
        "diff_db": 10 * (L[y == 1].mean(0) - L[y == 0].mean(0)),
        "welch_t": stats.ttest_ind(L[y == 1], L[y == 0], equal_var=False).statistic})
    return {"kind": "eda_psd", "df": df, "nperseg": PSD_BASELINE_NPERSEG, "fs": FS}


def prep_history(arm, rs, noise_sd):
    T, F = [], []
    for r in rs:
        t = pd.DataFrame(r["trials"]).sort_values("number")
        t["running_best"] = np.maximum.accumulate(t["value"].values)
        t["fold"] = r["fold"]
        T.append(t[["fold", "number", "value", "failed", "running_best"]])
        F.append({"fold": r["fold"], "best_value": r["best_value"], "chosen_trial": r["chosen_trial"],
                  "chosen_value": r["chosen_value"], "outer_auc": r["AUC"]})
    return {"kind": "optuna_history", "label": METHOD_LABEL[arm], "trials": pd.concat(T),
            "folds": pd.DataFrame(F), "noise_sd": float(noise_sd) if noise_sd is not None else np.nan,
            "tpe_startup": TPE_STARTUP}


def prep_landscape(arm, rs):
    T = pd.concat([pd.DataFrame(r["trials"]).assign(fold=r["fold"]) for r in rs])
    T = T[~T["failed"]][["fold", "value", "neurons", "latent", "beta", "n_params", "nperseg"]]
    C = pd.DataFrame([{"fold": r["fold"], "value": r["chosen_value"], "neurons": r["neurons"],
                       "latent": r["latent"], "beta": r["beta"], "n_params": r["n_params"],
                       "nperseg": int(r["feature"][3:]) if r["feature"].startswith("psd") else np.nan}
                      for r in rs])
    return {"kind": "landscape", "label": METHOD_LABEL[arm], "trials": T, "chosen": C,
            "has_nperseg": arm == "vae_psd"}


def prep_selected(results, arms):
    rows = []
    for a in arms:
        for r in _rs(results, a):
            rows.append({"method": a, "label": METHOD_LABEL[a], "fold": r["fold"],
                         "neurons": r["neurons"], "latent": r["latent"], "beta": r["beta"],
                         "nperseg": int(r["feature"][3:]) if r["feature"].startswith("psd") else np.nan})
    return {"kind": "selected_hparams", "df": pd.DataFrame(rows)}


def prep_noise(noise, results):
    rows = []
    for a in ARMS:
        if not noise.get(a):
            continue
        for n in noise[a]:
            rows.append({"method": a, "label": METHOD_LABEL[a], "group": "Fixed config\n1×3-fold CV",
                         "value": n["old_1x3"]})
            rows.append({"method": a, "label": METHOD_LABEL[a],
                         "group": f"Fixed config\n{INNER_REPEATS}×{INNER_SPLITS}-fold CV",
                         "value": n["new_repeated"]})
        r1 = results.get(a, {}).get(NOISE_FLOOR_FOLD)
        if r1 is not None:
            for t in r1["trials"]:
                if not t["failed"]:
                    rows.append({"method": a, "label": METHOD_LABEL[a],
                                 "group": f"Optuna trials\n(fold {NOISE_FLOOR_FOLD})", "value": t["value"]})
    return {"kind": "noise_floor", "df": pd.DataFrame(rows)} if rows else None


def prep_roc(results, methods, kind):
    C, M, A = [], [], []
    for m in methods:
        rs = _rs(results, m)
        for r in rs:
            C.append(pd.DataFrame({"method": m, "label": METHOD_LABEL[m], "fold": r["fold"],
                                   "fpr": r["fpr"], "tpr": r["tpr"]}))
            A.append({"method": m, "label": METHOD_LABEL[m], "fold": r["fold"], "auc": r["AUC"]})
        g, mt, sd = _mean_roc(rs)
        M.append(pd.DataFrame({"method": m, "label": METHOD_LABEL[m], "fpr": g,
                               "mean_tpr": mt, "sd_tpr": sd}))
    return {"kind": kind, "curves": pd.concat(C), "mean": pd.concat(M), "auc": pd.DataFrame(A)}


def prep_scores(results, methods):
    rows = []
    for m in methods:
        for r in _rs(results, m):
            rows.append(pd.DataFrame({"method": m, "label": METHOD_LABEL[m], "fold": r["fold"],
                                      "sample_idx": r["te_idx"], "y": r["y_test"],
                                      "log_ratio": norm_scores(r)}))
    return {"kind": "score_distributions", "df": pd.concat(rows)}


def prep_cm_pooled(results, methods):
    rows = []
    for m in methods:
        yp, pp, _, _ = pooled(_rs(results, m))
        cm = confusion_matrix(yp, pp, labels=[0, 1])
        rows.append({"title": f"{METHOD_LABEL[m]}", "subtitle": f"accuracy = {np.mean(yp == pp):.3f}",
                     "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]})
    return {"kind": "cm_grid", "df": pd.DataFrame(rows), "ncols": 4,
            "suptitle": "Pooled confusion matrices (sum of the outer test folds)"}


def prep_cm_folds(arm, rs):
    rows = [{"title": f"Fold {r['fold']}", "subtitle": f"F1 = {r['F1']:.2f}",
             "TN": r["TN"], "FP": r["FP"], "FN": r["FN"], "TP": r["TP"]} for r in rs]
    return {"kind": "cm_grid", "df": pd.DataFrame(rows), "ncols": 5,
            "suptitle": f"Per-fold confusion matrices — {METHOD_LABEL[arm]}"}


def prep_metrics(results, methods):
    rows = [{"method": m, "label": METHOD_LABEL[m], "short": METHOD_SHORT[m], "fold": r["fold"],
             "metric": met, "value": r[met]}
            for m in methods for r in _rs(results, m) for met in METRICS]
    return {"kind": "metric_boxplots", "df": pd.DataFrame(rows), "metrics": METRICS}


def prep_paired(results, methods, ref, ratio):
    P, S = [], []
    for m in methods:
        if m == ref:
            continue
        for met in ("AUC", "BalancedAcc", "F1"):
            d = np.array([results[ref][f][met] - results[m][f][met] for f in sorted(results[ref])])
            mean, lo, hi, p = nb_corrected(d, ratio)
            for f, v in zip(sorted(results[ref]), d):
                P.append({"other": m, "label": METHOD_LABEL[m], "metric": met, "fold": f, "diff": v})
            S.append({"other": m, "label": METHOD_LABEL[m], "metric": met, "mean": mean,
                      "ci_lo": lo, "ci_hi": hi, "p": p})
    return {"kind": "paired_diffs", "points": pd.DataFrame(P), "summary": pd.DataFrame(S),
            "ref_label": METHOD_LABEL[ref]}


def prep_inner_outer(results, arms):
    rows = [{"method": a, "label": METHOD_LABEL[a], "fold": r["fold"], "inner_auc": r["inner_auc_mean"],
             "oof_auc": r["oof_auc"], "outer_auc": r["AUC"]} for a in arms for r in _rs(results, a)]
    return {"kind": "inner_vs_outer", "df": pd.DataFrame(rows)}


def prep_threshold(results, arms):
    rows = [{"method": a, "label": METHOD_LABEL[a], "fold": r["fold"],
             "oof_recall": r["oof_recall"], "test_recall": r["Recall"],
             "oof_specificity": r["oof_specificity"], "test_specificity": r["Specificity"]}
            for a in arms for r in _rs(results, a)]
    return {"kind": "threshold_transfer", "df": pd.DataFrame(rows)}


def _latent_df(r):
    L = r["latent_2d"]
    return pd.concat([pd.DataFrame({"fold": r["fold"], "group": g, "pc1": L[g][:, 0], "pc2": L[g][:, 1]})
                      for g in LAT_STYLE if len(L[g])])


def prep_latent(arm, rs):
    rs = [r for r in rs if r.get("latent_2d") is not None]
    info = pd.DataFrame([{"fold": r["fold"], "neurons": r["neurons"], "latent": r["latent"],
                          "evr1": r["latent_2d"]["evr"][0], "evr2": r["latent_2d"]["evr"][1],
                          "outer_auc": r["AUC"]} for r in rs])
    return {"kind": "latent", "label": METHOD_LABEL[arm], "points": pd.concat([_latent_df(r) for r in rs]),
            "info": info}


def prep_training(arm, rs):
    rows = []
    for r in rs:
        h = r.get("loss_history")
        if h and h["epoch"]:
            rows.append(pd.DataFrame({"fold": r["fold"], "epoch": h["epoch"], "rec": h["rec"],
                                      "kl": h["kl"], "loss": h["loss"], "beta": r["beta"]}))
    return {"kind": "training_curves", "label": METHOD_LABEL[arm], "df": pd.concat(rows)} if rows else None


def prep_main(results, complete):
    arms = [a for a in ARMS if a in complete]
    if not arms:
        return None
    main = MAIN_METHOD if MAIN_METHOD in arms else arms[0]
    second = SECOND_METHOD if (SECOND_METHOD in complete and SECOND_METHOD != main) else None
    rs = _rs(results, main)
    yp, pp, sp, _ = pooled(rs)
    hist = pd.DataFrame({"y": yp, "log_ratio": sp})
    aucs = np.array([r["AUC"] for r in rs])
    rep = rs[int(np.argmin(np.abs(aucs - np.median(aucs))))]   # fold with median outer AUC
    lat = _latent_df(rep)

    conv = []
    for a in arms:
        curves = [np.maximum.accumulate(pd.DataFrame(r["trials"]).sort_values("number")["value"].values)
                  for r in _rs(results, a)]
        n = min(len(c) for c in curves)
        M = np.array([c[:n] for c in curves])
        k = M.shape[0]
        half = stats.t.ppf(0.975, k - 1) * M.std(0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(n)
        conv.append(pd.DataFrame({"method": a, "label": METHOD_LABEL[a], "trial": np.arange(1, n + 1),
                                  "mean": M.mean(0), "ci_lo": M.mean(0) - half, "ci_hi": M.mean(0) + half}))

    box = []
    for m in complete:
        y_, _, s_, _ = pooled(_rs(results, m))
        box.append(pd.DataFrame({"method": m, "short": METHOD_SHORT[m].replace("\n", " "),
                                 "y": y_, "log_ratio": s_}))

    cms = []
    for m in [main] + ([second] if second else []):
        y_, p_, _, _ = pooled(_rs(results, m))
        cm = confusion_matrix(y_, p_, labels=[0, 1])
        cms.append({"method": m, "label": METHOD_LABEL[m], "TN": cm[0, 0], "FP": cm[0, 1],
                    "FN": cm[1, 0], "TP": cm[1, 1]})

    train = []
    for a in arms:
        H = [r["loss_history"] for r in _rs(results, a) if r.get("loss_history") and r["loss_history"]["epoch"]]
        if not H:
            continue
        R = np.log(np.maximum(np.array([h["rec"] for h in H]), 1e-12))
        train.append(pd.DataFrame({"method": a, "label": METHOD_LABEL[a], "epoch": H[0]["epoch"],
                                   "geo_mean": np.exp(R.mean(0)),
                                   "lo": np.exp(R.mean(0) - R.std(0)), "hi": np.exp(R.mean(0) + R.std(0))}))
    return {"kind": "main_multipanel", "main_label": METHOD_LABEL[main], "main": main,
            "rep_fold": rep["fold"], "rep_info": {"neurons": rep["neurons"], "latent": rep["latent"],
                                                  "auc": rep["AUC"]},
            "hist": hist, "latent": lat, "conv": pd.concat(conv), "box": pd.concat(box),
            "cms": pd.DataFrame(cms), "train": pd.concat(train) if train else pd.DataFrame()}


# ----------------------------------------------------------------------
# PLOT functions (tidy data -> figure)
# ----------------------------------------------------------------------
def p_eda_psd(d, name):
    df = d["df"]
    f = df["freq_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, 7.4), sharex=True)
    ax = axes[0]
    for cls, c, lab in (("normal", COLOR_NORMAL, "Normal"), ("damage", COLOR_DAMAGE, "Damage")):
        m, s = df[f"{cls}_mean_db"], df[f"{cls}_sd_db"]
        ax.fill_between(f, m - s, m + s, color=c, alpha=0.30, linewidth=0)
        ax.plot(f, m, color=c, lw=1.6, label=f"{lab} (mean ± 1 SD)")
    ax.set_ylabel("PSD [dB]")
    top_legend(ax)
    panel(ax, 0, f"Mean log-PSD per class (Welch, nperseg = {d['nperseg']})")
    ax = axes[1]
    ax.axhline(0, color=LIGHT_GRAY, lw=0.8)
    ax.plot(f, df["diff_db"], color=BLACK, lw=1.2, label="Damage − Normal [dB]")
    ax.set_ylabel("Difference [dB]")
    ax.set_xlabel(f"Frequency [Hz] (FS = {d['fs']:g} Hz)")
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(f, df["welch_t"], color=MSSP_DARK, lw=0.9, alpha=0.85, label="Welch t-statistic")
    ax2.set_ylabel("t-statistic", color=MSSP_DARK)
    lim1 = 1.1 * np.nanmax(np.abs(df["diff_db"]))
    lim2 = 1.1 * np.nanmax(np.abs(df["welch_t"]))
    ax.set_ylim(-lim1, lim1)
    ax2.set_ylim(-lim2, lim2)          # zero of both axes aligned
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    top_legend(ax, h1 + h2, l1 + l2)
    panel(ax, 1, "Class difference (exploratory, all samples)")
    finish(fig, None, name)


def p_optuna_history(d, name):
    T, F = d["trials"], d["folds"]
    fig, axes, first_col, last_row = portrait_grid(len(F), ncols=2, row_h=1.85)
    for i, (ax, (_, fr)) in enumerate(zip(axes, F.iterrows())):
        t = T[T["fold"] == fr["fold"]].sort_values("number")
        ok, bad = t[~t["failed"].astype(bool)], t[t["failed"].astype(bool)]
        if np.isfinite(d["noise_sd"]):
            ax.axhspan(fr["best_value"] - d["noise_sd"], fr["best_value"], color=MSSP_LIGHT,
                       alpha=0.35, lw=0, label="Best − 1 SD (noise floor)")
        ax.scatter(ok["number"], ok["value"], s=8, color=MSSP_MID, alpha=0.75, edgecolor="none", label="Trial")
        if len(bad):
            ax.scatter(bad["number"], np.full(len(bad), ok["value"].min()), marker="x", s=12,
                       color=GRAY, label="Failed trial")
        ax.plot(t["number"], t["running_best"], color=MSSP_DARK, lw=1.6, label="Running best")
        ax.axvline(d["tpe_startup"] - 0.5, color=GRAY, ls=":", lw=1.0, label="End of random start-up")
        ch = t[t["number"] == fr["chosen_trial"]]
        ax.scatter(ch["number"], ch["value"], marker="*", s=120, color=MSSP_DARK, edgecolor=BLACK,
                   linewidth=0.8, zorder=5, label="Chosen (1-SE rule)")
        if last_row[i]:
            ax.set_xlabel("Trial")
        if first_col[i]:
            ax.set_ylabel("Objective")
        panel(ax, i, f"{d['label']}, fold {int(fr['fold'])} (outer AUC = {fr['outer_auc']:.3f})")
    h, l = axes[0].get_legend_handles_labels()
    rb = bottom_legend(fig, h, l, ncol=3)
    finish(fig, None, name, rect_bottom=rb)


def p_landscape(d, name):
    T, C = d["trials"], d["chosen"]
    cols = [("neurons", "Neurons N", 2), ("latent", "Latent dimension", 2),
            ("beta", "β (KL weight)", 10), ("n_params", "Trainable parameters", 10)]
    if d["has_nperseg"]:
        cols.append(("nperseg", "PSD nperseg", 2))
    fig, axes, first_col, _ = portrait_grid(len(cols), ncols=2, row_h=2.75)
    rng = np.random.default_rng(SEED)
    ylo, yhi = T["value"].min(), T["value"].max()
    pad = 0.04 * (yhi - ylo if yhi > ylo else 1.0)
    for i, (ax, (c, lab, base)) in enumerate(zip(axes, cols)):
        x = T[c].astype(float).values
        if c in ("neurons", "latent", "nperseg"):
            x = x * (1 + rng.uniform(-0.07, 0.07, len(x)))
        ax.scatter(x, T["value"], color=[fold_color(f) for f in T["fold"]], s=12, alpha=0.7,
                   edgecolor="white", linewidth=0.3)
        ax.scatter(C[c].astype(float), C["value"], marker="*", s=130, color=MSSP_DARK,
                   edgecolor=BLACK, linewidth=0.8, zorder=5)
        ax.set_xscale("log", base=base)
        if c in OPTION_TICKS:
            value_ticks(ax, "x", OPTION_TICKS[c])
        ax.set_ylim(ylo - pad, yhi + pad)
        ax.set_xlabel(lab)
        ax.set_ylabel("Inner objective")
        panel(ax, i, f"{d['label']}: objective vs {lab.lower()}")
    folds = sorted(T["fold"].unique())
    h = [Line2D([], [], marker="o", ls="", color=fold_color(f), label=f"Fold {f}") for f in folds]
    h.append(Line2D([], [], marker="*", ls="", markersize=11, color=MSSP_DARK,
                    markeredgecolor=BLACK, label="Chosen"))
    rb = bottom_legend(fig, h, [x.get_label() for x in h], ncol=6)
    finish(fig, None, name, rect_bottom=rb)


def p_selected(d, name):
    D = d["df"]
    arms = list(dict.fromkeys(D["method"]))
    specs = [("neurons", "Neurons N", 2), ("latent", "Latent dimension", 2),
             ("beta", "β (KL weight)", 10), ("nperseg", "PSD nperseg", 2)]
    fig, axes, _, last_row = portrait_grid(4, ncols=2, row_h=2.75)
    for j, a in enumerate(arms):
        s = D[D["method"] == a]
        x = s["fold"] + (j - (len(arms) - 1) / 2) * 0.25
        for ax, (c, _, _) in zip(axes, specs):
            v = s[c].astype(float)
            if v.notna().any():
                ax.scatter(x, v, s=40, color=METHOD_COLOR[a], marker=METHOD_MARKER[a],
                           edgecolor=BLACK, linewidth=0.6, label=s["label"].iloc[0], zorder=3)
    folds = sorted(D["fold"].unique())
    for i, (ax, (c, lab, base)) in enumerate(zip(axes, specs)):
        ax.set_yscale("log", base=base)
        if c in OPTION_TICKS:
            value_ticks(ax, "y", OPTION_TICKS[c])
        ax.set_xticks(folds)
        ax.set_ylabel(lab)
        if last_row[i]:
            ax.set_xlabel("Outer fold")
        panel(ax, i, f"Selected {lab.lower()} per outer fold")
    h, l = axes[0].get_legend_handles_labels()
    rb = bottom_legend(fig, h, l)
    finish(fig, None, name, rect_bottom=rb)


def p_noise(d, name):
    D = d["df"]
    arms = list(dict.fromkeys(D["method"]))
    fig, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 4.4), squeeze=False)
    for i, (ax, a) in enumerate(zip(axes[0], arms)):
        s = D[D["method"] == a]
        groups = list(dict.fromkeys(s["group"]))
        data = [s[s["group"] == g]["value"].dropna().values for g in groups]
        labels = [f"{g}\nSD = {np.std(v, ddof=1):.4f}" if len(v) > 1 else g for g, v in zip(groups, data)]
        mssp_boxplot(ax, data, [MSSP_LIGHT, MSSP_MID, MSSP_DARK][:len(data)], labels, show_fliers=False)
        jitter_points(ax, data)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("Inner objective")
        panel(ax, i, f"Noise floor vs trial spread - {s['label'].iloc[0]}")
    finish(fig, None, name)


def p_roc_grid(d, name):
    C, M, A = d["curves"], d["mean"], d["auc"]
    methods = _keep(dict.fromkeys(A["method"]))
    fig, axes, first_col, last_row = portrait_grid(len(methods), ncols=2, row_h=3.45)
    for i, (ax, m) in enumerate(zip(axes, methods)):
        for _, g in C[C["method"] == m].groupby("fold"):
            ax.plot(g["fpr"], g["tpr"], color=MSSP_LIGHT, lw=0.8, alpha=0.9)
        ax.plot([], [], color=MSSP_LIGHT, lw=0.8, label="Individual folds")
        mm = M[M["method"] == m]
        a = A[A["method"] == m]["auc"]
        ax.fill_between(mm["fpr"], np.clip(mm["mean_tpr"] - mm["sd_tpr"], 0, 1),
                        np.clip(mm["mean_tpr"] + mm["sd_tpr"], 0, 1),
                        color=MSSP_LIGHT, alpha=0.45, linewidth=0, label="± 1 SD")
        ax.plot(mm["fpr"], mm["mean_tpr"], color=MSSP_DARK, lw=2.0,
                label=f"Mean (AUC = {a.mean():.3f} ± {a.std(ddof=1):.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=7)
        panel(ax, i, f"ROC - {mm['label'].iloc[0]}")
    finish(fig, None, name)


def p_roc_comparison(d, name):
    M, A = d["mean"], d["auc"]
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    for m in _keep(dict.fromkeys(A["method"])):
        mm = M[M["method"] == m]
        a = A[A["method"] == m]["auc"]
        ax.plot(mm["fpr"], mm["mean_tpr"], color=METHOD_COLOR[m], ls=METHOD_LS[m],
                lw=2.3 if m.startswith("vae") else 1.7,
                label=f"{mm['label'].iloc[0]} (AUC = {a.mean():.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=LIGHT_GRAY, lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    single_title(ax, "Mean outer-test ROC - all methods")
    finish(fig, None, name)


def p_scores(d, name):
    D = d["df"]
    methods = _keep(dict.fromkeys(D["method"]))
    fig, axes, first_col, last_row = portrait_grid(len(methods), ncols=2, row_h=2.7)
    for i, (ax, m) in enumerate(zip(axes, methods)):
        s = D[D["method"] == m]
        bins = np.linspace(s["log_ratio"].min(), s["log_ratio"].max(), 30)
        ax.hist(s[s["y"] == 0]["log_ratio"], bins=bins, density=True, alpha=0.65, color=COLOR_NORMAL,
                edgecolor="white", linewidth=0.6, label="Normal")
        ax.hist(s[s["y"] == 1]["log_ratio"], bins=bins, density=True, alpha=0.60, color=COLOR_DAMAGE,
                edgecolor="white", linewidth=0.6, label="Damage")
        ax.axvline(0, color=BLACK, ls="--", lw=1.0, label="Threshold")
        ax.set_xlabel(LOG_RATIO_LABEL)
        if first_col[i]:
            ax.set_ylabel("Density")
        panel(ax, i, f"Score distribution - {s['label'].iloc[0]}")
    h, l = axes[0].get_legend_handles_labels()
    rb = bottom_legend(fig, h, l)
    finish(fig, None, name, rect_bottom=rb)


def p_cm_grid(d, name):
    D = d["df"]
    D = D[D["title"].isin(_keep_labels(list(D["title"])))]
    per_fold = len(D) > 4
    if per_fold:   # per-fold matrices: 5 x 2 portrait grid
        fig, axes, first_col, last_row = portrait_grid(len(D), ncols=2, row_h=1.78, width=5.4)
        fs = 8
    else:
        fig, axes, first_col, last_row = portrait_grid(len(D), ncols=2, row_h=3.2, width=6.4)
        fs = 10
    for i, (ax, (_, r)) in enumerate(zip(axes, D.iterrows())):
        draw_cm(ax, [[r["TN"], r["FP"]], [r["FN"], r["TP"]]], show_ylabel=first_col[i],
                show_xlabel=last_row[i] or not per_fold, fontsize=fs)
        if per_fold:
            ax.tick_params(labelsize=8)
        panel(ax, i, f"{r['title']} ({r['subtitle']})")
    finish(fig, None, name)


def p_metrics(d, name):
    D = d["df"]
    methods = _keep(dict.fromkeys(D["method"]))
    short = [D[D["method"] == m]["short"].iloc[0] for m in methods]
    metrics = d["metrics"]
    fig, axes, first_col, last_row = portrait_grid(len(metrics), ncols=2, row_h=2.2)
    for i, (ax, met) in enumerate(zip(axes, metrics)):
        data = [D[(D["method"] == m) & (D["metric"] == met)]["value"].values for m in methods]
        mssp_boxplot(ax, data, [METHOD_COLOR[m] for m in methods], short, show_fliers=False)
        jitter_points(ax, data, s=7)
        ax.scatter(range(1, len(data) + 1), [np.mean(v) for v in data], marker="D", s=20,
                   color="white", edgecolor=BLACK, linewidth=0.9, zorder=5)
        ax.axhline(0.9, color=GRAY, ls=":", lw=1.0)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_ylabel(met)
        panel(ax, i, met)
    h = [Patch(facecolor=METHOD_COLOR[m], alpha=0.8, edgecolor=GRAY,
               label=D[D["method"] == m]["label"].iloc[0]) for m in methods]
    h += [Line2D([], [], marker="D", ls="", color="white", markeredgecolor=BLACK, label="Mean"),
          Line2D([], [], ls=":", color=GRAY, label="0.9 reference")]
    rb = bottom_legend(fig, h, [x.get_label() for x in h], ncol=3)
    finish(fig, None, name, rect_bottom=rb)


def p_paired(d, name):
    P, S = d["points"], d["summary"]
    S = S[S["other"].isin(PLOT_METHODS)]
    P = P[P["other"].isin(PLOT_METHODS)]
    metrics = list(dict.fromkeys(S["metric"]))
    others = list(dict.fromkeys(S["other"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 1.0 * len(others) + 2.4),
                             sharey=True, squeeze=False)
    for i, (ax, met) in enumerate(zip(axes[0], metrics)):
        for j, o in enumerate(others):
            p = P[(P["metric"] == met) & (P["other"] == o)]["diff"].values
            yj = j + np.random.default_rng(SEED + j).uniform(-0.15, 0.15, len(p))
            ax.scatter(p, yj, s=24, color=METHOD_COLOR[o], edgecolor="white", linewidth=0.4,
                       alpha=0.9, zorder=3)
            s = S[(S["metric"] == met) & (S["other"] == o)].iloc[0]
            ax.errorbar(s["mean"], j, xerr=[[s["mean"] - s["ci_lo"]], [s["ci_hi"] - s["mean"]]],
                        fmt="D", color=BLACK, ms=5, capsize=4, lw=1.2, zorder=4)
            ax.annotate(f"p = {s['p']:.3f}", (s["ci_hi"], j), xytext=(4, 7), textcoords="offset points",
                        fontsize=8, color=GRAY)
        ax.axvline(0, color=GRAY, ls="--", lw=0.9)
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels([S[S["other"] == o]["label"].iloc[0] for o in others])
        ax.set_xlabel(f"{d['ref_label']} − method")
        panel(ax, i, f"Paired per-fold difference in {met}")
    finish(fig, None, name)


def p_inner_outer(d, name):
    D = d["df"]
    D = D[D["method"].isin(PLOT_METHODS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    vals = np.concatenate([D["inner_auc"], D["oof_auc"], D["outer_auc"]])
    lo = max(0.0, np.nanmin(vals) - 0.05)
    for i, (ax, col, lab) in enumerate(zip(axes, ("inner_auc", "oof_auc"),
                                           ("Inner AUC of chosen trial (Optuna)",
                                            "Out-of-fold AUC (threshold calibration)"))):
        for m in dict.fromkeys(D["method"]):
            s = D[D["method"] == m]
            ax.scatter(s[col], s["outer_auc"], s=46, color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                       edgecolor=BLACK, linewidth=0.6, label=s["label"].iloc[0], zorder=3)
        ax.plot([lo, 1], [lo, 1], ls="--", color=GRAY, lw=0.9)
        ax.set_xlim(lo, 1.005)
        ax.set_ylim(lo, 1.005)
        ax.set_aspect("equal")
        ax.set_xlabel(lab)
        ax.set_ylabel("Outer-test AUC")
        ax.legend(loc="lower right")
        panel(ax, i, "Inner (Optuna) AUC vs outer-test AUC" if i == 0 else "Out-of-fold AUC vs outer-test AUC")
    finish(fig, None, name)


def p_threshold(d, name):
    D = d["df"]
    arms = _keep(dict.fromkeys(D["method"]))
    fig, axes = plt.subplots(len(arms), 2, figsize=(12, 3.5 * len(arms)), squeeze=False)
    k = 0
    for row, a in zip(axes, arms):
        s = D[D["method"] == a]
        f = s["fold"].values
        for ax, oo, tt, lab in ((row[0], "oof_recall", "test_recall", "Recall"),
                                (row[1], "oof_specificity", "test_specificity", "Specificity")):
            ax.bar(f - 0.2, s[oo], 0.4, color=MSSP_LIGHT, edgecolor="white", linewidth=0.6,
                   label="Out-of-fold (threshold set here)")
            ax.bar(f + 0.2, s[tt], 0.4, color=MSSP_DARK, edgecolor="white", linewidth=0.6,
                   label="Outer test")
            ax.set_ylim(0, 1.22)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_xticks(f)
            ax.set_xlabel("Outer fold")
            ax.set_ylabel(lab)
            panel(ax, k, f"{s['label'].iloc[0]} - {lab.lower()}")
            k += 1
        row[0].legend(loc="upper left", ncol=2)
    finish(fig, None, name)


def p_latent(d, name):
    P, I = d["points"], d["info"]
    fig, axes, first_col, last_row = portrait_grid(len(I), ncols=2, row_h=1.9)
    for i, (ax, (_, r)) in enumerate(zip(axes, I.iterrows())):
        s = P[P["fold"] == r["fold"]]
        for g, kw in LAT_PLOT_STYLE.items():
            q = s[s["group"] == g]
            if len(q):
                ax.scatter(q["pc1"], q["pc2"], **kw)
        ax.set_xlabel(f"PC1 ({r['evr1']:.0%})", labelpad=1)
        ax.set_ylabel(f"PC2 ({r['evr2']:.0%})", labelpad=1)
        panel(ax, i, f"{d['label']}, fold {int(r['fold'])} (N={int(r['neurons'])}, "
                     f"z={int(r['latent'])}), outer-test samples")
    h, l = axes[0].get_legend_handles_labels()
    rb = bottom_legend(fig, h, l)
    finish(fig, None, name, rect_bottom=rb)


def p_training(d, name):
    D = d["df"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for fold, g in D.groupby("fold"):
        axes[0].plot(g["epoch"], g["rec"], color=fold_color(fold), lw=1.3, label=f"Fold {fold}")
        axes[1].plot(g["epoch"], g["kl"], color=fold_color(fold), lw=1.3)
    for i, (ax, lab) in enumerate(zip(axes, ("Reconstruction loss (MSE)", "KL divergence"))):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)
        panel(ax, i, f"{d['label']} - training {lab.lower()}")
    axes[0].legend(ncol=2, fontsize=7)
    finish(fig, None, name)


def p_main(d, name):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    # (a) score distribution
    ax = axes[0, 0]
    H = d["hist"]
    bins = np.linspace(H["log_ratio"].min(), H["log_ratio"].max(), 30)
    ax.hist(H[H["y"] == 0]["log_ratio"], bins=bins, density=True, alpha=0.65, color=COLOR_NORMAL,
            edgecolor="white", linewidth=0.6, label="Normal")
    ax.hist(H[H["y"] == 1]["log_ratio"], bins=bins, density=True, alpha=0.60, color=COLOR_DAMAGE,
            edgecolor="white", linewidth=0.6, label="Damage")
    ax.axvline(0, color=BLACK, ls="--", lw=1.0, label="Threshold")
    ax.set_xlabel(LOG_RATIO_LABEL)
    ax.set_ylabel("Density")
    top_legend(ax)
    panel(ax, 0, f"Pooled outer-test anomaly-score distribution - {d['main_label']}")

    # (b) latent space of the representative fold: outer-test samples only
    ax = axes[0, 1]
    L = d["latent"]
    for g, kw in LAT_PLOT_STYLE.items():
        q = L[L["group"] == g]
        if len(q):
            ax.scatter(q["pc1"], q["pc2"], **kw)
    ax.set_xlabel("Latent PC1")
    ax.set_ylabel("Latent PC2")
    top_legend(ax)
    panel(ax, 1, f"Latent space (PCA of z_mean), outer-test samples of fold {d['rep_fold']}")

    # (c) Optuna convergence with 95% CI
    ax = axes[0, 2]
    C = d["conv"]
    conv_methods = _keep(dict.fromkeys(C["method"]))
    for m in conv_methods:
        s = C[C["method"] == m]
        ax.fill_between(s["trial"], np.clip(s["ci_lo"], 0, 1), np.clip(s["ci_hi"], 0, 1),
                        color=METHOD_COLOR[m], alpha=0.25, linewidth=0)
        ax.plot(s["trial"], s["mean"], color=METHOD_COLOR[m], lw=2.3, label=s["label"].iloc[0])
    h, l = ax.get_legend_handles_labels()
    h.append(Patch(facecolor=MSSP_LIGHT, alpha=0.5, edgecolor="none"))
    l.append("95% CI")
    ax.set_xlabel("Optuna trial")
    ax.set_ylabel("Running-best inner objective")
    top_legend(ax, h, l)
    panel(ax, 2, "Optimization convergence (mean over outer folds, 95% CI)")

    # (d) boxplots normal / damage per method
    ax = axes[1, 0]
    B = d["box"]
    methods = _keep(dict.fromkeys(B["method"]))
    data, labels, colors = [], [], []
    pairs = [(MSSP_DARK, MSSP_LIGHT), (MSSP_MAIN, MSSP_MID)]
    for j, m in enumerate(methods):
        s = B[B["method"] == m]
        cn, cd = pairs[j % 2]
        for cls, lab, c in ((0, "Normal", cn), (1, "Damage", cd)):
            data.append(s[s["y"] == cls]["log_ratio"].values)
            labels.append(lab)
            colors.append(c)
    mssp_boxplot(ax, data, colors, labels)
    ax.axhline(0, color=BLACK, ls="--", lw=1.0)
    ax.tick_params(axis="x", labelsize=7 if len(data) > 4 else 9)
    for j, m in enumerate(methods):   # method name centred under each Normal/Damage pair
        ax.text(2 * j + 1.5, -0.13, B[B["method"] == m]["short"].iloc[0], ha="center", va="top",
                fontsize=8, fontweight="bold", color=BLACK, transform=ax.get_xaxis_transform())
    ax.set_ylabel(LOG_RATIO_LABEL)
    panel(ax, 3, "Anomaly-score distributions per method (outer test)")

    # (e) confusion matrices (inset layout of the reference)
    ax = axes[1, 1]
    ax.axis("off")
    ax.grid(False)
    CM = d["cms"]
    CM = CM[CM["method"].isin(PLOT_METHODS)]
    pos = [[0.02, 0.12, 0.45, 0.75], [0.53, 0.12, 0.45, 0.75]] if len(CM) == 2 else [[0.25, 0.12, 0.5, 0.75]]
    for k, (_, r) in enumerate(CM.iterrows()):
        ia = ax.inset_axes(pos[k])
        draw_cm(ia, [[r["TN"], r["FP"]], [r["FN"], r["TP"]]], show_ylabel=(k == 0),
                show_yticklabels=(k == 0))
        ia.set_title(r["label"], fontsize=10)
    panel(ax, 4, "Pooled confusion matrices")

    # (f) training history
    ax = axes[1, 2]
    T = d["train"]
    if len(T):
        for m in _keep(dict.fromkeys(T["method"])):
            s = T[T["method"] == m]
            ax.fill_between(s["epoch"], s["lo"], s["hi"], color=METHOD_COLOR[m], alpha=0.25, linewidth=0)
            ax.plot(s["epoch"], s["geo_mean"], color=METHOD_COLOR[m], lw=2.0, label=s["label"].iloc[0])
        ax.set_yscale("log")
        top_legend(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training reconstruction loss")
    panel(ax, 5, "Training history (geometric mean ± 1 SD over folds)")

    finish(fig, None, name)


PLOTTERS = {
    "eda_psd": p_eda_psd, "optuna_history": p_optuna_history, "landscape": p_landscape,
    "selected_hparams": p_selected, "noise_floor": p_noise, "roc_grid": p_roc_grid,
    "roc_comparison": p_roc_comparison, "score_distributions": p_scores, "cm_grid": p_cm_grid,
    "metric_boxplots": p_metrics, "paired_diffs": p_paired, "inner_vs_outer": p_inner_outer,
    "threshold_transfer": p_threshold, "latent": p_latent, "training_curves": p_training,
    "main_multipanel": p_main,
}


def make_all_figures(results, noise, noise_sd, complete, feats, freqs, y, ratio):
    arms = [a for a in ARMS if a in complete]
    jobs = [("fig00_main_multipanel", safe_prep(prep_main, results, complete)),
            ("fig01_eda_psd", safe_prep(prep_eda, feats, freqs, y))]
    for a in arms:
        rs = _rs(results, a)
        jobs += [(f"fig02_optuna_history_{a}", safe_prep(prep_history, a, rs, noise_sd.get(a))),
                 (f"fig03_landscape_{a}", safe_prep(prep_landscape, a, rs)),
                 (f"fig10_confusion_per_fold_{a}", safe_prep(prep_cm_folds, a, rs)),
                 (f"fig15_latent_pca_{a}", safe_prep(prep_latent, a, rs)),
                 (f"fig16_training_curves_{a}", safe_prep(prep_training, a, rs))]
    if arms:
        jobs += [("fig04_selected_hparams", safe_prep(prep_selected, results, arms)),
                 ("fig13_inner_vs_outer", safe_prep(prep_inner_outer, results, arms)),
                 ("fig14_threshold_transfer", safe_prep(prep_threshold, results, arms))]
    jobs += [("fig05_noise_floor", safe_prep(prep_noise, noise, results)),
             ("fig06_roc_per_method", safe_prep(prep_roc, results, complete, "roc_grid")),
             ("fig07_roc_comparison", safe_prep(prep_roc, results, complete, "roc_comparison")),
             ("fig08_score_distributions", safe_prep(prep_scores, results, complete)),
             ("fig09_confusion_pooled", safe_prep(prep_cm_pooled, results, complete)),
             ("fig11_metric_boxplots", safe_prep(prep_metrics, results, complete))]
    vae_complete = [m for m in complete if m.startswith("vae")]
    if vae_complete and len(complete) > 1:
        ref = max(vae_complete, key=lambda m: np.mean([results[m][f]["AUC"] for f in results[m]]))
        jobs.append(("fig12_paired_differences", safe_prep(prep_paired, results, complete, ref, ratio)))
    for name, data in jobs:
        if data is not None:
            emit(name, data)
    write_panel_index()


# ======================================================================
# 11. REPORT
# ======================================================================
def load_all_results():
    results = {m: {} for m in METHOD_ORDER}
    for fold in RUN_FOLDS:
        b = load_ckpt(f"baselines_fold{fold:02d}")
        if b:
            for m, r in b.items():
                if m in BASELINES:
                    results[m][fold] = r
        for a in ARMS:
            r = load_ckpt(f"fold_{a}_{fold:02d}")
            if r:
                results[a][fold] = r
    results["vae_nested"] = build_nested(results)
    noise = {a: load_ckpt(f"noise_floor_{a}") for a in ARMS}
    return results, noise


def build_report(feats, freqs, y, outer_splits):
    print("\n" + "=" * 72)
    print("BUILDING REPORT")
    print("=" * 72)
    results, noise = load_all_results()
    complete = [m for m in METHOD_ORDER if len(results[m]) == len(RUN_FOLDS)]
    partial = [m for m in METHOD_ORDER if 0 < len(results[m]) < len(RUN_FOLDS)]
    if partial:
        print(f"Incomplete methods (excluded from comparisons): "
              + ", ".join(f"{m} ({len(results[m])}/{len(RUN_FOLDS)})" for m in partial))
    n_te = np.mean([len(outer_splits[f - 1][1]) for f in RUN_FOLDS])
    n_tr = np.mean([len(outer_splits[f - 1][0]) for f in RUN_FOLDS])
    ratio = n_te / n_tr

    # ---- noise floor table ----
    noise_sd = {}
    for a in ARMS:
        if noise.get(a):
            N = pd.DataFrame(noise[a])
            noise_sd[a] = float(N["new_repeated"].std(ddof=1)) if len(N) > 1 else np.nan
            line = (f"Noise floor {METHOD_LABEL[a]} (fold {NOISE_FLOOR_FOLD}, {len(N)} reps): "
                    f"1x3 SD={N['old_1x3'].std(ddof=1):.4f} | {INNER_REPEATS}x{INNER_SPLITS} "
                    f"SD={noise_sd[a]:.4f}")
            r1 = results[a].get(NOISE_FLOOR_FOLD)
            if r1:
                v = np.array([t["value"] for t in r1["trials"] if not t["failed"]])
                line += (f" | Optuna trials SD={v.std(ddof=1):.4f}, best-median="
                         f"{v.max() - np.median(v):.4f}")
            print(line)
            N.to_csv(os.path.join(TAB_DIR, f"noise_floor_{a}.csv"), index=False)

    # ---- per-fold tables ----
    for m in METHOD_ORDER:
        if not results[m]:
            continue
        cols = ["fold", "AUC"] + METRICS[1:] + ["threshold", "oof_auc", "TN", "FP", "FN", "TP"]
        extra = ["neurons", "latent", "beta", "feature", "n_params", "chosen_value", "best_value",
                 "inner_auc_mean", "inner_f1_mean"] if m.startswith("vae") else []
        if m == "vae_nested":
            extra.append("selected_arm")
        df = pd.DataFrame([{c: results[m][f].get(c) for c in cols + extra} for f in sorted(results[m])])
        df.to_csv(os.path.join(TAB_DIR, f"per_fold_{m}.csv"), index=False)
        if m.startswith("vae"):
            print(f"\nPer-fold - {METHOD_LABEL[m]}")
            with pd.option_context("display.width", 250, "display.max_columns", 40):
                print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not complete:
        print("No complete method yet - summary and comparison plots skipped.")
        return

    # ---- summary table ----
    rows = []
    for m in complete:
        rs = [results[m][f] for f in sorted(results[m])]
        yp, pp, sp, _ = pooled(rs)
        boot = bootstrap_pooled(yp, sp, pp)
        cm = confusion_matrix(yp, pp, labels=[0, 1])
        acc_lo, acc_hi = wilson_ci(int(np.sum(yp == pp)), len(yp))
        row = {"method": METHOD_LABEL[m]}
        for met in METRICS:
            v = np.array([r[met] for r in rs])
            mean, lo, hi, _ = nb_corrected(v, ratio)
            row[f"{met}_mean"], row[f"{met}_sd"] = mean, v.std(ddof=1) if len(v) > 1 else np.nan
            row[f"{met}_ci_lo"], row[f"{met}_ci_hi"] = lo, hi
        row.update({"pooled_AUC": roc_auc_score(yp, sp),
                    "pooled_AUC_boot_lo": boot["AUC"][0], "pooled_AUC_boot_hi": boot["AUC"][1],
                    "pooled_Acc": float(np.mean(yp == pp)), "pooled_Acc_wilson_lo": acc_lo,
                    "pooled_Acc_wilson_hi": acc_hi,
                    "pooled_F1": f1_score(yp, pp), "pooled_F1_boot_lo": boot["F1"][0],
                    "pooled_F1_boot_hi": boot["F1"][1],
                    "pooled_TN": cm[0, 0], "pooled_FP": cm[0, 1], "pooled_FN": cm[1, 0], "pooled_TP": cm[1, 1]})
        rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(TAB_DIR, "summary_all_methods.csv"), index=False)

    print("\n" + "=" * 72)
    print(f"SUMMARY - mean ± SD across {len(RUN_FOLDS)} outer folds "
          f"[Nadeau-Bengio corrected 95% CI]")
    print("=" * 72)
    show = pd.DataFrame({"Method": S["method"]})
    for met in METRICS:
        show[met] = [f"{a:.3f}±{b:.3f} [{c:.2f},{d:.2f}]" for a, b, c, d in
                     zip(S[f"{met}_mean"], S[f"{met}_sd"], S[f"{met}_ci_lo"], S[f"{met}_ci_hi"])]
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 30):
        print(show[["Method", "AUC", "Accuracy", "BalancedAcc", "F1"]].to_string(index=False))
        print()
        print(show[["Method", "Precision", "Recall", "Specificity", "MCC"]].to_string(index=False))
    print("\nPooled over all outer test samples (scores normalized by each fold's threshold):")
    for _, r in S.iterrows():
        print(f"  {r['method']:<32} AUC={r['pooled_AUC']:.3f} [{r['pooled_AUC_boot_lo']:.3f},"
              f"{r['pooled_AUC_boot_hi']:.3f}]  Acc={r['pooled_Acc']:.3f} [{r['pooled_Acc_wilson_lo']:.3f},"
              f"{r['pooled_Acc_wilson_hi']:.3f}]  F1={r['pooled_F1']:.3f}  "
              f"CM=[{r['pooled_TN']} {r['pooled_FP']}; {r['pooled_FN']} {r['pooled_TP']}]")

    # ---- pairwise tests ----
    pr = []
    for i, a in enumerate(complete):
        for b in complete[i + 1:]:
            ya, pa, _, ia = pooled([results[a][f] for f in sorted(results[a])])
            yb, pb, _, ib = pooled([results[b][f] for f in sorted(results[b])])
            assert np.array_equal(ia, ib)
            nb_, nc_, p_mc = mcnemar_exact(ya, pa, pb)
            row = {"A": METHOD_LABEL[a], "B": METHOD_LABEL[b],
                   "McNemar_A_only_correct": nb_, "McNemar_B_only_correct": nc_, "McNemar_p": p_mc}
            for met in ("AUC", "BalancedAcc", "F1"):
                d = np.array([results[a][f][met] - results[b][f][met] for f in sorted(results[a])])
                mean, lo, hi, p = nb_corrected(d, ratio)
                row.update({f"{met}_diff": mean, f"{met}_ci_lo": lo, f"{met}_ci_hi": hi,
                            f"{met}_p_corrected_t": p, f"{met}_p_wilcoxon": safe_wilcoxon(d)})
            pr.append(row)
    if pr:
        P = pd.DataFrame(pr)
        P.to_csv(os.path.join(TAB_DIR, "pairwise_tests.csv"), index=False)
        print("\nPairwise comparisons (A - B), AUC per fold, corrected t and Wilcoxon; McNemar on pooled preds:")
        for _, r in P.iterrows():
            print(f"  {r['A']:<30} vs {r['B']:<26} dAUC={r['AUC_diff']:+.3f} "
                  f"[{r['AUC_ci_lo']:+.3f},{r['AUC_ci_hi']:+.3f}] p_t={r['AUC_p_corrected_t']:.3f} "
                  f"p_W={r['AUC_p_wilcoxon']:.3f} | McNemar p={r['McNemar_p']:.3f}")
        print("  (No multiple-comparison correction applied; interpret p-values accordingly.)")

    # ---- hyperparameter stability ----
    for a in ARMS:
        if a in complete:
            H = pd.DataFrame([{k: results[a][f][k] for k in ("fold", "neurons", "latent", "beta", "feature")}
                              for f in sorted(results[a])])
            print(f"\nChosen hyperparameters - {METHOD_LABEL[a]}:")
            print(H.to_string(index=False))
    if "vae_nested" in complete:
        sel = [results["vae_nested"][f]["selected_arm"] for f in sorted(results["vae_nested"])]
        print(f"\nvae_nested picked: psd in {sel.count('vae_psd')} folds, raw in {sel.count('vae_raw')} folds")

    # ---- figures + plot data ----
    print("\nSaving figures (MSSP style) and plot data ...")
    make_all_figures(results, noise, noise_sd, complete, feats, freqs, y, ratio)
    print(f"Figures: ./{FIG_DIR}/   Plot data: ./{PLOT_DATA_DIR}/   Tables: ./{TAB_DIR}/")

    print("\n" + "=" * 72)
    print("CONCLUSION (outer-test estimates; every method above is reported)")
    print("=" * 72)
    for _, r in S.iterrows():
        tgt = "yes" if (r["AUC_mean"] > 0.9 and r["F1_mean"] > 0.9) else "no"
        print(f"  {r['method']:<32} AUC={r['AUC_mean']:.3f}  Acc={r['Accuracy_mean']:.3f}  "
              f"F1={r['F1_mean']:.3f}  (AUC>0.9 & F1>0.9: {tgt})")
    print("  Caveat: normal and damaged records come from different seasonal periods; any method's")
    print("  separation may partly reflect environmental (temperature) differences, not damage alone.")


# ======================================================================
# 12. MAIN
# ======================================================================
def main():
    for d in (OUTPUT_DIR, CKPT_DIR, FIG_DIR, TAB_DIR, PLOT_DATA_DIR):
        os.makedirs(d, exist_ok=True)
    apply_mssp_style()
    if REPLOT_ONLY:
        print(f"REPLOT_ONLY: redrawing figures from ./{PLOT_DATA_DIR}/")
        replot_all()
        return
    sys.stdout = Tee(os.path.join(OUTPUT_DIR, "run_log.txt"))
    t_start = time.time()
    print("\n" + "=" * 72)
    print(f"Z24 SHALLOW VAE - EXTENDED EXPERIMENT - started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print(f"TF {tf.__version__} | Optuna {optuna.__version__} | NumPy {np.__version__} | "
          f"QUICK_TEST={QUICK_TEST} | REPORT_ONLY={REPORT_ONLY}")
    print(f"folds={RUN_FOLDS} | trials={N_TRIALS} | inner={INNER_REPEATS}x{INNER_SPLITS} | "
          f"epochs={EPOCHS} | cross-fit={FINAL_CV_SPLITS} | 1-SE={ONE_SE_RULE} | "
          f"PSD nperseg={PSD_NPERSEG_OPTIONS} | FS={FS} Hz (labels only)")
    config = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, bool, list, dict, type(None)))}
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    X, y = load_dataset()
    feats, freqs = build_features(X)
    print("Features: " + ", ".join(f"{k}{v.shape}" for k, v in feats.items()))
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    outer_splits = list(outer_cv.split(X, y))

    saved = load_ckpt("outer_splits")
    if saved is not None and not all(np.array_equal(a[1], b[1]) for a, b in zip(saved, outer_splits)):
        raise RuntimeError("Outer splits differ from checkpoint - data or seed changed. "
                           "Use a new OUTPUT_DIR.")
    save_ckpt("outer_splits", outer_splits)

    if not REPORT_ONLY:
        print("\n[E1] Baselines")
        for fold in RUN_FOLDS:
            tr, te = outer_splits[fold - 1]
            try:
                b = run_baselines(fold, feats, y, tr, te)
                print(f"  fold {fold:2d}: " + "  ".join(f"{METHOD_LABEL[m]} AUC={b[m]['AUC']:.3f}"
                                                        for m in BASELINES))
            except Exception:
                print(f"[ERROR] baselines fold {fold}:\n{traceback.format_exc()}")

        print("\n[E2] Noise floor")
        for a in ARMS:
            try:
                run_noise_floor(a, feats, y, outer_splits)
            except Exception:
                print(f"[ERROR] noise floor {a}:\n{traceback.format_exc()}")

        storage = get_storage()
        for a in ARMS:
            print(f"\n[{'E3' if a == 'vae_psd' else 'E4'}] {METHOD_LABEL[a]} - nested CV")
            for fold in RUN_FOLDS:
                tr, te = outer_splits[fold - 1]
                print(f"\n>>> {a} outer fold {fold}/{N_OUTER} | elapsed {fmt_time(time.time() - t_start)}")
                try:
                    run_vae_fold(a, fold, feats, y, tr, te, storage)
                except Exception:
                    print(f"[ERROR] {a} fold {fold} - will be retried on next run:\n{traceback.format_exc()}")
                    free_tf()
            try:
                build_report(feats, freqs, y, outer_splits)   # intermediate report per arm
            except Exception:
                print(f"[ERROR] report:\n{traceback.format_exc()}")
    else:
        build_report(feats, freqs, y, outer_splits)

    print(f"\nTotal time: {fmt_time(time.time() - t_start)} | finished {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
