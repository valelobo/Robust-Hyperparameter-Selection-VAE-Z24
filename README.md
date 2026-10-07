# Robust Hyperparameter Selection for Real-Time Monitoring: Shallow VAEs on Bridge Damage Detection

Code for the paper *"Robust Hyperparameter Selection for Real-Time Monitoring: Shallow VAEs on Bridge Damage Detection"*
(L. Lobo, D. Ruiz, D. Barreto, A. Freitas, E. Lima e Silva), submitted to *Mechanical Systems and Signal Processing*.

The repository implements an **epistemic-aware framework** for selecting the hyperparameters and the input
representation of a shallow variational autoencoder (VAE) for unsupervised damage detection, and applies it to
the Z24 bridge benchmark.

## Method in short

- **Model:** shallow VAE, `input -> Dense(N, ReLU) -> (mu, log sigma^2) -> Dense(N, ReLU) -> input`, trained on
  healthy data only (full batch, Adam, 800 epochs). Anomaly score = reconstruction error through the latent mean.
- **Inputs:** raw 1000-sample windows, or their log power spectral density (Welch, segment length searched).
- **Nested cross-validation:** 10 stratified outer folds. In each outer-train set, Optuna (TPE, 150 trials) searches
  `N`, latent size, KL weight `beta` and the Welch segment length.
- **Objective:** mean of `0.5 * (AUC + F1 at the Youden threshold)` over repeated stratified 2x5 inner CV.
- **Selection:** one-standard-error rule, which picks the model with the fewest parameters within one SE of the best.
- **Threshold:** Youden threshold on out-of-fold scores of 5 cross-fitted models, fixed before the outer test.
- **Noise floor:** one fixed configuration retrained 20 times (1x3 vs 2x5 inner CV). This measures how much the
  objective changes from training and resampling alone.
- **Baseline:** z-distance of the log-PSD to the healthy mean.

## Requirements

Python 3 with:

numpy, scipy, pandas, pyarrow, scikit-learn, tensorflow, optuna, matplotlib

Everything runs on CPU.

## Data

The Z24 data belong to **KU Leuven** and are **not included** in this repository. Request access from KU Leuven.
The script expects a file `Z24_R3V_Feb_to_Aug.npz` next to it, containing:

- `X`: array of shape `(n_windows, 1000)`, accelerations of channel R3V (100 Hz, non-overlapping 10 s windows)
- `y`: labels, `0` = healthy (PDT scenario 1, 4 August 1998), `1` = damaged (scenario 6, 18 August 1998,
  pier lowered by 95 mm)

## Usage

Run the script from its own folder, because the output folder is created relative to the working directory:

    cd Code
    python claudev2.py

Run modes are set at the top of the script:

| Flag | Effect |
|---|---|
| `QUICK_TEST = True` | Short smoke test of the whole pipeline, written to a separate folder |
| `REPORT_ONLY = True` | No training: rebuild tables, figures and plot data from saved checkpoints |
| `REPLOT_ONLY = True` | Only redraw the figures from the saved plot data (seconds) |

A full run took about 20 hours on CPU. Every Optuna trial (SQLite) and every finished fold (pickle) is
checkpointed, so an interrupted run resumes where it stopped.

## Outputs

`z24_vae_overnight/` contains:

- `checkpoints/`: Optuna database and per-fold results
- `tables/`: per-fold metrics, summary with corrected confidence intervals, pairwise tests, noise floor
- `plot_data/`: the data behind every figure (`.pkl` + CSV)
- `run_log.txt`, `config.json`: full log and configuration of the run

## Citation

If you use this code, please cite the paper (reference to be added after publication).

## Acknowledgements

This work was supported by CAPES and CNPq (Brazil). The Z24 benchmark is described in
Maeck & De Roeck, *Mechanical Systems and Signal Processing* 17(1), 2003, 127–131.
