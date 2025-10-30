# Deep Learning for Time Series 

This README accompanies the `TSA4.ipynb` notebook in this workspace. It provides a high-level, descriptive walkthrough of the methods, modeling choices, and the analysis pipeline used in the notebook. The goal of the notebook is to explore, model, and evaluate forecasting and anomaly-detection techniques on a time series dataset (`ELECTRIC.csv`) using deep learning models and classical approaches.

> Note: This README focuses on explanation and interpretation of methods used and results produced (including visual artifacts in the `images/` folder). It intentionally omits step-by-step replication instructions.

## Project summary

The notebook performs a typical time series modeling pipeline:
- Data ingestion and exploratory data analysis (EDA) to check trends, seasonality, missing values, and distributional properties.
- Time series preprocessing (resampling, cleaning, imputation, scaling) and feature construction (lags, rolling statistics, time-based features).
- Creation of supervised training windows (sliding windows / sequence-to-one and sequence-to-sequence formats) suitable for neural network models.
- Training and evaluation of two deep learning models: an LSTM-based forecasting model and a Temporal Convolutional Network (TCN) model. Additional experiments include generative adversarial approaches and classical anomaly detection.
- Visual and quantitative evaluation (loss curves, prediction vs. ground truth plots, residual diagnostics) and artifact saving (trained model weights).

The repository contains model artifacts and visualizations:
- `best_lstm_model.pth` — best LSTM weights found during training.
- `best_tcn_model.pth` — best TCN weights found during training.
- `images/` — contains plotting artifacts used in the notebook.


Included images (referenced below):

- `images/LSTM forecasting.png` — LSTM model forecast vs. ground truth (training/validation/test split visualizations).
- `images/TCN forecasting.png` — TCN model forecast visualization.
- `images/generative adversarial networks.png` — visualization illustrating GAN-based anomaly-detection experiments (e.g., reconstruction / discriminator-based anomaly scores and detected anomalies).
- `images/isolation forest for anomaly detection.png` — anomaly detection visualization using an Isolation Forest (classical unsupervised method) with detected anomalies highlighted.

(If an image does not display in your viewer, check spaces in filenames or open via your file manager — the images are in the `images/` folder.)

## Data and EDA (conceptual)

Dataset: `ELECTRIC.csv` — a recorded electrical time-series (likely energy consumption or a component measurement) used for forecasting and anomaly detection.

Typical EDA steps implemented:
- Time index parsing and frequency inference (daily/hourly/minutely). Checking for missing timestamps.
- Visual inspection (line plots) to identify trend and seasonality.
- Stationarity checks (visual and statistical tests such as rolling mean/variance inspection and Augmented Dickey-Fuller where applicable).
- Histogram and kde plots for distributional understanding; autocorrelation (ACF) and partial autocorrelation (PACF) to inspect lag relationships.

Why these matter: identifying seasonality and autocorrelation shapes model choices (e.g., need for differencing, choice of sequence length, whether to incorporate exogenous features).

## Preprocessing and Feature Engineering

Common preprocessing steps used in the notebook:
- Missing value handling: forward/backward filling, interpolation, or removal depending on gap sizes.
- Resampling: if the dataset had an irregular sampling rate, it was resampled to a consistent frequency.
- Scaling: normalization or standardization (e.g., MinMaxScaler or StandardScaler) applied to input features before feeding neural models.
- Windowing: converting the series into overlapping sequences with a specified input sequence length (lookback) and prediction horizon. Both sequence-to-one and sequence-to-sequence formats may be used depending on model output design.
- Additional features: calendar features (hour, day-of-week), rolling aggregates (mean, std), and lag features to capture short-term dependencies.

Design tradeoffs documented in the notebook:
- Lookback length vs. model complexity: longer lookbacks give more context but increase memory and training time.
- Whether to normalize using global fit (train+val+test) vs. train-only fit — typically, the scaler is fit only on training data to avoid information leakage.

## Models and architectures

Two main neural architectures were used and compared:

1) LSTM (Long Short-Term Memory)
- Architecture: stacked LSTM layers (one or more recurrent layers) followed by dense output layers.
- Input shape: (batch_size, sequence_length, n_features).
- Output: either a single-step forecast (regression) or multi-step sequence predictions depending on setup.
- Loss: Mean Squared Error (MSE) or Mean Absolute Error (MAE), depending on experiment.
- Optimizer: typically Adam with a small learning rate (e.g., 1e-3 to 1e-4), and weight decay optionally.
- Regularization: dropout in recurrent layers and/or L2 weight regularization.
- Early stopping and model checkpointing were used to preserve the best model on validation loss.

Why LSTM: LSTMs capture long-term dependencies in sequences via gating mechanisms; they’re a natural baseline for many time-series forecasting tasks.

2) TCN (Temporal Convolutional Network)
- Architecture: causal/inflated 1D convolutional layers with residual connections, dilations to capture large receptive fields, and a final dense head for prediction.
- TCNs process sequences in a fully-convolutional manner and can be faster to train than recurrent models while offering comparable receptive fields.
- Key hyperparameters: number of filters, kernel size, dilation schedule, and dropout.

Why TCN: TCNs often match or outperform RNNs on sequence modeling tasks while being easier to parallelize and sometimes more stable to train.

3) Additional experiments
- GAN-based anomaly detection: the notebook includes experiments using generative adversarial networks for anomaly detection — for example, using reconstruction error or discriminator scores to flag anomalous segments (see `images/generative adversarial networks.png`).
- Classical anomaly detection: Isolation Forest-based anomaly detection is used as a complementary unsupervised baseline and is visualized in `images/isolation forest for anomaly detection.png`.

## Training procedure (typical setup)

Core training loop elements used in the notebook:
- Train / validation / test splits kept chronological to avoid leakage.
- Batch sampling with sequence windows; careful shuffling applied only at the batch level, preserving temporal order within windows.
- Loss function: MSE (primary) and sometimes MAE for robustness to outliers.
- Early stopping based on validation loss with patience to avoid overfitting.
- Model checkpointing: best validation model saved as `best_lstm_model.pth` and `best_tcn_model.pth`.

Hyperparameter choices discussed:
- Learning rate scheduling (reduce-on-plateau) to refine convergence.
- Batch size tradeoffs: larger batches increase GPU throughput but may reduce generalization.
- Number of epochs determined by early stopping and convergence behavior on validation loss.

## Evaluation and metrics

Quantitative metrics considered for forecasting tasks:
- MSE (Mean Squared Error) — penalizes large errors strongly.
- RMSE (Root Mean Squared Error) — interpretable in original units.
- MAE (Mean Absolute Error) — robust to outliers.
- MAPE (Mean Absolute Percentage Error) — percentage error, avoid if series has values near zero.

For anomaly detection tasks:
- Precision/Recall and F1 if labeled anomalies are available.
- Visual inspection and manual verification are used where ground truth is scarce.

Qualitative evaluation:
- Forecast plots overlaying predictions and ground truth (see `images/LSTM forecasting.png` and `images/TCN forecasting.png`).
- Loss curves (training vs. validation) to detect overfitting or underfitting.
- Residual plots and autocorrelation of residuals to check whether remaining signal exists.

## Visual artifacts (what to look for)

- `images/LSTM forecasting.png`: This figure typically shows the ground truth series in one color and the LSTM predictions overlaid. Look for how well the model captures peaks, troughs, and seasonal patterns. Gaps between prediction and truth indicate model error — check whether these correspond to high-variance or rare events.

- `images/TCN forecasting.png`: Same conceptual layout for the TCN model. Compare TCN vs LSTM in terms of sharpness of peaks, phase alignment, and smoothness of predictions.

-- `images/generative adversarial networks.png`: Visual outputs from GAN-based anomaly-detection experiments. These illustrate typical anomaly signals (reconstruction residuals, discriminator scores, or flagged sequences) used to identify outliers in the series.

- `images/isolation forest for anomaly detection.png`: Shows detected outliers colored/highlighted on the original timeseries. Isolation Forest is an unsupervised tree-based method that isolates anomalies by random partitioning; its visual outputs help identify segments that deviate from normal behavior.

## Artifacts and files

- `TSA4.ipynb` — the interactive notebook containing code, experiments, and plotting. It is the primary companion to this README.
- `ELECTRIC.csv` — the dataset used for experiments.
- `best_lstm_model.pth` and `best_tcn_model.pth` — saved model weights from the experiments.
- `images/` — visual outputs used by the notebook and referenced throughout this README.

## Interpretation and lessons learned

- Model comparison: LSTM and TCN both offer strengths. LSTMs are powerful for capturing long-term dependencies with gating, while TCNs usually train faster and handle long contexts through dilations. The notebook documents differences in forecasting quality and training behavior.
- Data quality matters: missing data, non-stationarity, and rare events materially affect forecasting models. Preprocessing and careful window design are crucial.
-- GANs and classical anomaly detection methods provide complementary perspectives: here GANs were applied for anomaly detection (e.g., via reconstruction/discriminator signals) while Isolation Forests serve as a fast unsupervised baseline.

## LSTM Forecasting
![Error for LSTM forecasting](images/Error%20for%20LSTM%20forecasting.png)

## TCN Forecasting
![Error for TCN forecasting](images/error%20for%20TCN%20forecasting.png)

## GAN Model
![Generative Adversarial Networks](images/generative%20adversarial%20networks.png)

## Isolation Forest
![Isolation Forest for Anomaly Detection](images/isolation%20forest%20for%20anomaly%20detection.png)
