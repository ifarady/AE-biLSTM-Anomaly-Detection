# Autoencoder with 2-layer LSTM for Anomaly Detection in ECG Signals

This repository provides an implementation for detecting anomalies in ECG signals using an autoencoder with a 2-layer LSTM (Long Short-Term Memory) network. The architecture includes LSTM layers in both the encoder and decoder sections. While it is possible to experiment with additional layers, the results from this study suggest that utilizing 2 layers yields optimal performance for anomaly detection in ECG signals. The code demonstrates the successful adaptation of 2D data into 1D data to effectively detect anomalies. Additionally, users can easily adjust the anomaly detection threshold.

![Alt Text](https://github.com/ifarady/AE-biLSTM-Anomaly-Detection/blob/main/figs/ae-2lstm.png)
*Structure of LSTM-Autoencoder with compressing features.*

In this work, we utilize the [ECG5000 dataset](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000). The ECG5000 dataset originates from a 20-hour ECG recording named "chf07" within the BIDMC Congestive Heart Failure Database, as published by Goldberger et al. in 2000. The recording is sourced from a patient diagnosed with severe congestive heart failure. This dataset comprises 5,000 individual heartbeats extracted from the original recording. These heartbeats have been pre-processed to have equal lengths through interpolation and are further labeled with automated anomaly annotations. Initially presented in the paper "A general framework for never-ending learning from time series streams"

![Alt Text](https://github.com/ifarady/AE-biLSTM-Anomaly-Detection/blob/main/figs/fig2.png)
*ECG signals from ECG500 dataset.*

Figure below shows the prediction results of anomaly and normal ECG datasets using an autoencoder with a 2-layer LSTM. The threshold for this prediction can be customized to enhance the quality of results.

<image src="https://github.com/ifarady/AE-biLSTM-Anomaly-Detection/blob/main/figs/fig4.png" alt="Distribution density of anomaly and normal signals with AE-2 layer LSTM" width="400"/>
*Distribution density of anomaly and normal signals with AE-2 layer LSTM.*

## Requirements
- Python 3
- PyTorch
- Pandas
- Matplotlib

## Credit
This code is a modification of the original code available at [https://github.com/shobrook/sequitur](https://github.com/shobrook/sequitur). The initial implementation and credit for the original code go to its respective author(s). I have made modifications for experimental and implementation purposes.

**Note:**
This code is intended for experimental and implementation purposes only. Please refer to the original source for licensing and usage information.
