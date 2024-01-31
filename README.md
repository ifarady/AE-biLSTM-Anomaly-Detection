# Autoencoder with 2-layer LSTM for Anomaly Detection in ECG Signals

This repository provides an implementation for detecting anomalies in ECG signals using an autoencoder with a 2-layer LSTM (Long Short-Term Memory) network. The architecture includes LSTM layers in both the encoder and decoder sections. While it is possible to experiment with additional layers, the results from this study suggest that utilizing 2 layers yields optimal performance for anomaly detection in ECG signals. The code demonstrates the successful adaptation of 2D data into 1D data to effectively detect anomalies. Additionally, users can easily adjust the anomaly detection threshold.

![Alt Text](url)
*Caption: Your caption here.*


## Requirements
- Python 3
- PyTorch
- Standard Library
- Pandas
- Matplotlib

## Credit
This code is a modification of the original code available at [https://github.com/shobrook/sequitur](https://github.com/shobrook/sequitur). The initial implementation and credit for the original code go to its respective author(s). I have made modifications for experimental and implementation purposes.

**Note:**
This code is intended for experimental and implementation purposes only. Please refer to the original source for licensing and usage information.
