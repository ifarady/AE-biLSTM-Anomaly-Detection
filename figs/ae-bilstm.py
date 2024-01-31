# This code is a modification of the original code from:
#   https://github.com/shobrook/sequitur
#
# This code is intended for experimental and implementation purposes only.
# All credit for the original code goes to its author(s).
#
# Please refer to the original source for licensing and usage information.

!nvidia-smi

!pip install -qq liac-arff
# import pandas as pd
# from liac_arff import loadarff

# # Load ARFF file
# data, meta = loadarff('path/to/your/arff/file.arff')

# # Convert to DataFrame
# df = pd.DataFrame(data)

!pip install -q -U watermark



# Commented out IPython magic to ensure Python compatibility.
# %reload_ext watermark
# %watermark -v -p numpy,pandas,torch,liac-arff

# Commented out IPython magic to ensure Python compatibility.
import torch

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
import arff
# from arff import a2p
# from liac_arff import a2p
# from liac-arff import loadarff


# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

!gdown --id 16MIleqoIr1vYxlGk4GKnGmrsCPuWkkpT

!unzip -qq ECG5000.zip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset = pd.DataFrame(arff.load(open('Training Dataset.arff')))
# print(dataset)
with open('ECG5000_TRAIN.arff') as f:
  data = arff.load(f)
  train = pd.DataFrame(data['data'])
  # train = pd.DataFrame(arff.load(open(f)))
  # train = a2p.load(f)

with open('ECG5000_TEST.arff') as f:
  data = arff.load(f)
  test = pd.DataFrame(data['data'])

import pandas as pd
df = pd.concat([train, test])
# df = train.append(test)
df = df.sample(frac=1.0)
df.shape

df.head()

CLASS_NORMAL = 1

class_names = ['Normal','R on T','PVC','SP','UB']

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns
# print(new_columns)

df.target.value_counts()

df.target.shape

class_counts = df['target'].value_counts()
ax = sns.countplot(x='target', data=df)
ax.set_xticklabels(class_names)
plt.show()

def plot_time_series_class(data, class_name, ax, n_steps=10):
  time_series_df = pd.DataFrame(data)

  smooth_path = time_series_df.rolling(n_steps).mean()
  path_deviation = 2 * time_series_df.rolling(n_steps).std()

  under_line = (smooth_path - path_deviation)[0]
  over_line = (smooth_path + path_deviation)[0]

  ax.plot(smooth_path, linewidth=2)
  ax.fill_between(
    path_deviation.index,
    under_line,
    over_line,
    alpha=.125
  )
  ax.set_title(class_name)



classes = df.target.unique()

fig, ax = plt.subplots(len(classes), sharex=True, sharey=True)
fig.set_size_inches(15, 15)

for i, cls in enumerate(classes):
    data = df.loc[df.target == cls, df.columns != 'target'].sample(1).iloc[0, :186]
    ax[i].plot(data)
    ax[i].set_title( class_names[i])
    # ax[i].set_yticklabels([])

plt.show()



# classes = df.target.unique()

# fig, axs = plt.subplots(
#   nrows=len(classes) // 3 + 1,
#   ncols=3,
#   sharey=True,
#   figsize=(14, 8)
# )

# for i, cls in enumerate(classes):
#   ax = axs.flat[i]
#   data = df[df.target == cls] \
#     .drop(labels='target', axis=1) \
#     .mean(axis=0) \
#     .to_numpy()
#   plot_time_series_class(data, class_names[i], ax)

# fig.delaxes(axs.flat[-1])
# fig.tight_layout();

import matplotlib.pyplot as plt

font_size = 20

classes = df.target.unique()

fig, axs = plt.subplots(
  nrows=len(classes) // 3 + 1,
  ncols=3,
  sharey=True,
  figsize=(14, 8)
)

for i, cls in enumerate(classes):
  ax = axs.flat[i]
  data = df[df.target == cls] \
    .drop(labels='target', axis=1) \
    .mean(axis=0) \
    .to_numpy()
  plot_time_series_class(data, class_names[i], ax)

  # Set the font size for titles, labels, and tick labels
  ax.set_title(class_names[i], fontsize=font_size)
  ax.xaxis.set_tick_params(labelsize=font_size)
  ax.yaxis.set_tick_params(labelsize=font_size)

fig.delaxes(axs.flat[-1])
fig.tight_layout()

plt.show()  # Display the plot

normal_df = df[df.target == str(CLASS_NORMAL)].drop(labels='target', axis=1)
normal_df.shape

anomaly_df = df[df.target != str(CLASS_NORMAL)].drop(labels='target', axis=1)
anomaly_df.shape

train_df, val_df = train_test_split(
  normal_df,
  test_size=0.15,
  random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
  val_df,
  test_size=0.33,
  random_state=RANDOM_SEED
)

print("Train Set:")
print(train_df.index.value_counts())
print("\nValidation Set:")
print(val_df.index.value_counts())
print("\nTest Set:")
print(test_df.index.value_counts())

def create_dataset(df):

  sequences = df.astype(np.float32).to_numpy().tolist()

  dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

  n_seq, seq_len, n_features = torch.stack(dataset).shape

  return dataset, seq_len, n_features

train_dataset, seq_len, n_features = create_dataset(train_df)
val_dataset, _, _ = create_dataset(val_df)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)

class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x

model = RecurrentAutoencoder(seq_len, n_features, 128)
model = model.to(device)

def train_model(model, train_dataset, val_dataset, n_epochs):
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  for epoch in range(1, n_epochs + 1):
    model = model.train()

    train_losses = []
    for seq_true in train_dataset:
      optimizer.zero_grad()

      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      loss.backward()
      optimizer.step()

      train_losses.append(loss.item())

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:

        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)

        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history

import torch

# Provide the full path to "model.pth" in your Google Drive
model_path = '/content/Lmodel150epochs.pth'


# Load the model on the CPU
model = torch.load(model_path, map_location=torch.device('cpu'))

# model, history = train_model(
#   model,
#   train_dataset,
#   val_dataset,
#   n_epochs=150
# )
model.eval()

# train_loss = history['train_loss']
# val_loss = history['val_loss']
# epochs = range(1, len(train_loss) + 1)

# # Plot the loss curves
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss, label='Train Loss')
# plt.plot(epochs, val_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Loss over Training Epochs')
# plt.grid(True)
# plt.show()

# ax = plt.figure().gca()

# ax.plot(history['train'])
# ax.plot(history['val'])
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'val'])
# plt.title('Loss over training epochs')
# plt.show();

from google.colab import drive
drive.mount('/content/drive')

MODEL_PATH = '/Lmodel150epochs.pth'

torch.save(model, MODEL_PATH)

# from google.colab import files

# # Specify the path of the file you want to download
# file_path = '/content/model.pth'

# # Download the file
# files.download(file_path)

# !gdown --id 1jEYx5wGsb7Ix8cZAw3l5p5pOwHs3_I9A
# model = torch.load('model.pth')
# model = model.to(device)

def predict(model, dataset):
  predictions, losses = [], []
  criterion = nn.L1Loss(reduction='sum').to(device)
  with torch.no_grad():
    model = model.eval()
    for seq_true in dataset:
      seq_true = seq_true.to(device)
      seq_pred = model(seq_true)

      loss = criterion(seq_pred, seq_true)

      predictions.append(seq_pred.cpu().numpy().flatten())
      losses.append(loss.item())
  return predictions, losses

_, losses = predict(model, train_dataset)

ax = sns.distplot(losses, bins=50, kde=True,color='green');
# ax.set_xlabel('Reconstruction Loss')
ax.set_xlabel('Reconstruction Loss',fontsize=20)
ax.set_ylabel('Density', fontsize=20)

THRESHOLD = 35

predictions, pred_losses = predict(model, test_normal_dataset)
ax = sns.distplot(pred_losses, bins=50, kde=True,color='green');
ax.set_xlabel('Reconstruction Loss',fontsize=20)
ax.set_ylabel('Density', fontsize=20)

correct = sum(l <= THRESHOLD for l in pred_losses)
print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')

anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

predictions, pred_losses = predict(model, anomaly_dataset)
ax = sns.distplot(pred_losses, bins=50, kde=True,color='green');
ax.set_xlabel('Reconstruction Loss',fontsize=20)
ax.set_ylabel('Density', fontsize=20)

correct = sum(l > THRESHOLD for l in pred_losses)
print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')

def plot_prediction(data, model, title, ax,legend_loc='lower right'):
  predictions, pred_losses = predict(model, [data])

  ax.plot(data, label='original')
  ax.plot(predictions[0], label='reconstructed')
  ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
  ax.legend(loc=legend_loc)

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'green'])
fig, axs = plt.subplots(
  nrows=2,
  ncols=6,
  sharey=True,
  sharex=True,
  figsize=(22, 8)
)

for i, data in enumerate(test_normal_dataset[:6]):
  plot_prediction(data, model, title='Normal', ax=axs[0, i])

for i, data in enumerate(test_anomaly_dataset[:6]):
  plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

fig.tight_layout()
plt.show()

import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 20})  # Adjust the fontsize as needed

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'green'])

def plot_prediction(data, model, title, ax, legend_loc='lower right'):
    predictions, pred_losses = predict(model, [data])

    ax.plot(data, label='original')
    ax.plot(predictions[0], label='reconstructed')


    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})', fontsize=20)  # Adjust the fontsize as needed


    ax.legend(loc=legend_loc)

fig, axs = plt.subplots(
  nrows=2,
  ncols=6,
  sharey=True,
  sharex=True,
  figsize=(22, 8)
)

for i, data in enumerate(test_normal_dataset[:6]):
  plot_prediction(data, model, title='Normal', ax=axs[0, i])

for i, data in enumerate(test_anomaly_dataset[:6]):
  plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

fig.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})  # Adjust the fontsize as needed

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['blue', 'green'])

def plot_prediction(data, model, title, ax, legend_loc='lower right'):
    predictions, pred_losses = predict(model, [data])

    ax.plot(data, label='original')
    ax.plot(predictions[0], label='reconstructed')

    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})', fontsize=20)  # Adjust the fontsize as needed

    # Increase the font size for the legend labels
    legend = ax.legend(loc=legend_loc)
    for label in legend.get_texts():
        label.set_fontsize(16)  # Adjust the fontsize as needed

fig, axs = plt.subplots(
  nrows=2,
  ncols=6,
  sharey=True,
  sharex=True,
  figsize=(22, 8)
)

for i, data in enumerate(test_normal_dataset[:6]):
  plot_prediction(data, model, title='Normal', ax=axs[0, i])

for i, data in enumerate(test_anomaly_dataset[:6]):
  plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

fig.tight_layout()
plt.show()



