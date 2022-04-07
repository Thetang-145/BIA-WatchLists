import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc, rcParams
import math
import os

import pandas as pd
import numpy as np
import tqdm
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALLETE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALLETE))

rcParams['figure.figsize'] = 12, 8

tqdm.pandas()

pl.seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("configured device:", device)

def Load_data(path):
    # Load watch price
    watch = pd.read_csv(path)
    watch['Date'] = pd.to_datetime(watch['Date'], dayfirst=True)
    watch['Price'] = pd.to_numeric(watch['Price'])
    watch = watch.resample('D', on='Date', convention='s').mean()
    watch['Date'] = watch.index
    watch.reset_index(drop=True, inplace=True)
    watch.fillna(method='ffill', inplace=True)
    watch["Prev_Price"] = watch.shift(1)["Price"]
    watch["Price_Change"] = watch.progress_apply(
        lambda row: 0 if np.isnan(row["Prev_Price"]) else row["Price"] - row["Prev_Price"],
        axis = 1
    )
    # Creat watch feature
    rows = []
    for _, row in tqdm(watch.iterrows(), total=watch.shape[0]):
        row_data = dict(
            day_of_week = row.Date.dayofweek,
            day_of_month = row.Date.day,
            week_of_year = row.Date.week,
            month = row.Date.month,
            price_change = row.Price_Change,
            price = row.Price
        )
        rows.append(row_data)
    return pd.DataFrame(rows)

def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, days_pred):

    sequences = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - sequence_length - days_pred)):

        sequence = input_data[i:i+sequence_length]

        label_position = i + sequence_length + days_pred
        label = input_data.iloc[label_position][target_column]
        
        sequences.append((sequence, label))

    return sequences

class TS_Dataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        
        sequence, label = self.sequences[idx]

        return dict(
            sequence = torch.Tensor(sequence.to_numpy()).to(device),
            label = torch.tensor(label).float().to(device)
        )

class PriceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size=8):
        super().__init__
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        
    def prepare_data(self):
        self._has_prepared_data = True

    def setup(self):
        self.train_dataset = TS_Dataset(self.train_sequences)
        self.test_dataset = TS_Dataset(self.test_sequences)
        self.prepare_data()
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4,
            shuffle=False
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False
        )

class PricePredictionModel(nn.Module):

    def __init__(self, n_features, n_hidden=128, n_layers=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size =n_hidden,
            batch_first = True,
            num_layers = n_layers,
            dropout = 0.2
        )
        self.regressor = nn.Linear(n_hidden, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]

        return self.regressor(out)

class PricePredictor(pl.LightningModule):

    def __init__(self, n_features: int):
        super().__init__()
        self.model = PricePredictionModel(n_features).to(device)
        self.criterion = nn.MSELoss().to(device)

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))

        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss    

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        loss, outputs = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adagrad(self.parameters(), lr=1e-3)

def descale(descaler, values):
    values_2d = np.array(values)[:, np.newaxis]
    return descaler.inverse_transform(values_2d).flatten()

# Main
def Train_eval_model(
    path, 
    model_name,
    TRAIN_RATIO = 0.8, 
    SEQUENCE_LENGTH = 30, 
    DAYS_PREDICTION = 30,
    N_EPOCHS = 50,
    BATCH_SIZE = 64
    ):
    features_df = Load_data(path)
    train_size = int(len(features_df) * TRAIN_RATIO)
    train_df, test_df = features_df[:train_size], features_df[train_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_df)

    train_df = pd.DataFrame(
        scaler.transform(train_df),
        index=train_df.index,
        columns=train_df.columns
    )
    test_df = pd.DataFrame(
        scaler.transform(test_df),
        index=test_df.index,
        columns=test_df.columns
    )

    train_sequences = create_sequences(train_df, "price", SEQUENCE_LENGTH, DAYS_PREDICTION)
    test_sequences = create_sequences(test_df, "price", SEQUENCE_LENGTH, DAYS_PREDICTION)

    data_module = PriceDataModule(train_sequences, test_sequences, batch_size=BATCH_SIZE)
    data_module.setup()

    model = PricePredictor(n_features=train_df.shape[1])

    checkpoint_callback = ModelCheckpoint(
        dirpath="results/checkpoints",
        filename = model_name,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("results/lightning_logs", name="model_name")

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

    trainer.fit(model, data_module)

    trained_model = PricePredictor.load_from_checkpoint(
        "results/checkpoints/"+model_name+".ckpt",
        n_features=train_df.shape[1]
    )

    trained_model.freeze()

    test_dataset = TS_Dataset(test_sequences)
    predictions = []
    labels = []
    for item in tqdm(test_dataset):
        sequence = item['sequence']
        label = item['label']
        _, output = trained_model(sequence.unsqueeze(dim=0))
        predictions.append(output.item())
        labels.append(label.item())

    descaler = MinMaxScaler()
    descaler.min_, descaler.scale_ = scaler.min_[-1], scaler.scale_[-1]

    predictions_descaled = descale(descaler, predictions)
    labels_descaled = descale(descaler, labels)

    # Graph plotting
    test_data = features_df[train_size:]
    test_sequences_data = test_data.iloc[SEQUENCE_LENGTH:]

    test_dates = matplotlib.dates.date2num(test_sequences_data.index.tolist())
    all_dates = matplotlib.dates.date2num(features_df.index.tolist())
    plt.plot_date(test_dates[DAYS_PREDICTION:], predictions_descaled, '-', label='predicted')
    plt.plot_date(all_dates, features_df.price.tolist(), '-', label='real')
    plt.xticks(rotation=45)
    plt.savefig('results/test.png')


    return scaler


# Watch model looping
datasets_path = "Datasets/Price_chart_Game"
dir_list = os.listdir(datasets_path)
Model_Price_df = pd.DataFrame()
for brand in os.listdir(datasets_path):
    for model in os.listdir(datasets_path+'/'+brand):
        path = datasets_path+'/'+brand+'/'+model+'/price.csv'
        Train_eval_model(path, model)
        break
    break



