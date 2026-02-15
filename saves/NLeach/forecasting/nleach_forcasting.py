#import libraries
import os
import pickle
import random
import time
import re
import itertools
import math
import json
from pathlib import Path
import warnings
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
parent_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(parent_dir))

# globally ignore all warnings
warnings.filterwarnings("ignore")

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,TensorDataset,Subset
import torch.multiprocessing as mp
from torchinfo import summary


from utils import preprocessing as ppsr
from utils.potsimloader import potsimloader as psl
from utils import split
from models import forecast_enc_transformer
from sklearn.preprocessing import MinMaxScaler
from utils import scaler
from training import train
from testing import evaluate
from sklearn.metrics import r2_score
pl.enable_string_cache()

#####################

def set_seed(seed):
    torch.manual_seed(seed)  # Set seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch GPU (if using CUDA)
    np.random.seed(seed)  # Set seed for NumPy
    random.seed(seed)  # Set seed for Python's random module
set_seed(42)

#####################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################

# Data Preparation
# def generate_treatments(n_values):
#     combis = list(itertools.product(n_values, repeat=3))
#     return ["-".join(map(str, t)) for t in combis if sum(t) <= 400]


# def join_potsim_yearly(data_dir, save_dir=Path("data"),save=True):
#     if not save:
#         return
#     data_dir = Path(data_dir).resolve()
#     files = os.listdir(data_dir)
#     pattern = re.compile(r"^potsim_\d{4}\.parquet$")
#     files = sorted([file for file in files if pattern.match(file)])
#     files = [data_dir / file for file in files]
#     df = pl.scan_parquet(files)
#     filepath = save_dir / "potsim.parquet"
#     df.sink_parquet(
#         filepath,
#         statistics=True,
#         compression="zstd",
#         compression_level=1,
#         row_group_size=1_000_000,
#     )


# usecols = ['Year', 'Date', 'Treatment', 'NFirstApp','PlantingDay', 'IrrgDep',
#            'IrrgThresh', 'DayAfterPlant', 'NApp', 'NLeach', 'NPlantUp', 'NTotL1', 
#            'NTotL2', 'Irrg', 'SWatL1', 'SWatL2', 'Rain', 'SolarRad', 'AirTempC']
   
# mask = (
#     ((pl.col("NFirstApp") == "Npl") & (pl.col("DayAfterPlant") >= -1)) |
#     ((pl.col("NFirstApp") == "Npre") & (pl.col("DayAfterPlant") >= -37))
# )

# potsim_yearly_dir = Path("data/potsim_yearly/")
# weather_file = Path("data") / "weather.parquet"
# data_file = Path("data") / "potsim.parquet"
# #Turn the save option to "True" to compile the dataset.
# join_potsim_yearly(potsim_yearly_dir,save=False)


# n_values = [0, 56, 112, 168, 196]
# treatments = generate_treatments(n_values)
# scenario_filter= {
#     "Year": list(range(2014, 2024)),
#     "Treatment": treatments,
#     "PlantingDay": [29, 43],
#     "IrrgDep": [30],
#     "IrrgThresh": [70],
#     "NFirstApp": ["Npl"]
# }
# train_years = list(range(2014, 2018))
# val_years = list(range(2018, 2020))
# test_years = list(range(2020, 2023))

# data = psl.read_data(
#     dataset_path=data_file,
#     weather_path=weather_file,
#     usecols=usecols,
#     lazy=True,
#     as_pandas=False,
# )
# data = psl.apply_filter(data, filters=scenario_filter, lazy=False, as_pandas=False)
# df = data.to_pandas()

#####################

#The above code is bypassed as we have already run it and saved the data in the xls file,shown below. 
#You can uncomment the above lines annd can run it to get the same xls file

######################

df=pd.read_excel("../../../data/forecasting_data_subset.xlsx")


#Feature Processing and data scaling

scaler=MinMaxScaler()
X,y= df[['PlantingDay', 'DayAfterPlant', 'NApp','Rain', 'SolarRad', 'AirTempC']], df['NLeach']
y=np.array(y).reshape(y.shape[0],1)
req_cols=['Year','Treatment','PlantingDay','DayAfterPlant','NApp','Rain','SolarRad','AirTempC','NLeach']
data_cols=df[req_cols]
copy_data=data_cols.copy()
copy_data.iloc[:,3:].fillna(0,inplace=True)
copy_data.iloc[:,3:]=scaler.fit_transform(copy_data.iloc[:,3:])

#######################

def get_ip_op(data_cols):
    #This definition will divide the data such that every example covers crop cycle length of features, the final 
    #result having the shape (#examples,#crop cycle or episode length, #features)
    inp_data=[]
    op_data=[]
    op_data_l=[]
    for yr in data_cols['Year'].unique():
        for tmt in data_cols['Treatment'].unique():
              for days in data_cols['PlantingDay'].unique():
                  base=data_cols[(data_cols['Year']==yr)&(data_cols['Treatment']==tmt)&(data_cols['PlantingDay']==days)]
                  inp_data.append(base.iloc[:,3:-1].to_numpy())
                  op_data.append(base['NLeach'].to_numpy())
                  op_data_l.append(base['NLeach'].iloc[-1])
              
    return inp_data,op_data,op_data_l

########################

def get_scen(data_cols,year,tmt,pl_day):
    #Retreives a particular scenario from given data. Helpful during inference. 
    base=data_cols[(data_cols['Year']==year)&(data_cols['Treatment']==tmt)&(data_cols['PlantingDay']==pl_day)]
    inp=base.iloc[:,3:-1].to_numpy()
    op=base['NLeach'].to_numpy()
    #Converting them to tensors
    inp=torch.tensor(inp).float()
    op=torch.tensor(op).float()

              
    return inp,op

#########################

#Converting to tensors and data types
inp,op,op_l=get_ip_op(copy_data)
op=np.array(op,dtype=np.float32)
op_l=np.array(op_l,dtype=np.float32)
#getting input and output tensors
inp=torch.tensor(inp)
op=torch.tensor(op).unsqueeze(2)
inp=inp.float()
op=op.float()

##########################

#Hyperparametrs setup

window_size = 40
batch_size  = 32
epochs      = 30
lr          = 0.001

###########################

# Dataset setup

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_in = inp.float().to(device)    # [N_eps,155,4]
data_op = op.float().to(device)     # [N_eps,155,1]
N_eps   = data_in.size(0)
H       = data_op.size(1) - window_size  # forecast horizon

#Checking the shapes of input and output
# print(data_in.shape,data_op.shape)

class TuberDataset(Dataset):
    def __init__(self, X, Y, w):
        self.X = X
        self.Y = Y
        self.w = w
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        x = self.X[idx, :self.w, :]     # [w,5]
        y = self.Y[idx, self.w:, :]     # [H,1]
        return x, y

full_ds     = TuberDataset(data_in, data_op, window_size)
idx         = torch.randperm(N_eps)

n_tr        = int(0.7*N_eps)
n_val       = int(0.15*N_eps)
train_idx   = idx[:n_tr]
val_idx     = idx[n_tr:n_tr+n_val]
test_idx    = idx[n_tr+n_val:]


train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=batch_size)
test_loader  = DataLoader(Subset(full_ds, test_idx),  batch_size=1)



#######################

# instantiate the model.
F     = data_in.size(2)
model = forecast_enc_transformer.ForecastTransformerEncOnly(num_features=F,
                                   horizon=H).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr)
crit  = nn.MSELoss()


#######################

#Model Training

# train_losses = []
# val_losses = []


# for epoch in range(1, epochs+1):
#     # Training
#     model.train()
#     tot_loss = 0.0
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         pred = model(x)
#         loss = crit(pred, y)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         tot_loss += loss.item()
#     avg_train = tot_loss / len(train_loader)
#     train_losses.append(avg_train)  # <-- Track training loss

#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for x_val, y_val in val_loader:
#             x_val, y_val = x_val.to(device), y_val.to(device)
#             pred_val = model(x_val)
#             loss_val = crit(pred_val, y_val)
#             val_loss += loss_val.item()
#     avg_val = val_loss / len(val_loader)
#     val_losses.append(avg_val)      # <-- Track validation loss

#     print(f"Epoch {epoch}/{epochs} ▶︎ train loss: {avg_train:.4f} | val loss: {avg_val:.4f}")



#######################

# ########################

# # #Reteiving the saved weights for this model
saved_model_file="../NLeach_forecasting_EncoderOnlyTransformer.pth"

# # #Loading the model with saved weights
model.load_state_dict(torch.load(saved_model_file))


#########################
 # VISUALIZATION ON A SINGLE EPISODE

def plot_episode(model,data,year,tmt,pl,w):

    """
    inp_full: [N_eps,#crop_cycle,4], op_full: [N_eps,#crop_cycle,1]
    """
    inp_full,op_full=get_scen(data,year,tmt,pl)
    model.eval()
    device = next(model.parameters()).device

    # select one example
    x = inp_full[:w, :].unsqueeze(0).to(device)  # (1, w, 5)
    with torch.no_grad():
        pred = model(x)                                 # (1, H, 1)

    # convert to numpy
    pred   = pred.squeeze(0).squeeze(-1).cpu().numpy()    # (H,)
    actual = op_full[w:].cpu().numpy()           # (H,)

    
    # plot
    plt.figure()
    plt.plot(range(w, w+len(pred)), pred,   label='Predicted')
    plt.plot(range(w, w+len(actual)), actual, label='Actual')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.title(f'Forecasting Treatment:',)
    plt.legend()
    plt.show()

    return pred,actual

#########################


year_inp=int(input("Enter Year -one value from 2014 through 2023 |  "))
treat_inp=str(input("Enter Treatment-from these combinations:56,112,168,0,196, eg: 56-0-56 |   "))
pl_inp=int(input("Enter one planting day value out of  29 and  43 |  "))




pred,act=plot_episode(model,copy_data,year_inp,treat_inp,pl=pl_inp, w=window_size)

#########################