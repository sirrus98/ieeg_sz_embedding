#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import matplotlib.pyplot as plt
# import pytorch_lightning.callbacks
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'widget')

# In[2]:
print('=========Loading imports=========')
import warnings
import numpy as np
from os.path import join as ospj
import argparse

# third party imports
import torch

import lightning as L

# local imports
from model_config import MODEL_CONFIG
# from model_v4 import VITEEG
from model_v5 import MultivarWav2Vec2
from model_wrapper import ModelV3WrapperWithQueue
from dataset import SzDatasetRegsTenS
from torch.utils.data import DataLoader
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
# warnings
warnings.filterwarnings("ignore")

from config import CONFIG

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
parser.add_argument("--logs_dir", type=str, default="logs")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--subsample", type=float, default=1)
parser.add_argument("--shuffle", type=bool, default=False)

args = parser.parse_args()

# %%
# SET ARGS
# ckpt_dir_path = ospj(CONFIG.ckpt_dir, "checkpoints_10s")
# log_name = "log_10s"
# random_seed = 42

ckpt_dir_path = ospj(CONFIG.ckpt_dir, args.ckpt_dir)
log_name = args.logs_dir
random_seed = args.seed
subsample = args.subsample
shuffle = args.shuffle

# In[3]:
torch.manual_seed(random_seed)
np.random.seed(random_seed)
determine_generator = torch.Generator()
determine_generator.manual_seed(random_seed)
torch.set_float32_matmul_precision('medium')

if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    # True ensures the algorithm selected by CUFA is deterministic
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    # False ensures CUDA select the same algorithm each time the application is run
    torch.backends.cudnn.benchmark = False

# In[4]:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('=========Loading Data=========')
print(f"Device: {device}")
# %%
dataset = SzDatasetRegsTenS(transform=False, subsample=subsample, shuffle=shuffle)
dataloader = DataLoader(dataset, batch_size=MODEL_CONFIG.batch_size, shuffle=False, num_workers=4, persistent_workers=True)

model_nn = MultivarWav2Vec2(MODEL_CONFIG, subsample=subsample)
model = ModelV3WrapperWithQueue(model_nn)

# In[5]:
checkpoint_callback = pl_callbacks.ModelCheckpoint(monitor='train_loss',
                                                   filename='model_epoch-{epoch:02d}-{train_loss:.5f}',
                                                   save_top_k=-1,
                                                   every_n_epochs=20,
                                                #    enable_version_counter=True,
                                                   dirpath=ckpt_dir_path)

early_stopping = pl_callbacks.EarlyStopping(monitor="train_loss",
                                            mode="min",
                                            patience=15)
csv_logger = pl_loggers.CSVLogger(CONFIG.logs_dir,
                                  name=log_name)

trainer = L.Trainer(log_every_n_steps=5,
                    logger=csv_logger,
                    max_epochs=500,
                    # callbacks=[checkpoint_callback,early_stopping],
                    callbacks=[checkpoint_callback],
                    accelerator='gpu',
                    devices=1,)
trainer.fit(model=model, train_dataloaders=dataloader)
