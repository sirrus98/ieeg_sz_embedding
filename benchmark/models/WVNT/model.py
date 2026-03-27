# Scientific computing imports
import numpy as np
import scipy as sc
import pandas as pd
from scipy.linalg import hankel
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

# Imports for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from funcs.utils import num_wins, MovingWinClips
import warnings

class DSOSDbase:
    def __init__(self, fs, w_size, w_stride):
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.is_fitted = False
    
    def _fit_scaler(self, x):
        self.scaler = RobustScaler().fit(x)

    def _scaler_transform(self, x):
        col_names = x.columns
        return pd.DataFrame(self.scaler.transform(x),columns=col_names)
    
    def get_win_times(self, n_samples):
        win_len = int(self.w_size * self.fs)
        step = int(self.w_stride * self.fs)
        n_windows = (n_samples - win_len) // step + 1
        return np.arange(n_windows) * self.w_stride
    
    def get_win_times(self, n_samples):
        time_arr = np.arange(n_samples) / self.fs
        n_windows = (n_samples - int(self.w_size * self.fs)) // int(self.w_stride * self.fs) + 1
        return time_arr[:n_windows * int(self.w_stride * self.fs):int(self.w_stride * self.fs)]
    
    def get_onset_and_spread(self,sz_prob,threshold=None,
                             ret_smooth_mat = False,
                             filter_w = 5, # seconds
                             rwin_size = 5, # seconds
                             rwin_req = 4 # seconds
                             ): 
        if threshold is None:
            threshold = self.threshold

        sz_clf = (sz_prob>threshold).reset_index(drop=True)
        filter_w_idx = np.floor((filter_w - self.w_size)/self.w_stride).astype(int) + 1
        sz_clf = pd.DataFrame(sc.ndimage.median_filter(sz_clf,size=filter_w_idx,mode='nearest',axes=0,origin=0),columns=sz_prob.columns)
        seized_idxs = np.any(sz_clf,axis=0)
        rwin_size_idx = np.floor((rwin_size - self.w_size)/self.w_stride).astype(int) + 1
        rwin_req_idx = np.floor((rwin_req - self.w_size)/self.w_stride).astype(int) + 1
        sz_spread_idxs_all = sz_clf.rolling(window=rwin_size_idx,center=False).apply(lambda x: (x == 1).sum()>rwin_req_idx).dropna().reset_index(drop=True)
        sz_spread_idxs = sz_spread_idxs_all.loc[:,seized_idxs]
        extended_seized_idxs = np.any(sz_spread_idxs,axis=0)
        first_sz_idxs = sz_spread_idxs.loc[:,extended_seized_idxs].idxmax(axis=0)
        
        if sum(extended_seized_idxs) > 0:
            # Get indices into the sz_prob matrix and times since start of matrix that the seizure started
            sz_idxs_arr = np.array(first_sz_idxs)
            sz_order = np.argsort(first_sz_idxs)
            sz_idxs_arr = first_sz_idxs.iloc[sz_order].to_numpy()
            sz_ch_arr = first_sz_idxs.index[sz_order].to_numpy()
            # sz_times_arr = self.get_win_times(len(sz_clf))[sz_idxs_arr]
            # sz_times_arr -= np.min(sz_times_arr)
            # sz_ch_arr = np.array([s.split("-")[0] for s in sz_ch_arr]).flatten()
        else:
            sz_ch_arr = []
            sz_idxs_arr = np.array([])
        sz_idxs_df = pd.DataFrame(sz_idxs_arr.reshape(1,-1),columns=sz_ch_arr)
        if ret_smooth_mat:
            missing_rows = rwin_size_idx-1
            last_valid_row = sz_spread_idxs_all.iloc[-1]  # Last row of the smoothed matrix
            padding = pd.DataFrame([last_valid_row] * missing_rows, columns=sz_spread_idxs_all.columns)

            # Append the propagated values to restore alignment
            sz_spread_idxs_all_padded = pd.concat([sz_spread_idxs_all, padding], ignore_index=True)
            return sz_idxs_df,sz_spread_idxs_all_padded
        else:
            return sz_idxs_df

    def fit(self,X):
        print("Must define a fit function")
        return None

    def forward(self,X):
        print("Must define a forward function")
        assert self.is_fitted, "Must fit model before running inference"
        return None
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def __str__(self):
        print('Base')

class LiNDD(DSOSDbase):
    def __init__(self, fs = 128, w_size=1, w_stride = 0.5):
        super().__init__(fs=fs,w_size=w_size,w_stride=w_stride)
        self.model = LinearRegression(fit_intercept=False)
    
    def fit(self, X):
        self._fit_scaler(X)
        nX = self._scaler_transform(X)
        self.model.fit(nX.iloc[:-1,:],nX.iloc[1:,:])

    def forward(self, X):
        ch_names = X.columns
        nX = self._scaler_transform(X)
        y = self.model.predict(nX.iloc[:-1,:])
        se = pd.DataFrame((nX.iloc[1:,:]-y)**2,columns=ch_names)
        mse = se.rolling(int(self.w_size*self.fs),min_periods=int(self.w_size*self.fs),center=False).mean()
        mse_wins = mse.iloc[::int(self.w_stride*self.fs)].reset_index(drop=True)
        mse_wins = mse_wins[~mse_wins.isna().any(axis=1)]
        # smooth_mse_wins = pd.DataFrame(sc.ndimage.uniform_filter1d(mse_wins,20,axis=0,mode='constant'),columns=ch_names)
        return mse_wins

class NDDmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NDDmodel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1,:])
        return out
    
    def __str__(self):
         return "NDD" 

class NDDIso(nn.Module):
    def __init__(self, num_channels, hidden_size):
        super(NDDIso, self).__init__()
        self.num_channels = num_channels
        self.lstms = nn.ModuleList([nn.LSTM(1, hidden_size, batch_first=True) for _ in range(num_channels)])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_channels)])

    def forward(self, x):
        outputs = []
        for i in range(self.num_channels):
            out, _ = self.lstms[i](x[:, :, i].unsqueeze(-1))  # LSTM input shape: (batch_size, seq_len, 1)
            out = self.fcs[i](out[:, -1, :])  # FC input shape: (batch_size, hidden_size)
            outputs.append(out.unsqueeze(1))  # Add channel dimension back

        # Concatenate outputs along channel dimension
        output = torch.cat(outputs, dim=1).squeeze()  # shape: (batch_size, num_channels, 1)
        return output
    
    def __str__(self):
         return "NDDIso"

class WVNT(nn.Module):
    def __init__(self, mdl, device = None, use_cuda = False):
        super().__init__() 
        self.is_fitted = False
        self.use_cuda = use_cuda
        if device is not None:
            self.device = device
        else:
            if self.use_cuda and not torch.cuda.is_available():
                warnings.warn("CUDA is not available, using CPU instead.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.mdl = mdl
        self.mdl.to(self.device)
        self.mdl.eval()
    
    def __str__(self) -> str:
        return "WVNT"

    def forward(self, x):
        n_ch = x.shape[1]
        # x should be samples x channels
        x = torch.tensor(x.T).float().unsqueeze(-1).to(self.device)
        with torch.no_grad():
            y = self.mdl.predict_proba_temp(x).cpu().numpy()[:,1]
        return y.reshape(1,n_ch)
    
    def __call__(self, *args):
        return self.forward(*args)

class NDD(DSOSDbase):
    def __init__(self, hidden_size = 10, fs = 128,
                  train_win = 12, pred_win = 1,
                  w_size = 1, w_stride = 0.5,
                  num_epochs = 10, batch_size = 'full',
                  lr = 0.01,
                  model_class = NDDmodel,
                  use_cuda = False):
        super().__init__(fs=fs,w_size=w_size,w_stride=w_stride)
        self.hidden_size = hidden_size
        self.train_win = train_win
        self.pred_win = pred_win
        self.w_size = w_size
        self.w_stride = w_stride
        self.fs = fs
        self.use_cuda = use_cuda
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_class = model_class
        self.is_fitted = False

        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available, using CPU instead.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    
    def _prepare_segment(self, data, ret_time=False):
        data_ch = data.columns.to_list()
        data_np = data.to_numpy()

        j = int(self.w_size*self.fs-(self.train_win+self.pred_win)+1)

        nwins = num_wins(data_np.shape[0],self.fs,self.w_size,self.w_stride)
        data_mat = torch.zeros((nwins,j,(self.train_win+self.pred_win),data_np.shape[1]))
        for k in range(len(data_ch)): # Iterating through channels
            samples = MovingWinClips(data_np[:,k],self.fs,self.w_size,self.w_stride)
            for i in range(samples.shape[0]):
                clip = samples[i,:]
                mat = torch.tensor(hankel(clip[:j],clip[-(self.train_win+self.pred_win):]))
                data_mat[i,:,:,k] = mat
        time_mat = MovingWinClips(np.arange(len(data))/self.fs,self.fs,self.w_size,self.w_stride)
        win_times = time_mat[:,0]
        data_flat = data_mat.reshape((-1,self.train_win + self.pred_win,len(data_ch)))
        input_data = data_flat[:,:-1,:].float()
        target_data = data_flat[:,-1,:].float()

        if ret_time:
            return input_data, target_data, win_times
        else:
            return input_data, target_data
    
    def _train_model(self,dataloader,criterion,optimizer):
        # Training loop
        tbar = tqdm(range(self.num_epochs),leave=False)
        for e in tbar:
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                del inputs, targets, outputs
            if e % 10 == 9:
                tbar.set_description(f"{loss.item():.4f}")
                del loss

    def _repair_data(self,outputs,X):
        nwins = num_wins(X.shape[0],self.fs,self.w_size,self.w_size)
        nchannels = X.shape[1]
        repaired = outputs.reshape((nwins,self.w_size*self.fs-(self.train_win + self.pred_win)+1,nchannels))
        return repaired

    def fit(self, X):
        input_size = X.shape[1]
        # Initialize the model
        self.model = self.model_class(input_size, self.hidden_size)
        self.model = self.model.to(self.device)
        # Scale the training data
        self._fit_scaler(X)
        X_z = self._scaler_transform(X)

        # Prepare input and target data for the LSTM
        input_data,target_data = self._prepare_segment(X_z)

        dataset = TensorDataset(input_data, target_data)
        if self.batch_size == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Train the model, this will just modify the model object, no returns
        self._train_model(dataloader,criterion,optimizer)
        self.is_fitted = True

    def forward(self, X):
        assert self.is_fitted, "Must fit model before running inference"
        X_z = self._scaler_transform(X)
        input_data,target_data, time_wins = self._prepare_segment(X_z,ret_time=True)
        self.time_wins = time_wins
        dataset = TensorDataset(input_data,target_data)
        if self.batch_size == 'full':
            batch_size = len(dataset)
        else:
            batch_size = self.batch_size
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        with torch.no_grad():
            self.model.eval()
            mse_distribution = []
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                mse = (outputs-targets)**2
                mse_distribution.append(mse)
                del inputs, targets, outputs, mse
        raw_mdl_outputs = torch.cat(mse_distribution).cpu().numpy()
        mdl_outs = raw_mdl_outputs.reshape((len(time_wins),-1,raw_mdl_outputs.shape[1]))
        raw_loss_mat = np.sqrt(np.mean(mdl_outs,axis=1)).T
        # loss_mat = sc.ndimage.uniform_filter1d(raw_loss_mat,20,origin=0,axis=1,mode='constant')
        # loss_mat = sc.ndimage.median_filter(raw_loss_mat,size=10,mode='constant',axes=1)
        self.feature_df = pd.DataFrame(raw_loss_mat.T,columns = X.columns)
        return self.feature_df