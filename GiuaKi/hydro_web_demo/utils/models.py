
import joblib
import torch
import torch.nn as nn
from xgboost import XGBRegressor

class CNNLSTM(nn.Module):
    def __init__(self, seq_feat_dim=6, conv_filters=16, lstm_hidden=64, dense_hidden=32, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(seq_feat_dim, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_hidden, batch_first=True)
        self.norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, dense_hidden), nn.ReLU(), nn.Linear(dense_hidden, 1))

    def forward(self, x_seq):
        x = x_seq.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.norm(h)
        h = self.dropout(h)
        return self.head(h).squeeze(1)

def load_xgb_model(path_json: str):
    model = XGBRegressor()
    model.load_model(path_json)
    return model

def load_dtree_pipeline(path_pkl: str):
    return joblib.load(path_pkl)

def load_cnn_lstm(path_pth: str, device: str = "cpu"):
    model = CNNLSTM().to(device)
    state = torch.load(path_pth, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
