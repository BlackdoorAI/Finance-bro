import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

def create_tensor_dataset(mode:str, lookbehind:int, limit = None, categories = 0):
    """
    Takes in the mode: ["static", "dynamic"]
    Returns a tensor dataset from all the merged frames
    If no limit is entered you get everything
    """
    data_directory = f"..\\ready_data\{mode}"
    iterator = list(os.listdir(data_directory))
    if limit == None:
        limit = len(iterator)
    atemporal_columns = ["dividend", "splits", "category", "size"]
    label_column = f"{mode}_quantile_shifted"
    feature_tensor = None
    total_label_tensor = None
    total_atemporal_tensor = None
    for filename in iterator[:limit]: #Iterates through the frames in the directory
        print(f"Catting {filename}")
        file_path = os.path.join(data_directory, filename) #Get the path to the frame
        frame = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if frame.empty:
            continue
        if categories > 0:
            label_tensor = torch.tensor(frame[[f"range{i}" for i in range(0,categories)]].values)
        else:
            label_tensor = torch.tensor(frame[label_column].values)
        atemporal_tensor = torch.tensor(frame[atemporal_columns].values)
        index = frame.index
        if categories > 0:
            frame.drop(columns=atemporal_columns+[f"range{i}" for i in range(0,categories)], inplace=True)
        else:
            frame.drop(columns=atemporal_columns+[label_column], inplace=True)
        sorted_columns = sorted(frame.columns, key=lambda x: int(x.split('-')[-1])) #Order the columns 
        sorted_frame = frame[sorted_columns]
        frame_tensor = torch.tensor(sorted_frame.values)
        frame_tensor = frame_tensor.view(-1,lookbehind, len(sorted_frame.columns)//lookbehind) #shape before = (rows, columns) | shape after = (rows, columns/lookbehind,lookbehind)
        if feature_tensor != None:
            feature_tensor = torch.cat([feature_tensor, frame_tensor], axis=0)
        else:
            feature_tensor = frame_tensor
        if total_label_tensor != None:
            total_label_tensor = torch.cat([total_label_tensor, label_tensor], axis=0)
        else:
            total_label_tensor = label_tensor
        if total_atemporal_tensor != None:
            total_atemporal_tensor = torch.cat([total_atemporal_tensor, atemporal_tensor], axis=0)
        else:
            total_atemporal_tensor = atemporal_tensor

    feature_tensor = feature_tensor.double()
    total_label_tensor = total_label_tensor.double()
    total_atemporal_tensor = total_atemporal_tensor.double()

    class LSTMDataset(Dataset):
        def __init__(self, features, atemporals, labels):
            # self.features = F.normalize(features, p=1, dim=0)
            self.atemporals = atemporals
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return (self.features[idx], self.atemporals[idx]), self.labels[idx]

    dataset = LSTMDataset(feature_tensor, total_atemporal_tensor, total_label_tensor)
    return dataset


    
