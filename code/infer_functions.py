import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import pandas as pd
from skorch import NeuralNetClassifier, NeuralNetRegressor
from sklearn.model_selection import GridSearchCV


def profile(values_tensor):
    """
    Returns the tensor with the profiles computed and concatted to it 
    Also normalizes along the lookbehind axis
    """
    profile_tensor = values_tensor/torch.max(values_tensor,dim=2)[0].unsqueeze(2)
    normalized_tensor = F.normalize(values_tensor, p=1, dim=1)
    cat_tensor = torch.cat((normalized_tensor, profile_tensor), dim=2)
    return cat_tensor

def multiples(values_tensor):
    """
    Refactors the data so that its in terms of the multiple current/prev
    """
    cats = [torch.ones(values_tensor.shape[0], 1, values_tensor.shape[2])]
    for i in range(1,values_tensor.shape[1]): #lookbehind
        cats.append((values_tensor[:,i,:]/values_tensor[:,i-1,:]).unsqueeze(1))
    multiples_tensor = torch.cat(tuple(cats), dim=1)
    return multiples_tensor

def create_tensor_dataset(mode:str, lookbehind:int, limit = None, categories = 0, verbose=False, averages = False, use_profile=False, use_multiples=False):
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
    if averages:
        label_column = f"{mode}_average"
    else:
        label_column = f"{mode}_quantile"
    feature_tensor = None
    total_label_tensor = None
    total_atemporal_tensor = None
    for filename in iterator[:limit]: #Iterates through the frames in the directory
        if verbose:
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
        sorted_columns = sorted(frame.columns, key=lambda x: int(x.split('-')[-1]), reverse=True) #Order the columns, we start with the last ones because LSTM
        sorted_frame = frame[sorted_columns]
        frame_tensor = torch.tensor(sorted_frame.values) #The current feature tensor 
        frame_tensor = frame_tensor.view(-1,lookbehind, len(sorted_frame.columns)//lookbehind) #shape before = (rows, columns) | shape after = (rows, lookbehind, columns/lookbehind)
        if use_profile: #Get a stat profile for each timestamp and concat it 
            frame_tensor = profile(frame_tensor)
        if use_multiples:
            frame_tensor = multiples(frame_tensor)
        else:
            frame_tensor = F.normalize(frame_tensor, p=1, dim=1)
        if feature_tensor != None: #Feauture tensor is the total tensor 
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
            self.features = features
            self.atemporals = atemporals
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return (self.features[idx], self.atemporals[idx]), self.labels[idx]

    dataset = LSTMDataset(feature_tensor, total_atemporal_tensor, total_label_tensor)
    return dataset

def split_tensor_dataset(Dataset):
    """
    Splits the dataset into the x_train and y_train needed for scikit learn.
    """
    x = Dataset.features
    y = Dataset.labels
    return x, y

# def grid_search(Model, parameters, x_train, y_train):
#     net = NeuralNetClassifier(Model)
#     gs = GridSearchCV(net, parameters, refit=False, cv=3, scoring='accuracy')
#     gs.fit(x_train, y_train)  # Assuming X_train and y_train are your data
#     return gs, gs.best_params_

def objective(trial, Model_Class, x_train, y_train, categories):
    # Define the parameters to search
    model_params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 100, 300),
        'layers': trial.suggest_int('layers', 2, 8),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
        "input" : x_train.shape[2],
        'categories': categories  # Just to pass it through
    }

    train_params = {
        'lr': trial.suggest_float('lr', 0.01, 0.1),
        
        'epochs': trial.suggest_int('epochs', 1, 3),
    }
    # Initialize the model with suggested parameters
    Model = Model_Class(**model_params)
    Model = Model.double()
    optimizer = optim.Adam(Model.parameters(), lr=train_params["lr"])
    if categories > 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    
    # Prepare the dataset and dataloader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=model_params['batch_size'], shuffle=True)
    
    # Train the model
    for epoch in range(train_params["epochs"]):
        Model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = Model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate the accuracy
    accuracy = evaluate_loss(Model, x_train, y_train, criterion)
    return accuracy

def grid_search(Model, x_train, y_train, n_trials = 20, categories=0):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, Model, x_train, y_train, categories), n_trials=n_trials)  # Adjust the number of trials as needed
    best_params = study.best_params
    return best_params

def evaluate_loss(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        loss = criterion(outputs.squeeze(), y)
    return loss.item()

    
