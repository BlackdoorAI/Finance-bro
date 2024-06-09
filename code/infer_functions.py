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
    Returns the tensor with the profiles 
    """
    profile_tensor = values_tensor/torch.max(values_tensor,dim=2)[0].unsqueeze(2)
    return profile_tensor

def multiples(values_tensor):
    # Initialize the list with all ones tensor in the first column
    cats = [torch.ones(values_tensor.shape[0], 1, values_tensor.shape[2])]

    # Compute the ratio for each consecutive pair of slices along dimension 1
    for i in range(1, values_tensor.shape[1]):
        # Calculate the ratio and avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        ratio = (values_tensor[:, i, :] + epsilon) / (values_tensor[:, i-1, :] + epsilon)
        
        # Clamp the values to prevent extremely large or small values affecting gradients
        clamped_ratio = torch.clamp(ratio, min=-10, max=10).unsqueeze(1)

        # Append the processed tensor to the list
        cats.append(clamped_ratio)

    # Concatenate all tensors in the list along dimension 1
    multiples_tensor = torch.cat(tuple(cats), dim=1)

    # Replace NaNs and Infs with a constant value (10.0) where needed
    # This step is a fallback, but typically the clamping should handle most issues
    multiples_tensor = torch.where(torch.isnan(multiples_tensor) | torch.isinf(multiples_tensor), 
                                   torch.tensor(10.0, dtype=torch.float64), 
                                   multiples_tensor)

    return multiples_tensor

def create_tensor_dataset(mode:str, lookbehind:int, measures:dict, limit = None, categories = 0, verbose=False, averages = False, use_profile=False, use_multiples=False):
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

        if mode == "together":
            dynamic_columns = frame.columns[:len(measures["dynamic"])] #We assume that the dynamic data is first
            static_columns = frame.columns[len(measures["dynamic"]):]                       
        sorted_columns = sorted(frame.columns, key=lambda x: int(x.split('-')[-1]), reverse=True) #Order the columns, we start with the last ones because LSTM
        sorted_frame = frame[sorted_columns]
        frame_tensor = torch.tensor(sorted_frame.values) #The current feature tensor 
        frame_tensor = frame_tensor.view(-1,lookbehind, len(sorted_frame.columns)//lookbehind) #shape before = (rows, columns) | shape after = (rows, lookbehind, columns/lookbehind)
        if use_profile: #Get a stat profile for each timestamp 
            profile_tensor = profile(frame_tensor)
        #Two ways to kind of normalize the data
        if use_multiples:
            frame_tensor = multiples(frame_tensor)
        else:
            frame_tensor = F.normalize(frame_tensor, p=1, dim=1)
        if use_profile: #If the profile was used just a little bit more data
            frame_tensor = torch.cat((frame_tensor, profile_tensor), dim=2)
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
    print(torch.isnan(feature_tensor).any(), torch.isnan(total_atemporal_tensor).any(), torch.isnan(total_label_tensor).any())
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

class Conditional_Hook:
    def __init__(self, module, counts):
        self.module = module
        self.counts = counts
        self.counter = 0
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # self.counter += 1
        # if self.counter % self.counts == 0:
        #     print(output)
        if input[0].isnan().any():
            print("Found nans")
            print(input)

    def remove(self):
        self.hook.remove()

def objective(trial, Model_Class, x_train, y_train, categories):
    # Define the parameters to search
    model_params = {
        'hidden_dim': trial.suggest_int('hidden_dim', 100, 300),
        'layers': trial.suggest_int('layers', 2, 3),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
        "input" : x_train.shape[2],
        'categories': categories  # Just to pass it through
    }

    train_params = {
        'lr': trial.suggest_float('lr', 0.001, 0.01),
        
        'epochs': trial.suggest_int('epochs', 1, 3),
    }
    # Initialize the model with suggested parameters
    Model = Model_Class(**model_params)
    Model = Model.double()
    hooker = Conditional_Hook(Model,100) #To debug
    optimizer = optim.Adam(Model.parameters(), lr=train_params["lr"])
    if categories > 0:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    
    # Prepare the dataset and dataloader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=model_params['batch_size'], shuffle=True)
    
    if any(torch.isnan(param).any() for param in Model.parameters()):
                print("Model initialized with Nans")

    # Train the model
    for epoch in range(train_params["epochs"]):
        Model.train()
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = Model(inputs)
            outputs = torch.clamp(outputs, min=-1e10, max=1e10)  # Avoiding extreme values that can lead to NaNs
            # Adding a small epsilon to avoid log(0) scenario in loss calculations if applicable
            loss = criterion(outputs.squeeze(), labels + 1e-5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), max_norm=1.0)
            optimizer.step()
            if torch.isnan(loss).any() or any(torch.isnan(param).any() for param in Model.parameters()):
                print("NaN detected in loss or parameters!")
                print(i)
                break  # Break if NaN is detected to avoid further corruption

    hooker.remove()
    
    
    # Evaluate the accuracy
    accuracy = evaluate_loss(Model, x_train, y_train, criterion, batch_size=model_params['batch_size'])
    return accuracy

def grid_search(Model, x_train, y_train, n_trials = 20, categories=0):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, Model, x_train, y_train, categories), n_trials=n_trials)  # Adjust the number of trials as needed
    best_params = study.best_params
    return best_params

def evaluate_loss(model, x, y, criterion, batch_size):
    model.eval()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    # with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        total_loss += loss.item() * inputs.size(0)  # Summing up loss weighted by batch size
    return total_loss / len(x)  # Return average loss

class DynamicLSTM(nn.Module):

    def __init__(self, hidden_dim, batch_size, layers, input, categories=0):
        super(DynamicLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.layers_num = layers
        
        #input is all the embedding vectors plus all the other variables
        self.lstm = nn.GRU(input, hidden_dim, num_layers=layers, batch_first=True) 
        self.hidden = (torch.zeros(layers,batch_size,hidden_dim),torch.zeros(layers,batch_size,hidden_dim))
        
        #Squeeeze them into 1 dimension
        if categories > 0:
            self.hidden2label = nn.Linear(hidden_dim, categories)
        else:
            self.hidden2label = nn.Linear(hidden_dim, 1)

    def forward(self, batch_tensor):
        lstm_out, self.hidden = self.lstm(batch_tensor)
        last_timestep_output = lstm_out[:, -1, :]
        sales = self.hidden2label(last_timestep_output)
        return sales
    
    def hidden_reset(self):
        #reset the hidden and cell state after each epoch
        self.hidden = (torch.zeros(self.layers_num,self.batch_size,self.hidden_dim),
                       torch.zeros(self.layers_num,self.batch_size,self.hidden_dim))
    def batch_reset(self,batch_size):
        self.hidden = (torch.zeros(self.layers_num,batch_size,self.hidden_dim),
                       torch.zeros(self.layers_num,batch_size,self.hidden_dim))
    def flatten_parameters(self):
        self.lstm.flatten_parameters()
