import os.path as p
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from definitions import *
from model_helper_functions import *
from dataset_helper_functions import *
from BiLSTM import BiLSTM
from DebatesDataset import DebatesDataset
from optuna.trial import TrialState


data, features = {}, {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data():
    dev_path = p.join(PROC_DATA_DIR_PATH, 'dev')
    features_path = p.join(PROC_DATA_DIR_PATH, 'features')

    data_paths = {
        'dev': [
            p.join(dev_path, 'dev.tsv'),
            p.join(dev_path, 'dev_spacy.pkl'),
            p.join(features_path, 'dev_stylometric_features.pkl'),
        ],
        'test': [
            p.join(POLIT_DATA_DIR_PATH, 'test', 'test_combined.tsv'),
            p.join(PROC_DATA_DIR_PATH, 'test', 'test_spacy.pkl'),
            p.join(features_path, 'test_stylometric_features.pkl'),

        ],
        'train': [
            p.join(POLIT_DATA_DIR_PATH, 'train', 'train_combined.tsv'),
            p.join(PROC_DATA_DIR_PATH, 'train', 'train_spacy.pkl'),
            p.join(features_path, 'train_stylometric_features.pkl')
        ],
        'val': [
            p.join(POLIT_DATA_DIR_PATH, 'val', 'val_combined.tsv'),
            p.join(PROC_DATA_DIR_PATH, 'val', 'val_spacy.pkl'),
            p.join(features_path, 'val_stylometric_features.pkl'),
        ],
    }

    for dtype, dpaths in data_paths.items():
        try:
            data[dtype] = pd.read_pickle(dpaths[1])
            features[dtype] = pd.read_pickle(dpaths[2])
        except Exception as e:
            print(e.args)
            exit()


def get_loaders():
    dev_df, test_df, train_df, val_df = data.values()

    dev_ids = dev_df['id'].values
    val_df = val_df.loc[~val_df['id'].isin(dev_ids)]

    val_worthy = val_df.loc[val_df['label'] == 1]
    val_unworthy = val_df.loc[val_df['label'] == 0].iloc[:len(val_worthy), :]

    balanced_val = val_worthy.append(val_unworthy).reset_index(drop=True)

    # TODO: probably not needed for optimization
    test_worthy = test_df.loc[test_df['label'] == 1]
    test_unworthy = test_df.loc[test_df['label'] == 0].iloc[:len(test_worthy), :]

    balanced_test = test_worthy.append(test_unworthy).reset_index(drop=True)

    train_dd = DebatesDataset(data=dev_df)
    val_dd = DebatesDataset(data=balanced_val)
    # test_dd = DebatesDataset(data=balanced_test)

    batch_size = 16
    train_loader = DataLoader(train_dd, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dd, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dd, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def objective(trial):
    train_loader, val_loader = get_loaders()

    pooling_strategy = trial.suggest_categorical('pooling_strategy', ['last_four', 'last_four_sum', 'second_last'])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam", "RMSprop", "SGD"])

    model = BiLSTM(device=device, emb_pooling_strat=pooling_strategy, dropout=dropout, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    n_epochs = 10
    # exp_path = p.join(EXP_DIR_PATH, 'bi-lstm', 'optimization')

    threshold = 0.5
    # training

    for epoch in range(n_epochs):
        model.train()
        for ids, sentences, labels in train_loader:
            labels = labels.float().to(device)

            optimizer.zero_grad()

            output = model(sentences)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()

        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for val_ids, val_sentences, val_labels in val_loader:
                val_labels = val_labels.float().to(device)
                output = model(val_sentences)

                # temp.extend(output.tolist())
                output = (output > threshold).int()
                y_pred.extend(output.tolist())
                y_true.extend(val_labels.tolist())

        accuracy = accuracy_score(y_true, y_pred)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    "Done."
    return accuracy


load_data()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2, timeout=600)


pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
