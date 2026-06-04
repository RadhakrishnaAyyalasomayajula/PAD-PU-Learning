import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator


def get_logistic_regression(X_train=None, y_train=None):
    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    calibrated_model = CalibratedClassifierCV(base_model, cv=5)
    if X_train is not None and y_train is not None:
        return calibrated_model.fit(X_train, y_train)
    return calibrated_model


def get_svm(X_train=None, y_train=None):
    base_model = SVC(probability=True, class_weight='balanced')
    calibrated_model = CalibratedClassifierCV(base_model, cv=5)
    if X_train is not None and y_train is not None:
        return calibrated_model.fit(X_train, y_train)
    return calibrated_model


def get_random_forest(X_train=None, y_train=None):
    base_model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    calibrated_model = CalibratedClassifierCV(base_model, cv=5)
    if X_train is not None and y_train is not None:
        return calibrated_model.fit(X_train, y_train)
    return calibrated_model


class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@variational_estimator
class BayesianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        x = self.blinear2(x)
        return x


def predict_mc_dropout(model, X_tensor, n_samples=20):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_tensor)
            probs = F.softmax(out, dim=1)
            preds.append(probs.cpu().numpy())
    return np.stack(preds, axis=0)


def predict_bnn(model, X_tensor, n_samples=20):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(X_tensor)
            probs = F.softmax(out, dim=1)
            preds.append(probs.cpu().numpy())
    return np.stack(preds, axis=0)


def get_model(name, input_dim=None, X_train=None, y_train=None, **kwargs):
    name = name.lower()
    if name == "logistic_regression":
        return get_logistic_regression(X_train, y_train)
    elif name == "svm":
        return get_svm(X_train, y_train)
    elif name == "random_forest":
        return get_random_forest(X_train, y_train)
    elif name == "mc_dropout":
        return MCDropoutMLP(input_dim=input_dim, **kwargs)
    elif name == "bnn":
        return BayesianMLP(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model name: {name}")
