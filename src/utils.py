# Defining key functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

# Set random seed
SEED = 42

def train_model(X, y):
    """
    Run random forest classification model on feature subset
    and retrieve cross validated ROC-AUC score
    """
    clf = RandomForestClassifier(random_state=SEED)
    kf = KFold(shuffle=True, n_splits=3, random_state=SEED)
    cv_roc_auc_score = round(cross_val_score(clf, X, y, cv=kf, 
                                             scoring="roc_auc", n_jobs=-1).mean(), 3)

    return cv_roc_auc_score

