from sklearn.metrics import \
    roc_auc_score as roc_auc, precision_score as precision, recall_score as recall, \
    mean_absolute_percentage_error as mape, r2_score as r2, root_mean_squared_error as rmse, balanced_accuracy_score, \
    make_scorer
from sklearn.ensemble import RandomForestRegressor as rf
from xgboost import XGBRegressor as xgb_reg, XGBClassifier as xgb_class


'''
    This file is reserved only for Machine Learning algorithm
    Deep Learning params will be in another file
'''

# Regression task

    # Scoring criteria
score_reg = {
    'r2': make_scorer(r2),
    'rmse': make_scorer(rmse),
    'mape': make_scorer(mape),
}

    # Algorithm
xgb_param_reg = {
    'n_estimators': [100],
    'learning_rate': [0.05],
    'max_depth' : [8, 10],
    'subsample' : [0.5]
}

    # Hyper params list
    # XGBoost Reg
algorithm_reg = {
    'XG_reg' : (xgb_reg(), xgb_param_reg)
    # 'RF' : (rf(), rf_param)
}

# Classification task

    # SCoring criteria
score_class = {
    'roc_auc': make_scorer(roc_auc),
    'precision': make_scorer(precision),
    'recall': make_scorer(recall),
    'accuracy': make_scorer(balanced_accuracy_score)
}

    # Algorithm

xgb_param_class = {
    'n_estimators': [100],
    'learning_rate': [0.05],
    'max_depth' : [8, 10],
    'subsample' : [0.5]
}

    # Hyper params list
    # XGBoost Reg
algorithm_class = {
    'XG_class' : (xgb_class(), xgb_param_class)
    # 'RF' : (rf(), rf_param)
}
