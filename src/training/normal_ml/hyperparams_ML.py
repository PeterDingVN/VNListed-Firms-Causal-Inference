from sklearn.metrics import \
    roc_auc_score as roc_auc, precision_score as precision, recall_score as recall, \
    mean_absolute_percentage_error as mape, r2_score as r2, root_mean_squared_error as rmse, balanced_accuracy_score as accuracy, \
    make_scorer
from sklearn.linear_model import LinearRegression as lrg
from sklearn.neighbors import KNeighborsClassifier as knn
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
    'algo__n_estimators': [100, 200, 300],
    'algo__learning_rate': [0.05, 0.03],
    'algo__max_depth' : [8],
    'algo__subsample' : [0.5]
}
lrg_param_reg = {
    'algo__fit_intercept': [True, False]
}

    # Hyper params list
algorithm_reg = {
    'XG_reg' : (xgb_reg(), xgb_param_reg),
    'Linear_reg': (lrg(), lrg_param_reg)
}



# Classification task
    # SCoring criteria
score_class = {
    'roc_auc': make_scorer(roc_auc),
    'precision': make_scorer(precision),
    'recall': make_scorer(recall),
    'accuracy': make_scorer(accuracy)
}

    # Algorithm
xgb_param_class = {
    'algo__n_estimators': [100, 200],
    'algo__learning_rate': [0.05, 0.03],
    'algo__max_depth' : [5, 10],
    'algo__subsample' : [0.5],
    'algo__scale_pos_weight': [0.1, 0.15, 0.2, 0.3, 0.5]
}


knn_param_class = {
    'algo__n_neighbors': [1, 3, 5, 7, 9],
    'algo__weights': ['uniform', 'distance']
}
    # Hyper params list
algorithm_class = {
    'XG_class' : (xgb_class(), xgb_param_class),
    'KNN_class': (knn(), knn_param_class)
}
