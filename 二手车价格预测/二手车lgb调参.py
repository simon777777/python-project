import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks import LearningRateScheduler
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model

tree_train = pd.read_csv('C:\\Users\\m1533\\Desktop\\研一下\\python\\项目\\结果\\tree_train.csv', sep=' ')
tree_test = pd.read_csv('C:\\Users\\m1533\\Desktop\\研一下\\python\\项目\\结果\\tree_test.csv', sep=' ')

X_data=tree_train.drop(['price', 'SaleID'], axis=1)
X_test=tree_test.drop(['price', ' SaleID'], axis=1)
Y_data=tree_train['price']
num5=[X_data.columns.get_loc(c) for c in Ca_feature]

param = {'boosting_type': 'gbdt',
         'objective': 'regression',

         'num_leaves': 31,
         'max_depth': -1,
         'learning_rate': 0.01,
         "lambda_l2": 0.5,
         'min_data_in_leaf': 20,
         "min_data_in_leaf": 20,
         "feature_fraction": 0.8,
         'subsample': 0.8,
         "bagging_freq": 1,
         "bagging_seed": 11,
         "metric": 'mae',
         }

data_train = lgb.Dataset(X_data, Y_data, silent=True)
cv_results = lgb.cv(
    param, data_train, num_boost_round=1000000, nfold=5, stratified=False, shuffle=True, metrics='mae',
    early_stopping_rounds=50, verbose_eval=1000, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['l1-mean']))
print('best cv score:', cv_results['l1-mean'][-1])

# 提高拟合程度调整max_depth num_leaves

from sklearn.model_selection import GridSearchCV
### 我们可以创建lgb的sklearn模型，使用上面选择的(学习率，评估器数目)
model_lgb = lgb.LGBMRegressor(boosting_type='gbdt', objective='mae',num_leaves=50,
                              learning_rate=0.1, n_estimators=111888, max_depth=6,
                              bagging_fraction = 0.8,feature_fraction = 0.8)

params_test1={
    'max_depth': range(3,30,5),
    'num_leaves':range(40, 400, 20)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_absolute_error', cv=5, verbose=1, n_jobs=4)

gsearch1.fit(X_data,Y_data)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

params_test2={
    'max_depth': [6,7,8],
    'num_leaves':[68,74,80,86,92]
}

gsearch2 = GridSearchCV(estimator=model_lgb, param_grid=params_test2, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch2.fit(df_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

## 降低过拟合
# 调整min_data_in_leaf和min_child_weight
# min_child_weight（最小海森值之和，采用rmse损失就是最小样本个数）=min_sum_hessian_in_leaf

params_test3={
    'min_child_samples': [18, 19, 20, 21, 22],
    'min_child_weight':[0.001, 0.002]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7,
                              metric='rmse', bagging_fraction = 0.8, feature_fraction = 0.8)
gsearch3 = GridSearchCV(estimator=model_lgb, param_grid=params_test3, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch3.fit(df_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

## 采样降低过拟合
# feature_fraction参数来进行特征的子抽样。这个参数可以用来防止过拟合及提高训练速度。
# bagging_fraction+bagging_freq参数必须同时设置，bagging_fraction相当于subsample样本采样，可以使bagging更快的运行，同时也可以降拟合。bagging_freq默认0，表示bagging的频率，0意味着没有使用bagging，k意味着每k轮迭代进行一次bagging。

params_test4={
    'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.1, n_estimators=43, max_depth=7,
                              metric='rmse', bagging_freq = 5,  min_child_samples=20)
gsearch4 = GridSearchCV(estimator=model_lgb, param_grid=params_test4, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch4.fit(df_train, y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

## 调整L1&L2

params_test6={
    'reg_alpha': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5],
    'reg_lambda': [0, 0.001, 0.01, 0.03, 0.08, 0.3, 0.5]
}
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=80,
                              learning_rate=0.b1, n_estimators=43, max_depth=7,
                              metric='rmse',  min_child_samples=20, feature_fraction=0.7)
gsearch6 = GridSearchCV(estimator=model_lgb, param_grid=params_test6, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
gsearch6.fit(df_train, y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

#贝叶斯全局优化

from sklearn.model_selection import cross_val_score
import lightgbm as lgb


def rf_cv_lgb(num_leaves, max_depth):
    # 建立模型
    model_lgb = lgb.LGBMRegressor(boosting_type='gbdt', objective='mae',
                                  n_estimators=120000,
                                  num_leaves=int(num_leaves), max_depth=int(max_depth),
                                  bagging_fraction=0.8, feature_fraction=0.8,
                                  bagging_freq=1, min_data_in_leaf=50,
                                  learning_rate=0.5,
                                  n_jobs=8, categorical_feature=num5
                                  )

    val = -cross_val_score(model_lgb, X_data, Y_data, cv=5, scoring='neg_mean_absolute_error').mean()
    return val


from bayes_opt import BayesianOptimization

bayes_lgb = BayesianOptimization(
    rf_cv_lgb,
    {
        'num_leaves': (30, 200),

        'max_depth': (6, 20)
    }
)

bayes_lgb.maximize()
print(bayes_lgb.max)
param=bayes_lgb.max['params']