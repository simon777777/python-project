from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# 划分训练及测试集
x_train,x_test,y_train,y_test = train_test_split( X, Y,test_size=0.3,random_state=1)
# 模型训练
clf=CatBoostRegressor(
            loss_function="MAE",
            eval_metric= 'MAE',
            task_type="CPU",
            od_type="Iter",
            random_seed=2022)

result = []
mean_score = 0
n_folds=5
kf = KFold(n_splits=n_folds ,shuffle=True,random_state=2022)
for train_index, test_index in kf.split(X):
    x_train = X.iloc[train_index]
    y_train = Y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]
    clf.fit(x_train,y_train)
    y_pred=clf.predict(x_test)
    print('验证集MAE:{}'.format(mean_absolute_error(np.expm1(y_test),np.expm1(y_pred))))
    mean_score += mean_absolute_error(np.expm1(y_test),np.expm1(y_pred))/ n_folds
    y_pred_final = clf.predict(test)
    y_pred_test=np.expm1(y_pred_final)
    result.append(y_pred_test)
# 模型评估
print('mean 验证集MAE:{}'.format(mean_score))
cat_pre=sum(result)/n_folds
ret=pd.DataFrame(cat_pre,columns=['price'])
ret.to_csv('/预测.csv')

from lightgbm.sklearn import LGBMRegressor

gbm = LGBMRegressor()
#转化object变量
X['notRepairedDamage'] = X['notRepairedDamage'].astype('float64')
test['notRepairedDamage'] = test['notRepairedDamage'].astype('float64')
result1 = []
mean_score1 = 0
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)
for train_index, test_index in kf.split(X):
    x_train = X.iloc[train_index]
    y_train = Y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = Y.iloc[test_index]
    gbm.fit(x_train, y_train)
    y_pred1 = gbm.predict(x_test)
    print('验证集MAE:{}'.format(mean_absolute_error(np.expm1(y_test), np.expm1(y_pred1))))
    mean_score1 += mean_absolute_error(np.expm1(y_test), np.expm1(y_pred1)) / n_folds
    y_pred_final1 = gbm.predict((test), num_iteration=gbm.best_iteration_)
    y_pred_test1 = np.expm1(y_pred_final1)
    result1.append(y_pred_test1)
# 模型评估
print('mean 验证集MAE:{}'.format(mean_score1))
cat_pre1 = sum(result1) / n_folds

# 加权融合
sub_Weighted = (1 - mean_score1 / (mean_score1 + mean_score)) * cat_pre1 + (
            1 - mean_score / (mean_score1 + mean_score)) * cat_pre