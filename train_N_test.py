import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor

reg = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0,
criterion='squared_error',
max_depth=None, max_features=1.0,
max_leaf_nodes=None, max_samples=None,
min_impurity_decrease=0.0,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0,
n_estimators=100, n_jobs=-1,
oob_score=False, random_state=3010,
verbose=0, warm_start=False)

train = pd.read_csv(r'./datasets/transformed_train.csv')
test = pd.read_csv(r'./datasets/transformed_test.csv')

# training
train = train.sample(frac=0.8, random_state=3010)
X, y = train.drop(columns='revenue'), train['revenue']
reg.fit(X, y)
# print(reg.score(X, y))

# prediction
X1 = test.drop(columns='id', inplace=False)
test['revenue'] = reg.predict(X1)
test = test.loc[:, ['id', 'revenue']]
test.to_csv(r'./Preds/preds_rf.csv', index=False)
