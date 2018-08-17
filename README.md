# housing-price-prediction--AI
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Import the raw data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()

print(train.dtypes)

# string label to categorical values
from sklearn.preprocessing import LabelEncoder

for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object: 
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
       


from sklearn.preprocessing import LabelEncoder

for i in range(test.shape[1]):
    if test.iloc[:,i].dtypes == object: 
        lbl = LabelEncoder()
        lbl.fit(list(test.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
       



print(train['GarageType'].unique())

print('training data+++++++++++++++++++++')
for i in np.arange(train.shape[1]):
    n = train.iloc[:,i].isnull().sum() 
    if n > 0:
        print(list(train.columns.values)[i] + ': ' + str(n) + ' nans')

print('testing data++++++++++++++++++++++ ')
for i in np.arange(test.shape[1]):
    n = test.iloc[:,i].isnull().sum() 
    if n > 0:
        print(list(test.columns.values)[i] + ': ' + str(n) + ' nans')

# keep ID for submission
train_ID = train['ID']
test_ID = test['ID']

# split data for training
y_train = train['SalePrice']
X_train = train.drop(['ID','SalePrice'], axis=1)
X_test = test.drop('ID', axis=1)


# dealing with missing data
Xmat = pd.concat([X_train, X_test])
Xmat = Xmat.drop(['LotFrontage','MasVnrArea','GarageYearBuilt'], axis=1)
Xmat = Xmat.fillna(Xmat.median())


print(Xmat.columns.values)
print(str(Xmat.shape[1]) + ' columns')

ax = sns.distplot(y_train)
plt.show()

y_train = np.log(y_train)

ax = sns.distplot(y_train)
plt.show()

X_train = Xmat.iloc[:train.shape[0],:]
X_test = Xmat.iloc[train.shape[0]:,:]
print(len(X_train.columns))
print(len(X_test.columns))


# Compute the correlation matrix
corr = X_train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')

ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()

print(ranking[:30])
len(X_train.columns)

X_train = X_train.iloc[:,ranking[:30]]
X_test = X_test.iloc[:,ranking[:30]]

# interaction between the top 2
X_train["Interaction"] = X_train["LivingArea"]*X_train["Quality"]
X_test["Interaction"] = X_test["LivingArea"]*X_test["Quality"]

# zscoring
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()
    
# heatmap
f, ax = plt.subplots(figsize=(11, 5))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.heatmap(X_train, cmap=cmap)
plt.show()

fig = plt.figure(figsize=(12,7))
for i in np.arange(30):
    ax = fig.add_subplot(5,6,i+1)
    sns.regplot(x=X_train.iloc[:,i], y=y_train)

plt.tight_layout()
plt.show()

# outlier deletion
Xmat = X_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(Xmat[(Xmat['LivingArea']>5) & (Xmat['SalePrice']<13)].index)
Xmat = Xmat.drop(Xmat[(Xmat['TotalBsmtArea']>8) & (Xmat['SalePrice']<12.5)].index)
Xmat = Xmat.drop(Xmat[(Xmat['1stFloorArea']>9) & (Xmat['SalePrice']<12.5)].index)
Xmat = Xmat.drop(Xmat[(Xmat['BsmtFinish1Area']>6) & (Xmat['SalePrice']<13)].index)
Xmat = Xmat.drop(Xmat[(Xmat['LotSize']>17) & (Xmat['SalePrice']<15)].index)
Xmat = Xmat.drop(Xmat[(Xmat['OpenPorchArea']>7) & (Xmat['SalePrice']<13)].index)
Xmat = Xmat.drop(Xmat[(Xmat['Quality']< -2.5) & (Xmat['SalePrice']<11)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GarageCars']> 0) & (Xmat['SalePrice']<11)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GarageCars']> 2.5) & (Xmat['SalePrice']<12)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GarageArea']> 0) & (Xmat['SalePrice']<11)].index)
Xmat = Xmat.drop(Xmat[(Xmat['YearBuilt']> -2.5) & (Xmat['SalePrice']<11)].index)
Xmat = Xmat.drop(Xmat[(Xmat['YearBuilt']<-2.5) & (Xmat['SalePrice']>12)].index)
Xmat = Xmat.drop(Xmat[(Xmat['YearRemodelled']<0) & (Xmat['SalePrice']<11)].index)
Xmat = Xmat.drop(Xmat[(Xmat['Neighborhood']< 2) & (Xmat['SalePrice']<11)].index)
Xmat = Xmat.drop(Xmat[(Xmat['MonthSold']< 2) & (Xmat['SalePrice']<11)].index)
# recover
y_train = Xmat['SalePrice']
X_train = Xmat.drop(['SalePrice'], axis=1)


import xgboost as xgb
from sklearn.model_selection import GridSearchCV

print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
reg_xgb.fit(X_train, y_train)
print(reg_xgb.best_score_)
print(reg_xgb.best_params_)

from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

scoresdepth = list()
y=train[target]
n_folds = 5

for i in range(3,30):
    clf = DecisionTreeRegressor(max_depth=i)
    #clf = clf.fit(x_train,y_train)
    #depth.append((i,clf.score(x_test,y_test)))
    #lasso.alpha = alpha
    this_scores = cross_val_score(clf, X, y, cv=n_folds, n_jobs=1)
    scoresdepth.append([i,np.mean(this_scores),np.std(this_scores)])

maxdepth=max(scoresdepth,key=lambda x: x[1])
print (maxdepth)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)
# define the grid search parameters
optimizer = ['SGD','Adam']
batch_size = [10, 30, 50]
epochs = [10, 50, 100]
param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)
reg_dl = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
reg_dl.fit(X_train, y_train)

print(reg_dl.best_score_)
print(reg_dl.best_params_)

# SVR
from sklearn.svm import SVR

reg_svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
reg_svr.fit(X_train, y_train)

print(reg_svr.best_score_)
print(reg_svr.best_params_)

# second feature matrix
X_train2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_train),
     #'NN': reg_dl.predict(X_train).ravel(),
     #'SVR': reg_svr.predict(X_train),
    })
X_train2.head()


# second-feature modeling using linear regression
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train2, y_train)

# prediction using the test set
X_test2 = pd.DataFrame( {'XGB': reg_xgb.predict(X_test),
     #'DL': reg_dl.predict(X_test).ravel(),
     #'SVR': reg_svr.predict(X_test),
    })

# Don't forget to convert the prediction back to non-log scale
y_pred = np.exp(reg.predict(X_test2))

# submission
submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": y_pred
})
submission.to_csv('houseprice.csv', index=False)
