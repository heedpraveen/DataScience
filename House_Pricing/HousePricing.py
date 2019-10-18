#Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn.model_selection import RandomizedSearchCV

#Reading csv file
train = pd.read_csv("train.csv")

#To view first 5 row(default)
train.head()

#To visualize in form of heatmap to check for null values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

#To check dimension
train.shape

#It gives null value count in each attribute
train.isnull().sum()

#Print information of index dtype and column dtypes, non null values
train.info()

#Creating Dataframe to check null value count, percentge of the same and count of instances
missing_value = pd.DataFrame(train.dtypes)
missing_value.rename(columns = {0:'DataTypes'},inplace=True)
missing_value['Null Value'] = train.isnull().sum()
missing_value['Percentage Missing Values'] = train.isna().mean().round(2)*100
missing_value['Count'] = train.count()

#Filling Missing Values
train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].mean())
train["MasVnrType"] = train["MasVnrType"].fillna(train["MasVnrType"].mode()[0])
train["MasVnrArea"] = train["MasVnrArea"].fillna(train["MasVnrArea"].mean())
train["BsmtQual"] = train["BsmtQual"].fillna(train["BsmtQual"].mode()[0])
train["BsmtCond"] = train["BsmtCond"].fillna(train["BsmtCond"].mode()[0])
train["BsmtExposure"] = train["BsmtExposure"].fillna(train["BsmtExposure"].mode()[0])
train["BsmtFinType1"] = train["BsmtFinType1"].fillna(train["BsmtFinType1"].mode()[0])
train["BsmtFinType2"] = train["BsmtFinType2"].fillna(train["BsmtFinType2"].mode()[0])
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])
train["FireplaceQu"] = train["FireplaceQu"].fillna(train["FireplaceQu"].mode()[0])
train["GarageType"] = train["GarageType"].fillna(train["GarageType"].mode()[0])
train["GarageFinish"] = train["GarageFinish"].fillna(train["GarageFinish"].mode()[0])
train["GarageQual"] = train["GarageQual"].fillna(train["GarageQual"].mode()[0])
train["GarageCond"] = train["GarageCond"].fillna(train["GarageCond"].mode()[0])
train.drop(["GarageYrBlt","PoolQC","Fence","MiscFeature"],axis=1,inplace=True)
train.drop(["Alley"],axis=1,inplace=True)

#Again checking with heatmap for null values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="YlGnBu")

#Column not needed so we are dropping
train.drop(['Id'],axis=1,inplace=True)

#Reading the cleaned test dataset
test = pd.read_csv("Cleaned_test.csv")

#Dimension of test dataset
test.shape

#As we don't know the value count of each attributes differ between test and train, so for good practise we concat both into new variable
df = pd.concat([train,test],axis=0)

#Dimension of both file
df.shape

#We are going to convert ctegorical to numerical, but for first we extracting categorical variable first
df.select_dtypes(include='object').columns
columns=['BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'BsmtQual', 'CentralAir', 'Condition1', 'Condition2', 'Electrical',
       'ExterCond', 'ExterQual', 'Exterior1st', 'Exterior2nd', 'FireplaceQu',
       'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual',
       'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual',
       'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSZoning',
       'MasVnrType', 'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle',
       'SaleCondition', 'SaleType', 'Street', 'Utilities']

#Length of categorical variable
len(columns)

#To check unique value count in each attributes
cate = pd.DataFrame(df.select_dtypes(include='object').nunique())

#Function to convert ctegorical to numerical, here am using one-hot encoding
def one_hot_encode(col):
    final_df = df
    i=0
    for fields in col:
        
        print(fields)
        df1=pd.get_dummies(df[fields],drop_first=True)
        df.drop([fields],axis=1,inplace=True)
        
        if i==0:
            final_df = df1.copy()
        else:
            final_df=pd.concat([final_df,df1],axis=1)
            
        i = i+1
    final_df = pd.concat([df,final_df],axis=1)
    return final_df

#Function call
df = one_hot_encode(columns)

#Dimension
df.shape

#Removing duplicates
df = df.loc[:,~df.columns.duplicated()]

#Dimension 
df.shape

#Dropping ID column
df.drop(["Id"],axis=1,inplace=True)

#Splitting train and test data from df
Train = df.iloc[:1460,:]
Test = df.iloc[1460:,:]

#Dimension of both
Train.shape,Test.shape

#Dropping Saleprice from test data
Test.drop(['SalePrice'],axis=1,inplace=True)

#Assigning x_train and y_train from Train data
x_train = Train.drop(["SalePrice"],axis=1)
y_train = Train["SalePrice"]

#Dimension
x_train.shape,y_train.shape

#Instantiating XGBoost Regressor
XGB= xgboost.XGBRegressor()

#HyperParameter Tuning
booster = ['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100,500,900,1100,1500]
max_depth = [2,3,5,10,15]
learning_rate = [0.05,0.1,0.15,0.20]
​
hyperparameter_grid={
    'n_estimators' : n_estimators,
    'booster': booster,
    'base_score' : base_score,
    'max_depth' : max_depth,
    'learning_rate' : learning_rate
}

#Using RndomizedSearchCV from sklearn.model_selection
random_CV = RandomizedSearchCV(estimator=XGB,param_distributions=hyperparameter_grid,cv=5,n_iter=50,scoring='neg_mean_absolute_error',n_jobs=4,return_train_score=True,random_state=44)

#Fit the model to check for best estimators anf it takes time to fit the model in hyperparameter
random_CV.fit(x_train,y_train)

#To get best estimator
random_CV.best_estimator_

#Model training and predicting
XGB = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.05, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=1100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
XGB.fit(x_train,y_train)
y_pred = XGB.predict(Test)
y_pred

#Saving the ouput to csv file
pred = pd.DataFrame(y_pred)
sub = pd.read_csv('sample_submission.csv')
dataset = pd.concat([sub['Id'],pred],axis=1)
dataset.columns = ["Id","SalePrice"]
dataset.to_csv('SubmissionXGHP.csv',index=False)
​
