#Importing library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read the required csv file
test = pd.read_csv("Test.csv")

#To check dimension of the dataset
test.shape

#List first 5 row by default
test.head()

#Creating DataFrame to check Null values, Percentage of Null Values and its count 
dataframe = pd.DataFrame(test.dtypes)
dataframe.rename(columns={0:"DataType"},inplace=True)
dataframe['Null Values'] = test.isnull().sum()
dataframe["Percentage of Missing Values"] = test.isna().mean().round(2)*100
dataframe["Count"] = test.count()

#Print index dtype and column dtypes, Non Null Value count 
test.info()

#To visulize the null value in form of Heatmap
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')

#Filling Missing Values
test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].mean())
test["MasVnrType"] = test["MasVnrType"].fillna(test["MasVnrType"].mode()[0])
test["MasVnrArea"] = test["MasVnrArea"].fillna(test["MasVnrArea"].mean())
test["BsmtQual"] = test["BsmtQual"].fillna(test["BsmtQual"].mode()[0])
test["BsmtCond"] = test["BsmtCond"].fillna(test["BsmtCond"].mode()[0])
test["BsmtExposure"] = test["BsmtExposure"].fillna(test["BsmtExposure"].mode()[0])
test["BsmtFinType1"] = test["BsmtFinType1"].fillna(test["BsmtFinType1"].mode()[0])
test["BsmtFinType2"] = test["BsmtFinType2"].fillna(test["BsmtFinType2"].mode()[0])
test["Electrical"] = test["Electrical"].fillna(test["Electrical"].mode()[0])
test["FireplaceQu"] = test["FireplaceQu"].fillna(test["FireplaceQu"].mode()[0])
test["GarageType"] = test["GarageType"].fillna(test["GarageType"].mode()[0])
test["GarageFinish"] = test["GarageFinish"].fillna(test["GarageFinish"].mode()[0])
test["GarageQual"] = test["GarageQual"].fillna(test["GarageQual"].mode()[0])
test["GarageCond"] = test["GarageCond"].fillna(test["GarageCond"].mode()[0])
test.drop(["GarageYrBlt","PoolQC","Fence","MiscFeature"],axis=1,inplace=True)
test.drop(["Alley"],axis=1,inplace=True)
test["Utilities"] = test["Utilities"].fillna(test["Utilities"].mode()[0])
test["Exterior1st"] = test["Exterior1st"].fillna(test["Exterior1st"].mode()[0])
test["Exterior2nd"] = test["Exterior2nd"].fillna(test["Exterior2nd"].mode()[0])
test["BsmtFinSF1"] = test["BsmtFinSF1"].fillna(test["BsmtFinSF1"].mean())
test["BsmtFinSF2"] = test["BsmtFinSF2"].fillna(test["BsmtFinSF2"].mean())
test["BsmtUnfSF"] = test["BsmtUnfSF"].fillna(test["BsmtUnfSF"].mean())
test["TotalBsmtSF"] = test["TotalBsmtSF"].fillna(test["TotalBsmtSF"].mean())
test["BsmtFullBath"] = test["BsmtFullBath"].fillna(test["BsmtFullBath"].mean())
test["BsmtHalfBath"] = test["BsmtHalfBath"].fillna(test["BsmtHalfBath"].mean())
test["KitchenQual"] = test["KitchenQual"].fillna(test["KitchenQual"].mode()[0])
test["Functional"] = test["Functional"].fillna(test["Functional"].mode()[0])
test["GarageCars"] = test["GarageCars"].fillna(test["GarageCars"].mean())
test["GarageArea"] = test["GarageArea"].fillna(test["GarageArea"].mean())
test["SaleType"] = test["SaleType"].fillna(test["SaleType"].mode()[0])

#Checking for change in dimension after filling values inplace of null
test.shape

#Exporting to new csv file
test.to_csv("Cleaned_test.csv",index=False)

