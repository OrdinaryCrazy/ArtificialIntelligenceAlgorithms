#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from pandas import Series, DataFrame


# In[2]:


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# In[3]:


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    '''
        数据存取函数，用于从网络上取数据并解压到本地
    '''
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[4]:


def load_housing_data(housing_path=HOUSING_PATH):
    '''
        数据加载
    '''
    csv_path=os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[5]:


fetch_housing_data()
housing = load_housing_data()
# 查看数据的前５条
housing.head()


# In[6]:


# 简单的统计信息
housing.info()


# In[7]:


# 查看类别分布情况
housing['ocean_proximity'].value_counts()


# In[8]:


# 查看数值统计信息
housing.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()


# In[10]:


# generate test data set
def splitTrainTest(data, testRatio):
    '''
        划分训练集测试集
    '''
    shuffledIndices = np.random.permutation(len(data))
    testSetSize = int(len(data) * testRatio)
    testIndices = shuffledIndices[:testSetSize]
    trainIndices = shuffledIndices[testSetSize:]
    return data.iloc[trainIndices], data.iloc[testIndices]


# In[11]:


trainSet, testSet = splitTrainTest(housing, 0.2)
print(len(trainSet) , "train +", len(testSet), "test")


# In[12]:


import hashlib
def testSetCheck(identifier, testRatio, hash):
#   hash.digest() 
#   返回摘要，作为二进制数据字符串值
    return hash(np.int64(identifier)).digest()[-1] < 256 * testRatio

def splitTrainTestByID(data, testRatio, IDColum, hash=hashlib.md5):
    indices = data[IDColum]
    inTestSet = indices.apply(lambda ID: testSetCheck(ID, testRatio, hash))
    return data.loc[~inTestSet], data.loc[inTestSet]


# In[13]:


housingWithID = housing.reset_index()   # add an 'index' column
trainSet, testSet = splitTrainTestByID(housingWithID, 0.2, "index")


# In[14]:


from sklearn.model_selection import train_test_split
trainSet, testSet = train_test_split(housing, test_size=0.2, random_state=42)


# In[15]:


# stratified sampling
housing["income_category"] = np.ceil(housing["median_income"] / 1.5)
housing["income_category"].where(housing["income_category"] < 5, 5.0, inplace=True)


# In[16]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for trainIndex, testIndex in split.split(housing, housing["income_category"]):
    strataTrainSet = housing.loc[trainIndex]
    strataTestSet = housing.loc[testIndex]
for set in (strataTrainSet, strataTestSet):
    set.drop(["income_category"], axis=1, inplace=True)


# In[17]:


# discover and visualize the data to gain insights
housing_v = strataTrainSet.copy()
housing_v.plot(kind="scatter", x="longitude", y="latitude")


# In[18]:


housing_v.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[19]:


housing_v.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
               s=housing_v["population"]/100, label="population",
               c="median_house_value", cmap=plt.get_cmap("gist_rainbow"), colorbar=True
              )
plt.legend()


# In[20]:


# standard correlation coefficient
corrMatrix = housing_v.corr()
corrMatrix["median_house_value"].sort_values(ascending=False)


# In[23]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value",
              "median_income",
              "total_rooms",
              "housing_median_age"
             ]
scatter_matrix(housing_v[attributes], figsize=(12, 8))


# In[25]:


housing_v.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.2)


# In[28]:


# attribute combinations
housing_v["rooms_per_household"] = housing_v["total_rooms"] / housing_v["households"]
housing_v["bedrooms_per_room"] = housing_v["total_bedrooms"] / housing_v["total_rooms"]
housing_v["population_per_household"] = housing_v["population"] / housing_v["households"]
corrMatrix = housing_v.corr()
corrMatrix["median_house_value"].sort_values(ascending=False)


# In[32]:


housing_predictors = strataTrainSet.drop("median_house_value", axis=1)
housing_labels = strataTrainSet["median_house_value"].copy()


# In[34]:


# Data Cleaning

# housing_predictors.dropna(subset=["total_bedrooms"]) # Get rid of the missing values instances
# housing_predictors.drop("total_bedrooms", axis=1)    # Get rid of the whole attribute
# median = housing_predictors["total_bedrooms"].median()
# housing_predictors['total_bedrooms'].fillna(median)  # Set the missing values to some value

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_predictors_num = housing_predictors.drop("ocean_proximity", axis=1)
imputer.fit(housing_predictors_num)


# In[36]:


imputer.statistics_


# In[38]:


housing_predictors_num.median().values


# In[ ]:


X = imputer.transform(housing_predictors_num)
housing_tr = DataFrame(X, columns=housing_predictors_num.columns)

