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


# In[21]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value",
              "median_income",
              "total_rooms",
              "housing_median_age"
             ]
scatter_matrix(housing_v[attributes], figsize=(12, 8))


# In[22]:


housing_v.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.2)


# In[23]:


# attribute combinations
housing_v["rooms_per_household"] = housing_v["total_rooms"] / housing_v["households"]
housing_v["bedrooms_per_room"] = housing_v["total_bedrooms"] / housing_v["total_rooms"]
housing_v["population_per_household"] = housing_v["population"] / housing_v["households"]
corrMatrix = housing_v.corr()
corrMatrix["median_house_value"].sort_values(ascending=False)


# In[24]:


housing_predictors = strataTrainSet.drop("median_house_value", axis=1)
housing_labels = strataTrainSet["median_house_value"].copy()


# In[25]:


# Data Cleaning

# housing_predictors.dropna(subset=["total_bedrooms"]) # Get rid of the missing values instances
# housing_predictors.drop("total_bedrooms", axis=1)    # Get rid of the whole attribute
# median = housing_predictors["total_bedrooms"].median()
# housing_predictors['total_bedrooms'].fillna(median)  # Set the missing values to some value

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_predictors_num = housing_predictors.drop("ocean_proximity", axis=1)
imputer.fit(housing_predictors_num)


# In[26]:


imputer.statistics_


# In[27]:


housing_predictors_num.median().values


# In[28]:


X = imputer.transform(housing_predictors_num)
housing_tr = DataFrame(X, columns=housing_predictors_num.columns)


# In[31]:


# handling text and categorical attributes
# 类别编码
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_predictors_cat = housing_predictors["ocean_proximity"]
housing_predictors_cat_encoded = encoder.fit_transform(housing_predictors_cat)
housing_predictors_cat_encoded


# In[33]:


encoder.classes_


# In[35]:


housing_predictors_cat_encoded.reshape(-1, 1)


# In[41]:


# one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto')
housing_predictors_cat_1hot = encoder.fit_transform(housing_predictors_cat_encoded.reshape(-1,1))
housing_predictors_cat_1hot


# In[42]:


housing_predictors_cat_1hot.toarray()


# In[49]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_predictors_cat_1hot = encoder.fit_transform(housing_predictors_cat)
housing_predictors_cat_1hot


# In[50]:


encoder = LabelBinarizer(sparse_output=True)
housing_predictors_cat_1hot = encoder.fit_transform(housing_predictors_cat)
housing_predictors_cat_1hot


# In[52]:


# Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

room_ix, bedroom_ix, population_ix, household_ix = 3, 4, 5, 6

class combinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, room_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room :
            bedrooms_per_room = X[:, bedroom_ix] / X[:, room_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = combinedAttributeAdder(add_bedrooms_per_room=False)
housing_predictors_extra_attribs = attr_adder.transform(housing_predictors.values)
housing_predictors_extra_attribs


# In[54]:


# transformation pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

numerical_pipeline = Pipeline([#('name', estimator())
                                ('imputer', SimpleImputer(strategy="median")),
                                ('attribs_adder', combinedAttributeAdder()),
                                ('std_scaler', StandardScaler())
                                ])
housing_predictors_num_tr = numerical_pipeline.fit_transform(housing_predictors_num)


# In[96]:


from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_predictors_num)
cat_attribs = ["ocean_proximity"]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', combinedAttributeAdder()),
    ('std_scaler', StandardScaler()),
])

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
        self.enc = LabelBinarizer(sparse_output=self.sparse_output) # 这里LabelBinarizer只接受self和X两个参数
    def fit(self, X, y=None):
        return self.enc.fit(X)
    def transform(self, X, y=None):
#         return enc.fit_transform(X)
        return self.enc.transform(X)

    
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
#     ('label_binarizer', LabelBinarizer()),
    ('label_binarizer', CustomLabelBinarizer()),
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])


# In[97]:


list(housing_predictors_num)


# In[98]:


housing_predictors_prepared = full_pipeline.fit_transform(housing_predictors)
housing_predictors_prepared[1]


# In[99]:


housing_predictors_prepared.shape


# In[100]:


# Training and Evaluation
from sklearn.linear_model import LinearRegression

LRClassifier = LinearRegression()
LRClassifier.fit(housing_predictors_prepared, housing_labels)


# In[102]:


some_data = housing_predictors.iloc[:5]
some_label = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", LRClassifier.predict(some_data_prepared))
print("Labels:\t\t",list(some_label))


# In[86]:


list(housing_predictors)


# In[106]:


from sklearn.metrics import mean_squared_error
housing_predictions = LRClassifier.predict(housing_predictors_prepared)
LR_MSE = mean_squared_error(housing_labels, housing_predictions)
LR_RMSE = np.sqrt(LR_MSE)
LR_RMSE


# In[107]:


from sklearn.tree import DecisionTreeRegressor

treeRegressor = DecisionTreeRegressor()
treeRegressor.fit(housing_predictors_prepared, housing_labels)
housing_predictions = treeRegressor.predict(housing_predictors_prepared)
TREE_MSE = mean_squared_error(housing_labels, housing_predictions)
TREE_RMSE = np.sqrt(TREE_MSE)
TREE_RMSE


# In[108]:


# Cross Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(treeRegressor, housing_predictors_prepared, housing_labels,
                        scoring="neg_mean_squared_error", cv=10
                        )
rmse_scores = np.sqrt(-scores)


# In[110]:


rmse_scores


# In[112]:


rmse_scores.mean()


# In[114]:


rmse_scores.std()


# In[118]:


# from sklearn.externals import joblib
import joblib
joblib.dump(treeRegressor, "housing_treeregress.pkl")
# loaded = joblib.load("housing_treeregress.pkl")


# In[119]:


# model fine-tune
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

forestRegressor = RandomForestRegressor()

paramGrid = [
    {"n_estimators":[3, 10, 30], "max_features":[2, 4, 6, 8]},
    {"bootstrap":[False], "n_estimators":[3, 10], "max_features":[2, 3, 4]},
]

gridSearch = GridSearchCV(forestRegressor, param_grid=paramGrid, cv=5, scoring="neg_mean_squared_error")

gridSearch.fit(housing_predictors_prepared, housing_labels)


# In[120]:


gridSearch.best_params_


# In[121]:


gridSearch.best_estimator_


# In[124]:


cvres = gridSearch.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]) :
    print(np.sqrt(-mean_score), params)


# In[125]:


# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
# Ensemble Methods
feature_importances = gridSearch.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_household", "population_per_household"]
# extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
attributes = num_attribs + extra_attribs + list(encoder.classes_)
sorted(zip(feature_importances, attributes), reverse=True)


# In[126]:


final_model = gridSearch.best_estimator_

X_test = strataTestSet.drop("median_house_value", axis=1)
Y_test = strataTestSet["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[127]:


final_rmse

