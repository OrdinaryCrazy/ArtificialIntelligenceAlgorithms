#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib
import pandas as pd
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


# In[ ]:




