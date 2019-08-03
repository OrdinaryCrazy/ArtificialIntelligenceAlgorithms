#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# In[2]:


X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100,1)

plt.scatter(X, y, color='r')
plt.show()


# In[3]:


X_b = np.c_[np.ones((100,1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


# In[4]:


theta_best


# In[5]:


X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict


# In[6]:


plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0,2,0,15])
plt.show()


# In[7]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_


# In[8]:


lin_reg.predict(X_new)


# In[9]:


eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations) :
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
theta


# In[10]:


n_epochs = 50
t0, t1 = 5, 10 # learning schedule hyperparameters

def learningSchedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs) :
    for i in range(m):
        randomIndex = np.random.randint(m)
        xi = np.array([X_b[randomIndex]])
        yi = np.array([y[randomIndex]])
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learningSchedule(epoch * m + i)
        theta = theta - eta * gradients


# In[11]:


theta


# In[12]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter_no_change=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())


# In[13]:


sgd_reg.intercept_, sgd_reg.coef_


# In[26]:


# polynomial regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X,y,c='r')


# In[27]:


from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]


# In[28]:


X_poly[0]


# In[29]:


lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_


# In[54]:


# learning curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [],[]
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_pred = model.predict(X_train[:m])
        y_val_pred = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_pred, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_pred, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right")

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.show()


# In[78]:


from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline((
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("sgd_reg", LinearRegression()),
))

plot_learning_curves(polynomial_regression, X, y)


# In[79]:


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])


# In[81]:


sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])


# In[82]:


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
lasso_reg.predict([[1.5]])


# In[84]:


# ElasticNet
from sklearn.linear_model import ElasticNet
elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_reg.fit(X, y)
elastic_reg.predict([[1.5]])


# In[96]:


# early stopping
from sklearn.base import clone
sgd_reg = SGDRegressor(n_iter_no_change=1, warm_start=True, penalty=None,
                       learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_poly_train, X_poly_val, y_train, y_val = train_test_split(
    scaler.fit_transform(X_poly), y.ravel(), test_size = 0.2, random_state = 42
)
for epoch in range(1000):
    sgd_reg.fit(X_poly_train, y_train)
    y_val_pred = sgd_reg.predict(X_poly_val)
    val_error = mean_squared_error(y_val_pred, y_val)
#     print(val_error)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)


# In[98]:


# logistic regression
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())


# In[99]:


X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)


# In[102]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(X, y)


# In[105]:


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:,0], "b--",label="Not Iris-Virginica")
plt.legend(loc="best")


# In[106]:


X = iris["data"][:,(2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




