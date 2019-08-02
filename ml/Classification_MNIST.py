#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
# MINIST = fetch_mldata('MNIST original')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


# In[2]:


X.shape


# In[3]:


y.shape


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# In[5]:


y[36000]


# In[6]:


train_ins = 60000
X_train, X_test, Y_train, Y_test = X[:train_ins], X[train_ins:], y[:train_ins], y[train_ins:]
# shuffle
import numpy as np
shuffle_index = np.random.permutation(train_ins)
X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]


# In[7]:


# binary classifier --> 5-detector
Y_train_5 = (Y_train == '5')
Y_test_5 = (Y_test == '5')
from sklearn.linear_model import SGDClassifier

SGD_C = SGDClassifier(random_state=42)
SGD_C.fit(X_train, Y_train_5)

SGD_C.predict([some_digit])


# In[8]:


# performance measure
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, Y_train_5) :
    clone_clf = clone(SGD_C)
    X_train_folds = X_train[train_index]
    Y_train_folds = Y_train_5[train_index]
    X_test_fold = X_train[test_index]
    Y_test_fold = Y_train_5[test_index]
    
    clone_clf.fit(X_train_folds, Y_train_folds)
    Y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(Y_pred == Y_test_fold)
    print(n_correct / len(Y_pred))


# In[9]:


from sklearn.model_selection import cross_val_score
cross_val_score(SGD_C, X_train, Y_train_5, cv=3, scoring="accuracy")


# In[10]:


from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator) :
    def fit(self, X, y=None) :
        pass
    def predict(delf, X) :
        return np.zeros((len(X), 1), dtype=bool)

never5_C = Never5Classifier()
cross_val_score(never5_C, X_train, Y_train_5, cv=3, scoring="accuracy")


# In[11]:


# confusion matrix
from sklearn.model_selection import cross_val_predict
Y_train_pred = cross_val_predict(SGD_C, X_train, Y_train_5,cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_train_5, Y_train_pred)


# In[12]:


confusion_matrix(Y_train_5, Y_train_5)


# In[13]:


# each row in a confusion matrix represens an actual class, while each column represents a predicted class
# precision: the accuracy of the positive prediction
# recall: the ratio of positive instances that are correctly detected by the classifier
from sklearn.metrics import precision_score, recall_score
precision_score(Y_train_5, Y_train_pred)


# In[14]:


recall_score(Y_train_5, Y_train_pred)


# In[15]:


from sklearn.metrics import f1_score
f1_score(Y_train_5, Y_train_pred)


# In[16]:


y_scores = SGD_C.decision_function([some_digit])
y_scores


# In[17]:


y_scores = cross_val_predict(SGD_C, X_train, Y_train_5, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(Y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "b--", label="Precision")
    plt.plot(threshold, recall[:-1],    "g-",  label="Recall"   )
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])
plot_precision_recall_vs_threshold(precision, recall, threshold)
plt.show()


# In[18]:


# ROC Curve
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_train_5, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel("false_positive_rate")
    plt.ylabel("true_positive_rate")
    
plot_roc_curve(fpr, tpr)
plt.show()


# In[19]:


from sklearn.metrics import roc_auc_score
roc_auc_score(Y_train_5, y_scores)


# In[20]:


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, Y_train_5, cv=3, method="predict_proba")
fpr_forest, tpr_forest, threshold_forest = roc_curve(Y_train_5, y_probas_forest[:, 1])
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()


# In[21]:


roc_auc_score(Y_train_5, y_probas_forest[:, 1])


# In[22]:


SGD_C.fit(X_train, Y_train)
SGD_C.predict([some_digit])


# In[23]:


some_digit_scores = SGD_C.decision_function([some_digit])
some_digit_scores


# In[24]:


np.argmax(some_digit_scores)


# In[25]:


SGD_C.classes_


# In[26]:


# one-versus-one and one-versus-all
from sklearn.multiclass import OneVsOneClassifier
OvO_C = OneVsOneClassifier(SGDClassifier(random_state=42))
OvO_C.fit(X_train, Y_train)
OvO_C.predict([some_digit])


# In[27]:


len(OvO_C.estimators_)


# In[28]:


forest_clf.fit(X_train, Y_train)
forest_clf.predict_proba([some_digit])


# In[29]:


cross_val_score(SGD_C, X_train, Y_train, cv=3, scoring="accuracy")


# In[30]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(SGD_C, X_train_scaled, Y_train, cv=3, scoring="accuracy")


# In[31]:


Y_train_pred = cross_val_predict(SGD_C, X_train_scaled, Y_train, cv=3)


# In[32]:


conf_mx = confusion_matrix(Y_train, Y_train_pred)
conf_mx


# In[33]:


plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()


# In[34]:


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


# In[35]:


# multi-label classification
from sklearn.neighbors import KNeighborsClassifier

Y_train_large = (Y_train.astype(np.int64) >= 7)
Y_train_odd = (Y_train.astype(np.int64) % 2 == 1)
Y_multilabel = np.c_[Y_train_large, Y_train_odd]

KNN_C = KNeighborsClassifier()
KNN_C.fit(X_train, Y_multilabel)


# In[36]:


Y_train_KNN_pred = cross_val_predict(KNN_C, X_train[:10000], Y_train[:10000], cv=3)


# In[38]:


f1_score(Y_train[:10000], Y_train_KNN_pred, average="macro")


# In[40]:


# multi-output classification
noise_train = np.random.randint(0, 100, (len(X_train), 784))
noise_test = np.random.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise_train
X_test_mod = X_test + noise_test
Y_train_mod = X_train
Y_test_mod = X_test


# In[46]:


plt.imshow(X_test_mod[1000].reshape(28,28), cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# In[53]:


KNN_C.fit(X_train_mod, Y_train_mod)
clean_digit = KNN_C.predict([X_test_mod[1000]])
plt.imshow(clean_digit.reshape(28,28), cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


# In[ ]:




