#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head(5)


# In[2]:


### Data conversion for network perception

# Replacing NaN Age with mean
# Replacing another NaN by ffill

mean_age = df[["Age"]].mean()

df = df.fillna({'Age':mean_age})
df = df.fillna(method = "ffill")

df.isnull().sum()


# In[3]:


# Replacing "Embarked" and "Sex" columns with integer  encoding

label_encoder = preprocessing.LabelEncoder()

df['gender_type'] = label_encoder.fit_transform(df["Sex"])
df['embark_type'] = label_encoder.fit_transform(df["Embarked"])

# Delete unsable features

del df['Ticket']
del df['Sex']
del df['Embarked']
del df['Cabin']
del df['Name']
del df['PassengerId']

df.head(10)


# In[4]:


# Extract Survived feature as target array

y = df['Survived']
del df['Survived']
 


# In[5]:


# Convert all to [0 - 1] 

for i in df.columns:
    a = max(df[i])
    b = min(df[i])
    df[i] = (df[i] - b) / a
    
df.head(10)


# In[6]:


# Train/test splitting

x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size = 0.2, random_state = 13)

# Build network model

model = keras.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dropout(.75),
    layers.Dense(128, activation="relu"),
    layers.Dropout(.1),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="Adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])


# In[7]:


# Fit it!

history = model.fit(x_train,
                    y_train,
                    epochs = 60,
                    batch_size = 32,
                    validation_data = (x_test, y_test))

history_dict = history.history
history_dict.keys()


# In[8]:


# Visualization of training process, epoch/loss

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Visualization of training process, epoch/accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



# In[9]:


# Prediction and accuracy

pred = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test_acc: {test_acc}")


# In[10]:


# Optimal metric of success

p = np.around(pred)
recall_score(y_test, p)


# In[11]:


# Now. Testing and making prediction & submission

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head(5)


# In[12]:


mean_age = test[["Age"]].mean()

test = test.fillna({'Age':mean_age})
test = test.fillna(method = "ffill")

label_encoder = preprocessing.LabelEncoder()

test['gender_type'] = label_encoder.fit_transform(test["Sex"])
test['embark_type'] = label_encoder.fit_transform(test["Embarked"])

del test['Ticket']
del test['Sex']
del test['Embarked']
del test['Cabin']
del test['Name']
del test['PassengerId']

for i in test.columns:
    a = max(test[i])
    b = min(test[i])
    test[i] = (test[i] - b) / a


# In[13]:


test.isnull().sum()


# In[14]:


sub = np.round(model.predict(test))


# In[15]:


submission = pd.read_csv('/kaggle/input/titanic/test.csv')

del submission['Ticket']
del submission['Sex']
del submission['Embarked']
del submission['Cabin']
del submission['Name']
del submission['Pclass']
del submission['SibSp']
del submission['Parch']
del submission['Fare']
del submission['Age']
                  
submission.head()


# In[16]:


submission["Survived"] = np.round(model.predict(test)).astype(int)
submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[17]:


submission.head()

