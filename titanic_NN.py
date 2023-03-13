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

df = pd.read_csv('train.csv')
df.head(5)


# In[2]:


### Data conversion for network perception

# Replacing NaN Age with mean
# Replacing another NaN by ffill

mean_age = df[["Age"]].mean()
df = df.fillna({'Age':mean_age})
df = df.fillna(method="ffill")
df.isnull().sum()


# In[3]:


# Replacing "Embarked" and "Sex" columns with integer  encoding

label_encoder=preprocessing.LabelEncoder()
df['gender_type']=label_encoder.fit_transform(df["Sex"])
df['embark_type']=label_encoder.fit_transform(df["Embarked"])

# Delete unsable features

del df['Ticket']
del df['Sex']
del df['Embarked']
del df['Cabin']
del df['Name']

df.head(20)


# In[4]:


# Convert to [0 - 1] by basic math

a = df['Parch'] / max(df.Parch)
b = df['embark_type'] / max(df.embark_type)
c = df['Fare'] / max(df.Fare)
d = (df['Pclass'] - min(df.Pclass)) / max(df.Pclass) 
e = df['SibSp'] / max(df.SibSp)
f = (df['PassengerId'] - min(df.PassengerId)) / max(df.PassengerId)
g = (df['Age'] - min(df.Age)) / max(df.Age)

# Except gender, it is binary 0/1 yet

h = df['gender_type']

# Extract Survived feature as target array

y = df['Survived']
 


# In[11]:


# Merging columns into an array
# Except Y
x = np.column_stack((f, d, g, e, c, a, h, b))


# In[13]:


# Train/test splitting

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

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


# In[14]:


# Fit it!

history = model.fit(x_train,
                    y_train,
                    epochs=60,
                    batch_size=32,
                    validation_data=(x_test, y_test))

history_dict = history.history
history_dict.keys()


# In[15]:


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



# In[16]:


# Prediction and accuracy

pred = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"test_acc: {test_acc}")


# In[17]:


# Optimal metric of success

p = np.around(pred)
recall_score(y_test, p)

