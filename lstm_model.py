import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

#read the csv file
data =pd.read_csv("data/temperature.csv")
print(data.head())

#extract temperature
temperatures=data['temperature'].values
print(temperatures)

#visualize the temperature
plt.plot(temperatures)
plt.title("Temperature over time")
plt.xlabel("Day")
plt.ylabel("Temperature")
plt.show()

#sequence creation code
window_size=7
x=[]
y=[]
for i in range(len(temperatures)- window_size):
    x.append(temperatures[i:i+ window_size])
    y.append(temperatures[i+ window_size])


#convert to numpy
x=np.array(x)
y=np.array(y)
print("x shape:",x.shape)
print("y shape:",y.shape)


x=x.reshape((x.shape[0],x.shape[1],1))
print(x.shape)


#splitting training and testing data
train_size = int(len(x) * 0.8)

x_train = x[:train_size]
x_test = x[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

#RNN
model = models.Sequential([
    layers.LSTM(32),
    layers.Dense(16, activation="relu"),
    layers.Dense(1)
])

#compile
model.compile(
    optimizer="adam",
    loss="mse"
)


#summary
model.summary()

#train
history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(x_test, y_test)
)


#plot training loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Training Loss", "Validation Loss"])
plt.show()

#making prediction
predictions = model.predict(x_test)
for actual, predicted in zip(y_test, predictions):
    print(
        "Actual:",
        actual,
        "Predicted:",
        predicted[0]
    )

#plot actual and predicted
plt.plot(y_test, label="Actual")
plt.plot(predictions, label="Predicted")

plt.title("LSTM Temperature Prediction")

plt.xlabel("Test Sample")
plt.ylabel("Temperature")

plt.legend()

plt.show()