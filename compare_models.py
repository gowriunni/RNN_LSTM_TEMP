import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.metrics import mean_absolute_error, mean_squared_error


#Read the CSV file
data = pd.read_csv("data/temperature.csv")
print(data.head())


#Extract temperature
temperatures = data["temperature"].values
print(temperatures)


#Create sequences
window_size = 7
x = []
y = []
for i in range(len(temperatures) - window_size):

    x.append(
        temperatures[i:i + window_size]
    )

    y.append(
        temperatures[i + window_size]
    )



# Convert to NumPy
x = np.array(x)
y = np.array(y)
print("x shape:", x.shape)
print("y shape:", y.shape)



#Reshape for RNN/LSTM
x = x.reshape(
    (x.shape[0], x.shape[1], 1)
)

print("x shape after reshape:", x.shape)



#Split training and testing data
train_size = int(len(x) * 0.8)

x_train = x[:train_size]
x_test = x[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))



#Create RNN model
rnn_model = models.Sequential([
    layers.SimpleRNN(
        32,
        activation="tanh"
    ),
    layers.Dense(
        16,
        activation="relu"
    ),
    layers.Dense(1)

])


#Compile RNN
rnn_model.compile(

    optimizer="adam",

    loss="mse"

)


#Create LSTM model
lstm_model = models.Sequential([
    layers.LSTM(32),

    layers.Dense(
        16,
        activation="relu"
    ),

    layers.Dense(1)

])


#Compile LSTM
lstm_model.compile(

    optimizer="adam",

    loss="mse"
)


#Train RNN
print("\n")
print("Training RNN")
print("")
rnn_history = rnn_model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(x_test, y_test),
    verbose=1
)


#Train LSTM
print("\n")
print("Training LSTM")
print("")
lstm_history = lstm_model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=4,
    validation_data=(x_test, y_test),
    verbose=1
)


#Make RNN predictions
rnn_predictions = rnn_model.predict(x_test)


#Make LSTM predictions
lstm_predictions = lstm_model.predict(x_test)


#Calculate MAE
rnn_mae = mean_absolute_error(
    y_test,
    rnn_predictions
)
lstm_mae = mean_absolute_error(
    y_test,
    lstm_predictions
)


print("\n")
print("MAE Comparison")
print("")
print("RNN MAE:", rnn_mae)
print("LSTM MAE:", lstm_mae)


#Calculate RMSE
rnn_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rnn_predictions
    )
)
lstm_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lstm_predictions
    )
)
print("\n==============================")
print("RMSE Comparison")
print("==============================")
print("RNN RMSE:", rnn_rmse)
print("LSTM RMSE:", lstm_rmse)


#Print prediction comparison
print("\n==============================")
print("Prediction Comparison")
print("==============================")
for i in range(len(y_test)):

    print(
        f"Actual: {y_test[i]:.2f} | "
        f"RNN: {rnn_predictions[i][0]:.2f} | "
        f"LSTM: {lstm_predictions[i][0]:.2f}"
    )


#Plot actual vs RNN vs LSTM
plt.figure(figsize=(10, 5))
plt.plot(
    y_test,
    label="Actual"
)
plt.plot(
    rnn_predictions,
    label="RNN"
)
plt.plot(
    lstm_predictions,
    label="LSTM"
)
plt.title(
    "RNN vs LSTM Temperature Prediction"
)
plt.xlabel("Test Sample")
plt.ylabel("Temperature")
plt.legend()
plt.show()


#Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(
    rnn_history.history["loss"],
    label="RNN Training Loss"
)
plt.plot(
    lstm_history.history["loss"],
    label="LSTM Training Loss"
)
plt.title(
    "RNN vs LSTM Training Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()