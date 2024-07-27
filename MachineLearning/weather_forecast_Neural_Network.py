import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers 
        self.alpha = alpha
        self.W = []
        self.b = []
        for i in range(0, len(layers)-1):
            w_ = np.random.randn(layers[i], layers[i+1])
            b_ = np.zeros((layers[i+1], 1))
            self.W.append(w_ / np.sqrt(layers[i]))
            self.b.append(b_)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))
    
    def fit_partial(self, X, y):
        A = [X]
        out = A[-1] 
        for i in range(0, len(self.layers) - 1):
            out = self.sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        
        y = y.reshape(-1, 1)
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]

        dW = []
        db = []
        for i in reversed(range(0, len(self.layers)-1)):
            dw_ = np.dot((A[i]).T, dA[-1] * self.sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * self.sigmoid_derivative(A[i+1]), axis=0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * self.sigmoid_derivative(A[i+1]), self.W[i].T)

            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
            
        dW = dW[::-1]
        db = db[::-1]
        
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * np.clip(dW[i], -1, 1)
            self.b[i] = self.b[i] - self.alpha * np.clip(db[i], -1, 1)

    def fit(self, X, y, epochs, verbose):
        loss_history = []
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))
                loss_history.append(loss)
        return loss_history

    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = self.sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return X

    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        y_predict = np.clip(y_predict, 1e-6, 1 - 1e-6)  # tránh log(0)
        return -np.mean(y*np.log(y_predict) + (1-y)*np.log(1-y_predict))

def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length)]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def temperature():
    df = pd.read_csv('data\old_weather_data.csv')
    data = df[['tavg']].values

    return data

def humidity():
    df = pd.read_csv('data\weather_data.csv')
    data = df[['Humidity']].values

    return data

def get_data(scaled_data, sequence_length):
    X, y = create_sequences(scaled_data, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=60)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    return X_train, X_test, y_train, y_test

def get_train_data(model, X_train, y_train, scaler):
    y_train_pred = model.predict(X_train)
    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))

    y_train_pred = y_train_pred.round(1)
    return y_train_pred, y_train_true

def get_test_data(model, X_test, y_test, scaler):
    y_test_pred = model.predict(X_test)
    y_test_pred = scaler.inverse_transform(y_test_pred)
    y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    y_test_pred = y_test_pred.round(1)
    return y_test_pred, y_test_true

def get_percentage_correct(y_train_pred, y_train_true, y_test_pred, y_test_true, threshold):
    train_errors = np.abs(y_train_pred - y_train_true)
    test_errors = np.abs(y_test_pred - y_test_true)

    train_correct = np.sum(train_errors <= threshold)
    test_correct = np.sum(test_errors <= threshold)

    percentage_train_correct = (train_correct / len(y_train_true)) * 100
    percentage_test_correct = (test_correct / len(y_test_true)) * 100

    print(f"Percentage of Training Predictions: {percentage_train_correct:.2f}%")
    print(f"Percentage of Test Predictions: {percentage_test_correct:.2f}%")

def plot_a_graph2(y_test_pred, y_test_true, title, feature):
    plt.plot(y_test_true, color='blue', label='Actual')
    plt.plot(y_test_pred, color='red', linestyle='--', label="Predict")
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel(feature)
    plt.legend()
    plt.show()

def table_of_test_data(y_test_pred, y_test_true):    
    y_test_pred = np.array(y_test_pred).flatten()
    y_test_true = np.array(y_test_true).flatten()
    df = pd.DataFrame({
        'y_test_pred': y_test_pred,
        'y_test_true': y_test_true
    })
    print(df)

def model_of_temperature(scaler):
    # Temperature
    data_temperature = temperature()    
    threshold = 1

    scaled_data_temperature = scaler.fit_transform(data_temperature)

    sequence_length = 400
    X_train, X_test, y_train, y_test = get_data(scaled_data_temperature, sequence_length)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    model = NeuralNetwork(layers=[sequence_length, 400, 1], alpha=0.0005)
    print(repr(model))

    model.fit(X_train, y_train, epochs=1000, verbose=10)
    y_train_pred, y_train_true = get_train_data(model, X_train, y_train, scaler)
    y_test_pred, y_test_true = get_test_data(model, X_test, y_test, scaler)

    get_percentage_correct(y_train_pred, y_train_true, y_test_pred, y_test_true, threshold)
    table_of_test_data(y_test_pred, y_test_true)

    plot_a_graph2(y_test_pred, y_test_true, "Temperature Prediction", "Temperature")


def model_of_humidity(scaler):
    data_humidity = humidity()
    scaled_data_humidity = scaler.fit_transform(data_humidity)

    sequence_length = 200
    threshold = 1

    X_train, X_test, y_train, y_test = get_data(scaled_data_humidity, sequence_length)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    model = NeuralNetwork(layers=[sequence_length, 100, 1], alpha=0.0001)
    print(repr(model))

    model.fit(X_train, y_train, epochs=1000, verbose=10)

    y_train_pred, y_train_true = get_train_data(model, X_train, y_train, scaler)
    y_test_pred, y_test_true = get_test_data(model, X_test, y_test, scaler)

    get_percentage_correct(y_train_pred, y_train_true, y_test_pred, y_test_true, threshold)

    table_of_test_data(y_test_pred, y_test_true)
    plot_a_graph2(y_test_pred, y_test_true, "Humidity Prediction", "Humidity")



if __name__ == "__main__":
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_of_temperature(scaler)
    # # model_of_humidity(scaler)
    
