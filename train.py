import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import random
import joblib  # for saving the scaler



def create_keras_model(input_shape, l2_regularization):
    """Create a Keras sequential model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_shape,),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def train_model(model, X_train, y_train, epochs, batch_size):
    """Train the Keras model."""
    model.compile(loss=tf.keras.losses.mse,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mse', 'mae'])
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return RMSE and R-squared values."""

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred

def plot_results(actual_values, predicted_values, file_name):
    """Plot the results."""
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_values, predicted_values, label='Average Predicted vs. Actual')
    min_val = min(actual_values)
    max_val = max(actual_values)
    plt.plot([min_val, max_val], [min_val, max_val], 'red', label='Line of perfect predictions')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. Actual Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'static/images/{file_name}', format='jpeg')

    plt.show()


def trainMDD():
    # Initialize random seeds
    SEED_VALUE = 20
    tf.random.set_seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

    # Load and preprocess data

    scaler_X = MinMaxScaler()

    scaler_Y = MinMaxScaler()

    data = pd.read_excel("data/MDD.xlsx")

    X = data.drop(['MDD'], axis=1)
    y = np.array(data['MDD']).reshape(-1, 1)

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y)

    joblib.dump(scaler_X, 'data/model1/scaler_X.pkl')

    # Save the target variable scaler, if you have one
    joblib.dump(scaler_Y, 'data/model1/scaler_y.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=SEED_VALUE)

    # Regularization parameter
    L2_REGULARIZATION = 0.001
    EPOCHS = 200
    BATCH_SIZE = 32
    # Create and train the model
    model = create_keras_model(X_train.shape[1], L2_REGULARIZATION)
    train_model(model, X_train, y_train, EPOCHS, BATCH_SIZE)
    
    model.save('data/model1/model_1.h5')

    # Evaluate the model
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)


    # Display results
    results_df = pd.DataFrame({'Metric': ['Average RMSE', 'Average R-squared'],
                               'Value': [rmse, r2]}).round(3)
    
    # Plotting results
    plot_results(scaler_Y.inverse_transform(y_test).flatten(), scaler_Y.inverse_transform(y_pred).flatten(), 'plot1.jpeg')

def trainOMC():
    # Initialize random seeds
    SEED_VALUE = 19
    tf.random.set_seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

    # Load and preprocess data

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    data = pd.read_excel("data/omc.xlsx")

    X = data.drop(['OMC'], axis=1)
    y = np.array(data['OMC']).reshape(-1, 1)

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=SEED_VALUE)

    # Save the feature scaler
    joblib.dump(scaler_X, 'data/model2/scaler_X.pkl')

    # Save the target variable scaler, if you have one
    joblib.dump(scaler_Y, 'data/model2/scaler_y.pkl')


    # Regularization parameter
    L2_REGULARIZATION = 0.003

    EPOCHS = 200
    BATCH_SIZE = 16

    # Create and train the model

    model = create_keras_model(X_train.shape[1], L2_REGULARIZATION )
    train_model(model, X_train, y_train, EPOCHS, BATCH_SIZE)
    
    model.save('data/model2/model_2.h5')

    # Evaluate the model
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    # Display results
    results_df = pd.DataFrame({'Metric': ['Average RMSE', 'Average R-squared'],
                               'Value': [rmse, r2]}).round(3)
    
    # Plotting results
    plot_results(scaler_Y.inverse_transform(y_test).flatten(), scaler_Y.inverse_transform(y_pred).flatten(), 'plot2.jpeg')


def trainUCS():
    # Initialize random seeds
    SEED_VALUE = 94
    tf.random.set_seed(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)

    # Load and preprocess data

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    data = pd.read_excel("data/Model UCS.xlsx")

    X = data.drop(['UCS(kPa)'], axis=1)
    y = np.array(data['UCS(kPa)']).reshape(-1, 1)

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_Y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=SEED_VALUE)

    # Save the feature scaler
    joblib.dump(scaler_X, 'data/model3/scaler_X.pkl')

    # Save the target variable scaler, if you have one
    joblib.dump(scaler_Y, 'data/model3/scaler_y.pkl')


    # Regularization parameter
    L2_REGULARIZATION = 0.001

    EPOCHS = 200
    BATCH_SIZE = 32

    # Create and train the model

    model = create_keras_model(X_train.shape[1], L2_REGULARIZATION )
    train_model(model, X_train, y_train, EPOCHS, BATCH_SIZE)
    
    model.save('data/model3/model_3.h5')

    # Evaluate the model
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    # Display results
    results_df = pd.DataFrame({'Metric': ['Average RMSE', 'Average R-squared'],
                               'Value': [rmse, r2]}).round(3)
    
    # Plotting results
    plot_results(scaler_Y.inverse_transform(y_test).flatten(), scaler_Y.inverse_transform(y_pred).flatten(), 'plot3.jpeg')


# Run the function
trainMDD()
trainOMC()
trainUCS()