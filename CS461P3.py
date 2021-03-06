# CS461PR by Alex Arbuckle #


# Import <
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts

# >


# Main <
if (__name__ == '__main__'):

    # Data Preprocessing <
    dataset = pd.read_csv('CS461P3.csv')
    dataset = np.array(pd.get_dummies(dataset, drop_first = True))

    x, y = dataset[:, 3:], dataset[:, :3]
    xTrain, xTest, yTrain, yTest = tts(x, y, test_size = 0.15, random_state = 0)

    # >

    # Building <
    nn = tf.keras.models.Sequential()

    nn.add(tf.keras.layers.Dense(units = 60))
    nn.add(tf.keras.layers.Dense(units = 30))

    nn.add(tf.keras.layers.Dense(units = 3))

    # >

    # Training <
    nn.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

    nn.fit(validation_split = 0.15,
           x = xTrain, y = yTrain,
           batch_size = 32,
           epochs = 100,
           verbose = 2)

    # >

    # Predicting <
    yPred = nn.predict(xTest)
    np.set_printoptions(precision = 2)
    output = pd.DataFrame(yPred, columns = ['math', 'reading', 'writing'])

    print()
    print(output)

    # >

# >
