import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Sequential

"""
this is a tutorial for a classification of the titanic dataset
using a 4 layer MLP (multilayer perceptron)
each sample (row) in the dataset consists of different features (characteristics) about the sample

for example:
    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
    2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
    3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
    4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S

we will try to use the features of a passenger on the titanic (name, age, sex, etc.)
to try to predict whether or not that person lived or died when the boat crashed
"""


def preprocess_data():
    """
    1. load the data from the csv file (using pandas)
    2. try to express features as numbers
        AI models can only understand information as numbers
    3. if data is irrelevant or cannot become a number the remove it
    4. if data is missing from the dataset (NA value)
        replace it with some other value so it is still partially useful
    5. split x (features) from y (target prediction)
    6. normalize the dataset
        AI learns better from normalized data
    """

    data = pd.read_csv("passengers.csv")

    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "male" else 0)
    data = data.drop(labels=["PassengerId", "Name"], axis=1)

    potential_col = ["Parch", "Ticket", "Fare", "Cabin", "Embarked"]
    data = data.drop(labels=potential_col, axis=1)
    data = data.fillna(0)

    y = data["Survived"].to_numpy().astype(float)

    x = data.drop(labels=["Survived"], axis=1).to_numpy().astype(float)
    x = np.reshape(x, (-1, 1, x.shape[-1]))

    norm = Normalization()
    norm.adapt(x)

    return x, y, norm


def build_model(norm):
    """
    this model is a neural network
    every node in the network multiplies its input with a set of weights
    then adds up those values and applies an activation function

    the activation function can help keep output values within boundaries
    the goal of the network is to change its weights
        until it finds a nonlinear function that can approximate the data
        think y = mx + b but way bigger ... (y = mx + ... + ... + ... + ... + b)

    this network is made up of 4 layers of [32, 16, 16, 1] node each
        the last node gives the output value

    the network doesnt change the weights randomly, it uses an optimizer to do it efficiently
    the model calculates its loss (a measure of how wrong the predictions were)
    and then uses the optimizer to figure out which weights to change and by how much.
    since we are doing binary classification (2 classes) we use binary_crossentropy

    repeat until you are happy with the model
    """

    model = Sequential(
        [
            norm,
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_hist(hist):
    """
    this method takes the model's training history and plots it on a graph
    that way you can see how the accuracy changes over time.
    it plots the training history and validation history

    the validation set is a subset of the training data
    which is not fed to the model for optimization.
    instead it is used to intermediately test the model.
    if your model has to train on data to make a prediction then it isnt good.
    you want a model that can train on data to make predictions on unseen data
    """

    ax = pd.DataFrame(hist.history).plot(figsize=(8, 5))
    ax.set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
    plt.show()


def main():
    """
    steps in a machine learning project:

    1. preprocess the data
    2. split the data into training and testing datasets
    3. create your model (or try out different machine learning models)

        4. train the model on the data
        5. modify the model and or the data if needed
        6. repeat 4,5,6

    7. evaluate the model on the test data
    """

    x, y, norm = preprocess_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    model = build_model(norm)

    model.summary()

    hist = model.fit(x_train, y_train, validation_split=0.15, epochs=20, batch_size=32)
    plot_hist(hist)

    print()
    print(model.evaluate(x_test, y_test, return_dict=True))


if __name__ == "__main__":
    main()
