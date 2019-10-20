"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

To know which GPU to use, you can check it with the command

$ nvidia-smi
"""

import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, regularizers
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
TARGET_COL = 'AdoptionSpeed'


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='./dataset', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--hidden_layer_sizes', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropout)
    return args


def process_features(df, one_hot_columns, numeric_columns, embedded_columns, test=False):
    direct_features = []

    # Create one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    # TODO Create and append numeric columns
    # Don't forget to normalize!
    # ....
    
    scaler = MinMaxScaler()
    direct_features.append(scaler.fit_transform(df[numeric_columns]))
    
    
    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
       features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None
    
    return features, targets



def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join(dataset_dir, 'train.csv')), test_size=0.2)
    
    test_dataset = pandas.read_csv(os.path.join(dataset_dir, 'test.csv'))
    
    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))
    
    return dataset, dev_dataset, test_dataset

def build_model(nlabels, direct_features_input_shape, embedded_columns):
    embedding_layers = []
    inputs = []
    for embedded_col, max_value in embedded_columns.items():
        input_layer = layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / 4)
        embedding_layers.append(
            tf.squeeze(layers.Embedding(input_dim=max_value, output_dim=embedding_size)(input_layer), axis=-2))
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)
        
    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [direct_features_input])

    dense1 = layers.Dense(100, activation='relu')(features)
    dropout1 = layers.Dropout(0.3)(dense1)
    dense2 = layers.Dense(50, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.5)(dense2)
    output_layer = layers.Dense(nlabels, activation='softmax')(dropout2)

    model = models.Model(inputs=inputs, outputs=output_layer)

    return model


def main():
    tf.keras.backend.clear_session()
    args = read_args()
    #dataset_dir = ''
    batch_size = 64
    #epochs = 20
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]
    
    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in ['Gender', 'Color1','Color2', 'Color3', 'Type']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1','Breed2']
    }
    numeric_columns = ['Age', 'Fee']
    
    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
    X_test, y_test = process_features(test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)
    
    # Create the tensorflow Dataset
    
    # TODO shuffle the train dataset!
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(5).batch(batch_size)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).shuffle(5).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).shuffle(5).batch(batch_size)
    
    # TODO: Build the Keras model
    
    model = build_model(nlabels, direct_features_input_shape, embedded_columns)   

    print(model.summary())
    
    model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True),
                  metrics=['accuracy']) 
        
    #predictions = model.predict(test_ds)
    #print(predictions)
    #mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_size', args.hidden_layer_sizes)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numerical_columns', numeric_columns)  # Not using these yet
        mlflow.log_param('epochs', args.epochs)

        # Train
        history = model.fit(train_ds, 
                            epochs=args.epochs, 
                            validation_data=dev_ds,
                            verbose=1);

        # TODO: analyze history to see if model converges/overfits
        
        # TODO: Evaluate the model, calculating the metrics.
        # Option 1: Use the model.evaluate() method. For this, the model must be
        # already compiled with the metrics.
        #performance = model.evaluate(X_test, y_test)

        loss, accuracy = 0, 0
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('loss', loss)
        mlflow.log_metric('accuracy', accuracy)
        
        # Option 2: Use the model.predict() method and calculate the metrics using
        # sklearn. We recommend this, because you can store the predictions if
        # you need more analysis later. Also, if you calculate the metrics on a
        # notebook, then you can compare multiple classifiers.
        
        #predictions = 'No prediction yet'
        #predictions = model.predict(test_ds)
        predictions = numpy.argmax(model.predict(test_ds), axis=1)

        predictions=predictions.flatten()
        results = pandas.Series(predictions,name="AdoptionSpeed")
        submission = pandas.concat([pandas.Series(test_dataset.PID,name = "PID"),results],axis = 1)
        submission.to_csv("result_submission.csv",index=False)

        # TODO: Convert predictions to classes
        # TODO: Save the results for submission
        # ...
        print(predictions)

        
    print('All operations completed')

if __name__ == '__main__':
    main()

    