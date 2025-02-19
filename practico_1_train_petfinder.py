"""Exercise 1

Usage:

$ CUDA_VISIBLE_DEVICES=2 python practico_1_train_petfinder.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

To know which GPU to use, you can check it with the command

$ nvidia-smi
"""
# Seed value (can actually be different for each attribution step)
seed_value= 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy
numpy.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

import argparse

import os
import mlflow

import pandas

from tensorflow.keras.utils import model_to_dot

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, regularizers
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import SVG

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

    df['SQAge'] = df['Age']  ** 2
    
    df["Description"] = df["Description"].fillna("")
    df["len_des"] = df["Description"].apply(lambda x: len(x))
    df["count_word"] = df["Description"].apply(lambda x: len(x.split(" ")))
    
    scaler = StandardScaler()
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

def build_model(nlabels, direct_features_input_shape, embedded_columns, hidden_layer_sizes, dropout):
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
    
    next_layer = features
    ##next_layer = BatchNormalization(momentum=0)(features)
    for x, hidden_layer_num in enumerate(hidden_layer_sizes):
        dense1 = layers.Dense(hidden_layer_num, activation='relu', )(next_layer)
        dropout1 = layers.Dropout(dropout[x])(dense1)
        next_layer = dropout1
        
    output_layer = layers.Dense(nlabels, activation='softmax')(next_layer)

    model = models.Model(inputs=inputs, outputs=output_layer)

    return model

def plot_history(history):
    hist = pandas.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['loss'],
           label='Loss')
    plt.plot(hist['epoch'], hist['val_loss'],
           label='Validation Loss')
   
    plt.legend()
    plt.savefig('output/loss.png')
    mlflow.log_artifact('output/loss.png') 

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['val_accuracy'],
           label = 'Validation Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],
           label = 'Accuracy')
    
    plt.legend()
    plt.savefig('output/accuracy.png')
    mlflow.log_artifact('output/accuracy.png') 
    

def main():
    tf.keras.backend.clear_session()
    args = read_args()
    batch_size = 64
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]
    
    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in [ 'Color1','Color2', 'Color3', 'Type']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1','Breed2']
    }
    numeric_columns = ['Age', 'Fee', 'SQAge',  "count_word"]
    
    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numeric_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numeric_columns, embedded_columns)
    X_test, y_test = process_features(test_dataset, one_hot_columns, numeric_columns, embedded_columns, test=True)
    
    # Create the tensorflow Dataset
    
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).shuffle(5).batch(batch_size)
    
    model = build_model(nlabels, direct_features_input_shape, embedded_columns, args.hidden_layer_sizes, args.dropout)   

    print(model.summary())
    
    model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  metrics=['accuracy']) 
        
    #predictions = model.predict(test_ds)
    #print(predictions)
    mlflow.set_experiment(args.experiment_name)
    
    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_size', args.hidden_layer_sizes)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numerical_columns', numeric_columns)
        mlflow.log_param('epochs', args.epochs)
        mlflow.log_param('Dropout', args.dropout)

        # Train
        history = model.fit(train_ds, 
                            epochs=args.epochs, 
                            validation_data=dev_ds,
                            verbose=1);

        

        plot_history(history)
        
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
        print(predictions)

        
    print('All operations completed')

if __name__ == '__main__':
    main()

    