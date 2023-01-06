#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
#addded the os library
import os
#matplotlib inline
# To add new datasets upload files and add them to datasets list entirely written by us
#test is to determine which method of getting data: 0 is for number of data using os list ; 1 is for pre-defined list
def dataset_gathering(test,quantity,list):
    if test == 0:
        dataset = os.listdir('archive/stocks')
        for i in range(quantity):
            dataset[i] = "archive/stocks/"+dataset[i]
    if test == 1:
        dataset = list
    return dataset

def train_model(filename,num_epoch,num_batch,num_layers,dropout):
    global p
    # reading dataset
    df = pd.read_csv(filename,na_values=["null"],index_col="Date",parse_dates=True,infer_datetime_format=True)
    df.head()
    #printing head and any null value column
    print("Dataframe Shape: ", df. shape)
    print("Null Value Present: ", df.isnull().values.any())
    # dataset transformation
    df = df['Open'].values
    df = df.reshape(-1, 1)
    dataset_train0 = np.array(df[:int(df.shape[0]*0.8)])
    dataset_test0 = np.array(df[int(df.shape[0]*0.8):])
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_train = scaler.fit_transform(dataset_train0)
    dataset_test = scaler.transform(dataset_test0)
    def create_dataset(df):
        x = []
        y = []
        for i in range(50, df.shape[0]):
            x.append(df[i-50:i, 0])
            y.append(df[i, 0])
        x = np.array(x)
        y = np.array(y)
        return x,y

    x_train, y_train = create_dataset(dataset_train)
    x_test, y_test = create_dataset(dataset_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # LSTM model
    # completely redefined the add the number of layer starting here
    # Allows channing the number of layers written as an input to the function train_model
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.15))
    
    if (num_layers > 2):
        for x in range(int((num_layers-2))):
            model.add(LSTM(units=100,return_sequences=True))
            model.add(Dropout(dropout))
        for x in range(1):
            model.add(LSTM(units=100))
            model.add(Dropout(dropout))
        model.add(Dense(units=1))
    else:
        model.add(LSTM(units=100))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        
    # until here

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #added the epoch and batch sizes being a function of the input parameters of the traiing data
    history=model.fit(x_train, y_train, epochs = num_epoch, batch_size =num_batch)
    model.save('stock_prediction.' + filename.split('.txt')[0])
    model = load_model('stock_prediction.' + filename.split('.txt')[0])
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1,1))
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = model.predict(x_test)
    train_max=dataset_test0.max()
    train_min=dataset_test0.min()
    # Rescale the data back to the original scale
    y_test = y_test*(train_max - train_min) + train_min
    y_pred = y_pred*(train_max - train_min) + train_min
    y_train = y_train*(train_max - train_min) + train_min

    # Plotting the results
    #changed data plotted to mimic the last 50 days
    fig1, ax1 = plt.subplots()
    ax1.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test.flatten(), marker='.', label="true")
    ax1.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred.flatten(), 'r', marker='.', label="prediction")
    ax1.plot(np.arange(0, len(y_train)), y_train.flatten(), 'g', marker='.', label="history")
    ax1.legend()
    fig1.savefig('{}_num6'.format(filename.split('.csv')[0]))


# main function to run training fully written by use
# test is to determine which method of getting data: 0 is for number of data using os list ; 1 is for pre-defined list
# datasets is the dataset inputed into the training section
# quanitiy is the number of listed os directory stocks needed to be run
# num of epoch is the number of epoch we wish to run
# num of batches is the number of batches we wish to run
# num of layer is the number of layer we wish to run
# dropout is the value of droupouts we wish to run
def training(test,datasets,quantity,num_epoch,num_batch,num_layers,dropout):
    if test == 0:
        for i in range(4):
            print('Trained using ' + datasets[i])
            train_model(datasets[i])
    if test == 1:
        for i in datasets:
            print('Trained using ' + i)
            train_model(i,num_epoch,num_batch,num_layers,dropout)
datasets = dataset_gathering(1,0,['archive/stocks/ANTE.csv'])

training(1,datasets,0,50,32,4,.15)
