import numpy as np
import pandas as pd
import os
import time
# from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from tensorflow.keras import regularizers
# from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def mat_sum(mat1,mat2,mat3,mat4,mat5,mat6,mat7):
    mat1 = mat1.sum(axis = 1)
    mat2 = mat2.sum(axis = 1)
    mat3 = mat3.sum(axis = 1)
    mat4 = mat4.sum(axis = 1)
    mat5 = mat5.sum(axis = 1)
    mat6 = mat6.sum(axis = 1)
    mat7 = mat7.sum(axis = 1)
    return mat1,mat2,mat3,mat4,mat5,mat6,mat7

def mat_sumT(mat1,mat2,mat3,mat4,mat5,mat6,mat7):
    mat1 = mat1.sum(axis = 0)
    mat2 = mat2.sum(axis = 0)
    mat3 = mat3.sum(axis = 0)
    mat4 = mat4.sum(axis = 0)
    mat5 = mat5.sum(axis = 0)
    mat6 = mat6.sum(axis = 0)
    mat7 = mat7.sum(axis = 0)
    return mat1,mat2,mat3,mat4,mat5,mat6,mat7

def buildTrain(train, past=7, future=1 , hour = 24):#grid為每行的index
    X_train,Y_train = [],[]
    for i in range(int(np.round(len(train)/hour)-past-future)):
        X_train.append(train[i*hour:i*hour+past*hour])
        Y_train.append(train[i*hour+past*hour:i*hour+past*hour+future*hour])
    return X_train , Y_train

def judge(cpred,gpred):
    if gpred < 0:
        gpred = 0
    number = cpred - gpred
    if number > 0:#如果消耗大於生產
        action = 1
        value = number
    elif number == 0:
        action = 0
    else:
        action = -1
        value = abs(number)
    # value = np.round(value)
    return action , value
    
def buildModel(shape):
    # model = Sequential()
    # model.add(GRU(1,input_length=shape[1], input_dim=shape[2]))
    # model.add(Dropout(0.1))
    # model.add(Dense(1))
    model = Sequential()
    # model.add(GRU(units = 5, input_length=shape[1], input_dim=shape[2]))
    # model.add(Dropout(0.05))
    model.add(GRU(units = 5, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    model.add(Dropout(0.05))
    model.add(GRU(units = 5))
    model.add(Dropout(0.05))
    model.add(Dense(24))
    model.compile(loss='mse', optimizer='adam')
    return model

def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val

# def normalize(data):
#     x = (x-np.min(data))/(np.max(data)-np.min(data))
    
if __name__ == '__main__':
    path = './training_data'
    data_list = os.listdir(path)
    gen_ori, con_ori = [],[]
    Gen_data, Con_data, Gen_label, Con_label = [], [], [], []

    # min_max_scaler = preprocessing.MinMaxScaler() 
    
    for i in range(len(data_list)):
        data = pd.read_csv(os.path.join(path, data_list[i]), header=None)#[1:,1:]
        data = data.drop(labels = [0],axis = 1)
        data = data.drop(labels = [0],axis = 0)
        # data = normalize(data)
        data = np.array(data)[0:,0:]
        data = np.stack(data).astype(None)
        gen, con = data[:,0],data[:,1]

        gen_data, gen_label = buildTrain(gen,7,1,24)
        con_data, con_label = buildTrain(con,7,1,24)
        #training data========================================================
        Gen_data.append(gen_data)
        Gen_label.append(gen_label)
        Con_data.append(con_data)
        Con_label.append(con_label)

    Gen_data = np.vstack(Gen_data)
    Con_data = np.vstack(Con_data)
    #=======================以天為計算總消耗總產量==============================
    # G1,G2,G3,G4,G5,G6,G7 = np.split(Gen_data,7,axis=1)
    # C1,C2,C3,C4,C5,C6,C7 = np.split(Con_data,7,axis=1)
    # G1,G2,G3,G4,G5,G6,G7 = mat_sum(G1,G2,G3,G4,G5,G6,G7)
    # C1,C2,C3,C4,C5,C6,C7 = mat_sum(C1,C2,C3,C4,C5,C6,C7)
    # Gen_data = np.transpose(np.vstack((G1,G2,G3,G4,G5,G6,G7)))
    # Gen_data = Gen_data.resha
    # Con_data = np.transpose(np.vstack((C1,C2,C3,C4,C5,C6,C7)))
    
    
    #==========================================================================
    Gen_label = np.vstack(Gen_label)
    Con_label = np.vstack(Con_label)

    #label為總生產或總耗電量
    # Gen_label = np.sum(np.vstack(Gen_label), axis=-1)[:, np.newaxis]
    # Con_label = np.sum(np.vstack(Con_label), axis=-1)[:, np.newaxis]
    
    gen_merge = np.hstack((Gen_data,Gen_label))
    con_merge = np.hstack((Con_data,Con_label))
    #==================================normalize=========================
    Gscaler = StandardScaler().fit(gen_merge)   
    Cscaler = StandardScaler().fit(con_merge)
    # G_scaler= preprocessing.MinMaxScaler(feature_range=(0, 1))
    # G_scaled = G_scaler.fit_transform(gen_merge)
    # C_scaler= preprocessing.MinMaxScaler(feature_range=(0, 1))
    # C_scaled = C_scaler.fit_transform(con_merge)
    # gen_merge = normalize(gen_merge)
    Nor_gen_merge = Gscaler.transform(gen_merge)
    # gen_merge = scaler.fit_transform(Gen_data)
    # Gen_scaled = scaler.fit_transform(gen_merge)
    Nor_con_merge = Cscaler.transform(con_merge)
    # con_merge = scaler.fit_transform(Con_data)
    # con_merge = normalize(con_merge)
    #=========================================================================
    # nGen_data,nGen_label = np.hsplit(gen_merge,[7,])
    # nCon_data,nCon_label = np.hsplit(con_merge,[7,])
    nGen_data,nGen_label = np.hsplit(Nor_gen_merge,[168,])
    nCon_data,nCon_label = np.hsplit(Nor_con_merge,[168,])
    nGen_data = np.reshape(nGen_data, (nGen_data.shape[0], nGen_data.shape[1], 1))#2dim->3dim
    nCon_data = np.reshape(nCon_data, (nCon_data.shape[0], nCon_data.shape[1], 1))#2dim->3dim
    # G_data, G_label = shuffle(Gen_data, Gen_label)
    # C_data, C_label = shuffle(Con_data, Con_label)
    # GX_train, GY_train, GX_val, GY_val = splitData(nGen_data, nGen_label, 0.1)
    # CX_train, CY_train, CX_val, CY_val = splitData(nCon_data, nCon_label, 0.1)
    GX_train, GY_train, GX_val, GY_val = splitData(nGen_data, nGen_label, 0.1)
    CX_train, CY_train, CX_val, CY_val = splitData(nCon_data, nCon_label, 0.1)
    #================生產train==================
    # print(GX_train.shape)
    
    Gen_model = buildModel(GX_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    Gen_model.fit(GX_train, GY_train, epochs=10, batch_size=32, validation_data=(GX_val, GY_val), callbacks=[callback])
    Gen_model.save('GenModel.h5')
    #================消耗train==================
    # CX_train = np.reshape(CX_train, (CX_train.shape[0], CX_train.shape[1], 1))
    Con_model = buildModel(CX_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    Con_model.fit(CX_train, CY_train, epochs=10, batch_size=32, validation_data=(CX_val, CY_val), callbacks=[callback])
    Con_model.save('ConModel.h5')
    #===========================================
    