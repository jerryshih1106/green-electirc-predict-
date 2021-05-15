import model
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import os
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# You should not modify this part.
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


# def output(date,act,value,path,predict):
def output(date,path,Cpred,Gpred):
    import pandas as pd
    lastdate = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    date = lastdate + datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%d")
    data = []
    for i in range(24):
        if Gpred[i]<0:
            Gpred[i] = 0
        # print(C_pred[0][i])
        # print(G_pred[0][i])
        act,val = model.judge(C_pred[i],G_pred[i])
        if act == -1:
            text_data = [date+ " {:02d}".format(i)+":00:00", "sell", 2.2, val]
            data.append(text_data)
        if act == 1:
            if Gpred[i]>0:
                text_data = [date+ " {:02d}".format(i)+":00:00", "sell", 2.8, Gpred[i]]#若能賣得比台電高就是賺
                data.append(text_data)
            text_data1 = [date+ " {:02d}".format(i)+":00:00", "buy", 2.2, val]#比台電低的價格買缺少的電
            
            data.append(text_data1)
        else:
            text_data = [date+ " {:02d}".format(i)+":00:00", "buy", 0, 0]

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)
    return

def denorm(x,oridata):
    x = x * (np.max(oridata)-np.min(oridata))+np.min(oridata)
    return x
if __name__ == "__main__":
    args = config()
    
    G_model = keras.models.load_model('GenModel.h5')
    C_model = keras.models.load_model('ConModel.h5')
    #================================pre=========================================
    path = './training_data'
    data_list = os.listdir(path)
    gen_ori, con_ori = [],[]
    Gen_data, Con_data, Gen_label, Con_label = [], [], [], []
    for i in range(len(data_list)):
        data = pd.read_csv(os.path.join(path, data_list[i]), header=None)#[1:,1:]
        data = data.drop(labels = [0],axis = 1)
        data = data.drop(labels = [0],axis = 0)
        # data = normalize(data)
        data = np.array(data)[0:,0:]
        data = np.stack(data).astype(None)
        gen, con = data[:,0],data[:,1]
        #============================normalize================================
        # gen_n = normalize(gen)
        # con_n = normalize(con)
        gen_data, gen_label = model.buildTrain(gen,7,1,24)
        con_data, con_label = model.buildTrain(con,7,1,24)
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
    # G1,G2,G3,G4,G5,G6,G7 = model.mat_sum(G1,G2,G3,G4,G5,G6,G7)
    # C1,C2,C3,C4,C5,C6,C7 = model.mat_sum(C1,C2,C3,C4,C5,C6,C7)
    # Gen_data = np.transpose(np.vstack((G1,G2,G3,G4,G5,G6,G7)))
    # # Gen_data = Gen_data.resha
    # Con_data = np.transpose(np.vstack((C1,C2,C3,C4,C5,C6,C7)))
    #==========================================================================
    Gen_label = np.vstack(Gen_label)
    Con_label = np.vstack(Con_label)
    #label為總生產或總耗電量
    # Gen_label = np.sum(np.vstack(Gen_label), axis=-1)[:, np.newaxis]
    # Con_label = np.sum(np.vstack(Con_label), axis=-1)[:, np.newaxis]
    
    gen_merge = np.hstack((Gen_data,Gen_label))
    con_merge = np.hstack((Con_data,Con_label))
    #===============================normalize=================================
    # G_scaler= preprocessing.MinMaxScaler(feature_range=(0, 1))
    # G_scaled = G_scaler.fit_transform(gen_merge)
    # C_scaler= preprocessing.MinMaxScaler(feature_range=(0, 1))
    # C_scaled = C_scaler.fit_transform(con_merge)
    Gscaler = StandardScaler()
    Ggen_merge = Gscaler.fit_transform(gen_merge)
    Cscaler = StandardScaler()
    Cgen_merge = Cscaler.fit_transform(con_merge)
        # gen_merge = normalize(gen_merge)
    # Ngen_merge = scaler.fit_transform(gen_merge)
    # gen_merge = scaler.fit_transform(Gen_data)
    # Ggen_merge = scaler.fit(gen_merge)
    # Gen_scaled = scaler.fit_transform(gen_merge)
    # Ncon_merge = scaler.fit_transform(con_merge)
    # con_merge = scaler.fit_transform(Con_data)
    # Ccon_merge = scaler.fit(con_merge)
    # con_merge = normalize(con_merge)
    # Gscaler = MinMaxScaler(feature_range = (0,1)).fit(gen_merge)
    # Cscaler = MinMaxScaler(feature_range = (0,1)).fit(con_merge) 
    #================================pre========================================
    #================================Gen每天總消耗產出==========================
    GValue = np.array(pd.read_csv(args.generation, header=None))[1:,1:]
    GValue = np.stack(GValue).astype(None)[:,0]
    # G1,G2,G3,G4,G5,G6,G7 = np.split(GValue,7,axis=0)
    # G1,G2,G3,G4,G5,G6,G7 = model.mat_sumT(G1,G2,G3,G4,G5,G6,G7)
    # G8 = 0
    # GValue = np.vstack((G1,G2,G3,G4,G5,G6,G7,G8))
    #================================Con每天總消耗產出==========================
    CValue = np.array(pd.read_csv(args.consumption, header=None))[1:,1:]
    CValue = np.stack(CValue).astype(None)[:,0] 
    # C1,C2,C3,C4,C5,C6,C7 = np.split(CValue,7,axis=0)
    # C1,C2,C3,C4,C5,C6,C7 = model.mat_sumT(C1,C2,C3,C4,C5,C6,C7)
    # C8 = 0
    # CValue = np.vstack((C1,C2,C3,C4,C5,C6,C7,C8))
    #================================Con=======================================
    date = np.array(pd.read_csv(args.consumption, header=None))[-1,0]
    GVdata, CVdata = GValue[np.newaxis, :], CValue[np.newaxis, :]
    GVlabel, CVlabel = GValue, CValue
    #===============================normalize=================================
    # gen_merg,con_merg = pre()
    # GVdata = np.reshape(GVdata,(1,8))
    # CVdata = np.reshape(CVdata,(1,8))
    Gzero = np.zeros([1,24])
    Czero = np.zeros([1,24])
    TGVdata = np.hstack((GVdata,Gzero))
    TCVdata = np.hstack((CVdata,Czero))
    GVdata = Gscaler.transform(TGVdata)
    CVdata = Cscaler.transform(TCVdata)
    # for i in range(24):
    GVdata = np.delete(GVdata, np.s_[168:192], axis=1)
    CVdata = np.delete(CVdata, np.s_[168:192], axis=1)
    # GVdata = np.reshape(GVdata,(1,7,1))
    # CVdata = np.reshape(CVdata,(1,7,1))
    GVdata = np.reshape(GVdata,(1,168,1))
    CVdata = np.reshape(CVdata,(1,168,1))
    #=========================================================================
    Gpred = G_model.predict(GVdata)
    Cpred = C_model.predict(CVdata)
    #================================denorm====================================
    # G_pred = denorm(Gpred,gen_merge)
    Gzero = np.zeros([1,168])
    Czero = np.zeros([1,168])
    Gpred = np.hstack((Gpred,Gzero))
    Cpred = np.hstack((Cpred,Czero))
    # Gpred = np.reshape(Gpred,(24))
    # Cpred = np.reshape(Cpred,(24))
    G_pred = Gscaler.inverse_transform(Gpred)
    # C_pred = denorm(Cpred,con_merge)
    C_pred = Cscaler.inverse_transform(Cpred)
    G_pred = np.delete(G_pred, np.s_[24:192], axis=1)
    C_pred = np.delete(C_pred, np.s_[24:192], axis=1)
    G_pred = np.reshape(G_pred,(24))
    C_pred = np.reshape(C_pred,(24))
    # #======================================================================
    #=================================判斷===================================
    # print(G_pred)
    # action,value = model.judge(C_pred[0],G_pred[0])
    # print(value[0])
    output(date,args.output,C_pred,G_pred)