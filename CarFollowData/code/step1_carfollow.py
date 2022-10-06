import pandas as pd
import numpy as np
import csv
import glob
import os

def noIntersec(sign_num,traj_num,map_num):
    traj_num['rowIndex'] = np.arange(0,len(traj_num),1)
    avtraj = traj_num[traj_num.is_sdc == 1]
    u,index = np.unique(sign_num['x'],return_index=True)
    siginalCdx, siginalCdy= sign_num.iloc[index]['x'],sign_num.iloc[index]['y']
    stopSignCdx, stopSignCdy= map_num['x'][map_num['Lane_name']=='StopSign'],map_num['y'][map_num['Lane_name']=='StopSign']
    allx=siginalCdx.append(stopSignCdx)
    ally = siginalCdy.append(stopSignCdy)
    
    x = np.array(traj_num['center_x'])
    y = np.array(traj_num['center_y'])
    
    r=13
    for i in range(len(allx)):

        h = allx.iloc[i]
        k = ally.iloc[i]
        m = ((x-h) **2 + (y-k) **2).tolist()
        p = [x for x, y in enumerate(m) if y <= r**2]
        asp = traj_num[traj_num['rowIndex'].isin(p)].index
        traj_num = traj_num.drop(asp)
        
    return traj_num
    

def roadline(RLX, RLY, av_X, av_Y):
    for x,y in zip(RLX,RLY):
        p = abs(y - av_Y)
        px = abs(x - av_X)
        for i,j in zip(p,px):
            if i < 0.15 and j < 0.15:
                return True
                break


def check_lineCenter_point(LX,LY,av_X,av_Y):
    dx= []
    dy = []
    for i in range(len(av_X)):
        dmin = 10000
        for j in range(len(LX)):
            d = np.sqrt((LX[j] - av_X[i]) ** 2 + (LY[j] - av_Y[i]) ** 2)
            #print(d)
            if dmin > d:
                dmin = d
                x = LX[j]
                y = LY[j]
        #d1.append(dmin)
        dx.append(x)
        dy.append(y)
    return dx,dy

def calcDiff(x,y):
    dic = max(np.sqrt((np.diff(x)) ** 2 + (np.diff(y)) ** 2))
    return dic


def difference(av_X,av_Y):
    d = np.sqrt((av_X[0] - av_X[-1]) ** 2 + (av_Y[0] - av_Y[-1]) ** 2)
    D = sum(np.sqrt(np.diff(av_X) ** 2 + np.diff(av_Y) ** 2))
    return D-d

#load data
def loadData(path,name,sfile_num,efile_num):
    df1=pd.DataFrame()
    # create list of filename
    filenames = list()
    n_files = len(glob.glob1(os.getcwd() + path,"*.csv"))
    for i in range(sfile_num,efile_num):
        file = 'file'+str(i)+name
        filenames.append(file)

    for idx, fname in enumerate(filenames):
        df  = load_dataset(os.getcwd() + path + fname)
        df1 = pd.concat([df1, df], ignore_index=True)
    p = df1.iloc[0]['scenario_id_num']
    s = df1.iloc[-1]['scenario_id_num']
    return df1,s,p

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def main(l,sfile_num,efile_num):
    path = "/traj/"
    traj,s,p = loadData(path,'_traj.csv',sfile_num,efile_num)
    
    path = "/map/"
    map,s,p = loadData(path,'_maps.csv',sfile_num,efile_num)
    
    path = "/sign/"
    sign,s,p = loadData(path,'_signs.csv',sfile_num,efile_num)
    
    
    
    file_av = pd.DataFrame([])
    file_map = pd.DataFrame([])
    file_sign = pd.DataFrame([])
    
    for sid in range(p,s+1):
        #if p+100 ==sid:
            #break
        #print(sid)
        map_num = map[map.scenario_id_num == sid]
        traj_num = traj[traj.scenario_id_num == sid]
        ax,ay = traj_num.loc[traj_num.is_sdc == 1,'center_x'].to_numpy(), traj_num.loc[traj_num.is_sdc == 1,'center_y'].to_numpy()
        sign_num = sign[sign.scenario_id_num == sid]
        
        #delete intersection points
        if len(sign_num) != 0:
            traj_num = noIntersec(sign_num,traj_num,map_num)
        
        traj_num = traj_num[traj_num.is_sdc == 1]
        #after delete intersec points, no traj left
        if len(traj_num) < 2:
            continue
            
        RLX = map_num[map_num['Lane_name']=='RoadLine']['x'].to_numpy()
        RLY = map_num[map_num['Lane_name']=='RoadLine']['y'].to_numpy()
    
        av_X = traj_num['center_x'].to_numpy()
        av_Y = traj_num['center_y'].to_numpy()
        
        # pass LC case
        if roadline(RLX,RLY,av_X,av_Y):
            continue
        
        # pass turning case
        diff = difference(av_X,av_Y)
        if diff > 1.2:
            continue
            
        diff = difference(ax,ay)
        if diff > 0.6:
            continue
        
        LX = map_num[map_num['Lane_name']=='LaneCenter']['x'].to_numpy()
        LY = map_num[map_num['Lane_name']=='LaneCenter']['y'].to_numpy()
        
        # pass stop av case
        x,y = check_lineCenter_point(LX,LY,av_X,av_Y)
        maxdiff = calcDiff(x,y)
        if  maxdiff < 0.2:
            continue
            
        file_av = file_av.append(traj.loc[traj.scenario_id_num == sid])
        file_map = file_map.append(map.loc[map.scenario_id_num == sid])
        file_sign = file_sign.append(sign.loc[sign.scenario_id_num == sid])
        print(sid)
    
    file_av.to_csv('s1_traj/traj'+str(l)+'.csv')
    file_map.to_csv('s1_map/map'+str(l)+'.csv')
    file_sign.to_csv('s1_sign/sign'+str(l)+'.csv')
    
if __name__ == '__main__':
    #make a folder
    #change your video folder name or path here
    datapath = '{}'.format('s1_traj')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    datapath = '{}'.format('s1_map')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    datapath = '{}'.format('s1_sign')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    
    #change your first file number and last file number(for example, if you want to scan file1_traj until file2_traj, sfile_num will equal to 1, and efile_num will equal to 2)
    sfile_num= 1
    efile_num = 2
    for l in range(1,2):
        main(l,sfile_num,efile_num)
        sfile_num+=1
        efile_num+=1
