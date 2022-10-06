import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import glob
import os


def difference(x,y):
    d = np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2)
    D = sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    return D-d

def distance(carX1,carY1,carX2,carY2):
    return np.sqrt((carX1 - carX2)**2 + (carY1 - carY2)**2)

def getLaneCenter(LC,car,av_x,av_y):
    '''
    find related lane center
    '''
    lineID1 = []
    lx1 = [[av_x[0],av_y[0]]]
    print(len(av_y))
    for id in LC:
        LCinfo = LC[id]
        lx = LCinfo['x'].to_numpy()
        ly = LCinfo['y'].to_numpy()
        if len(av_y) >0:
            df1 = np.sqrt((lx - av_x[0]) ** 2 + (ly - av_y[0]) ** 2)
            if min(df1)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_y) >41:
            df2 = np.sqrt((lx - av_x[40]) ** 2 + (ly - av_y[40]) ** 2)
            if min(df2)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_y) >71:
            df6 = np.sqrt((lx - av_x[70]) ** 2 + (ly - av_y[70]) ** 2)
            if min(df6)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_y) >101:
            df3 = np.sqrt((lx - av_x[100]) ** 2 + (ly - av_y[100]) ** 2)
            if min(df3)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_y) >1301:
            df7 = np.sqrt((lx - av_x[130]) ** 2 + (ly - av_y[130]) ** 2)
            if min(df7)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_y) >151:
            df4 = np.sqrt((lx - av_x[150]) ** 2 + (ly - av_y[150]) ** 2)
            if min(df4)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_y) >191:
            df5 = np.sqrt((lx - av_x[189]) ** 2 + (ly - av_y[189]) ** 2)
            if min(df5)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)
        if len(av_x)-1:
            df8 = np.sqrt((lx - av_x[len(av_x)-1]) ** 2 + (ly - av_y[len(av_y)-1]) ** 2)
            if min(df8)<1.5 and difference(lx,ly) < 1 and not id in lineID1:
                lineID1.append(id)

    #print(len(av_y))
    lineID = []
    for w in lineID1:
        LC_x = LC[w]['x'].to_numpy()
        LC_y = LC[w]['y'].to_numpy()
        x = []
        for z in range(len(LC_x)):
            df = np.sqrt((LC_x[z] - av_x) ** 2 + (LC_y[z] - av_y) ** 2)
            if min(df) < 1.5:
                x.append(LC_x[z])
        #print(w)
        #print(len(x))
        if len(x) > 35:
            lineID.append(w)
            
    # find related car
    carID = []
    for q in lineID:
        LC_x = LC[q]['x'].to_numpy()
        LC_y = LC[q]['y'].to_numpy()
        #print(len(LC_x))
        for obj_id in car:
            #if obj_id != 511:
                #continue
            hcarinfo = car[obj_id]
            x = hcarinfo['center_x'].to_numpy()
            y = hcarinfo['center_y'].to_numpy()
            leng = len(LC_x)-len(x)
            #print(len(x))
            if len(LC_x) > len(x):
                for j in range(leng):
                    hmd = np.sqrt((LC_x[j:len(x)+j] - x) ** 2 + (LC_y[j:len(y)+j] - y) ** 2)
                    #print(min(hmd),difference(x,y))
                    if min(hmd) < 1.5 and difference(x,y)<1.5 and not obj_id in carID:
                        carID.append(obj_id)
            else:
                if id in carID:
                    continue
                for k in range(len(x)):
                    hmd = np.sqrt((LC_x - x[k]) ** 2 + (LC_y - y[k]) ** 2)
                    #print(min(hmd),difference(x,y))
                    if min(hmd) < 1.5 and difference(x,y)<1.5 and not obj_id in carID:
                        carID.append(obj_id)
                        
            
    return carID,lineID
    
def delete(list):
    for j in list:
        del j[1]
    return list



def check_LaneCenter_point(LC,lineID,car,carID):
    '''
    check if cars did lane change
    '''
    lcID = []
    lctime = []
    JoinCar = []
    JCTime = []
    lx = []
    ly = []
    for h in range(len(lineID)):
        LCinfo = LC[lineID[h]]
        lx1 = LCinfo['x'].to_list()
        ly1 = LCinfo['y'].to_list()
        lx+=lx1
        ly+=ly1
    #print(len(lx))

    for q in carID:
        #q = 5
        x = car[q]['center_x'].to_list()
        y = car[q]['center_y'].to_list()
        time = car[q]['time'].to_list()
        dm = []
        for i in range(len(x)):
            d = []
            for j in range(len(lx)):
                d.append(np.sqrt((lx[j] - x[i]) ** 2 + (ly[j] - y[i]) ** 2))
            dmin = min(d)
            dm.append(dmin)
        
        #print(dm)
        for g,item in enumerate(dm):
            if item < 1.2:
                index = g
                break
                
        res = any(item < 1 for item in dm)
        
        if res == True:
            result_list = [dm[f] for f in range(index,len(dm)) if dm[f] > 1.5 and dm[f] < 4]
            Joincar_time = time[index]
            JoinCar.append(q)
            JCTime.append(Joincar_time)
        else:
            lcID.append(q)
            lctime.append(time[0])
            continue
        if len(result_list)>10:
            index_time_point = dm.index(result_list[0])
            time_point = time[index_time_point]
            lcID.append(q)
            lctime.append(time_point)
    return lcID,lctime,JoinCar,JCTime
    
def findLDorFLX(av_time,car,av_x,av_y,lcID,lctime,JoinCar,JCTime):
    '''
    identify leader car or follow car
    '''
    leaderID = []
    followID = []
    glbLc= []
    glbFlc = []
    det = []
    for t in range(len(av_time)):
        if t > len(av_time):
            break
        follow_car = []
        leader_car = []
        for jctime,JoinCarID in zip(JCTime,JoinCar):
            if JoinCarID in det:
                continue
            carX = car[JoinCarID]['center_x'].to_list()
            carY = car[JoinCarID]['center_y'].to_list()
            car_time = car[JoinCarID]['time'].to_list()
            index_car_time = car_time.index(jctime)
            if av_time[t] in car_time[index_car_time:len(car_time)]:
                if JoinCarID in lcID:
                    index_LCID = lcID.index(JoinCarID)
                    if lctime[index_LCID] == av_time[t]:
                        if not JoinCarID in det:
                            det.append(JoinCarID)
                        continue
                index_car = car_time.index(av_time[t])
                if av_x[0]-av_x[1] < 0:
                    if av_x[t] > carX[index_car]:
                        follow_car.append([JoinCarID,abs(carX[index_car] - av_x[t])])
                        if not JoinCarID in followID:
                            followID.append(JoinCarID)
                    else:
                        leader_car.append([JoinCarID,abs(carX[index_car] - av_x[t])])
                        if not JoinCarID in leaderID:
                            leaderID.append(JoinCarID)
                else:
                    if av_x[t] < carX[index_car]:
                        follow_car.append([JoinCarID,abs(carX[index_car] - av_x[t])])
                        if not JoinCarID in followID:
                            followID.append(JoinCarID)
                    else:
                        leader_car.append([JoinCarID,abs(carX[index_car] - av_x[t])])
                        if not JoinCarID in leaderID:
                            leaderID.append(JoinCarID)
                        
        
        fc = sorted(follow_car,key=lambda x: x[1])
        lc = sorted(leader_car,key=lambda x: x[1])
        for j in fc:
            del j[1]
        for j in lc:
            del j[1]
        glbLc.append(lc)
        glbFlc.append(fc)
        
    return glbLc,glbFlc,leaderID,followID

def findLDorFLY(av_time,car,av_x,av_y,lcID,lctime,JoinCar,JCTime):
    '''
    identify leader car or follow car
    '''
    leaderID = []
    followID = []
    glbLc= []
    glbFlc = []
    det = []
    av_time = av_time.tolist()
    for t in range(len(av_time)):
        if t > len(av_time):
            break
        follow_car = []
        leader_car = []
        for jctime,JoinCarID in zip(JCTime,JoinCar):
            if JoinCarID in det:
                continue
            carX = car[JoinCarID]['center_x'].to_list()
            carY = car[JoinCarID]['center_y'].to_list()
            car_time = car[JoinCarID]['time'].to_list()
            index_car_time = car_time.index(jctime)
            if av_time[t] in car_time[index_car_time:len(car_time)]:
                if JoinCarID in lcID:
                    index_LCID = lcID.index(JoinCarID)
                    if lctime[index_LCID] == av_time[t]:
                        det.append(JoinCarID)
                        continue
                index_car = car_time.index(av_time[t])
                if av_y[0]-av_y[1] < 0:
                    #print('3')
                    if av_y[t] > carY[index_car]:
                        follow_car.append([JoinCarID,abs(carY[index_car] - av_y[t])])
                        if not JoinCarID in followID:
                            followID.append(JoinCarID)
                    else:
                        #print('leader_car: '+ str(carID[q]))
                        leader_car.append([JoinCarID,abs(carY[index_car] - av_y[t])])
                        if not JoinCarID in leaderID:
                            leaderID.append(JoinCarID)
                else:
                    #print('4')
                    if av_y[t] < carY[index_car]:
                        follow_car.append([JoinCarID,abs(carY[index_car] - av_y[t])])
                        if not JoinCarID in followID:
                            followID.append(JoinCarID)
                    else:
                        leader_car.append([JoinCarID,abs(carY[index_car] - av_y[t])])
                        if not JoinCarID in leaderID:
                            leaderID.append(JoinCarID)
                    
            
        fc = sorted(follow_car,key=lambda x: x[1])
        lc = sorted(leader_car,key=lambda x: x[1])
        for j in fc:
            del j[1]
        for j in lc:
            del j[1]
        glbLc.append(lc)
        glbFlc.append(fc)
        
        
    return glbLc,glbFlc,leaderID,followID
        

def trackcars(leaderID,ldc,ldav):
    #track all cars
    pair = []
    Final_hc = []
    if len(leaderID) < 3:
        return None,None
    for i in range(len(leaderID)):
        if leaderID[i] == ldav:
            continue
        LD = [leaderID[i]]
        lead_car = []
        #print(leaderID[i])
        for j in range(len(ldc)):
            if LD in ldc[j]:
                index_car = ldc[j].index(LD)
                if index_car +1 != len(ldc[j]):
                    lead_car.append(ldc[j][index_car+1][0])
        #print(lead_car)
        items = Counter(lead_car).keys()
        if len(items) == 1 and len(lead_car) >= 150:
            pair.append([leaderID[i],lead_car[0]])
            if not leaderID[i] in Final_hc:
                Final_hc.append(leaderID[i])
            if not lead_car[0] in Final_hc:
                Final_hc.append(lead_car[0])
    if not pair or not Final_hc:
        return None,None
    return pair, Final_hc

def trackav(av_time,ldc):
    #print(ldc)
    if len(ldc) > 150:
        counter = 0
        val_c = 0
        bool = True
        for t in range(len(av_time)):
            if t+1 == len(av_time) or t == len(ldc):
                break
            if not ldc[t]:
                counter +=1
                continue
            if not ldc[t+1]:
                continue
            if ldc[t][0][0] != ldc[t+1][0][0]:
                bool = False
                break
            ldav = ldc[t][0][0]
            val_c+=1
        print(bool,counter,val_c)
        if counter > 50:
            return False,0
        elif bool == False:
            return False,0
        elif val_c > 150:
            return True,ldav
        else:
            return False,0
    else:
        return False,0

def calculate_cv(car,index):
    '''
    each car’s own trajectory curvature
    '''
    x,y = car[index]['center_x'].to_numpy(),car[index]['center_y'].to_numpy()
    x_t = np.gradient(x)
    y_t = np.gradient(y)
    vel = np.array([[x_t[i], y_t[i]] for i in range(x_t.size)])
    speed = np.sqrt(x_t * x_t + y_t * y_t)
    tangent = np.array([1/speed] * 2).transpose() * vel
    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5

    return curvature_val
    

def calculate_cv_av(traj_av,av_x,av_y):

    x,y = av_x,av_y
    x_t = np.gradient(x)
    y_t = np.gradient(y)
    vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])
    speed = np.sqrt(x_t * x_t + y_t * y_t)
    tangent = np.array([1/speed] * 2).transpose() * vel
    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    traj_av['curvature_val'] = curvature_val
    
    return traj_av

def loadData(path,name,g,f):
    df1=pd.DataFrame()
    # create list of filename
    filenames = list()
    n_files = len(glob.glob1(os.getcwd() + path,"*.csv"))
    for i in range(g,f):
        file = 'file'+str(i)+name
        filenames.append(file)

    for idx, fname in enumerate(filenames):
        df  = load_dataset(os.getcwd() + path + fname)
        df1 = pd.concat([df1, df], ignore_index=True)
    return df1


def load_dataset(path):
    df = pd.read_csv(path)
    return df
    
def main(traj1,map1,scenid):
    final_data = pd.DataFrame([])
    mapdata = pd.DataFrame([])
    traj = traj1[traj1.scenario_id_num == scenid]
    del traj['Unnamed: 0']
    del traj['Unnamed: 0.1']
    map = map1[map1.scenario_id_num == scenid]
        
    traj_av = traj[traj.is_sdc == 1]
    
    #print(traj['object_id'].unique())
    
    
    av_x = traj_av['center_x'].to_numpy()
    av_y = traj_av['center_y'].to_numpy()
    
    LC = defaultdict()
    car = defaultdict()
        
    uniqueMapID = map.loc[map.Lane_name == 'LaneCenter','id'].unique()
    #mapIDcount = map.loc[map.Lane_name == 'LaneCenter','id'].nunique()

    hum_carID = traj.loc[traj.is_sdc == 0, 'object_id'].unique()
        
    for i,id in enumerate(uniqueMapID):
        LC[id] = map[map.id == id]
        
    for j,obj_id in enumerate(hum_carID):
        car[obj_id]=traj[traj.object_id == obj_id]
        
    #find related lane center
    carID,lineID = getLaneCenter(LC,car,av_x,av_y)
    '''
    plt.figure(figsize=(17, 9), dpi=80)
    #plt.scatter(map['x'], map['y'], c='c', s=0.5, label='LaneCenter')
    for q in range(len(lineID)):
        plt.scatter(LC[lineID[q]]['x'], LC[lineID[q]]['y'], c='b', s=1, label='LaneCenter')
    
    for w in carID:
        carX = car[w]['center_x'].to_numpy()
        carY = car[w]['center_y'].to_numpy()
        plt.scatter(carX[0], carY[0], c='k', s=3, label='av')
        #plt.scatter(carX[-1], carY[-1], c='g', s=3, label='av')
        plt.annotate(w,(carX[0], carY[0]),fontsize=7,color ='k', ha='center', va='center')
        #plt.annotate(w,(carX[-1], carY[-1]),fontsize=7,color ='k', ha='center', va='center')
    plt.scatter(av_x, av_y, c='red', s=0.5, label='av')
    #plt.scatter(av_x[0], av_y[0], c='b', s=3, label='av')
    plt.title('sid'+str(scenid))
    #plt.savefig('s2_trj/map'+str(scenid)+'.png')
    #plt.close()
    plt.show()
    '''
    av_time = traj_av['time'].to_numpy()
    
    print('CarID: ',carID)
    print('lineID: ',lineID)
    
    if carID:
        lcID,lctime,JoinCar,JCTime = check_LaneCenter_point(LC,lineID,car,carID)
        print('lane change car: ',lcID)
        print('new car for lane: ',JoinCar)
        
        # To the x axis
        if abs(av_x[-1]-av_x[0])>50:
            #print('1')
            ldc,flc,leaderID,followID = findLDorFLX(av_time,car,av_x,av_y,lcID,lctime,JoinCar,JCTime)
        # To the y axis
        else:
            #print('2')
            ldc,flc,leaderID,followID = findLDorFLY(av_time,car,av_x,av_y,lcID,lctime,JoinCar,JCTime)
        
        print('leaderID: ',leaderID)
        print('followID: ',followID)
        #print(ldc)
        #print(flc)
        if leaderID:
            useful,ldav = trackav(av_time,ldc)
            print('AV Leader: ',ldav)
            print('useful data: ',useful)
            if useful == True:
                plt.figure(figsize=(17, 9), dpi=80)
                plt.scatter(map['x'], map['y'], c='c', s=0.5, label='LaneCenter')
                #for q in range(len(lineID)):
                    #plt.scatter(LC[lineID[q]]['x'], LC[lineID[q]]['y'], c='b', s=1, label='LaneCenter')
                for w in JoinCar:
                    carX = car[w]['center_x'].to_numpy()
                    carY = car[w]['center_y'].to_numpy()
                    plt.scatter(carX[0], carY[0], c='k', s=3, label='av')
                    #plt.scatter(carX[-1], carY[-1], c='g', s=3, label='av')
                    plt.annotate(w,(carX[0], carY[0]),fontsize=7,color ='k', ha='center', va='center')
                    #plt.annotate(w,(carX[-1], carY[-1]),fontsize=7,color ='k', ha='center', va='center')
                plt.scatter(av_x, av_y, c='red', s=0.5, label='av')
                #plt.scatter(av_x[0], av_y[0], c='b', s=3, label='av')
                plt.title('sid'+str(scenid))
                plt.savefig('picture/'+str(scenid)+'.png')
                plt.close()
                #plt.show()
                leaderPair,Final_useful_leadercar = trackcars(leaderID,ldc,ldav)
                followPair, Final_useful_followcar = trackcars(followID,flc,ldav)
                
                print(Final_useful_leadercar,Final_useful_followcar)
                
                if Final_useful_leadercar:
                    Final_useful_leadercar.reverse()
                traj_av = calculate_cv_av(traj_av,av_x,av_y)
                
                if Final_useful_leadercar:
                    e = 0
                    for index in Final_useful_leadercar:
                        cv = calculate_cv(car,index)
                        list = [abs(e-len(Final_useful_leadercar)-1) for a in range(len(car[index]))]
                        #car[index]['newIndex'] = list
                        data = pd.DataFrame.from_dict(car[index])
                        data['curvature_val'] = cv
                        data['newIndex'] = list
                        final_data = final_data.append(data)
                        e+=1
                cv = calculate_cv(car,ldav)
                list = [1 for a in range(len(car[ldav]))]
                data = pd.DataFrame.from_dict(car[ldav])
                data['curvature_val'] = cv
                data['newIndex'] = list
                final_data = final_data.append(data)
                
                list = [0 for a in range(len(traj_av))]
                traj_av['newIndex'] = list
                final_data = final_data.append(traj_av)
                mapdata = mapdata.append(map)
                
                if Final_useful_followcar:
                    e = -1
                    for index in Final_useful_followcar:
                        cv = calculate_cv(car,index)
                        list = [e for a in range(len(car[index]))]
                        #car[index]['newIndex'] = list
                        data = pd.DataFrame.from_dict(car[index])
                        data['curvature_val'] = cv
                        data['newIndex'] = list
                        final_data = final_data.append(data)
                        e-=1
    else:
        print('PASS')
    
    return final_data,mapdata
   
    #print(lineID)
    
if __name__ == '__main__':
 
    g = 85
    f = 86
    for l in range(85,100):
        path = "/s1_traj/"
        traj = loadData(path,'_traj.csv',g,f)
        
        path = "/s1_map/"
        map = loadData(path,'_map.csv',g,f)
        
        #Filename_traj = 's1_traj/file0_traj.csv'
        #Filename_map = 's1_map/file0_map.csv'
            
        #traj = pd.read_csv(Filename_traj)
        #map = pd.read_csv(Filename_map)
        
        #df = pd.DataFrame([])
        #mappp = pd.DataFrame([])
        
       
        #sid = traj.scenario_id_num.unique()
        #print(len(sid))
        '''
        for scenid in sid:
            if scenid == 58 or scenid == 727 or scenid == 651 or scenid == 458 or scenid == 462:
                continue
            print(scenid)
            #main(traj,map,scenid)
            dfdata,mapdata = main(traj,map,scenid)
            #df = df.append(dfdata)
            #mappp = mappp.append(mapdata)
        '''
        df = pd.DataFrame([])
        mappp = pd.DataFrame([])
        
        sid = traj.scenario_id_num.unique()
        #print(len(sid))
        #sid = [5531]

        for scenid in sid:
            print(scenid)
            #main(traj,map,scenid)
            dfdata,mapdata = main(traj,map,scenid)
            df = df.append(dfdata)
            mappp = mappp.append(mapdata)

        #df,mapdata = main(traj,map,3)
        #df1.to_csv('s2_trj/3.csv',index = False)
        df.to_csv('s2car_new/trj'+str(l)+'.csv',index = False)
        mappp.to_csv('s2map_new/map'+str(l)+'.csv',index = False)
        g+=1
        f+=1
    
    
    
    
