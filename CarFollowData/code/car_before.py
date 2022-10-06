import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
import glob
import os
import random

def find_av_lane(avID,car,LC,LCID,LCx,LCy):
    '''
    find av lane ID
    '''
    avx,avy = car[avID]['center_x'].to_numpy(),car[avID]['center_y'].to_numpy()
    
    nearx = []
    neary = []
    dnmin = []
    for x,y in zip(avx,avy):
        d = []
        for lx,ly in zip(LCx,LCy):
            d.append(np.sqrt((lx - x) ** 2 + (ly - y) ** 2))
        dmin = min(d)
        dnmin.append(dmin)
        index = d.index(dmin)
        nearx.append(LCx[index])
        neary.append(LCy[index])
    
    LaneID = []
    for nx, ny in zip(nearx,neary):
        for lid in LCID:
            lx,ly = LC[lid]['x'].to_numpy(),LC[lid]['y'].to_numpy()
            if nx in lx and ny in ly:
                if not lid in LaneID:
                    LaneID.append(lid)

    LaneList = []
    for lid in LaneID:
        lx,ly = LC[lid]['x'].to_numpy(),LC[lid]['y'].to_numpy()
        cont = 0
        for nx,ny in zip(nearx,neary):
            for lcx,lcy in zip(lx,ly):
                if nx == lcx and ny ==lcy:
                    cont+=1
        LaneList.append([lid,cont])
    print('LaneList: ',LaneList)
    avLaneID = []
    for nlid in LaneList:
        if nlid[1] > 15:
            avLaneID.append(nlid[0])
            
    return avLaneID

def find_car_lane(LC,LCID,LCx,LCy,avx,avy):
    '''
    find av lane ID
    '''
    d = []
    for lx,ly in zip(LCx,LCy):
        d.append(np.sqrt((lx - avx) ** 2 + (ly - avy) ** 2))
    dmin = min(d)
    #print(min(d))
    index = d.index(dmin)
    nearx = LCx[index]
    neary=LCy[index]

    LaneID = []
    for lid in LCID:
        lx,ly = LC[lid]['x'].to_numpy(),LC[lid]['y'].to_numpy()
        if nearx in lx and neary in ly:
            if not lid in LaneID:
                LaneID.append(lid)
    
    return LaneID

def find_car_on_lane(LaneID,car,carID,avID,LC):
    rCar = []
    frCar = []
    erCar = []
    dn = []
    #lx = []
    #ly = []
    for i in range(len(LaneID)):
        if i == 0:
            lx = LC[LaneID[i]]['x'].to_numpy()
            ly = LC[LaneID[i]]['y'].to_numpy()
        if i > 0:
            lx2 = LC[LaneID[i]]['x'].to_numpy()
            ly2 = LC[LaneID[i]]['y'].to_numpy()
            lx = np.append(lx, lx2)
            ly = np.append(ly, ly2)
            
    #lx,ly = LC[LaneID]['x'],LC[LaneID]['y']
    
    for id in carID:
        if id == avID:
            continue
        x,y = car[id]['center_x'].to_numpy(),car[id]['center_y'].to_numpy()
        fd = np.sqrt((lx - x[0]) ** 2 + (ly - y[0]) ** 2)
        if min(fd) < 1.5:
            frCar.append(id)
        ed = np.sqrt((lx - x[-1]) ** 2 + (ly - y[-1]) ** 2)
        if min(ed) < 1.5:
            erCar.append(id)
        for cx,cy in zip(x,y):
            d = np.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            if min(d) < 1.5:
                rCar.append(id)
                break
    return rCar,frCar,erCar

def find_related_car_at_first_end(frCar,erCar,car,LCID,LC,LCx,LCy):
    #find related car at first s
    StartLaneID = []
    EndLaneID = []
    for fid in frCar:
        avx,avy = car[fid]['center_x'].to_numpy(),car[fid]['center_y'].to_numpy()
        StartLaneID.append(find_car_lane(LC,LCID,LCx,LCy,avx[0],avy[0]))
    for eid in erCar:
        avx,avy = car[eid]['center_x'].to_numpy(),car[eid]['center_y'].to_numpy()
        EndLaneID.append(find_car_lane(LC,LCID,LCx,LCy,avx[-1],avy[-1]))
    
    return StartLaneID,EndLaneID
    
def find_LC_time(car,rCar,LC,relatedLane):
    LCTimeID = []
    dn = []
    for i in range(len(relatedLane)):
        if i == 0:
            lx = LC[relatedLane[i]]['x'].to_numpy()
            ly = LC[relatedLane[i]]['y'].to_numpy()
        if i > 0:
            lx2 = LC[relatedLane[i]]['x'].to_numpy()
            ly2 = LC[relatedLane[i]]['y'].to_numpy()
            lx = np.append(lx, lx2)
            ly = np.append(ly, ly2)
    for id in rCar:
        dmin = []
        time = car[id]['time'].to_list()
        x,y = car[id]['center_x'].to_numpy(),car[id]['center_y'].to_numpy()
        for cx,cy in zip(x,y):
            d = np.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            dmin.append(min(d))
        #print(id)
        #print(dmin)
        list = [nd for nd in dmin if nd < 1.2]
        if len(list) != 0:
            indexS = dmin.index(list[0])
            indexE = dmin.index(list[-1])
            t1,t2 = time[indexS],time[indexE]
            LCTimeID.append([id,t1,t2])
    return LCTimeID
        

def plot(LaneID,LC,car,LCTimeID,scenid,LCx,LCy,av_x,av_y,avID):
    color = []
    for k in range(len(LCTimeID)):
        color.append(["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])])
    plt.figure(figsize=(17, 9), dpi=80)
    plt.scatter(LCx,LCy, c='c', s=1)
    #for id in LaneID:
        #plt.plot(LC[id]['x'], LC[id]['y'], c='c')
    for LCid,col in zip(LCTimeID,color):
        ctime = car[LCid[0]]['time'].to_list()
        tsindex = ctime.index(LCid[1])
        teindex = ctime.index(LCid[2])
        x = car[LCid[0]]['center_x'].to_list()
        y = car[LCid[0]]['center_y'].to_list()
        plt.scatter(x[tsindex:teindex], y[tsindex:teindex], c=col, s=1,label = str(LCid[0]))
        
    plt.scatter(av_x, av_y, c='r', s=1,label = str(avID[0]))
    
    plt.legend()
    plt.title('Sid'+str(scenid))
    #plt.show()
    plt.savefig('CF_pic/Sid'+str(scenid)+'.png')
    plt.close()
'''
def plot_lane(LaneID,LC,scenid):
    color = []
    for k in range(len(LaneID)):
        color.append(["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])])
    plt.figure(figsize=(17, 9), dpi=80)
    for id,col in zip(LaneID,color):
        plt.scatter(LC[id]['x'], LC[id]['y'], c = col,s = 1, label = str(id))
    plt.legend()
    plt.title('Sid'+str(scenid))
    #plt.show()
    plt.savefig('new_picture/Sid'+str(scenid)+'_LaneID.png')
    plt.close()
'''
def find_subject_leader_y(car,nearAV,av_x,av_y,time,avID,lcID):

    targetx = av_x
    targety = av_y
    target_time = time
   
    leaderID = []
    followID = []
    glbLc= []
    glbFlc = []
    LCid = []
    
    for t in time:
        follow_car = []
        leader_car = []
        for LCTimeID in lcID:
            if LCTimeID[0] in LCid:
                continue
            carX1 = car[LCTimeID[0]]['center_x'].to_list()
            carY1 = car[LCTimeID[0]]['center_y'].to_list()
            carHeading = car[LCTimeID[0]]['heading'].to_list()
            car_time1 = car[LCTimeID[0]]['time'].to_list()
            index_car_time = car_time1.index(LCTimeID[1])
            index_car_time_leave = car_time1.index(LCTimeID[2])
            if t in target_time and t in car_time1[index_car_time:index_car_time_leave+1]:
                index_car = target_time.index(t)
                index_car1 = car_time1.index(t)
                if targety[-1]-targety[0] < 0:
                    if targety[index_car] < carY1[index_car1] and abs(carY1[index_car1] - targety[index_car]) > 7:
                        follow_car.append([LCTimeID[0],abs(carY1[index_car1] - targety[index_car])])
                        if not LCTimeID[0] in followID:
                            followID.append(LCTimeID[0])
                    else:
                        leader_car.append([LCTimeID[0],abs(carY1[index_car1] - targety[index_car])])
                        if not LCTimeID[0] in leaderID:
                            leaderID.append(LCTimeID[0])
                elif targety[-1]-targety[0] > 0 and abs(carY1[index_car1] - targety[index_car]) > 7:
                    if targety[index_car] > carY1[index_car1]:
                        follow_car.append([LCTimeID[0],abs(carY1[index_car1] - targety[index_car])])
                        if not LCTimeID[0] in followID:
                            followID.append(LCTimeID[0])
                    else:
                        leader_car.append([LCTimeID[0],abs(carY1[index_car1] - targety[index_car])])
                        if not LCTimeID[0] in leaderID:
                            leaderID.append(LCTimeID[0])

            
        fc = sorted(follow_car,key=lambda x: x[1])
        lc = sorted(leader_car,key=lambda x: x[1])
        for j in fc:
            del j[1]
        for j in lc:
            del j[1]
        glbLc.append(lc)
        glbFlc.append(fc)
  
    return glbLc,glbFlc,leaderID,followID,nearAV[0]

def find_subject_leader_x(car,nearAV,av_x,av_y,time,avID,lcID):

    targetx = av_x
    targety = av_y
    target_time = time

    leaderID = []
    followID = []
    glbLc= []
    glbFlc = []
    LCid = []
    for t in time:
        follow_car = []
        leader_car = []
        for LCTimeID in lcID:
            if LCTimeID[0] in LCid:
                continue
            carX1 = car[LCTimeID[0]]['center_x'].to_list()
            carY1 = car[LCTimeID[0]]['center_y'].to_list()
            carHeading = car[LCTimeID[0]]['heading'].to_list()
            car_time1 = car[LCTimeID[0]]['time'].to_list()
            index_car_time = car_time1.index(LCTimeID[1])
            index_car_time_leave = car_time1.index(LCTimeID[2])
            if t in target_time and t in car_time1[index_car_time:index_car_time_leave+1]:
                index_car = target_time.index(t)
                index_car1 = car_time1.index(t)
                if targetx[-1]-targetx[0] < 0 and abs(carX1[index_car1] - targetx[index_car]) > 7:
                    #print('1',t)
                    if targetx[index_car] < carX1[index_car1]:
                        follow_car.append([LCTimeID[0],abs(carX1[index_car1] - targetx[index_car])])
                        if not LCTimeID[0] in followID:
                            followID.append(LCTimeID[0])
                    else:
                        leader_car.append([LCTimeID[0],abs(carX1[index_car1] - targetx[index_car])])
                        if not LCTimeID[0] in leaderID:
                            leaderID.append(LCTimeID[0])
                            
                elif targetx[-1]-targetx[0] > 0 and abs(carX1[index_car1] - targetx[index_car]) > 7:
                    #print('2',t)
                    if targetx[index_car] > carX1[index_car1]:
                        follow_car.append([LCTimeID[0],abs(carX1[index_car1] - targetx[index_car])])
                        if not LCTimeID[0] in followID:
                            followID.append(LCTimeID[0])
                    else:
                        leader_car.append([LCTimeID[0],abs(carX1[index_car1] - targetx[index_car])])
                        if not LCTimeID[0] in leaderID:
                            leaderID.append(LCTimeID[0])
             
        fc = sorted(follow_car,key=lambda x: x[1])
        lc = sorted(leader_car,key=lambda x: x[1])
        for j in fc:
            del j[1]
        for j in lc:
            del j[1]
        #print(fc)
        glbLc.append(lc)
        glbFlc.append(fc)
        

    return glbLc,glbFlc,leaderID,followID,nearAV[0]
    


def find_pair(final_array,avID):
    lead_car0 = []
    lead_car100 = []
    follow_100 = []
    for j in range(len(final_array)):
        if avID[0] in final_array[j]:
            index_car = final_array[j].index(avID[0])
        if index_car != 0:
            if index_car-2 >-1:
                lead_car100.append([final_array[j][index_car-2],final_array[j][index_car-1]])
            elif index_car-2 < 0 and index_car-1 <= len(final_array[j])-1:
                lead_car100.append([None,final_array[j][index_car-1]])
            else:
                lead_car100.append([None,None])
        else:
            lead_car100.append([None,None])
        
        if index_car == 0:
            lead_car0.append([None,avID[0]])
        elif index_car-1 <= len(final_array[j])-1 and len(final_array[j]) != 1:
            lead_car0.append([final_array[j][index_car-1],avID[0]])
        else:
            lead_car0.append([None,avID[0]])
            
        if index_car+1 <= len(final_array[j])-1:
            follow_100.append([avID[0],final_array[j][index_car+1]])
        else:
            follow_100.append([None,None])
   
    return lead_car100,lead_car0,follow_100

def speed(speedx,speedy):
    speed1 = np.sqrt(speedx**2 + speedy ** 2)
    return speed1


def main(vec,map,scenid):
    trj = vec.loc[vec.scenario_id_num == scenid]
    mp = map.loc[map.scenario_id_num == scenid]
    pssdata = pd.DataFrame([])
    carID = trj.loc[trj.object_type_id == 1,'object_id'].unique()
    #create car dictionary
    car = defaultdict()
    for id in carID:
        car[id] = trj.loc[(trj.object_id == id)]
    LCID = mp.loc[mp.Lane_name == 'LaneCenter','id'].unique()
    #create lane center dictionary
    LC = defaultdict()
    for lcid in LCID:
        LC[lcid] = mp.loc[mp.id == lcid]
    #get av ID
    avID = trj.loc[trj.is_sdc == 1,'object_id'].unique()
    if len(avID) == 0:
        return pssdata
    print('avID: ',avID[0])
    LCx = mp.loc[mp.Lane_name == 'LaneCenter','x'].to_numpy()
    LCy = mp.loc[mp.Lane_name == 'LaneCenter','y'].to_numpy()
    av_x = car[avID[0]]['center_x'].to_numpy()
    av_y = car[avID[0]]['center_y'].to_numpy()
    av_time = car[avID[0]]['time'].to_list()
    
    data = pd.DataFrame([])
    LaneID = find_av_lane(avID[0],car,LC,LCID,LCx,LCy)
    print('avLaneID: ',LaneID)
    #plot(LaneID,LC,13,nearx,neary)
    rCar,frCar,erCar = find_car_on_lane(LaneID,car,carID,avID,LC)
    #rCar.append(avID[0])
    print('CarID: ',rCar)
    print(frCar,erCar)
    if len(rCar) <= 1:
        print('PASS')
        return data
    
    '''
    StartLaneID,EndLaneID = find_related_car_at_first_end(frCar,erCar,car,LCID,LC,LCx,LCy)
    print(StartLaneID,EndLaneID)
    
    for sID in StartLaneID:
        for slid in sID:
            if slid in LaneID:
                continue
            else:
                lax = LC[slid]['x'].to_list()
                if len(lax) <50:
                    continue
                else:
                    LaneID.append(slid)
   
    for eID in EndLaneID:
        for elid in eID:
            if elid in LaneID:
                continue
            else:
                lax = LC[elid]['x'].to_list()
                if len(lax) <50:
                    continue
                else:
                    LaneID.append(elid)
    '''
    LCTimeID = find_LC_time(car,rCar,LC,LaneID)
    print('LCTimeID: ',LCTimeID)
    if len(LCTimeID) == 1 or len(LCTimeID) < 2:
        print('PASS')
        return data
    #laneID = find_lane(relatedLane,LC,LCID)
    #print(laneID)
    if abs(av_x[-1]-av_x[0])<abs(av_y[-1]-av_y[0]):
        if len(LCTimeID)>1:
            print('1')
            glbLc,glbFlc,leaderID,followID,target = find_subject_leader_y(car,rCar,av_x,av_y,av_time,avID,LCTimeID)
            
    else:
        if len(LCTimeID)>1:
            print('2')
            glbLc,glbFlc,leaderID,followID,target = find_subject_leader_x(car,rCar,av_x,av_y,av_time,avID,LCTimeID)

    
    final_array = []
    for i,j in zip(glbLc,glbFlc):
        array = []
        for k in range(len(i)):
            lc = np.flip(i)
            array.append(lc[k][0])
        array.append(avID[0])
        for l in range(len(j)):
            array.append(j[l][0])
        final_array.append(array)
    print(final_array)
    
    lead_car100,lead_car0,follow_100 = find_pair(final_array,avID)
    print(len(lead_car100))
    print(len(lead_car0))
    print(len(follow_100))
    
    print(lead_car100)
    print(lead_car0)
    print(follow_100)
    
    cont = False
    for lc0 in lead_car0:
        if lc0[0] != None:
            cont = True
    if cont == False:
        print('PASS')
        return data
        
    leader100 = False
    for lc1 in lead_car100:
        if lc1[0] != None:
            leader100 = True
    
    f_100 = False
    for lc2 in follow_100:
        if lc2[0] != None:
            f_100 = True
            
    plot(LaneID,LC,car,LCTimeID,scenid,LCx,LCy,av_x,av_y,avID)
    
    print(leader100,f_100)
    data100 = pd.DataFrame(columns=['scenario_id', 'scenario_id_num','time', 'subject_id','subject_is_sdc','subject_x','subject_y','subject_length','subject_speed','Position_num','leader_id','leader_is_sdc','leader_x','leader_y','leader_length','leader_speed','leader_position'])
    data0 = pd.DataFrame(columns=['scenario_id', 'scenario_id_num','time', 'subject_id','subject_is_sdc','subject_x','subject_y','subject_length','subject_speed','Position_num','leader_id','leader_is_sdc','leader_x','leader_y','leader_length','leader_speed','leader_position'])
    data_100 = pd.DataFrame(columns=['scenario_id', 'scenario_id_num','time', 'subject_id','subject_is_sdc','subject_x','subject_y','subject_length','subject_speed','Position_num','leader_id','leader_is_sdc','leader_x','leader_y','leader_length','leader_speed','leader_position'])

    for i,j,k,t in zip(lead_car100,lead_car0,follow_100,av_time):
        count = av_time.index(t)
        if i[0] != None and i[1] != None:
            indexl200 = car[i[0]]['time'].to_list().index(t)
            indexf200 = car[i[1]]['time'].to_list().index(t)
            speedx = car[i[0]]['velocity_x'].to_list()[indexl200]
            speedy = car[i[0]]['velocity_y'].to_list()[indexl200]
            speedx1 = car[i[1]]['velocity_x'].to_list()[indexf200]
            speedy1 = car[i[1]]['velocity_y'].to_list()[indexf200]
            s = speed(speedx,speedy)
            s1 = speed(speedx1,speedy1)
            data100.loc[count] = [car[i[1]]['scenario_id'].to_list()[indexf200]] + [car[i[1]]['scenario_id_num'].to_list()[indexf200]] + [t] + [i[1]] + [car[i[1]]['is_sdc'].to_list()[indexf200]] + [car[i[1]]['center_x'].to_list()[indexf200]] + [car[i[1]]['center_y'].to_list()[indexf200]] + [car[i[1]]['length'].to_list()[indexf200]] + [s1] + [100] +[i[0]] + [car[i[0]]['is_sdc'].to_list()[indexl200]] + [car[i[0]]['center_x'].to_list()[indexl200]] + [car[i[0]]['center_y'].to_list()[indexl200]] + [car[i[0]]['length'].to_list()[indexl200]] + [s]+[200]
        elif i[1] != None and leader100 == False:
            pass
        elif i[0] == None and i[1] != None and leader100 == True:
            indexf200 = car[i[1]]['time'].to_list().index(t)
            speedx1 = car[i[1]]['velocity_x'].to_list()[indexf200]
            speedy1 = car[i[1]]['velocity_y'].to_list()[indexf200]
            s1 = speed(speedx1,speedy1)
            data100.loc[count] = [car[i[1]]['scenario_id'].to_list()[indexf200]] + [car[i[1]]['scenario_id_num'].to_list()[indexf200]] + [t] + [i[1]] + [car[i[1]]['is_sdc'].to_list()[indexf200]] + [car[i[1]]['center_x'].to_list()[indexf200]] + [car[i[1]]['center_y'].to_list()[indexf200]] + [car[i[1]]['length'].to_list()[indexf200]] + [s1] + [100] +[None] + [None] + [None] + [None] + [None] + [None]+[None]
        elif i[0] == None and i[1] == None:
            pass
        if j[0] != None and j[1] != None:
            indexl0 = car[j[0]]['time'].to_list().index(t)
            indexf0 = car[j[1]]['time'].to_list().index(t)
            speedxj = car[j[0]]['velocity_x'].to_list()[indexl0]
            speedyj = car[j[0]]['velocity_y'].to_list()[indexl0]
            speedxj1 = car[j[1]]['velocity_x'].to_list()[indexf0]
            speedyj1 = car[j[1]]['velocity_y'].to_list()[indexf0]
            sj = speed(speedxj,speedyj)
            sj1 = speed(speedxj1,speedyj1)
            data0.loc[count] = [car[j[1]]['scenario_id'].to_list()[indexf0]] + [car[j[1]]['scenario_id_num'].to_list()[indexf0]] + [t] + [j[1]] + [car[j[1]]['is_sdc'].to_list()[indexf0]] + [car[j[1]]['center_x'].to_list()[indexf0]] + [car[j[1]]['center_y'].to_list()[indexf0]] + [car[j[1]]['length'].to_list()[indexf0]] + [sj1] + [0] +[j[0]] + [car[j[0]]['is_sdc'].to_list()[indexl0]] + [car[j[0]]['center_x'].to_list()[indexl0]] + [car[j[0]]['center_y'].to_list()[indexl0]] + [car[j[0]]['length'].to_list()[indexl0]] + [sj]+[10]
        elif j[0] == None and j[1] != None:
            indexl0 = car[j[1]]['time'].to_list().index(t)
            speedxj1 = car[j[1]]['velocity_x'].to_list()[indexl0]
            speedyj1 = car[j[1]]['velocity_y'].to_list()[indexl0]
            sj1 = speed(speedxj1,speedyj1)
            data0.loc[count] = [car[j[1]]['scenario_id'].to_list()[indexl0]] + [car[j[1]]['scenario_id_num'].to_list()[indexl0]] + [t] + [j[1]] + [car[j[1]]['is_sdc'].to_list()[indexl0]] + [car[j[1]]['center_x'].to_list()[indexl0]] + [car[j[1]]['center_y'].to_list()[indexl0]] + [car[j[1]]['length'].to_list()[indexl0]] + [sj1] + [0] +[None] + [None] + [None] + [None] + [None] + [None]+[None]
            
        if k[0] != None and k[1] != None:
            indexl100 = car[k[0]]['time'].to_list().index(t)
            indexf100 = car[k[1]]['time'].to_list().index(t)
            speedxk = car[k[0]]['velocity_x'].to_list()[indexl100]
            speedyk = car[k[0]]['velocity_y'].to_list()[indexl100]
            speedxk1 = car[k[1]]['velocity_x'].to_list()[indexf100]
            speedyk1 = car[k[1]]['velocity_y'].to_list()[indexf100]
            sk = speed(speedxk,speedyk)
            sk1 = speed(speedxk1,speedyk1)
            data_100.loc[count] = [car[k[1]]['scenario_id'].to_list()[indexf100]] + [car[k[1]]['scenario_id_num'].to_list()[indexf100]] + [t] + [k[1]] + [car[k[1]]['is_sdc'].to_list()[indexf100]] + [car[k[1]]['center_x'].to_list()[indexf100]] + [car[k[1]]['center_y'].to_list()[indexf100]] + [car[k[1]]['length'].to_list()[indexf100]] + [sk1] + [-100] +[k[0]] + [car[k[0]]['is_sdc'].to_list()[indexl100]] + [car[k[0]]['center_x'].to_list()[indexl100]] + [car[k[0]]['center_y'].to_list()[indexl100]] + [car[k[0]]['length'].to_list()[indexl100]] + [sk]+[-50]
        elif k[0] == None and k[1] != None:
            indexl100 = car[k[1]]['time'].to_list().index(t)
            speedxk1 = car[k[1]]['velocity_x'].to_list()[indexl100]
            speedyk1 = car[k[1]]['velocity_y'].to_list()[indexl100]
            sk1 = speed(speedxk1,speedyk1)
            data_100.loc[count] = [car[k[1]]['scenario_id'].to_list()[indexl100]] + [car[k[1]]['scenario_id_num'].to_list()[indexl100]] + [t] + [k[1]] + [car[k[1]]['is_sdc'].to_list()[indexl100]] + [car[k[1]]['center_x'].to_list()[indexl100]] + [car[k[1]]['center_y'].to_list()[indexl100]] + [car[k[1]]['length'].to_list()[indexl100]] + [sk1] + [-100] +[None] + [None] + [None] + [None] + [None] + [None]+[None]
        elif f_100 == False:
            pass
    
    data = data.append(data100)
    data = data.append(data0)
    data = data.append(data_100)
    return data
    '''
    data = pd.DataFrame([])
    if len(LCTimeID)> 1:
        for lcid in LCTimeID:
            ctime = car[lcid[0]]['time'].to_list()
            tsindex = ctime.index(lcid[1])
            teindex = ctime.index(lcid[2])
            ndata = pd.DataFrame(trj.loc[trj.object_id == lcid[0]])
            data = data.append(ndata[tsindex:teindex+1])
    if len(data) > 1:
        print('pushed')
        plot(LaneID,LC,car,LCTimeID,scenid,LCx,LCy)
        plot_lane(LaneID,LC,scenid)
    return data
    '''
if __name__=='__main__':
    
    Filename_C = 'final/new_format/car_follow_data_10.csv'
    car = pd.read_csv(Filename_C)
    sid = car.scenario_id_num.unique()
    for l in range(90,94):
        Filename_V = 's1_traj/file'+str(l)+'_traj.csv'
        Filename_M = 's1_map/file'+str(l)+'_map.csv'
        vec = pd.read_csv(Filename_V)
        del vec['Unnamed: 0']
        del vec['Unnamed: 0.1']
        map = pd.read_csv(Filename_M)
        final_data = pd.DataFrame([])
        for id in sid:
            print(id)
            data = main(vec,map,id)
            final_data = final_data.append(data)
        final_data.to_csv('CF_ahead_behind/file'+str(l)+'_traj.csv',index = False)
        
