import numpy as np
import pandas as pd
import csv
from collections import defaultdict

 
def acceleration(carID,car):
    '''
    cal acceleration(moving average[-0.3,0.3])
    '''
    acc = []
    #for id in carID:
    speed = car[carID]['subject_speed'].to_numpy()
    time = car[carID]['time'].to_numpy()
    acc.append(0.0)
    for i in range(len(speed)):
        if i+1 == len(speed):
            break
        acc.append((speed[i+1]-speed[i])/(time[i+1]-time[i]))
    return acc

def check_length(vec):
    #check car length (3.5 - 6.5)
    subject_del_id = vec.loc[(vec.subject_length >= 6.5) | (vec.subject_length <= 3), 'subject_id'].unique()
    leader_del_id = vec.loc[(vec.leader_length >= 6.5) | (vec.leader_length <= 3), 'subject_id'].unique()
    if len(subject_del_id)>len(leader_del_id) or len(subject_del_id) == len(leader_del_id):
        list = subject_del_id-leader_del_id
    else:
        list = leader_del_id-subject_del_id
    subject_del_id = subject_del_id.tolist()
    for id in list:
        if not id in subject_del_id:
            subject_del_id.append(id)
    return subject_del_id


def distance_leader(uniqueMapID,LC,fdx,fdy,speedf,timef,speedl,ldx,ldy,timel):
    print(len(fdx))
    final_dis_LaneCenter = []
    for t in timef:
        print(t)
        if not t in timel:
            print('None')
            final_dis_LaneCenter.append(None)
        elif t in timel:
            indexf = timef.tolist().index(t)
            indexl = timel.index(t)
            mind = 2
            x = 0
            #print('x = 0')
            for q in range(len(fdx)):
                d = np.sqrt((fdx[q] - ldx[indexl]) ** 2 + (fdy[q] - ldy[indexl]) ** 2)
                #print(d)
                if mind > d:
                    mind = d
                    x = fdx[q]
            minl = 2
            x1 = 0
            if x == 0:
                #print('x1 = 0')
                for a in range(len(ldx)):
                    d = np.sqrt((fdx[indexf] - ldx[a]) ** 2 + (fdy[indexf] - ldy[a]) ** 2)
                    #print(d)
                    if minl > d:
                        minl = d
                        x1 = ldx[a]
                        y1 = ldx[a]

            if x1 == 0 and x == 0:
                lineID = []
                lineID2 = []
                #print('x1 == 0 and x== 0')
                for id in uniqueMapID:
                    x4 = LC[id]['x'].to_numpy()
                    y4 = LC[id]['y'].to_numpy()
                    hmd = np.sqrt((x4 - fdx[indexf]) ** 2 + (y4 - fdy[indexf]) ** 2)
                    hmd1 = np.sqrt((x4 - ldx[indexl]) ** 2 + (y4 - ldy[indexl]) ** 2)
                    
                    Lindex1 = hmd.tolist().index(min(hmd))
                    lineID.append([id,min(hmd),Lindex1])
                    
                    Lindex2 = hmd1.tolist().index(min(hmd1))
                    lineID2.append([id,min(hmd1),Lindex2])
                sortedLineID = sorted(lineID,key=lambda x: x[1])
                sortedLineID2 = sorted(lineID2,key=lambda x: x[1])
                print('lineID: ',sortedLineID[0][0],sortedLineID2[0][0])
                
                if sortedLineID[0][0] == sortedLineID2[0][0]:
                    x5 = LC[sortedLineID[0][0]]['x']
                    y5 = LC[sortedLineID[0][0]]['y']
                    newX = x5[sortedLineID[0][2]:sortedLineID2[0][2]]
                    newY = y5[sortedLineID[0][2]:sortedLineID2[0][2]]
                    fdx1 = fdx[indexf:]
                    fdy1 = fdy[indexf:]
                    finalX = np.concatenate((fdx1,newX),axis=0)
                    finalY = np.concatenate((fdy1,newY),axis=0)
                    final_D = sum(np.sqrt(np.diff(finalX) ** 2 + np.diff(finalY) ** 2))
                    final_dis_LaneCenter.append(abs(final_D))
                else:
                    return None
                    
            if x != 0 and x1 == 0:
                #print('x != 0 and x1 == 0')
                indexfd = fdx.tolist().index(x)
                finalX, finalY = fdx[indexf:indexfd+1],fdy[indexf:indexfd+1]
                finalX = finalX.astype(float)
                finalY = finalY.astype(float)
                final_D = sum(np.sqrt(np.diff(finalX) ** 2 + np.diff(finalY) ** 2))
                final_dis_LaneCenter.append(abs(final_D))
            elif x == 0 and x1 != 0:
                #print('x == 0 and x1 != 0')
                lx = ldx[indexl]
                ly = ldy[indexl]
                md = []
                for nx,ny in zip(fdx,fdy):
                    md.append(np.sqrt((nx - lx) ** 2 + (ny - ly) ** 2))
                newidex = md.index(min(md))
                final_D = sum(np.sqrt(np.diff(fdx[indexf:newidex+1]) ** 2 + np.diff(fdy[indexf:newidex+1]) ** 2))
                final_dis_LaneCenter.append(abs(final_D))
    return final_dis_LaneCenter
        

def getEquidistantPoints(p1, p2, parts):
    return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))

    
def To_LaneCenter(car,cID,uniqueMapID,LC,lx,ly):
    '''
    deviation from lane center
    '''
    dis_sf_car = []
    min_distance = []
    carx = car[cID]['subject_x'].to_numpy()
    cary = car[cID]['subject_y'].to_numpy()
    #print(len(carx))
    for r in range(len(carx)):
        min_distance1 = []
        for id in uniqueMapID:
            x = LC[id]['x'].to_numpy()
            y = LC[id]['y'].to_numpy()
            hmd = np.sqrt((x - carx[r]) ** 2 + (y - cary[r]) ** 2)
            if min(hmd) < 1.5:
                distt = []
                dis_start= []
                dis_final = []
                for s in range(len(x)):
                    dist = np.sqrt((x[s]-carx[r])**2 +(y[s]-cary[r])**2)
                    distt.append([dist,x[s],y[s]])
                minDis = sorted(distt,key=lambda x: x[0])
                px1,py1,px2,py2 = minDis[0][1],minDis[0][2],minDis[1][1],minDis[1][2]
                listp = list(zip(*getEquidistantPoints((px1,py1),(px2,py2), 1000)))
                px,py = listp[0],listp[1]
                for h in range(len(px)):
                    diss = np.sqrt((px[h]-carx[r])**2 +(py[h]-cary[r])**2)
                    dis_final.append([diss,px[h],py[h]])
                min_dists = sorted(dis_final,key=lambda x: x[0])
                min_distance1.append([min_dists[0][0],min_dists[0][1],min_dists[0][2]])
        min_distanc = sorted(min_distance1,key=lambda x: x[0])
        d = []
        if len(min_distanc) == 0:
            min_distance.append(None)
            dis_sf_car.append([None,None])
            continue
        min_distance.append(min_distanc[0][0])
        dis_sf_car.append([min_distanc[0][1],min_distanc[0][2]])
    
    return min_distance,dis_sf_car

def leader_To_LaneCenter(car,cID,uniqueMapID,LC,lx,ly):
    '''
    deviation from lane center
    '''
    dis_sf_car = []
    min_distance = []

    carx = car[cID]['leader_x'].to_numpy()
    cary = car[cID]['leader_y'].to_numpy()
    #print(len(carx))
    for r in range(len(carx)):
        min_distance1 = []
        for id in uniqueMapID:
            x = LC[id]['x'].to_numpy()
            y = LC[id]['y'].to_numpy()
            hmd = np.sqrt((x - carx[r]) ** 2 + (y - cary[r]) ** 2)
            if min(hmd) < 1.5:
                distt = []
                dis_start= []
                dis_final = []
                for s in range(len(x)):
                    dist = np.sqrt((x[s]-carx[r])**2 +(y[s]-cary[r])**2)
                    distt.append([dist,x[s],y[s]])
                minDis = sorted(distt,key=lambda x: x[0])
                px1,py1,px2,py2 = minDis[0][1],minDis[0][2],minDis[1][1],minDis[1][2]
                listp = list(zip(*getEquidistantPoints((px1,py1),(px2,py2), 1000)))
                px,py = listp[0],listp[1]
                for h in range(len(px)):
                    diss = np.sqrt((px[h]-carx[r])**2 +(py[h]-cary[r])**2)
                    dis_final.append([diss,px[h],py[h]])
                min_dists = sorted(dis_final,key=lambda x: x[0])
                min_distance1.append([min_dists[0][0],min_dists[0][1],min_dists[0][2]])
        min_distanc = sorted(min_distance1,key=lambda x: x[0])
        d = []
        if len(min_distanc) == 0:
            min_distance.append(None)
            dis_sf_car.append([None,None])
            continue
        min_distance.append(min_distanc[0][0])
        dis_sf_car.append([min_distanc[0][1],min_distanc[0][2]])

    return min_distance,dis_sf_car

    
def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)
'''
def derived_speed(speed,cumLongtitude,time):
    derived_spd = []
    
    for l in range(len(cumLongtitude)):
        if l == 0:
            derived_spd.append(0.0)
        else:
            sp = abs(cumLongtitude[l]-cumLongtitude[l-1])/abs(time[l]-time[l-1])
            derived_spd.append(sp)
    return derived_spd
'''
def main(sid,vec1,map1):
    
    vec = vec1[vec1['scenario_id_num'] == sid]
    map = map1[map1['scenario_id_num'] == sid]
    
    
    del_id = check_length(vec)
    for k in del_id:
        vec = vec.loc[vec.subject_id != k]
        
    uniqueMapID = map.loc[map.Lane_name == 'LaneCenter','id'].unique()
    carID = vec.Position_num.unique()
    avID = vec.loc[vec['subject_is_sdc'] == 1,'subject_id'].unique()
    final_data = pd.DataFrame([])
    LC = defaultdict()
    car = defaultdict()
    if len(carID) == 1:
        print('PASS')
        return final_data
    for LaneID in uniqueMapID:
        LC[LaneID] = map[map.id == LaneID]
        
    LCx = map.loc[map.Lane_name == 'LaneCenter','x'].to_list()
    LCy = map.loc[map.Lane_name == 'LaneCenter','y'].to_list()
    
    for id in carID:
        car[id] = vec[vec.Position_num == id]
    
    delID = []
    for cid in carID:
        data = pd.DataFrame([])
        data = data.append(car[cid])
        
        acc = acceleration(cid,car)
        data['acceleration'] = acc
        
        min_distance,dis_sf_car = To_LaneCenter(car,cid,uniqueMapID,LC,LCx,LCy)
        
        min_distance_leader,dis_sf_car_leader = leader_To_LaneCenter(car,cid,uniqueMapID,LC,LCx,LCy)
            
        data['subject_lane_deviation'] = min_distance
        data['leader_lane_deviation'] = min_distance_leader
        dis_sf_car = np.array(dis_sf_car)
        data['Subject_laneCenterx'] = dis_sf_car[:,0]
        data['Subject_laneCentery'] = dis_sf_car[:,1]
        dis_sf_car_leader = np.array(dis_sf_car_leader)
        data['Leader_laneCenterx'] = dis_sf_car_leader[:,0]
        data['Leader_laneCentery'] = dis_sf_car_leader[:,1]

        final_data = final_data.append(data)
   
    car1 = defaultdict()
    for id in carID:
        car1[id] = final_data[final_data.Position_num == id]
    df = pd.DataFrame([])
    for ncid in carID:
        print(ncid)
        data1 = pd.DataFrame(car1[ncid])
        fdx = car1[ncid]['Subject_laneCenterx'].to_numpy()
        fdy = car1[ncid]['Subject_laneCentery'].to_numpy()
        speedf = car1[ncid]['subject_speed'].to_numpy()
        timef = car1[ncid]['time'].to_numpy()
        speedl1 = car1[ncid]['leader_speed'].to_numpy()
        ldx1 = car1[ncid]['Leader_laneCenterx'].to_numpy()
        ldy1 = car1[ncid]['Leader_laneCentery'].to_numpy()
        timel1 = car1[ncid]['time'].to_numpy()
        speedl = []
        ldx = []
        ldy= []
        timel = []
        for sp in range(len(speedl1)):
            if not np.isnan(speedl1[sp]):
                speedl.append(speedl1[sp])
                ldx.append(ldx1[sp])
                ldy.append(ldy1[sp])
                timel.append(timel1[sp])
        print(len(ldx))
        print(len(fdx))
        print(len(data1))
        final_dis_LaneCenter = distance_leader(uniqueMapID,LC,fdx,fdy,speedf,timef,speedl,ldx,ldy,timel)

        
        if final_dis_LaneCenter == None:
            print(str(ncid)+' PASS3')
            return df
            
        data1['distance_to_leader_LaneCenter'] = final_dis_LaneCenter
        df = df.append(data1)
    
    bad = pd.DataFrame([])
    newcarID = df.Position_num.unique()
    if len(newcarID) == 1 and newcarID[0] == 0:
        new_carID = []
        newcarIDq = df.loc[df.Position_num == 0, 'leader_id'].unique()
        for nq in newcarIDq:
            if not np.isnan(nq):
                new_carID.append(nq)
        if len(nq) == 1:
            return bad
    elif len(newcarID) == 1 and newcarID[0] == -100:
        nq = df.loc[df.Position_num == -100, 'subject_id'].unique()
        if len(nq) == 1:
            return bad
    elif not 0 in newcarID:
        return bad
            
    return df

    
if __name__ == '__main__':
    Filename_M = 'CF_ahead_behind/file98_traj.csv'
    Filename_V = 's1_map/file98_map.csv'

    vec = pd.read_csv(Filename_M)
    map = pd.read_csv(Filename_V)
    sid = vec.scenario_id_num.unique()
    print(len(sid))
    df = pd.DataFrame([])
    for i in range(len(sid)):
        if sid[i] in delid:
            continue
        print('now in '+str(sid[i]))
        final_data = main(sid[i],vec,map)
        df = df.append(final_data)
    df.to_csv('new_CF_ahead_behind/file98_traj.csv',index = False)

