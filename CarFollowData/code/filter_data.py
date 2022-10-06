import numpy as np
import pandas as pd
import csv
from collections import defaultdict
import matplotlib.pyplot as plt


def check(distance):
    for dis in distance:
        if dis == None:
            continue
        elif dis < 3 or dis > 100:
            return True
    return False
def check_leng(length):
    for dis in length:
        if dis == None:
            continue
        elif dis < 2 or dis > 6:
            return True
    return False
    
def main(vec,nsid):
    data = pd.DataFrame([])
    new_data = pd.DataFrame([])
    vec1 = vec[vec['scenario_id_num'] == nsid]
    laneID = vec1.laneID.unique()
    print(laneID)
    for LID in laneID:
        print('now in: ', LID)
        vec4 = vec1[vec1['laneID'] == LID]
        carID = vec4.newID.unique()
        car = defaultdict()
        for id in carID:
            car[id] = vec4[vec4.newID == id]
        for cid in carID:
            distance = car[cid]['distance_to_leader_LaneCenter']
            length = car[cid]['length']
            time = car[cid]['time']
            if LID == 0:
                if check(distance) or check_leng(length) or len(time)<150:
                    if cid == 0:
                        return data
                    if cid == 1:
                        return data
                    elif cid > 0:
                        print('check 1 drop: ',cid,cid+1)
                        vec4 = vec4.loc[(vec4.newID != cid)&(vec1.newID != cid+1)]
                    elif cid < 0:
                        if cid % 2 == 0:
                            print('check 2.3 drop: ',cid,cid+1)
                            vec4 = vec4.loc[(vec4.newID != cid)&(vec1.newID != cid+1)]
                        else:
                            print('check 2.3 drop: ',cid,cid-1)
                            vec4 = vec4.loc[(vec4.newID != cid)&(vec1.newID != cid-1)]
            elif LID > 0:
                if check(distance) or check_leng(length) or len(time)<150:
                    print('check 3 drop: ',cid,-cid)
                    vec4 = vec4.loc[(vec4.newID != cid)&(vec1.newID != -cid)]
                
            else:
                if check(distance) or check_leng(length) or len(time)<150:
                    print('check 4 drop: ',cid,-cid)
                    vec4 = vec4.loc[(vec4.newID != cid)&(vec1.newID != -cid)]
        new_data = new_data.append(vec4)
            
    return new_data


if __name__ == '__main__':
    
    for r in range(95,96):
        Filename_V = 'final/final'+str(r)+'.csv'
        vec = pd.read_csv(Filename_V)
        sid = vec.scenario_id_num.unique()
        df = pd.DataFrame([])
        for i in sid:
            print(i)
            datef = main(vec,i)
            df = df.append(datef)
            
        df.to_csv('final/final'+str(r)+'_new.csv',index = False)
    
