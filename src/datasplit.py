#Copyright 2020 Lingjiao Chen, version 0.1.
#All rights reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import numpy as np
import os
import glob

def data_split(datapath = '/Users/lingjiaochen/Documents/projects/MLMarket/MLMarket/MLAS/FER-2013/mlserviceperformance_FERPlus_Full/', 
               train_ratio ='0.5', 
               randseed = 0,
			   perclasssplit = False
               ):
    if(perclasssplit):
        trainindex,testindex = generate_index_class(datapath = datapath,
                                          train_ratio = train_ratio, 
                                          randseed = randseed)
    else:
        #print('random seed is',randseed)
        #print('train tario is',train_ratio)
        #print('datapath',datapath)
        trainindex,testindex = generate_index(datapath = datapath,
                                          train_ratio = train_ratio, 
                                          randseed = randseed)
    allpath = sorted(glob.glob(datapath+'*.txt'))
    for i in range(len(allpath)):
        onepath = allpath[i]
        filename = onepath.split('/')[-1]
        # 1/3: Get the directory
        train_dir =  datapath+'train/'
        test_dir = datapath+'test/'
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        # 2/3: Split data
        #print('data split path',onepath)
        data = np.loadtxt(onepath, dtype=str)
        train_data = data[trainindex]
        test_data = data[testindex]
        # 3/3: Save
        np.savetxt(train_dir+filename,train_data,'%s')
        np.savetxt(test_dir+filename,test_data,'%s')
        
def generate_index(datapath = '/Users/lingjiaochen/Documents/projects/MLMarket/MLMarket/MLAS/FER-2013/mlserviceperformance_FERPlus_Full/', 
                   train_ratio ='0.5', 
                   randseed = 0,):        
    allpath = sorted(glob.glob(datapath+'*.txt'))
    data_num = len(np.loadtxt(allpath[0], dtype=str))
    # generate random index
    np.random.seed(randseed)
    trainlast = int(data_num*train_ratio)
    shuffle_index = np.random.permutation(data_num)
    trainindex = shuffle_index[0:trainlast]
    testindex = shuffle_index[trainlast:]
    #print('data split test index',testindex)
    return trainindex, testindex
 
def generate_index_class(datapath = '/Users/lingjiaochen/Documents/projects/MLMarket/MLMarket/MLAS/FER-2013/mlserviceperformance_FERPlus_Full/', 
                   train_ratio ='0.5', 
                   randseed = 0,): 
    classindexlist = getlabelindexlist(datapath)
    trainindex = list()
    testindex = list()
    for i in range(len(classindexlist)):
        train1, test1 = partition_index(classindexlist[i],train_ratio,randseed)
        #if(i<5):
            #print('train1tolist',trainindex)
        trainindex = trainindex+train1.tolist()
        testindex = testindex + test1.tolist()
    #print('train index final',trainindex)
    return np.asarray(trainindex).flatten(), np.asarray(testindex).flatten().astype(int)

def getlabelindexlist(datapath):
    index = np.loadtxt(datapath+'Model0_TrueLabel.txt')
    labellist = list()
    #print('max of index',max(index))
    for i in range(int(max(index)+1)):
        labellist.append(list())
    for i in range(len(index)):
        label = int(index[i])
        labellist[label].append(i)
    return labellist
		
def partition_index(data_array, train_ratio=0.5, randseed = 0):
    #print('data array',data_array)
    data_num = len(data_array)
    np.random.seed(randseed)
    trainlast = int(data_num*train_ratio)
    shuffle_index = np.random.permutation(data_num)
    trainindex = shuffle_index[0:trainlast]
    testindex = shuffle_index[trainlast:]
    #print('train index',trainindex)
    #print('test index',testindex)
	
    train_data = np.asarray(data_array)[trainindex]
    test_data = np.asarray(data_array)[testindex]
    return train_data, test_data

def main():
    datapath = '/Users/lingjiaochen/Documents/projects/MLMarket/MLMarket/MLAS/FER-2013/mlserviceperformance_FERPlus_Full/'
    randseed = 0
    train_ratio = 0.5
    data_split(datapath= datapath,randseed= randseed,train_ratio=train_ratio)
    return
if __name__ == '__main__':
    main()


