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

from __future__ import division

#import time
#import matplotlib.pyplot as plt
import numpy
import numpy as np
from mlmodels import MLModels
from scipy.interpolate import UnivariateSpline
from datasplit import data_split
import multiprocessing
import csv


class OptimizerTemplate(object):
    def __init__(self):
        raise NotImplementedError
        
    def solve(self):
        raise NotImplementedError
        
    def setbudget(self, budget):
        raise NotImplementedError
        
    def getresult(self):
        raise NotImplementedError
    
    def getmarketinfo(self):
        return self.api_id_list, self.api_name_list, self.cost_list,       
        
class OptimizerFrugalML(OptimizerTemplate):
    ''' FrugalML optimizer. '''
    def __init__(self,
                 datapath='../dataset/mlserviceperformance_FERPLUS',
                 split = True,
                 train_ratio = 0.5,
                 test_eval = True,
                 randseed = 100,
                 method='FrugalML',
                 baseid=100):       
        self.datapath = datapath
        self.split = split
        self.train_ratio = train_ratio
        self.randseed = randseed
        self.test_eval = test_eval
        self._loadmetainfo()
        self.budget_num = 20 # number of budget splitting. internal params
        self.prob_interval = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,
                              0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
        self.method = method
        self.baseid = int(baseid)
    def solve(self):
        ''' generate the optimal strategy via FrugalML training algorithm. '''
        if(self.method=='FrugalMLFixBase'):
            return self._solve_fixbase()
        if(self.method=='FrugalMLQSonly'):
            return self._solve_qsonly()
        budget = self.budget
        randseed = self.randseed
        myoptimizer = optimizer_linear_offline_autobase(
                                       prob_interval=self.prob_interval,
                                       cost_vector_all = self.cost_list, 
                                       budget = budget, 
                                       budget_num = self.budget_num,
                                       model_id_all =self.api_id_list,
                                       use_context = True,
                                       datapath=self.datapath,
                                       context = list(range(self.label_num)),
                                       split = self.split,
                                       train_ratio = self.train_ratio,
                                       randseed = randseed,
                                       test_eval = self.test_eval)    
        result = myoptimizer.solve()
        self.result = result
        #print('direct result is',result)
        self.strategy = self.strategyparser(result)
        return result
    
    def setbudget(self, budget):
        self.budget = budget
    
    def getresult(self):
        return self.strategy
    
    def _solve_fixbase(self):
        baseid = self.baseid
        cost_vector= list()
        model_id = list()
        for i in range(len(self.api_id_list)):
            if(self.api_id_list[i]==baseid):
                basecost = self.cost_list[i]
            model_id.append(self.api_id_list[i])
            cost_vector.append(self.cost_list[i])
        if(basecost>self.budget):
            raise Exception("Base API cost (", 
                            basecost, ") is higher than budget ("
                            ,self.budget,")!") 
        budget_frac =[0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 
                      0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 
                      0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 
                      0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8,
                      0.825,  0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1]    
        myoptimizer = optimizer_linear_context_offline(base_id = [baseid],
                                           cost_vector = cost_vector, 
                                           budget = self.budget-basecost, 
                                           prob_interval = self.prob_interval,
                                           budget_frac=budget_frac,
                                           model_id=model_id,
                                           context = list(range(self.label_num)),
                                           datapath=self.datapath,
                                           split = self.split,
                                           train_ratio = self.train_ratio,
                                           randseed=self.randseed,
                                           test_eval=self.test_eval)
        result = myoptimizer.solve()
        strategy = dict()
        strategy['accuracy'] = result[1]
        strategy['rawresult'] = result        
        self.strategy = strategy
        return result

    def _solve_qsonly(self):   
        baseid = self.baseid
        cost_vector= list()
        model_id = list()
        for i in range(len(self.api_id_list)):
            if(self.api_id_list[i]==baseid):
                basecost = self.cost_list[i]
            model_id.append(self.api_id_list[i])
            cost_vector.append(self.cost_list[i])
        if(basecost>self.budget):
            raise Exception("Base API cost (", 
                            basecost, ") is higher than budget ("
                            ,self.budget,")!")
        myoptimizer = optimizer_linear_offline(base_id = [baseid],
                                           prob_interval=self.prob_interval,
                                           cost_vector = cost_vector, 
                                           budget = self.budget - basecost,
                                           model_id=model_id,
                                           datapath=self.datapath,
                                           split = self.split,
                                           train_ratio =self.train_ratio,
                                           randseed=self.randseed,
                                           test_eval=self.test_eval)
        result = myoptimizer.solve()
        strategy = dict()
        strategy['accuracy'] = result[1]
        strategy['rawresult'] = result        
        self.strategy = strategy
        return result
    
    
    def _loadmetainfo(self):
        # 1/2: load mata data file
        datapath = self.datapath
        meta = datapath+'/meta.csv'
        with open(meta, newline='') as f:
            reader = csv.reader(f)
            next(reader) # skip the first line
            data = list(reader)
        self.data = data
        # 2/2: generate meta info
        self.cost_list =  [float(w[2]) for w in data]
        self.api_id_list = [int(w[0]) for w in data]
        self.api_name_list = [w[1] for w in data]
        self.label_num = int(data[0][3])
        print('cost, api, name, and label', self.cost_list, self.api_id_list, self.api_name_list,self.label_num)

    def strategyparser(self, result):
        strategy = dict()
        # 1 store accuracy
        strategy['accuracy'] = result[1]
        # 2 store id, cost and budget
        strategy['API_id'] = self.api_id_list
        strategy['API_name'] = self.api_name_list        
        strategy['cost_list'] = self.cost_list
        strategy['budget'] = self.budget
        # 3 store p1
        k = len(self.cost_list)
        p1 = numpy.zeros((k,1))
        baseidlist = list()
        if(result[-1]=='one_model'):
            p1[result[-2]] = 1
            baseidlist.append(result[-2])
        else:
            p1[result[-2]] = result[-6]
            p1[result[-3]] = result[-7]
            baseidlist.append(result[-3])
            baseidlist.append(result[-2])            
        strategy['baseprob'] = p1
        # 4 store Q matrix
        L = self.label_num
        Q = numpy.zeros((k,L))
        for i in range(len(baseidlist)):
            for j in range(L):
                if(len(baseidlist)==2):
                    Q[baseidlist[i],j] = result[0][i][0][j][2]
                else:               
                    Q[baseidlist[i],j] = result[i][j][2]
        strategy['QualityThres'] = Q
        # 5 store p2 matrix for each base service
        p2matrix = dict()        
        for i in range(len(baseidlist)):
            temp  = numpy.zeros((k,L))
            for j in range(L):
                if(len(baseidlist)==2):
                    probvector = result[0][i][0][j][0]    
                else:
                    probvector = result[i][j][0]
                selfprob = 1-sum(probvector)
                for ttt in range(len(probvector)):
                    index = ttt
                    if(ttt>=baseidlist[i]):
                        index += 1
                    temp[index,j] = probvector[ttt]
                temp[baseidlist[i],j] = selfprob
            p2matrix['base_index_'+str(baseidlist[i])] = temp
        strategy['addonprob'] = p2matrix
        strategy['baseindex'] = baseidlist
        return strategy
        


class optimizer_single_offline(object):
    '''
    Use single model and skip the data point if out of budget
    '''
    def __init__(self,
                 cost = 6, 
                 budget = 5, 
                 model_id= 0,
                 datapath='path/to/imagenet/result/val_performance',
                 MLModelsClass = MLModels,
                 online = False,
                 num_of_label = None):
        self.cost = cost
        self.budget = budget
        self.model_id = [model_id]
        self.model = MLModelsClass([model_id],datapath)
    def setbudget(self,budget):
        self.budget = budget 
        
    def solve(self):
        if(self.budget>self.cost):
            prob = 1
        else:
            prob = self.budget/self.cost
        self.prob = prob
        active_index = [0]
        qvalue = 1
        Pi = np.ones(1)*self.prob
        opt = 1
        policy = (Pi, opt, qvalue, prob, active_index)
        baseid = self.model_id
        result = self.evalpolicy(policy=policy,baseid=baseid[0],
                                 policytype='q_value',
                                 modelid=self.model_id)
        opt = result
        return Pi, opt, qvalue, prob, active_index

    def evalpolicy(self,policy,baseid, policytype,modelid):
        acc = self.model.eval_policy(policy,baseid, modelid,
                                         policytype
                                         )
        return acc
    
class optimizer(object):
    def __init__(self):
        raise NotImplementedError

    def update_params(self):
        raise NotImplementedError
        
    def solve(self):
        raise NotImplementedError

    def solve_case1(self):
        raise NotImplementedError

    def solve_case2(self):
        raise NotImplementedError

    def solve_case3(self):
        raise NotImplementedError

    def setbudget(self,budget):
        self.budget = budget 
        
class optimizer_linear(optimizer):
    '''
    Core Internal Optimizer.
    '''
    def __init__(self,
                 weight_slop = np.ones((3,2)),
                 weight_intersect = np.ones((3,2)),
                 w_0_vector = np.ones(3),
                 b_0_vector = np.ones(3),
                 prob_interval=[0,0.3,0.6,1],
                 costvector = [4,6], 
                 budget = 5,
                 r_0 = 0
                 ):
        self.weight_slop = weight_slop
        self.weight_intersect = weight_intersect
        self.w_0_vector = w_0_vector
        self.b_0_vector = b_0_vector
        self.costvector = costvector
        self.budget = budget
        self.r_0 = r_0
        self.prob_interval = prob_interval
        self.eps=1e-9
        self.myinf = 1e9
   
    def update_params(self,                
                      weight_slop,
                      weight_intersect,
                      w_0_vector,
                      b_0_vector,
                      prob_interval=[0,0.3,0.6,1],
                      costvector = [4,6], 
                      budget = 5,
                      r_0 = 0):
        self.weight_slop = weight_slop
        self.weight_intersect = weight_intersect
        self.w_0_vector = w_0_vector
        self.b_0_vector = b_0_vector   
        self.budget = budget
        self.r_0 = r_0
        self.prob_interval = prob_interval        
    
    def solve(self):
        opt1, p1, activeindex1, Pi1 = self.solve_case1()
        opt2, p2, activeindex2, Pi2 = self.solve_case2()
        opt3, p3, activeindex3, Pi3 = self.solve_case3()
        #print('opt1,',opt1,p1,activeindex1, Pi1)
        #print('opt2,',opt2,p2,activeindex2, Pi2)
        #print('opt3,',opt3,p3,activeindex3, Pi3)
        #print(self.prob_interval)
        if(opt1>=opt2 and opt1 >= opt3):
            return opt1, p1, activeindex1, Pi1
        if(opt2>=opt1 and opt2 >= opt3):
            return opt2, p2, activeindex2, Pi2
        if(opt3>=opt1 and opt3 >= opt2):
            return opt3, p3, activeindex3, Pi3        
    
    def solve_case1(self):
        n = len(self.prob_interval)
        K = len(self.costvector)
        cost = self.costvector
        p = 0
        besti = 0
        Pi = np.zeros(K)
        opt = - self.myinf
        budget = self.budget
        for i in range(K):
            for j in range(n-1):
                left = self.prob_interval[j]
                right = self.prob_interval[j+1]
                w_i = self.weight_slop[i][j]
                b_i = self.weight_intersect[i][j]
                w_0 = self.w_0_vector[j]
                b_0 = self.b_0_vector[j]
                #print('CASE1 i and j',i,j)
                #print('CASE1 w_i, b_i')
                #print(w_i,b_i)
                #print('CASE1 w_0, b_0')
                #print(w_0,b_0)      
                opt1, p1 = self._solve_case1_interval(cost, left, right, w_i, b_i, w_0, b_0, budget, i)
            
                if(opt1>opt):
                    p = p1
                    opt = opt1
                    besti = i
        Pi[besti] = 1
        activeindex = list()
        activeindex.append(besti)        
        return opt, p, activeindex, Pi
    
    def solve_case2(self):
        n = len(self.prob_interval)
        K = len(self.costvector)
        cost = self.costvector
        p = 0
        opt = - self.myinf
        besti = 0
        Pi = np.zeros(K)        
        budget = self.budget
        for i in range(K):
            #print('i',i)
            for j in range(n-1):
                left = self.prob_interval[j]
                right = self.prob_interval[j+1]
                w_i = self.weight_slop[i][j]
                b_i = self.weight_intersect[i][j]
                w_0 = self.w_0_vector[j]
                b_0 = self.b_0_vector[j]
                #print(w_i,b_i)
                #print(w_0,b_0)
                #print('case 2 solver interval',cost, left, right, w_i, b_i, w_0, b_0, budget, i)
                opt1, p1 = self._solve_case2_interval(cost, left, right, w_i, b_i, w_0, b_0, budget, i)
                #print('i,j,opt1,p1',i,j,opt1,p1,left,right)
                if(opt1>opt):
                    #print('yes')
                    p = p1
                    opt = opt1
                    besti = i
        if(not(p==0)):
            Pi[besti] = budget/cost[besti]/p
        #print('opt,p,besti,Pi',opt, p, (besti), Pi)
        activeindex = list()
        activeindex.append(besti)
        return opt, p, activeindex, Pi

    def solve_case3(self):
        n = len(self.prob_interval)
        K = len(self.costvector)
        cost = self.costvector
        w_0 = self.w_0_vector
        b_0 = self.b_0_vector
        p = 0
        opt = - self.myinf
        besti = 0
        bestj = 0
        Pi = np.zeros(K)        
        budget = self.budget
        for i in range(K):
            for i2 in range(K):
                for j in range(n-1):
                    left = self.prob_interval[j]
                    right = self.prob_interval[j+1]
                    w_i = self.weight_slop[i][j]
                    b_i = self.weight_intersect[i][j]
                    w_j = self.weight_slop[i2][j]
                    b_j = self.weight_intersect[i2][j]
                    w_0 = self.w_0_vector[j]
                    b_0 = self.b_0_vector[j]
                    if(cost[i]>cost[i2]):    
                        opt1, p1 = self._solve_case3_interval(cost, left, right, w_i,b_i, w_j, b_j, w_0, b_0, budget, i, i2)
                        if(opt1>opt):
                            p = p1
                            opt = opt1
                            besti = i
                            bestj = i2
        c_i = cost[besti]
        c_j = cost[bestj]
        if(not(c_i==c_j) and not(p==0) ):
            Pi[besti] = (budget/p-c_j)/(c_i-c_j)
            Pi[bestj] = (c_i-budget/p)/(c_i-c_j)
        activeindex = list()
        activeindex.append(besti)
        activeindex.append(bestj)
        return opt, p, activeindex, Pi
        
    def _eval_quadratic_function(self, a1, a2, a3, x):
        return (a1*x*x+a2*x+a3)
    
    def _max_quadratic_function_within_range(self, a1, a2, a3, left, right):
        #print('a123,left, right', a1, a2, a3, left, right)
        if(left>right):
            return -self.myinf, None
        v1 = self._eval_quadratic_function(a1, a2, a3, left)
        v2 = self._eval_quadratic_function(a1, a2, a3, right)
        #print('v1',v1)
        #print('v2',v2)
        if(a1==0):
            v3 = -self.myinf
            x = None
        else:
            x = -a2/a1/2
            v3 = self._eval_quadratic_function(a1, a2, a3, x)
            #print('v3',v3)
            if(x<left or x>right):
                if(v1>v2):
                    return v1, left
                else:
                    return v2, right
        if(1):
            #print('optimal may be in the middle case2')
            if(v1>=v2 and v1>=v3):
                return v1, left
            if(v2>=v1 and v2>=v3):
                return v2, right
            if(v3>v1 and v3>v2):
                return v3, x       

    def _solve_case1_interval(self, cost, left, right, w_i, b_i, w_0, b_0, budget, index_i):
        a1 = w_i - w_0
        a2 = b_i - b_0
        a3 = self.r_0
        if(cost[index_i]==0):
            newright = right
        else:
            newright = min(right, budget/cost[index_i])
        #print('case1 params',a1, a2, a3, left, newright,index_i)
        #print('case1 solution',self._max_quadratic_function_within_range(a1, a2, a3, left, newright))
        return self._max_quadratic_function_within_range(a1, a2, a3, left, newright)

    def _solve_case2_interval(self, cost, left, right, w_i, b_i, w_0, b_0, budget, index_i):
        if(cost[index_i]<budget):
            v = -self.myinf
            x = None
            return v, x
        a1 = - w_0
        a2 = budget/cost[index_i]*w_i - b_0
        a3 = self.r_0 + budget/cost[index_i]*b_i
        #print('budget, w_i,cost_i',budget,w_i,cost[index_i],b_0)
        #print('a123 interval',a1,a2,a3)
        newleft = max(budget/cost[index_i],left)
        return self._max_quadratic_function_within_range(a1, a2, a3, newleft, right)
        
    def _solve_case3_interval(self, cost, left, right, w_i, b_i, w_j, b_j, w_0, b_0, budget, index_i, index_j):
        if(cost[index_i]<=cost[index_j]):
            v = -self.myinf
            x = None
            return v, x
        #print('CASE3 Params: cost, left, right, w_i, b_i, w_j, b_j, w_0, b_0, budget, index_i, index_j')        
        #print(cost, left, right, w_i, b_i, w_j, b_j, w_0, b_0, budget, index_i, index_j)
        c_i = cost[index_i]
        c_j = cost[index_j]
        a1 = (c_i*w_j-c_j*w_i)/(c_i-c_j) - w_0
        a2 = (budget*(w_i-w_j) - b_i*c_j + b_j* c_i ) /(c_i-c_j) - b_0
        a3 = self.r_0 + budget*(b_i-b_j)/(c_i-c_j)
        newleft = max(budget/c_i,left)
        newright = min(budget/c_j,right)
        #print('CASE3 a123 and left right:')
        #print(a1,a2,a3,newleft,newright)
        #print('CASE3 Output')
        result = self._max_quadratic_function_within_range(a1, a2, a3, newleft, newright)
        #print(result)
        return result
        
class optimizer_linear_offline(object):
    def __init__(self,
                 base_id = [100],
                 prob_interval=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.825,0.9,1],
                 cost_vector = [4,6], 
                 budget = 5, 
                 model_id=[0,2],
                 datapath='../dataset/mlserviceperformance_RAFDB',
                 MLModelsClass = MLModels,
                 online = False,
                 split = False,
                 train_ratio = 1,
                 randseed = 0,
                 test_eval=False,
                 context = [0,1]):
        self.base_id = base_id
        self.prob_interval = prob_interval
        self.cost_vector = cost_vector
        self.budget = budget
        self.model_id = model_id
        self.split = split
        self.train_ratio = 1
        self.test_eval = test_eval
        model_id_all =base_id+model_id
        
        
        if(online==False):
            self.base_model = MLModelsClass(ModelID=base_id,datapath=datapath,contextset=context)
            self.all_model = MLModelsClass(ModelID=model_id_all, datapath=datapath,contextset=context)
            if(split == True):
                #print('train ratio',train_ratio)
                data_split(datapath= datapath+'/',randseed= randseed,train_ratio=train_ratio)
                self.base_model = MLModelsClass(ModelID=base_id,datapath=datapath+'/train/',contextset=context)
                self.all_model = MLModelsClass(ModelID=model_id_all, datapath=datapath+'/train/',contextset=context)
                self.base_model_test = MLModelsClass(ModelID=base_id,datapath=datapath+'/test/',contextset=context)
                self.all_model_test = MLModelsClass(ModelID=model_id_all, datapath=datapath+'/test/',contextset=context)                

        else:
            self.base_model = MLModelsClass(baseid = base_id, ModelID=model_id) # Remember to add context!!!
            self.all_model = MLModelsClass(baseid = base_id, ModelID=model_id_all) # Same as the above line

        self.prob2qvalue = self.base_model.prob2qvalue
        #self.service_model = MLModelsClass(model_id,datapath)
        #print('type of MLModels',type(MLModels))
        #maxprint('prob_value',prob_value)
        self.optimizer = None
        self.update_optimizer_params()
        
        #print('my r_0',self.r_0)
        #print('weight slop',self.weight_slop)
        #print('weight intersect', self.weight_intersect)
        #print('weight 0,',self.w_0_vector)
        #print('weight b 0,',self.b_0_vector)
    def update_mlmodel(self,modelid = 100, reward = 1, qvalue = 0.9, basereward = 0):
        #print('update all model')
        self.all_model.update(modelid = modelid, reward = reward, qvalue = qvalue)
        #print('update all model base')
        self.all_model.update_base(reward = basereward, qvalue=qvalue)
        #print('update base model base')
        self.base_model.update_base(reward = basereward, qvalue=qvalue)
        
        if(self.base_id[0] == modelid):
            #print('update base model')
            self.base_model.update(modelid = modelid, reward = reward, qvalue = qvalue)
        return 0
            
    def update_optimizer_params(self):
        cost_vector = self.cost_vector
        prob_interval=self.prob_interval
        base_id = self.base_id
        model_id = self.model_id
        budget = self.budget
        self.r_0 = self.base_model.get_r_0()
        self.w_0_vector, self.b_0_vector = self.all_model.get_linear_coef(
                prob_interval=self.prob_interval,
                conf_id = base_id[0],
                model_id=base_id[0])
        self.weight_slop = np.zeros((len(cost_vector),len(prob_interval)-1))
        self.weight_intersect = np.zeros((len(cost_vector),len(prob_interval)-1))
        for i in range(len(cost_vector)):
            slop, intersect = self.all_model.get_linear_coef(
                    prob_interval=prob_interval,
                    conf_id = base_id[0],
                    model_id=model_id[i])
            self.weight_slop[i]=slop
            self.weight_intersect[i] = intersect
        if(self.optimizer == None):    
            self.optimizer = optimizer_linear(
                    weight_slop = self.weight_slop,
                    weight_intersect = self.weight_intersect ,
                    w_0_vector = self.w_0_vector,
                    b_0_vector = self.b_0_vector,
                    prob_interval=prob_interval,
                    costvector = cost_vector, 
                    budget = budget,
                    r_0 = self.r_0
                    )
        else:
            self.optimizer.update_params(
                    weight_slop = self.weight_slop,
                    weight_intersect = self.weight_intersect ,
                    w_0_vector = self.w_0_vector,
                    b_0_vector = self.b_0_vector,
                    prob_interval=prob_interval,
                    costvector = cost_vector, 
                    budget = budget,
                    r_0 = self.r_0
                    )
        #print('prob interval',self.prob_interval)
        #print('updated weight slop',self.weight_slop)
        #print('updated weight weight_intersect',self.weight_intersect)



    def setbudget(self,budget):
        self.budget = budget 
        self.optimizer.setbudget(budget)
        
    def solve(self):
        # Remember to bring back the q value as well. 
        # The internal method only returns p
        opt, prob, active_index, Pi = self.optimizer.solve()
        #print('prob',prob)
        qvalue = self.prob2qvalue([prob], self.base_id[0])
        policy = (Pi, opt, qvalue, prob, active_index)
        baseid = self.base_id[0]
        #print('acc estimated',opt)
        result = self.evalpolicy(policy=policy,baseid=baseid,
                                 policytype='q_value',
                                 modelid=self.model_id)
        opt = result
        if(self.budget<0):
            opt = 0
        return Pi, opt, qvalue, prob, active_index

    def evalpolicy(self,policy,baseid, policytype,modelid):
        if(self.budget<0):
            return 0       
        if(self.test_eval==False):
            acc = self.all_model.eval_policy(policy,baseid, modelid,
                                             policytype
                                             )
        else:
            acc = self.all_model_test.eval_policy(policy,baseid, modelid,
                                                  policytype
                                                  )
                
        #print('acc computeed on datasets',acc)
        return acc

class optimizer_linear_offline_autobase(object):
    def __init__(self,
                 prob_interval=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.825,0.9,1],
                 budget_frac =[0,0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
                 context = [0,1,2,3,4,5,6],
                 #context = [0,1],
                 spline_degree = 1,
                 cost_vector_all = [0,4,6], 
                 budget = 5,
                 budget_num = 20,
                 model_id_all = [0,2,100],
                 datapath='../dataset/mlserviceperformance_RAFDB',
                 MLModelsClass = MLModels,
                 use_context = False,
                 online = False,
                 split = True,
                 train_ratio = 0.5,
                 randseed=0,
                 test_eval=True):
        self.myinf = 1e9
        self.optimizer_linear_list = list()
        self.budget = budget
        self.cost_vector_all =cost_vector_all
        self.budget_num = budget_num
        budget_list = np.arange(0,budget_num)*max(cost_vector_all)*2/(budget_num-1)
        self.budget_list = np.sort(np.append(budget_list,budget) )
        #print('budgetlist',self.budget_list)
        
        for i in range(len(cost_vector_all)):
            base_id = model_id_all[i]
            cost_vector = cost_vector_all[:i] + cost_vector_all[i+1:]
            budget = self.budget - cost_vector_all[i]
            model_id = model_id_all[:i] + model_id_all[i+1:]
            #print('i, baseid, prob, cost, budget, model id, modelid all ')
            #print(base_id,prob_interval,cost_vector, budget,model_id,model_id_all)
            if(use_context == True):
                #print('base_id,prob_interval,budget_frac, cost_vector, budget,model_id,datapath,model_id_all')
                #print(base_id,prob_interval,budget_frac, cost_vector, budget,model_id,datapath,model_id_all)
                optimizer = optimizer_linear_context_offline(base_id = [base_id],
                                                     prob_interval=prob_interval,
                                                     budget_frac=budget_frac,
                                                     context = context,
                                                     spline_degree=spline_degree,
                                                     cost_vector = cost_vector, 
                                                     budget = budget, 
                                                     model_id= model_id,
                                                     datapath=datapath,
                                                     MLModelsClass = MLModelsClass,
                                                     online=online,
                                                     split = split,
                                                     train_ratio = train_ratio,
                                                     randseed=randseed,
                                                     test_eval=test_eval)
            else:  
                optimizer = optimizer_linear_offline(base_id = [base_id],
                                                     prob_interval=prob_interval,
                                                     cost_vector = cost_vector, 
                                                     budget = budget, 
                                                     model_id= model_id,
                                                     datapath=datapath,
                                                     MLModelsClass = MLModels,
                                                     online=online,
                                                     split = split,
                                                     train_ratio = train_ratio,
                                                     randseed=randseed,
                                                     test_eval=test_eval)
            self.optimizer_linear_list.append(optimizer)
        
    def setbudget(self,budget):
        self.budget = budget 
        cost_vector_all = self.cost_vector_all 
        budget_num = self.budget_num
        budget_list = np.arange(0,budget_num)*max(cost_vector_all)*2/(budget_num-1)
        self.budget_list = np.sort(np.append(budget_list,budget) )
        #print('budgetlist',self.budget_list)
        for i in range(len(cost_vector_all)):
            budget0 = self.budget - cost_vector_all[i]
            self.optimizer_linear_list[i].setbudget(budget0)
            

            
    def solve(self):
        result1 = self.best_one_base()
        #print('best one model and the base id',result1)
        result2 = self.best_two_base()
        #print('best two models',result2)
        if(result1[1]>=result2[1]):
            return result1
        else:
            return result2
        #Pi, opt, qvalue, prob, active_index
        
    def best_one_base(self):
        result = self.optimizer_linear_list[0].solve()
        best_i = 0
        for i in range(len(self.optimizer_linear_list)):
            temp = self.optimizer_linear_list[i].solve()
            #print('i and solution ',i,temp)
            if(temp[1]>result[1]):
                result= temp
                best_i = i
        result = list(result)
        result.append(best_i)
        result.append('one_model')
        return result
    
    def best_two_base(self):
        opt = -self.myinf
        z1 = 0
        z2 = 1
        B1 = 0
        B2 = 0
        best_i = -1
        best_j = -1
        n = len(self.optimizer_linear_list)
        B = self.budget
        self.func_eval()
        #print('finishe eval with n',n)
        for i in range(n):
            for j in range(n):
                if(i==j):
                    continue
                c_k1 = self.cost_vector_all[i]
                c_k2 = self.cost_vector_all[j]
                #print('i,j',i,j)
                #print('c_k1,c_k2',c_k1,c_k2)
                #print('B value',B)
                if(c_k1<B or c_k2>B):
                    continue
                B1t,B2t, B1_index_t, B2_index_t, opt1 = self.solve_k1_k2_best(i,j)
                #print('B1t,B2t, B1_index_t, B2_index_t, opt1',B1t,B2t, B1_index_t, B2_index_t, opt1)
                if(opt1>opt):
                    opt = opt1
                    B1 = B1t
                    B2 = B2t
                    best_i = i
                    best_j = j
                    B1_index = B1_index_t
                    B2_index = B2_index_t
        
        if(B1>B2):
            z1 = (B-B2)/(B1-B2)
            z2 = (B1-B)/(B1-B2)
        policy = list()
        
        if(not(best_i==-1)):
            policy.append(self.func[best_i][B1_index])
            policy.append(self.func[best_j][B2_index])        
        
        return (policy, opt, z1,z2, B1, B2,best_i,best_j,'two_model')
        
    def solve_k1_k2_best(self,k1,k2):
        B1_best = 0
        B2_best = 0
        B1_index = -1
        B2_index = -1
        opt = -self.myinf
        B = self.budget
        for i in range(len(self.budget_list)):
            for j in range(len(self.budget_list)):
                f1 = self.func[k1][i][1]
                f2 = self.func[k2][j][1]
                B1 = self.budget_list[i]
                B2 = self.budget_list[j]
                if(B1>B and B2<B):
                    opt1 = (B-B2)/(B1-B2) *f1 + (B1-B)/(B1-B2)*f2
                    #print('')
                    if(opt1>opt):
                        B1_best = B1
                        B2_best = B2
                        opt = opt1
                        B1_index = i
                        B2_index = j
        return B1_best, B2_best, B1_index, B2_index, opt
    
    def func_eval(self):
        n = len(self.optimizer_linear_list)
        m = len(self.budget_list)
        self.func = list()
        for i in range(n):
            result = list()
            #print('i and j in 2base model func eval',i)
            for j in range(m):
                budget = self.budget_list[j] - self.cost_vector_all[i]
                self.optimizer_linear_list[i].setbudget(budget)
                result.append(self.optimizer_linear_list[i].solve())
            self.func.append(result)
        #print('eval before')
        return 0
    

class optimizer_linear_context_offline(object):
    def __init__(self,
                 base_id = [100],
                 prob_interval=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                 budget_frac =[0,0.025,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],
                 context = [0,1,2,3,4,5,6],
                 #context = [0,1],
                 spline_degree = 1,
                 cost_vector = [4,6], 
                 budget = 5, 
                 model_id=[0,2],
                 #model_id_all = [0,2,100],
                 MLModelsClass = MLModels,
                 online=False,
                 datapath='../dataset/mlserviceperformance_RAFDB',
                 split = False,
                 randseed=0,
                 train_ratio=0.5,
                 test_eval=False,
                 num_budget_init=14,
                 pgd_maxiter=3500):
        self.num_thread_get_acc = 35
        self.num_thread = 15
        self.num_thread_pgd = num_budget_init	
        self.num_thread_init_opt = 30
        #self.num_thread_init_opt_weight = 36
        self.degree = spline_degree # degree of the spline used for budget allocation
        self.context = context # context set
        self.base_id = base_id
        self.prob_interval = prob_interval
        self.cost_vector = cost_vector
        self.budget = budget
        self.model_id = model_id
        self.base_model = MLModelsClass(ModelID=base_id,datapath=datapath,contextset=context)
        self.split = split
        self.randseed = randseed
        self.train_ratio = train_ratio
        self.test_eval = test_eval
        #self.service_model = MLModels(model_id,datapath)
        model_id_all = base_id+model_id
        #print('context_linear context number',context)
        #print('model_id_all',model_id_all)
        self.all_model = MLModelsClass(ModelID=model_id_all,datapath=datapath,contextset=context)
        if(split == True):
            data_split(datapath= datapath+'/',randseed= randseed,train_ratio=train_ratio)
            self.base_model = MLModelsClass(ModelID=base_id,datapath=datapath+'/train/',contextset=context)
            self.all_model = MLModelsClass(ModelID=model_id_all, datapath=datapath+'/train/',contextset=context)
            self.base_model_test = MLModelsClass(ModelID=base_id,datapath=datapath+'/test/',contextset=context)
            self.all_model_test = MLModelsClass(ModelID=model_id_all, datapath=datapath+'/test/',contextset=context)   
            self.context_prob_test = self.all_model_test.get_context_prob()
                
        self.context_prob = self.all_model.get_context_prob()
        #print('optimizer obtain context prob:',self.context_prob)
        #print('start init optimizer')
        self._optimizer_init()
        #print('finish initi optimizer')
        self.budget_frac = budget_frac
        #x0 = np.ones(len(context))/(len(context))
        #x0[0] = 1
        #np.random.seed(2)
        self.num_budget_init = num_budget_init
        np.random.seed(2)
        #x0 = np.random.normal(size=len(context))
        x0 = np.random.uniform(size=len(context))
        #x0[0] = 0.2
        #x0[1] = 0.8
        #print('initial x0',x0)
        self.pgd_maxiter = pgd_maxiter
        self.pgd = ProjectGradientDescent(x0=x0,maxiter = self.pgd_maxiter )
        #print('finish initial!')
        #print('my r_0',self.r_0)
        #print('weight slop',self.weight_slop)
        #print('weight intersect', self.weight_intersect)
        #print('weight 0,',self.w_0_vector)
        #print('weight b 0,',self.b_0_vector)


    def _optimizer_init(self):
        return self._optimizer_init_fast_parallel()

    def _optimizer_init_fast_parallel(self):
        # We divide this into 4 steps.
        pool = multiprocessing.Pool(processes=self.num_thread_init_opt)
        # 1/4: Get r_0 list
        #print('start get r_0')
        self.r_0 = (pool.map(self.base_model.get_r_0, self.context))
        #print('finish get r_0')
        # 2/4: Get w_0 and b_0 list
        #print('start get w_0 b_0')		
        self.w0_and_b0_list = (pool.map(self._get_w0_b0, self.context))
        #print('finish get w_0 b_0')
        # 3/4: Get weight slop and weight inter list
        #print('start get weight')
        self.weight_slop_and_inter_list = (pool.map(self._get_weight_slop_inter, self.context))        
        #print('finish get wieht')
        # 4/4: Generate optimizer list
        #print('start generate optimizers')
        self.optimizers = (pool.map(self.generate_optimizer_context,self.context))    
        #print('finish generate optimizers')
        pool.close()
        pool.join()
        return 0
    
    def _optimizer_init_slow(self):    
        self.optimizers = list()
        self.r_0 = list()
        self.w_0_vector = list()
        self.b_0_vector = list()
        self.weight_slop = list()
        self.weight_intersect = list()
        context = self.context
        pool = multiprocessing.Pool(processes=self.num_thread)
        self.optimizers = (pool.map(self._optimizer_init_context, context))
        pool.close()
        pool.join()
        #self.optimizers = [self._optimizer_init_context(context[i]) for i in range(len(self.context))]
        '''
        for i in range(len(self.context)):
            optimizer1 = self._optimizer_init_context(context[i])
            self.optimizers.append(optimizer1)
        '''
        return 0

    
    def _get_w0_b0(self,context):
        #print('w0 and b0 context',context)
        w_0_vector, b_0_vector = self.all_model.get_linear_coef(
                prob_interval=self.prob_interval,
                conf_id = self.base_id[0],
                model_id=self.base_id[0],context=context)    
        return w_0_vector, b_0_vector

    def _get_weight_slop_inter(self,context):
        #print('weight slop and inter context',context)
        cost_vector=  self.cost_vector
        prob_interval = self.prob_interval
        weight_slop = np.zeros((len(cost_vector),len(prob_interval)-1))
        weight_intersect = np.zeros((len(cost_vector),len(prob_interval)-1))
        for i in range(len(cost_vector)):
            slop, intersect = self.all_model.get_linear_coef(
                    prob_interval=prob_interval,
                    conf_id = self.base_id[0],
                    model_id=self.model_id[i],context=context)
            weight_slop[i]=slop
            weight_intersect[i] = intersect
        #print('finish one weight slop')
        return weight_slop,weight_intersect
        
    def generate_optimizer_context(self,i):
        optimizer = (optimizer_linear(
            weight_slop = self.weight_slop_and_inter_list[i][0],
            weight_intersect = self.weight_slop_and_inter_list[i][1] ,
            w_0_vector = self.w0_and_b0_list[i][0],
            b_0_vector = self.w0_and_b0_list[i][1],
            prob_interval=self.prob_interval,
            costvector = self.cost_vector, 
            budget = self.budget,
            r_0 = self.r_0[i]
            ))  
        return optimizer
                
    def _optimizer_init_context(self,context):
        r_0 = self.base_model.get_r_0(context)
        self.r_0.append(r_0)
        prob_interval= self.prob_interval
        base_id = self.base_id
        cost_vector = self.cost_vector
        model_id = self.model_id
        budget = self.budget
        w_0_vector, b_0_vector = self.all_model.get_linear_coef(
                prob_interval=prob_interval,
                conf_id = base_id[0],
                model_id=base_id[0],context=context)
        self.w_0_vector.append(w_0_vector)
        self.b_0_vector.append(b_0_vector)
        weight_slop = np.zeros((len(cost_vector),len(prob_interval)-1))
        weight_intersect = np.zeros((len(cost_vector),len(prob_interval)-1))
        for i in range(len(cost_vector)):
            slop, intersect = self.all_model.get_linear_coef(
                    prob_interval=prob_interval,
                    conf_id = base_id[0],
                    model_id=model_id[i],context=context)
            weight_slop[i]=slop
            weight_intersect[i] = intersect
        optimizer = (optimizer_linear(
                weight_slop = weight_slop,
                weight_intersect = weight_intersect ,
                w_0_vector = w_0_vector,
                b_0_vector = b_0_vector,
                prob_interval=prob_interval,
                costvector = cost_vector, 
                budget = budget,
                r_0 = r_0
                ))            
        self.w_0_vector.append(w_0_vector)
        self.b_0_vector.append(b_0_vector)
        self.weight_slop.append(weight_slop)
        self.weight_intersect.append(weight_intersect)
        return optimizer
        
    def setbudget(self,budget):
        self.budget = budget 
        #self.optimizer.setbudget(budget)
        
    def solve(self):
        #print('start get get_acc_budget_func_and_gradient')
        self.get_acc_budget_func_and_gradient()
        #return None
        #print('context prob',self.context_prob)
        #print('sum',sum(self.context_prob))
        self.solve_budget_allocate()
        result = self.construct_policy()
        return result
        # Remember to bring back the q value as well. 
        # The internal method only returns p
        #opt, prob, active_index, Pi = self.optimizer.solve()
        #qvalue = self.base_model.prob2qvalue(prob, self.base_id[0])
        #if(self.budget<0):
        #    opt = 0        
        #return Pi, opt, qvalue, prob, active_index    
    def get_acc_budget_func_and_gradient(self):
        return self.get_acc_budget_func_and_gradient_fast_parallel()


    def get_acc_budget_func_and_gradient_fast_parallel(self):
        #time1 = time.time()
        #print('generate zip')
        index = range(len(self.optimizers))
        optimizer_index_zip = zip(self.optimizers, index)
        #print('finish zip')
        pool = multiprocessing.Pool(processes=self.num_thread_get_acc)
        #print('computing acc')
        acc = pool.starmap(self.get_acc_budget_func_and_gradient_one_optimizer, optimizer_index_zip)
        #print('finish acc')
        pool.close()
        pool.join()
        #time2 = time.time()
        #print('acc compute time',time2-time1)
        xs =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        self.xs = xs
        self.acc = acc
        #time1 = time.time()
        self.spline = [self.get_acc_budget_func_and_gradient_fast_parallel_plot(i) for i in range(len(self.optimizers))]
        self.spline_grad = [self.get_acc_budget_func_and_gradient_fast_parallel_get_grad(k) for k in self.spline]

        #self.spline = pool.map(self.get_acc_budget_func_and_gradient_fast_parallel_plot,range(len(self.optimizers)))
        #self.spline_grad = pool.map(self.get_acc_budget_func_and_gradient_fast_parallel_get_grad,self.spline)
        '''
        budget_frac = self.budget_frac
        self.spline=list()
        self.spline_grad= list()
        self.acc = acc
        for i in range(len(self.optimizers)):
            acci = acc[i]
            ptrfunci = UnivariateSpline(x=budget_frac,y=acci,k=self.degree,s=0)
            plt.clf()
            plt.plot(xs, ptrfunci(xs), 'gx', lw=3)
            plt.plot(budget_frac, acci, 'ro', lw=3)
            plt.savefig('temp_inter_acc_{}'.format(i))
            self.spline.append(ptrfunci)
            self.spline_grad.append(ptrfunci.derivative())
        '''    
        #time2 = time.time()
        #print('inerpolation time',time2-time1)
        return self.func, self.func_grad
    
    def get_acc_budget_func_and_gradient_fast_parallel_plot(self,i):
            acci = self.acc[i]
            ptrfunci = UnivariateSpline(x=self.budget_frac,y=acci,k=self.degree,s=0)
            #plt.clf()
            #plt.plot(self.xs, ptrfunci(self.xs), 'gx', lw=3)
            #plt.plot(self.budget_frac, acci, 'ro', lw=3)
            #plt.savefig('temp_inter_acc_{}'.format(i))
            return ptrfunci

    def get_acc_budget_func_and_gradient_fast_parallel_get_grad(self,func1):
        return func1.derivative()
        
    def get_acc_budget_func_and_gradient_fast(self):
        #print('generate zip')
        index = range(len(self.optimizers))
        optimizer_index_zip = zip(self.optimizers, index)
        #print('finish zip')
        #print('computing acc')
        acc = [self.get_acc_budget_func_and_gradient_one_optimizer(optimizer, i) for (optimizer,i) in optimizer_index_zip]
        #print('finish acc')
        
        
        budget_frac = self.budget_frac
        self.spline=list()
        self.spline_grad= list()
        for i in range(len(self.optimizers)):
            acci = acc[i]
            ptrfunci = UnivariateSpline(x=budget_frac,y=acci,k=self.degree,s=0)
            plt.clf()
            #xs =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            #plt.plot(xs, ptrfunci(xs), 'gx', lw=3)
            #plt.plot(budget_frac, acci, 'ro', lw=3)
            #plt.savefig('temp_inter_acc_{}'.format(i))
            self.spline.append(ptrfunci)
            self.spline_grad.append(ptrfunci.derivative())
        return self.func, self.func_grad

    def get_acc_budget_func_and_gradient_one_optimizer(self,optimizer1,i):
        acci = list()
        #print(i, 'th optimizer start')
        acci = [self.get_acc_budget_func_and_gradient_one_optimizer_budget_frac(optimizer1,i,budget_frac) for budget_frac in self.budget_frac]  
        #print(i, 'th optimizer finish')
        return acci
    
    def get_acc_budget_func_and_gradient_one_optimizer_budget_frac(self,optimizer1,i,budget_frac):
        budget = self.budget
        if(self.context_prob[i]==0):
            optimizer1.setbudget(10)
        else:
            optimizer1.setbudget(budget_frac*budget/self.context_prob[i])
        opt, prob, active_index, Pi = optimizer1.solve()
        return opt
        
    def get_acc_budget_func_and_gradient_slow(self):
        budget_frac = self.budget_frac
        budget = self.budget
        acc = list()
        self.spline=list()
        self.spline_grad= list()
        bestacc = 0
        for i in range(len(self.optimizers)):
            #print('start solve',i)

            acci = list()
            for j in range(len(budget_frac)):
                if(self.context_prob[i]==0):
                    self.optimizers[i].setbudget(10)
                else:
                    self.optimizers[i].setbudget(budget_frac[j]*budget/self.context_prob[i])
                #print('start solve',i,'with ',j, 'th budget')
                opt, prob, active_index, Pi = self.optimizers[i].solve()
                #print('end solve',i,'with ',j, 'th budget')
                acci.append(opt)    
            acc.append(acci)
            #print('acc i',acci)
            #print('budget',budget_frac)
            bestacc += self.context_prob[i]*acci[-1]
            #print('bestacc after', i,'iter',bestacc)
            #print('arm choose', Pi)
            ptrfunci = UnivariateSpline(x=budget_frac,y=acci,k=self.degree,s=0)
            #plt.clf()
            #xs =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            #plt.plot(xs, ptrfunci(xs), 'gx', lw=3)
            #plt.plot(budget_frac, acci, 'ro', lw=3)
            #plt.savefig('temp_inter_acc_{}'.format(i))
            self.spline.append(ptrfunci)
            self.spline_grad.append(ptrfunci.derivative())
            #print('end solve',i)
        return self.func, self.func_grad
    
    def func(self,x):
        y = 0
        for i in range(len(self.spline)):
            y = y + self.spline[i](x[i]) * self.context_prob[i]
        return -y
    
    def func_grad(self,x):
        y = np.zeros(len(self.spline_grad[0](x)))
        for i in range(len(self.spline)):
            y[i] = self.spline_grad[i](x[i]) * self.context_prob[i]
        return -y        
        
    def solve_budget_allocate(self):
        return self.solve_budget_allocate_fast_parallel()
    
    def solve_budget_allocate_slow(self):
        opt_best = 1.1
        budgetallocate_best = None
        for i in range(self.num_budget_init):
            np.random.seed(i)
            #x0 = np.random.normal(size=len(context))
            x0 = np.random.uniform(size=len(self.context))
            #print(i, 'th initial x0',x0)
            self.pgd = ProjectGradientDescent(x0=x0,maxiter = self.pgd_maxiter )            
            self.pgd.update_obj_and_grad(self.func,self.func_grad)
            budgetallocate, opt = self.pgd.solve()
            if(opt_best>opt):
                budgetallocate_best = budgetallocate
                opt_best = opt
            #print(i, 'th allocataed budget',budgetallocate)
            #print(i, 'th opt',opt)
        self.budgetallocate = budgetallocate_best
        self.opt = opt_best
        #print('final allocataed budget',self.budgetallocate)
        #print('final opt',self.opt)        
        return 0

    def solve_budget_allocate_fast_parallel(self):
        x0_list = list()
        for i in range(self.num_budget_init):
            #np.random.seed(i)
            #x0 = np.random.normal(size=len(context))
            x0 = np.random.uniform(size=len(self.context))
            x0_list.append(x0)
        pool = multiprocessing.Pool(processes=self.num_thread_pgd)        
        result = pool.map(self.budget_allocate_onerun,x0_list)
        pool.close()
        pool.join()
        opt_best = 1.1
        #opt_index = -1
        budgetallocate_best = None
        for i in range(len(result)):
            opt = result[i][1]
            if(opt<opt_best):
                #opt_index = i
                opt_best = opt
                budgetallocate_best = result[i][0]
        self.budgetallocate = budgetallocate_best
        self.opt = opt_best
        #print('final opt index',opt_index)
        #print('final allocataed budget',self.budgetallocate)
        #print('final opt',self.opt)        
        return 0
    
    def budget_allocate_onerun(self,x0):
        pgd = ProjectGradientDescent(x0=x0,maxiter = self.pgd_maxiter )            
        pgd.update_obj_and_grad(self.func,self.func_grad)
        budgetallocate, opt = pgd.solve()
        return budgetallocate, opt
            
    
    def construct_policy(self):
        return self.construct_policy_fast_parallel()
    
    def construct_policy_slow(self):
        policy_all = list()
        budget = self.budget
        opt_total = 0
        for i in range(len(self.context_prob)):
            budgeti = self.budgetallocate[i]
            if(self.context_prob[i]==0):
                budgetii = 10
            else:
                budgetii = budgeti*budget/self.context_prob[i]
            #print('actual bodget for ', i, 'th context:',budgeti*budget)
            self.optimizers[i].setbudget(budgetii)
            opt, prob, active_index, Pi = self.optimizers[i].solve()
            qvalue = self.base_model.prob2qvalue(prob, self.base_id[0],
                                                         self.context[i])
            policy = (Pi,opt,qvalue,prob,active_index)
            #print('actual cost',np.inner(Pi, a)*self.context_prob[i]*prob   )
            baseid= self.base_id[0]
            policytype = 'q_value'
            modelid = self.model_id
            context = self.context[i]
            opt_real = self._evalpolicy(policy,baseid, policytype,modelid,context)
            #print('opt real',opt_real)
            if(self.test_eval==False):
                opt_total += opt_real*self.context_prob[i]
            else:
                opt_total += opt_real*self.context_prob_test[i]
            result = (Pi, opt, qvalue, prob, active_index)
            policy_all.append(result)
        #print('real total opt',opt_total)
        #print('context dist',self.context_prob)
        return policy_all, opt_total, self.opt


    def construct_policy_fast_parallel(self):
        #print('start construct')
        #pool = multiprocessing.Pool(processes=self.num_thread)
        index  =list(range(len(self.context_prob)))
        policy_all = [self.construct_policy_fast_onerun(i) for i in index]
        #policy_all =  (pool.map(self.construct_policy_fast_onerun, index))
        self.policy_all = policy_all

        opt_total = 0
        #print('finish constrct')
        #print('start compute the opt eval')
        #opt_real = (pool.map(self._compute_real_cost,index))
        opt_real = [self._compute_real_cost(i) for i in range(len(self.context_prob))]
        #pool.close()
        #pool.join()
        if(self.test_eval==False):
            opt_total = np.sum(np.asarray(opt_real)*np.asarray(self.context_prob))
        else:
            opt_total = np.sum(np.asarray(opt_real)*np.asarray(self.context_prob_test))
        '''
        for i in range(len(self.context_prob)):
            policy = policy_all[i]
            #a = np.asarray(self.cost_vector)
            #print('actual cost',np.inner(Pi, a)*self.context_prob[i]*prob   )
            baseid= self.base_id[0]
            policytype = 'q_value'
            modelid = self.model_id
            context = self.context[i]
            opt_real = self._evalpolicy(policy,baseid, policytype,modelid,context)
            #print('opt real',opt_real)
            if(self.test_eval==False):
                opt_total += opt_real*self.context_prob[i]
            else:
                opt_total += opt_real*self.context_prob_test[i]
        '''		
        #print('real total opt',opt_total)
        #print('context dist',self.context_prob)
        return policy_all, opt_total, self.opt

    def _compute_real_cost(self,i):
        policy = self.policy_all[i]
        #a = np.asarray(self.cost_vector)
        #print('actual cost',np.inner(Pi, a)*self.context_prob[i]*prob   )
        baseid= self.base_id[0]
        policytype = 'q_value'
        modelid = self.model_id
        context = self.context[i]
        opt_real = self._evalpolicy(policy,baseid, policytype,modelid,context)
        return opt_real

    def construct_policy_fast_onerun(self,i):
        budget = self.budget
        budgeti = self.budgetallocate[i]
        if(self.context_prob[i]==0):
            budgetii = 10
        else:
            budgetii = budgeti*budget/self.context_prob[i]
        #print('actual bodget for ', i, 'th context:',budgeti*budget)
        #print('start ',i,'th construct')
        self.optimizers[i].setbudget(budgetii)
        opt, prob, active_index, Pi = self.optimizers[i].solve()
        #print('finish', i,' th construct')
        qvalue = self.base_model.prob2qvalue(prob, self.base_id[0],
                                             self.context[i])
        result = (Pi, opt, qvalue, prob, active_index)
        return result
    def _evalpolicy(self,policy,baseid, policytype,modelid,context):
        #print('xaa',policytype)
        if(self.budget<0):
            return 0
        if(self.test_eval==False):
            acc = self.all_model.eval_policy(policy,baseid, modelid,
                                             policytype,context
                                             )
        else:
            acc = self.all_model_test.eval_policy(policy,baseid, modelid,
                                                  policytype,context
                                                  )
        return acc
        
class ProjectGradientDescent(object):
    def __init__(self, stepsize = 1e-2,maxiter = 1e4, x0 = np.ones(2)*0.03, B=1):
        self.x0 = x0
        self.stepsize = stepsize
        self.maxiter = maxiter
        self.tol = 1e-12
        self.constraint = B

    def obj(self,x):
        return -x[0]*x[0]-3/2*x[1]*x[1]-x[0] +x[1]
        
    def obj_gradient(self,x):
        grad = np.zeros(2)
        grad[0] = -2*x[0]-1
        grad[1] = -3*x[1]+1
        return grad
        
    def project(self, x_input):
        return self.project_efficient(x_input)  
    
    def project_efficient(self,x_input):
        n = len(x_input)
        y = x_input
        y_sort = np.sort(y)
        i = n-2
        while(i>=0):
            ti = (np.sum(y_sort[i+1:n])-1)/(n-i-1)
            if(ti>=y_sort[i]):
                break
            i = i-1
            if(i==-1):
                ti = (np.sum(y_sort)-1)/n
        t = np.ones(n)*ti
        x_projected = np.maximum(y-t,np.zeros(n))
        return x_projected
    
    def update_obj_and_grad(self,f,g):
        self.obj = f
        self.obj_gradient = g
        
    def solve(self):
        x = self.project(self.x0)
        diff = 1e6
        stepsize = self.stepsize
        f_value = self.obj(x)
        f_best = f_value
        x_best = x
        maxiter =self.maxiter
        i = 0
        while( (diff>self.tol or diff<-self.tol) and i<maxiter):
            x1 = x - stepsize * self.obj_gradient(x)
            x = self.project(x1)
            f_value_new = self.obj(x)
            diff = f_value_new-f_best
            if(f_value_new<f_best):
                x_best = x
                f_best =f_value_new
            i=i+1
        return x_best, f_best