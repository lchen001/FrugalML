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

import numpy as np
from mlmodels import MLModels
from optimizer import optimizer_single_offline
from optimizer import OptimizerFrugalML
import jsonpickle
optimizerset = {'FrugalML':0,'FrugalMLFixBase':1,'FrugalMLQSonly':2}
import argparse

class OptimizerEvaluator(object):
    def __init__(self,
                 datapath='../dataset/mlserviceperformance_FERPLUS',
                 split = True,
                 train_ratio = 0.5,
                 test_eval = True,
                 randseedset = [1,5,10,50,100],
                 baseid = 100,
                 optimizername = ['FrugalMLFixBase', 'FrugalMLQSonly','FrugalML'],
                 dataname='FERPLUS',
                 taskname='fer',
                 tradeoff_num=50):
        
        self.datapath = datapath
        self.split = split
        self.train_ratio = train_ratio
        self.randseedset = randseedset
        self.test_eval = test_eval
        self.baseid = baseid
        self.dataname = dataname
        self.taskname = taskname
        self.tradeoff_num = tradeoff_num

        self.optimizername = optimizername
        self.fulloptimizername = self._getoptimizername()
        print('Full Optimizer List',self.fulloptimizername)
        self.outputsavepath = '../output/'
        self.budget = 10 # Only used for semantic purpose.
    def run(self):
        for i in range(len(self.randseedset)):
            self.randseed = self.randseedset[i]
            self.optimizer_list = list()
            self.budget_list = list()
            print('name of optimizer', self.fulloptimizername)
            for j in range(len(self.fulloptimizername)):
                self.optimizer_list.append(self._gen_optimizer(self.fulloptimizername[j]))
            self.acc_budget_tradeoff()         
        return 0

    def acc_budget_tradeoff(self):
        acc_list = list()
        policy_list = list()
        for i in range(len(self.optimizer_list)):
            print(i, 'th optimizer', self.fulloptimizername[i])
            budgetfile = self.outputsavepath + self.dataname + '_split_'+str(self.split)+'_trainratio_'+str(self.train_ratio)+'_randseed_'+str(self.randseed)+'_testeval_'+str(self.test_eval)+'_optname_'+str(self.fulloptimizername[i])+'_budget.txt'
            accfile = self.outputsavepath + self.dataname + '_split_'+str(self.split)+'_trainratio_'+str(self.train_ratio)+'_randseed_'+str(self.randseed)+'_testeval_'+str(self.test_eval)+'_optname_'+str(self.fulloptimizername[i])+'_acc.txt'
            policyfile = self.outputsavepath + self.dataname + '_split_'+str(self.split)+'_trainratio_'+str(self.train_ratio)+'_randseed_'+str(self.randseed)+'_testeval_'+str(self.test_eval)+'_optname_'+str(self.fulloptimizername[i])+'_policy.txt'
            budget_list = self.budget_list[i]
            acc, policy = self._get_acc(i,budget_list)
            policy_list=policy
            print('acc',acc)
            acc_list.append(acc)
            np.savetxt(budgetfile,budget_list)
            np.savetxt(accfile,acc)
            self._write_policy(policyfile, policy_list)
        self.acc_list = acc_list
        return 0
    
    def _write_policy(self,filename,policy_list):
        policy_handle = open(filename,'w')
        print('len of write policy',len(policy_list))
        for www in range(len(policy_list)):
            policy_handle.writelines(jsonpickle.encode(policy_list[www],unpicklable=True))
            policy_handle.write('\n')
        policy_handle.close()		
        return 0    		

    def _read_policy(self,filename):
        policy_handle = open(filename,'r')
        data = policy_handle.readlines()
        print('len of policy',len(data))
        policy_list = list()
        for www in range(len(data)):
            policy_list.append(jsonpickle.decode(data[www]))
        policy_handle.close()	
        return policy_list
		
    def _get_acc(self,optimizer_index, budget_list):
        optimizer = self.optimizer_list[optimizer_index]
        acc = np.zeros(len(budget_list))
        result_list = list()
        for i in range(len(budget_list)):
            mybud = budget_list[i]
            optimizer.setbudget(mybud)
            print('use budget',mybud)
            result = optimizer.solve()
            acc[i] = result[1]
            result_list.append(result)
        return acc, result_list   
    
    def _getoptimizername(self):
        fullname = list()
        myoptimizer = OptimizerFrugalML(datapath=self.datapath,
                                        split=self.split, 
                                        train_ratio=self.train_ratio,
                                        method='FrugalML',
                                        baseid=self.baseid,
                                        randseed=1)
        api_ids, api_name, cost_list = myoptimizer.getmarketinfo()
        fullname = api_ids
        fullname = fullname + self.optimizername
        self.optimizertemp = myoptimizer
        self.cost_vector_all = cost_list
        self.api_ids = api_ids
        ## Model id to index dict
        self.model_id2index_dict = dict()
        for i in range(len(api_ids)):
            model_id = api_ids[i]
            self.model_id2index_dict[model_id] = i    
        return fullname
    
    def _gen_optimizer(self,name=100):
        Maxbudget = (sorted(self.cost_vector_all)[-2]+sorted(self.cost_vector_all)[-1])/2
        print('tested maxbudget',Maxbudget)
        num = self.tradeoff_num
        if(type(name) == int):
            modelid = name
            index = self.model_id2index_dict[modelid]
            cost = self.cost_vector_all[index]
            budget = self.budget
            optimizer = optimizer_single_offline(
                    cost = cost, 
                    budget = budget, 
                    model_id= modelid,
                    datapath=self.datapath,
                    MLModelsClass = MLModels,
                    online = False,
                    num_of_label = None)
            budget_list = np.arange(cost+0.001,Maxbudget*2+1,(Maxbudget*2-cost)/num)
 
        if(name in optimizerset):         
            optimizer = OptimizerFrugalML(datapath = self.datapath,
                                          split=self.split,
                                          train_ratio=self.train_ratio,
                                          method=name,
                                          baseid=self.baseid,
                                          randseed=self.randseed)
            budget_list = np.arange(min(self.cost_vector_all)+0.001,Maxbudget*2+1,(Maxbudget*2)/num)
        print('method name',name, type(name)==int)
        self.budget_list.append(budget_list)            
        return optimizer     

def evaluate_all_methods(datapath,
                         dataname,
                         task,
                         train_ratio,
                         randseedset,
                         baseid=100):
    myevaluator = OptimizerEvaluator(datapath=datapath,
                                     split = True,
                                     train_ratio=train_ratio,
                                     test_eval = True,
                                     randseedset = randseedset,
                                     baseid= baseid,
                                     dataname=dataname,
                                     taskname=task
                                     )
    myevaluator.run()
    return 
    
def main():
    '''
    Test optimizers
    '''   
    parser = argparse.ArgumentParser(description='FrugalML Evaluate')
    parser.add_argument('--task', type=str,help='task name',
                        default='fer')
    parser.add_argument('--datapath', type=str,help='Datapath',
                        default='../dataset/mlserviceperformance_RAFDB')
    parser.add_argument('--dataname', type=str,help='Datapath',
                        default='../dataset/mlserviceperformance_RAFDB')
    parser.add_argument('--train_ratio', type=float,help='training data ratio',
                        default=0.5)
    parser.add_argument('--baseid', type=int,help='Base id \
                        only valid for method {FrugalMLQSonly} \
                            and {FrugalMLFixBase}',
                        default=100)

    args = parser.parse_args()

    print('task:',args.task)
    print('datapath:',args.datapath)
    print('dataname:',args.dataname)
    print('train_ratio:',args.train_ratio)
    print('baseid (for baseline methods):',args.baseid)    

    task = args.task
    dataname = args.dataname 
    datapath= args.datapath 
    train_ratio = args.train_ratio
    baseid = args.baseid
    randseedset=[1,5,10,50,100]
    evaluate_all_methods(datapath=datapath,
                         dataname=dataname,
                         task=task,
                         train_ratio=train_ratio,
                         randseedset=randseedset,
                         baseid=baseid)

if __name__ == '__main__':
    main()
    

