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
from scipy import stats
#from branching import branching_method_online
#from sklearn.naive_bayes import CategoricalNB

class MLModels():
    def __init__(self, ModelID=[100],datapath='../dataset/mlserviceperformance_RAFDB',
                 contextset = [0,1,2,3,4,5,6]):
                 #contextset = [0,1]):
        baseid = ModelID[0]
        path1 = datapath+'/Model'
        #path2 = '_TotalCost.txt'
        #path3 = '_Cost.txt'
        path4 = '_TotalReward.txt'
        path5 = '_Reward.txt'
        path6 = '_ImageName.txt'
        path7 = '_Confidence.txt'
        path8 = '_PredictedLabel.txt'
        path9 = '_TrueLabel.txt'
        self.baseid = baseid
        self.ModelID = ModelID
        # Load Image Name
        Loadpath = path1+str(ModelID[0])+path6
        self.dataname = np.loadtxt(Loadpath, dtype=str)
        # Load True Label
        Loadpath = path1+str(ModelID[0])+path9
        self.truelabel = np.loadtxt(Loadpath, dtype=str)
        self.reward = list()
        self.confidence = list()
        self.prediction = list()
        self.context = list()
        self.probas = list()
        self.predefineddataorder = list()
        self.usepredefinedorder = True
        self.datacalls = 0
        self.n = len(ModelID)
        self.CombinedModel = None
        self.eps = 1e-6
        self.modeliddict = dict()
        for i in range(len(ModelID)):
            # Load Reward
            Loadpath = path1+str(ModelID[i])+path5
            self.reward.append(np.loadtxt(Loadpath))
            # Load Confidence
            Loadpath = path1+str(ModelID[i])+path7
            if(ModelID[i]==1):
                self.confidence.append(np.loadtxt(Loadpath)/100)
            else:                
                self.confidence.append(np.loadtxt(Loadpath))
            # Load Predicted Label
            Loadpath = path1+str(ModelID[i])+path8
            self.prediction.append(np.loadtxt(Loadpath))
            self.context.append(np.loadtxt(Loadpath)) # prediction value is viewed as a context
            # Load the accuracy of the model Label
            Loadpath = path1+str(ModelID[i])+path4
            TotalReward = np.loadtxt(Loadpath)
            self.probas.append(TotalReward[len(TotalReward)-1]/len(TotalReward))
            self.modeliddict[ModelID[i]] = i
        self.selecteddatapoint = -1
        self.best_proba = max(self.probas)
        self.SimpleBase = True
        self.contextset = contextset
        self._compute_context_prob()
        self.allone = np.ones(len(self.dataname))
        #self.construct_predefined_order()
    
    def _baseid2index(self):
        baseid = self.baseid
        for i in range(len(self.ModelID)):
            if(baseid == self.ModelID[i]):
                return i
    
    def get_context_prob(self):
        #print('context prob get', self.contextprob)
        return self.contextprob 
    
    def  _compute_context_prob(self):
        return self._compute_context_prob_fast()
    
    def  _compute_context_prob_slow(self):
        baseindex = self._baseid2index()
        contextprob = list()
        contextset = self.contextset
        total = len(self.dataname)
        for i in range(len(contextset)):
            c1 = 0
            context = contextset[i]
            for j in range(len(self.dataname)):
                if(self._iscontext(self.prediction[baseindex][j],context)):
                    c1+=1
            contextprob.append(c1/total)
        self.contextprob = contextprob
        #print('context prob', self.contextprob)
        #print('total',total)

    def  _compute_context_prob_fast(self):
        baseindex = self._baseid2index()
        contextprob = list()
        contextset = self.contextset
        total = len(self.dataname)
        for i in range(len(contextset)):
            c1 = 0
            context = contextset[i]
            CountMatrix = self._iscontext(self.prediction[baseindex],context)
            c1 = np.sum(CountMatrix)
            contextprob.append(c1/total)
        self.contextprob = contextprob
        #print('context prob', self.contextprob)
        #print('total',total)
        
    def get_linear_coef(self,
                        prob_interval,
                        conf_id,
                        model_id,
                        context=None):
        #print('prob_interval--------xxxxx',prob_interval)

        q_interval = self.prob2qvalue(prob_interval, conf_id,context)
        #print('prob_interval--------',prob_interval)
        #print('q_interval',q_interval)
        if(len(q_interval)==0):
            q_interval = np.ones(len(prob_interval))
        model_id_list = list()
        model_id_list.append(model_id)
        #print('model id array',model_id_list)
        BaseAccuracy, ModelAccuracy = self.accuracy_condition_score_list(
                ScoreRange=q_interval, 
                BaseID = conf_id, 
                ModelID=model_id_list,
                context=context)
        #print('my base, my model acc',BaseAccuracy, ModelAccuracy)

        return self._accuracy2linearweight(prob_interval=prob_interval,ModelAccuracy=ModelAccuracy[0])
    
    def get_r_0(self,context=None):
        a,b,c,d,e = self.compute_conditional_accuracy_among_model_inverse(ScoreBound=1.1, ModelID=[0,0],context=context)
        return a
    
    def prob2qvalue(self,prob_interval, conf_id,context=None):
        index = self.modeliddict[conf_id]
        Prob = self.confidence[index]
        #print('Prob',Prob)
        if(not(context==None)):
            Prob = self._get_prob_context(index, context)
        # Does not find any value within the context, i.e., the context never occurs
        if(len(Prob)==0):
            #print('Warning: Nothing in the context within the given prob')
            return []
        #print('max of prob',max(Prob))
        #print('prob_interval',prob_interval)
        #print('prob',Prob)
        #print('prob_interval',prob_interval)
        q_range = np.quantile(Prob, prob_interval,interpolation='lower')
        #print('q range',q_range)
        return q_range

    def _get_prob_context(self,index, context):
        Prob = self.confidence[index]
        ContextforProb = self.context[index]
        prob_context = list()
        for i in range(len(Prob)):
            if(ContextforProb[i]==context):
                prob_context.append(Prob[i])
        return prob_context
    
    def qvalue2prob(self,q_value,conf_id,context=None):
        index = self.modeliddict[conf_id]
        Prob = self.confidence[index]
        if(not(context==None)):
            Prob = self._get_prob_context(index, context)
        prob_of_q = stats.percentileofscore(Prob, q_value,kind='weak')/100

        return prob_of_q
    
    def _accuracy2linearweight(self, prob_interval,ModelAccuracy):
        n = len(prob_interval)
        weight_slop = np.zeros(n-1)
        weight_intersect = np.zeros(n-1)
        for i in range(n-1):
            if(prob_interval[i+1]==prob_interval[i]):
                weight_slop[i] = 0
                weight_intersect[i] = ModelAccuracy[i]
            else:
                weight_slop[i] = (ModelAccuracy[i+1]-ModelAccuracy[i])/(prob_interval[i+1]-prob_interval[i])
                weight_intersect[i] = ModelAccuracy[i] - weight_slop[i]*prob_interval[i]
        return weight_slop, weight_intersect
    
    def accuracy_condition_score_list(self, 
                                      ScoreRange=(0.1,0.5,0.9), 
                                      BaseID = 100, 
                                      ModelID=[100],
                                      context=None):
        BaseAccuracy = list()
        ModelAccuracy = list()
        #print('prob',self.probas)
        for j in ScoreRange:
            A1,A2,C,D,E = self.compute_conditional_accuracy_among_model_inverse(j,[self.modeliddict[BaseID],self.modeliddict[ModelID[0]]],context)            
            BaseAccuracy.append(A1)        
        for i in range(len(ModelID)):
            Model1 = list()
            for j in ScoreRange:
                A1,A2,C,D,E = self.compute_conditional_accuracy_among_model_inverse(j,[self.modeliddict[BaseID],self.modeliddict[ModelID[i]]],context)
                Model1.append(A2)
            ModelAccuracy.append(Model1)
        return BaseAccuracy, ModelAccuracy

    def accuracy_condition_score_list_cdf2pdf(self,
                                              probrange,
                                              base_acc,
                                              other_acc,
                                              diff = False,
                                              ):
        base_acc_pdf = self._accuracy_condition_score_cdf2pdf(probrange,base_acc)
        other_acc_pdf = list()
        for i in range(len(other_acc)):
            result = self._accuracy_condition_score_cdf2pdf(probrange,other_acc[i])
            if(diff==True):
                a = result
                b = base_acc_pdf
                resultnew = [i - j for i, j in zip(a, b)]
                other_acc_pdf.append(resultnew)
            else:
                other_acc_pdf.append(result)
        return base_acc_pdf,other_acc_pdf
    
    def _accuracy_condition_score_cdf2pdf(self,
                                          probrange,
                                          acc
                                          ):
        acc_pdf = list()
        acc_last = 0
        for i in range(len(probrange)):
            if(i==0):
                acc_pdf.append(acc[0])
                acc_last = acc[0]
            else:
                if(probrange[i]==probrange[i-1]):
                    acc_middle = acc_last
                else:
                    acc_middle = (acc[i]*probrange[i]-acc[i-1]*probrange[i-1])/(probrange[i]-probrange[i-1])
                    acc_last = acc_middle
                acc_pdf.append(acc_middle)
        return acc_pdf
    
    def compute_conditional_accuracy_among_model_inverse(self, ScoreBound=0.9, ModelID=[0,1],context=None):
        return self.compute_conditional_accuracy_among_model_inverse_fast(ScoreBound=ScoreBound,ModelID=ModelID,context=context)

    def compute_conditional_accuracy_among_model_inverse_slowloop(self, ScoreBound=0.9, ModelID=[0,1],context=None):
        index1 = ModelID[0]
        index2 = ModelID[1]
        Count = 0
        CountCorrect1 = 0
        CountCorrect2 = 0
        #print('class of confidence',type(self.confidence[index1]))
        for i in range(len(self.dataname)):
            if(self.confidence[index1][i]<=ScoreBound and self._iscontext(self.context[index1][i], context ) ) :
                Count = Count+1
                CountCorrect1 += self.reward[index1][i]
                CountCorrect2 += self.reward[index2][i]
        if Count ==0:
            Count = Count + 1
        #print('total count happens',Count)
        #result = self.compute_conditional_accuracy_among_model_inverse_fast(ScoreBound=ScoreBound,ModelID=ModelID,context=context)
        #print('count, correct1, correct2',Count, CountCorrect1, CountCorrect2)
        #print('fast result',result)
        return CountCorrect1/Count, CountCorrect2/Count, Count, CountCorrect1, CountCorrect2 

    def compute_conditional_accuracy_among_model_inverse_fast(self, ScoreBound=0.9, ModelID=[0,1],context=None):
        #print('start compute accuracin verse fast')
        index1 = ModelID[0]
        index2 = ModelID[1]
        Count = 0
        CountCorrect1 = 0
        CountCorrect2 = 0
        #time1 = time.time()
        ConditionMatrix = (np.logical_and(self.confidence[index1]<=ScoreBound,self._iscontext(self.context[index1], context )))
        #time2 = time.time()
        #print('logical time',time2-time1)
        Count = np.inner(ConditionMatrix, self.allone) #np.sum(ConditionMatrix)
        #time3 = time.time()
        #print('count time',time3-time2)
        CountCorrect1 = np.inner((self.reward[index1]),ConditionMatrix)
        CountCorrect2 = np.inner((self.reward[index2]),ConditionMatrix)
        
        #CountCorrect1 = np.sum(self.reward[index1]*ConditionMatrix) #np.inner((self.reward[index1]),ConditionMatrix)
        #CountCorrect2 = np.sum(self.reward[index2]*ConditionMatrix)#np.inner((self.reward[index2]),ConditionMatrix)
        #time4 = time.time()
        #print('correct time',time4-time3)
        if Count ==0:
            Count = Count + 1
        #print('total count happens',Count)
        return CountCorrect1/Count, CountCorrect2/Count, Count, CountCorrect1, CountCorrect2 


        
    def _iscontext(self,nowcontext, context=None):
        #print('nowcontext and context',nowcontext,context)
        if(context==None):
            return True
        else:
            return (nowcontext==context)
        
    def compute_prob_wrt_confidence(self,confidence_range=[0.9], BaseID=0,ModelID=0,context=None):
        index1 = self.modeliddict[BaseID]
        Prob=list()
        Count = 0
        ContextNumber = 0
        for i in range(len(self.dataname)):
            if(self._iscontext(self.context[index1][i],context)):
                ContextNumber += 1        
        for j in confidence_range:
            Count = 0
            for i in range(len(self.dataname)):
                if(self.confidence[index1][i]<=j and self._iscontext(self.prediction[index1][i],context)):
                    Count = Count+1
            Count = Count/ContextNumber
            Prob.append(Count)
        return Prob 
    
    def compute_prob_vs_score(self, ScoreRange=[0.1,0.3,0.9], BaseID=0,context=None):
        index1 = BaseID
        Prob=list()
        Count = 0
        ContextNumber = 0
        for i in range(len(self.dataname)):
            if(self._iscontext(self.context[index1][i],context)):
                ContextNumber += 1
        for j in ScoreRange:
            Count = 0
            for i in range(len(self.dataname)):
                if(self.confidence[index1][i]<=j and self._iscontext(self.context[index1][i],context) ):
                    Count = Count+1
            Count = Count/ContextNumber
            Prob.append(Count)
        return Prob



    def eval_policy(self,policy,baseid = 100, modelid = [0,2], policytype=None,
                    context=None):
        #print('type,',policytype)
        if(policytype=='q_value'):
            #print('enter')
            return self._eval_policy_qvalue(policy=policy,
                                            baseid = baseid, 
                                            modelid = modelid,
                                            policytype=policytype,
                                            context=context)


    def _eval_policy_qvalue(self,policy,baseid = 100, modelid = [0,2], policytype=None,
                    context=None):
        if(1):
            # policy of the type: Pi, opt, q value, prob_value, activeindex
            Pi = policy[0]
            opt = policy[1]
            q = policy[2]
            prob = policy[3]
            activeindex = policy[4]
            total = 0
            correct = 0
            #context=3
            #print('eval with q',q)
            #print('model dict',self.modeliddict)
            useML_bound = len(self.dataname)*prob
            useML = 0
            for i in range(len(self.dataname)):
                #print('sdfs',self.modeliddict[baseid])
                baseindex = self.modeliddict[baseid]
                prediction = self.prediction[baseindex][i]
                conf = self.confidence[self.modeliddict[baseid]][i]
                truelabel = int(self.truelabel[i])
                if(not(self._iscontext(self.context[baseindex][i], context ))):
                    continue
                #if(conf>q):
                if (conf>q or (conf==q and useML>=useML_bound)):
                    correct += self.reward[baseindex][i]                    
                else:
                    useML+=1
                    #print('use not base')
                    for j in ((activeindex)):
                        prob_call = Pi[j]
                        model = self.modeliddict[modelid[j]]
                        #print('model',model)
                        prediction_new = int(self.prediction[model][i])
                        #print('prob_call',prob_call)
                        #print('j',j)
                        #print('pred, label',prediction_new,(truelabel))
                        correct += prob_call*(self.reward[model][i])
                total += 1
            #print('mlmodels correct','total',correct,total)
            if(total == 0):
                return 0
            return correct/total
        
    def Compute_Conditional_Accuracy_AmongModel_Inverse(self, ScoreBound=0.9, ModelID=[0,1]):
        index1 = ModelID[0]
        index2 = ModelID[1]
        Count = 0
        CountCorrect1 = 0
        CountCorrect2 = 0
        for i in range(len(self.dataname)):
            if(self.confidence[index1][i]<=ScoreBound):
                Count = Count+1
                CountCorrect1 += self.reward[index1][i]
                CountCorrect2 += self.reward[index2][i]
                #if(self.prediction[index1][i] == float(self.truelabel[i])):
                #    CountCorrect1 = CountCorrect1 + 1
                #if(self.prediction[index2][i] == float(self.truelabel[i])):  
                #    CountCorrect2 = CountCorrect2 + 1
        if Count ==0:
            Count = Count + 1
        return CountCorrect1/Count, CountCorrect2/Count, Count, CountCorrect1, CountCorrect2 
    
    def Compute_Conditional_Accuracy(self, ScoreBound=0.9):
        Count = 0
        CountCorrect = 0
        for i in range(len(self.dataname)):
            if(self.confidence[0][i]>ScoreBound):
                Count = Count+1
                if(self.prediction[0][i] == float(self.truelabel[i])):
                    CountCorrect = CountCorrect + 1
        return CountCorrect/Count, Count, CountCorrect
    
    def Compute_Conditional_Accuracy_AmongModel(self, ScoreBound=0.9, ModelID=[0,1]):
        index1 = ModelID[0]
        index2 = ModelID[1]
        Count = 0
        CountCorrect1 = 0
        CountCorrect2 = 0
        for i in range(len(self.dataname)):
            if(self.confidence[index1][i]>ScoreBound):
                Count = Count+1
                if(self.prediction[index1][i] == float(self.truelabel[i])):
                    CountCorrect1 = CountCorrect1 + 1
                if(self.prediction[index2][i] == float(self.truelabel[i])):  
                    CountCorrect2 = CountCorrect2 + 1
        if Count ==0:
            Count = Count + 1
        return CountCorrect1/Count, CountCorrect2/Count, Count, CountCorrect1, CountCorrect2 

    def Compute_Conditional_Accuracy_AmongModel_List(self, ScoreRange=(0.1,0.5,0.9), BaseID = 0, ModelID=[0,1]):
        BaseAccuracy = list()
        ModelAccuracy = list()
        #print('prob',self.probas)
        for j in ScoreRange:
            A1,A2,C,D,E = self.Compute_Conditional_Accuracy_AmongModel_Inverse(j,[BaseID,ModelID[0]])            
            BaseAccuracy.append(A1)        
        for i in range(len(ModelID)):
            Model1 = list()
            for j in ScoreRange:
                A1,A2,C,D,E = self.Compute_Conditional_Accuracy_AmongModel_Inverse(j,[BaseID,ModelID[i]])
                Model1.append(A2)
            ModelAccuracy.append(Model1)
        return BaseAccuracy, ModelAccuracy