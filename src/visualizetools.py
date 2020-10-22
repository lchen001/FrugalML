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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import argparse

class VisualizeTool(object):
    def __init__(self, 
                 optimizer_name = ['100','0','1','2', 'FrugalML','FrugalMLQSOnly', 'FrugalMLFixBase'],
                 optimizer_legend = ['Base','Google Vision','Face++','MS Face', 'FrugalML','FrugalMLQSOnly', 'FrugalMLFixBase'],
                 optimizer_linemarker = ['X','^','v','s','o','*','d','p'],
                 optimizer_linestyle = ['-.','--','--','-.','-',':','--','-.'],
                                  
                 skip_optimizer=4,
                 skip_optimizer_shift_x = [0,0,0,0],
                 skip_optimizer_shift_y = [0,0,0,0],
                 figureformat = 'jpg',
                 figurefolder = "../figures/", #  folder to save the figures
                 dataname='FERPlus',                 
                 split = True,
                 train_ratio = 0.5,
                 randseedset=[1,5,10,50,100],
                 randseedsetstat = [1,5,10,50,100],
                 test_eval=True,
                 outputsavepath = '../output/', # folder to load the data
                 show_legend = True,
                 task = 'fer',
                 stat_efficiency = False,
                 train_ratio_set=[0.005,0.01,0.05,0.1,0.2,0.5],
                 data_size = 6358,
                 plot_frac = 0.82,
                 ):

        # Default values to set up the figures
        self.figuresize_tradeoff=(9,8)
        self.fontsize = 35
        self.linewidth = 4
        self.markersize = 8
        self.elinewidth = 2
        #self.colorset=['lime','violet','g','b','k','orange','r','yellow','m','pink','g']
        self.colorset=['lime','violet','g','b','r','k','orange','yellow','pink','g']

        self.optimizer_name = optimizer_name
        self.optimizer_legend = optimizer_legend
        self.optimizer_linemarker = optimizer_linemarker
        self.optimizer_name = optimizer_name
        self.optimizer_linestyle = optimizer_linestyle
        
        # control legend position for single API
        self.skip_optimizer_shift_x = skip_optimizer_shift_x 
        self.skip_optimizer_shift_y = skip_optimizer_shift_y
        self.skip_optimizer = skip_optimizer


        self.figureformat = figureformat
        self.folder = figurefolder
        self.dataname = dataname
        self.split = split
        self.train_ratio = train_ratio
        self.randseedset = randseedset
        self.randseedsetstat = randseedsetstat
        self.test_eval = test_eval
        self.outputsavepath = outputsavepath
        self.show_legend = show_legend
        self.task = task
        self.train_ratio_set = train_ratio_set
        self.stat_efficiency = stat_efficiency
        self.train_ratio_origin = train_ratio
        self.datasize = data_size
        self.plot_frac = plot_frac # fraction of the data to plot. This is used to mitigate the issue of too long flat line.

    def run(self):
        filename = self.folder+"Offlineaccbudget{}.{}".format(self.dataname,self.figureformat)            
        print("load acc and budget files...")        
        self.loadacc()
        print("finish loading!")
        print("plot accuracy budget tradeoff curve...")        
        self.plot_acc_bud_tradeoff(filename) 
        print("finish plotting!")
        if(self.stat_efficiency==True):
            self.plot_frac = 1
            self.randseedset = self.randseedsetstat
            filename = self.folder+"Offlineacctrainratio{}.{}".format(self.dataname,self.figureformat)                     
            print("plot accuracy sample size tradeoff curve...")    
            self.acc_vs_samplesize(filename)
            print("finish ploting!")
        return 0
    
    def loadacc(self):
        self.acc_list = list()
        self.budget_list = list()
        plot_frac = self.plot_frac
        for i in range(len(self.optimizer_name)):
            bud_opt_i = list()
            acc_opt_i = list()
            for j in range(len(self.randseedset)):
                randseed = self.randseedset[j]
                budgetfile = self.outputsavepath + self.dataname + '_split_'+str(self.split)+'_trainratio_'+str(self.train_ratio)+'_randseed_'+str(randseed)+'_testeval_'+str(self.test_eval)+'_optname_'+self.optimizer_name[i]+'_budget.txt'
                accfile = self.outputsavepath + self.dataname + '_split_'+str(self.split)+'_trainratio_'+str(self.train_ratio)+'_randseed_'+str(randseed)+'_testeval_'+str(self.test_eval)+'_optname_'+self.optimizer_name[i]+'_acc.txt'
                budget_list = np.loadtxt(budgetfile)
                budget_list = budget_list[0:math.ceil(plot_frac*len(budget_list))]
                bud_opt_i.append(budget_list)
                acc = np.loadtxt(accfile)
                acc = acc[0:math.ceil(plot_frac*len(acc))]
                acc_opt_i.append(acc)
            self.budget_list.append(bud_opt_i)
            self.acc_list.append(acc_opt_i)
        #print('budget list',self.budget_list)
        #print('acc_list',self.acc_list)
    
    def acc_vs_samplesize(self,filename):
        train_list = self.train_ratio_set
        budgetindex = 10
        #print(self.budget_list[5])
        print('sample size used budget',self.budget_list[5][0][budgetindex])
        acc_list = list()
        acc_std_list = list()
        for j in range(len(self.optimizer_legend)):
            acc_j_list = list()
            acc_std_j_list = list()
            for i in range(len(train_list)):
                self.train_ratio = train_list[i]
                self.loadacc()
                self._get_acc_mean()
                #print('len of acc list',len(self.acc_mean_list[j]))
                acc_i = self.acc_mean_list[j][budgetindex]
                acc_std = self.acc_std_list[j][budgetindex]
                acc_j_list.append(acc_i)
                acc_std_j_list.append(acc_std)
            acc_list.append(acc_j_list)
            acc_std_list.append(acc_std_j_list)
            self.train_ratio = self.train_ratio_origin
        #print('acc_vs_samplesize acc',acc_list)
        #print('train_list',train_list)
        fig = plt.figure(figsize=self.figuresize_tradeoff)
        fig, ax = plt.subplots(figsize=self.figuresize_tradeoff)        
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["lines.linewidth"] = self.linewidth
        plt.rcParams["lines.markersize"] = self.markersize
        plt.rcParams["font.sans-serif"] = 'Arial'        
        
        smallest = 100
        largest = -100
        for i in range(len(acc_list)):
            acc = acc_list[i]
            acc_std = acc_std_list[i]
            train_ratio = np.asarray(train_list)
            #print('bud',bud)
            #plt.plot(bud, acc_mean,self.optimizer_linemarker[i],color=self.colorset[i])   
            if(i>=self.skip_optimizer):
                smallest = min(smallest,np.min(np.asarray(acc)-np.asarray(acc_std)))
                largest = max(largest,np.max(np.asarray(acc)+np.asarray(acc_std)))
                plt.plot(np.asarray(train_ratio)*self.datasize, acc,
                             marker=self.optimizer_linemarker[i],
                             label=self.optimizer_legend[i],
                             color=self.colorset[i],
                             linestyle = self.optimizer_linestyle[i],
                            )
                plt.fill_between(np.asarray(train_ratio)*self.datasize, np.asarray(acc)-np.asarray(acc_std), np.asarray(acc)+np.asarray(acc_std),alpha=0.01,facecolor='lightgray')                
        plt.xlabel('Training Data Size')
        plt.ylabel('Accuracy') 
        plt.ylim(bottom = smallest-0.01)
   
        top = min(math.ceil(largest/0.05)*0.05,1)
        bottom = max(math.floor(smallest/0.05)*0.05,0)
        
        print('filename',filename)        
        if((top-bottom)/6>0.06):
            top = min(math.ceil(largest/0.1)*0.1,1)
            bottom = max(math.floor(smallest/0.1)*0.1,0)
            print('filename',filename)
            nbin = 0.1
        else:
            nbin = 0.05
            
        y_tick = np.arange(bottom, top+0.000001, nbin )
        
        ax.set_yticks(y_tick)
        
        
        plt.grid(True)
        if(self.show_legend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            #ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')            
            plt.legend(handles[::-1], labels[::-1], prop={'size': 35},markerscale=2, numpoints= 2,loc=8)
        plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       
                
        return 0
    
    def _get_acc_mean(self):
        acc_list = self.acc_list
        acc_mean_list = list()
        acc_std_list = list()
        smallest = 100
        for i in range(len(acc_list)):
            acc = np.asmatrix(acc_list[i])
            #print('acc',acc)
            acc_mean = np.mean(acc,0).tolist()[0]
            acc_std = np.std(acc,0)[0].tolist()[0]
            smallest = min(smallest,np.min(acc_mean))
            #print('acc mean',acc_mean)
            #print('acc std',acc_std)
            acc_mean_list.append(acc_mean)
            acc_std_list.append(acc_std)
        self.acc_mean_list = acc_mean_list
        self.acc_std_list = acc_std_list
        return 0
            
    def plot_acc_bud_tradeoff(self,filename):
        budget_list = self.budget_list
        acc_list = self.acc_list
        #print('acc_list len',len(acc_list))
        fig = plt.figure(figsize=self.figuresize_tradeoff)
        fig, ax = plt.subplots(figsize=self.figuresize_tradeoff)
        plt.rcParams.update({'font.size': self.fontsize})
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"
        plt.rcParams["lines.linewidth"] = self.linewidth
        plt.rcParams["lines.markersize"] = self.markersize
        plt.rcParams["font.sans-serif"] = 'Arial'        

        smallest = 100
        largest = 0
        for i in range(len(acc_list)):
            acc = np.asmatrix(acc_list[i])
            #print('acc',acc)
            acc_mean = np.mean(acc,0).tolist()[0]
            acc_std = np.std(acc,0)[0].tolist()[0]
            smallest = min(smallest,np.min(acc_mean))
            largest = max(largest,np.max(acc_mean))
            #print('acc mean',acc_mean)
            #print('acc std',acc_std)
            bud = np.asarray(budget_list[i][0])
            #print('bud',bud)
            #plt.plot(bud, acc_mean,self.optimizer_linemarker[i],color=self.colorset[i])   
            if(i>=self.skip_optimizer):
                #print("acc std",acc_std)
                plt.plot(bud, acc_mean,
                             marker=self.optimizer_linemarker[i],
                             label=self.optimizer_legend[i],
                             color=self.colorset[i],
                             linestyle = self.optimizer_linestyle[i]
                             )  
                plt.fill_between(bud, np.asarray(acc_mean)-np.asarray(acc_std), np.asarray(acc_mean)+np.asarray(acc_std),alpha=0.3,facecolor='lightgray')
            else:
                x1 = self.skip_optimizer_shift_x[i]
                y1 = self.skip_optimizer_shift_y[i]
                plt.text(bud[0]+x1,acc_mean[0]+y1,self.optimizer_legend[i],fontsize=35)
                
        if(self.show_legend):
            handles, labels = ax.get_legend_handles_labels()
            print("labels",labels)
            plt.legend(handles[::], labels[::],prop={'size': 35},markerscale=2, numpoints= 2,loc=7)

        plt.xlabel('Budget')
        plt.ylabel('Accuracy') 

        plt.ylim(bottom = smallest-0.01)
        if(self.task=='s2t' and self.dataname == 'COMMAND'):
            plt.ylim(bottom = smallest-0.015)

        top = min(math.ceil(largest/0.05)*0.05,1)
        bottom = max(math.floor(smallest/0.05)*0.05,0)
        
        #nbin = (top-bottom)/6
        print('top and bottom acc',top,bottom)
        print('filename',filename)        
        if((top-bottom)/6>0.06):
            top = min(math.ceil(largest/0.1)*0.1,1)
            bottom = max(math.floor(smallest/0.1)*0.1,0)
            print('filename',filename)
            nbin = 0.1
        else:
            nbin = 0.05
            
        y_tick = np.arange(bottom, top+0.000001, nbin )
        
        ax.set_yticks(y_tick)
        ax.tick_params(labelsize=35)

        plt.grid(True)
        #matplotlib.axis.YAxis.set_major_formatter(formatter='%.3f')
        #ax.locator_params(tight=True, nbins=9)
        ax.locator_params(axis='x', nbins=5)

        formatter = ticker.FormatStrFormatter('%0.2f')
        ax.yaxis.set_major_formatter(formatter)
        if(self.figureformat=='jpg'):
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight',dpi=600)       
        else:
            plt.savefig(filename, format=self.figureformat, bbox_inches='tight')       

    
def plot_offline(dataname = 'RAFDB', 
                 optimizer_name = ['100','0','1','2', 'FrugalML','FrugalMLQSonly', 'FrugalMLFixBase'],                 
                 optimizer_legend = ['GitHub (CNN)','Google Vision','Face++','MS Face','FrugalML','FrugalML(QS Only)', 'FrugalML (Base=GH)'],
                 skip_optimizer_shift_x=[1,-4,0.8,0.8],
                 skip_optimizer_shift_y=[-0.012,0.016,0.005,-0.012],
                 show_legend=True,
                 randseedset=[1,5,10,50,100],
                 stat_efficiency=True,
                 data_size=6358,
                 train_ratio_set=[0.01,0.05,0.1,0.2,0.3,0.5]):

   a = VisualizeTool(dataname=dataname,
                     optimizer_legend=optimizer_legend,
                     optimizer_name=optimizer_name,
                     skip_optimizer_shift_x=skip_optimizer_shift_x,
                     skip_optimizer_shift_y=skip_optimizer_shift_y,
                     show_legend=show_legend,
                     randseedset=randseedset,
                     stat_efficiency=stat_efficiency,
                     train_ratio_set=train_ratio_set,
                     data_size=data_size)
   a.run()
   
def main():   
    '''
    Test optimizers
    '''   
    parser = argparse.ArgumentParser(description='FrugalML Plot')
    parser.add_argument('--dataname', type=str,
                        help='datasetname.', default='RAFDB')
    parser.add_argument('--optimizer_name', type=str,help='list of \
                        optimizer name',
                        nargs='+',
                        default= ['100','0','1','2', 
                                  'FrugalML','FrugalMLQSonly', 
                                  'FrugalMLFixBase'],)
    parser.add_argument('--optimizer_legend', type=str,help='list of\
                        optimizer legend',
                        nargs='+',
                        default= ['GitHub (CNN)','Google Vision','Face++',
                                  'MS Face','FrugalML','FrugalML(QS Only)', 
                                  'FrugalML (Base=GH)'],)
    parser.add_argument('--skip_optimizer_shift_x', type=float,help='list of\
                        optimizer legend shift (x axis)',
                        nargs='+',
                        default= [1,-4,0.8,0.8],)
    parser.add_argument('--skip_optimizer_shift_y', type=float,help='list of\
                        optimizer legend shift (y axis)',
                        nargs='+',
                        default= [-0.012,0.016,0.005,-0.012],)        
        
    parser.add_argument('--show_legend', type=bool,
                        help='show legend or not',
                        default= True,) 
    parser.add_argument('--randseedset', type=str,help='random seed for\
                        testing training splitting',
                        nargs='+',                        
                        default= [1,5,10,50,100],) 
    parser.add_argument('--evalsampleefficiency', type=bool,help='evaluation\
                        of sample efficiency',
                        default= False,) 
    parser.add_argument('--datasize', type=int,help='datasize used for\
                        sample efficiency evaluation',
                        default= 6358,) 
        
    parser.add_argument('--train_ratio_set', type=float,help='train ratio for\
                        only valid for method {FrugalMLQSonly} \
                            and {FrugalMLFixBase}',
                        nargs='+',
                        default=[0.01,0.05,0.1,0.2,0.3,0.5])
    args = parser.parse_args()

    
    plot_offline(dataname = args.dataname, 
                 optimizer_name = args.optimizer_name, 
                 optimizer_legend = args.optimizer_legend,
                 skip_optimizer_shift_x= args.skip_optimizer_shift_x,
                 skip_optimizer_shift_y= args.skip_optimizer_shift_y,
                 show_legend = args.show_legend,
                 randseedset = args.randseedset,
                 stat_efficiency = args.evalsampleefficiency,
                 data_size = args.datasize,
                 train_ratio_set = args.train_ratio_set)
    plt.close('all')

if __name__ == '__main__':
    main()



