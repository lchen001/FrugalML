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

import time
from optimizer import OptimizerFrugalML
import argparse

def test_optimizer_FrugalML(datapath='../dataset/mlserviceperformance_RAFDB',
                            budget=5,
                            method='FrugalML',
                            baseid=100,
                            randseed=100):
    '''
    test the Optimizer of FrugalML
    '''
    myoptimizer = OptimizerFrugalML(datapath,split=True, 
                                    train_ratio=0.5,
                                    method=method,
                                    baseid=baseid,
                                    randseed=randseed)
    myoptimizer.setbudget(budget = budget)
    myoptimizer.solve()
    print('result',myoptimizer.getresult())
    
def main():
    '''
    Test optimizers
    '''   
    parser = argparse.ArgumentParser(description='FrugalML Test')
    parser.add_argument('--method', type=str,
                        help='Method name. Choices \
                            include {FrugalML},{FrugalMLQSonly} \
                            and {FrugalMLFixBase} ', default='FrugalML')
    parser.add_argument('--datapath', type=str,help='Datapath',
                        default='../dataset/mlserviceperformance_RAFDB')
    parser.add_argument('--budget', type=float,help='Budget value',
                        default=5)
    parser.add_argument('--baseid', type=float,help='Base id \
                        only valid for method {FrugalMLQSonly} \
                            and {FrugalMLFixBase}',
                        default=100)
    args = parser.parse_args()

    print('method:',args.method)
    print('datapath:',args.datapath)
    print('budget:',args.budget)
    
    time1 = time.time()
    test_optimizer_FrugalML(datapath=args.datapath, budget = args.budget,
                            method=args.method)        
    time2 = time.time()
    print('runtime',time2-time1)
    
      
if __name__ == '__main__':
    main()


