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

python3 visualizetools.py --dataname FERPLUS \
--optimizer_name '100' '0' '1' '2' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (CNN)' 'Google Vision' 'Face++' \
'MS Face' 'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.6 -4 0.8 0.8 \
--skip_optimizer_shift_y -0.012 0.015 -0.012 -0.008 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 6358 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 

python3 visualizetools.py --dataname EXPW \
--optimizer_name '100' '0' '1' '2' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (CNN)' 'Google Vision' 'Face++' \
'MS Face' 'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.6 -4 0.8 0.8 \
--skip_optimizer_shift_y -0.012 0.015 -0.012 -0.008 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 31510 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 

python3 visualizetools.py --dataname RAFDB \
--optimizer_name '100' '0' '1' '2' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (CNN)' 'Google Vision' 'Face++' \
'MS Face' 'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  1 -4 0.8 0.8 \
--skip_optimizer_shift_y -0.004 0.016 0.005 -0.012 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 15339 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 

python3 visualizetools.py --dataname AFFECTNET \
--optimizer_name '100' '0' '1' '2' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (CNN)' 'Google Vision' 'Face++' \
'MS Face' 'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.7 -5 0.7 0.8 \
--skip_optimizer_shift_y -0.012 -0.015 0.005 -0.012 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 287401 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 

python3 visualizetools.py --dataname YELP \
--optimizer_name '100' '0' '3' '4' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (Vader)' 'Google NLP' 'Baidu NLP' 'AMAZ COMP' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.1 -0.6 -1.2 0.1 \
--skip_optimizer_shift_y  -0.008 -0.040 -0.020 -0.008 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 20000 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 


python3 visualizetools.py --dataname IMDB \
--optimizer_name '100' '0' '3' '4' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (Vader)' 'Google NLP' 'Baidu NLP' 'AMAZ COMP' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.1 -0.4 -1.4 0.1 \
--skip_optimizer_shift_y  -0.008 -0.035 -0.015 -0.042 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 25000 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 

python3 visualizetools.py --dataname WAIMAI \
--optimizer_name '100' '0' '3' '4' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (Bixin)' 'Google NLP' 'Baidu NLP' 'AMAZ COMP' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.2 -0.3 -1.0 -0.5 \
--skip_optimizer_shift_y  -0.008 -0.030 -0.042 -0.048 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 62774 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 
				
python3 visualizetools.py --dataname SHOP \
--optimizer_name '100' '0' '3' '4' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (Bixin)' 'Google NLP' 'Baidu NLP' 'AMAZ COMP' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  0.1 -1.0 -0.2 -0.2 \
--skip_optimizer_shift_y  -0.008 -0.020 -0.025 -0.022 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 11987 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5 

python3 visualizetools.py --dataname DIGIT \
--optimizer_name '101' '0' '2' '5' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (DeepSpeech)' 'Google Speech' 'MS Speech' 'IBM Speech' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  1.9 34 10 0 \
--skip_optimizer_shift_y  -0.015 0.020 -0.060 -0.060 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 2000 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5
                          
python3 visualizetools.py --dataname AUDIOMNIST \
--optimizer_name '101' '0' '2' '5' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (DeepSpeech)' 'Google Speech' 'MS Speech' 'IBM Speech' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  2.5 -33 -5 0 \
--skip_optimizer_shift_y  -0.008 0.006 -0.014 -0.015 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 30000 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5

python3 visualizetools.py --dataname COMMAND \
--optimizer_name '101' '0' '2' '5' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (DeepSpeech)' 'Google Speech' 'MS Speech' 'IBM Speech' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  2.5 -33 -10 0 \
--skip_optimizer_shift_y  -0.012 0.009 -0.018 0.009 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 64727 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5

python3 visualizetools.py --dataname FLUENT \
--optimizer_name '101' '0' '2' '5' 'FrugalML' 'FrugalMLQSonly' 'FrugalMLFixBase' \
--optimizer_legend 'GitHub (DeepSpeech)' 'Google Speech' 'MS Speech' 'IBM Speech' \
'FrugalML' 'FrugalML(QS Only)' 'FrugalML (Base=GH)' \
--skip_optimizer_shift_x  1.2 -32 -10 1.2 \
--skip_optimizer_shift_y  -0.015 0.015 0.009 -0.015 \
--show_legend True \
--randseedset 1 5 10 50 100 \
--datasize 30043 \
--train_ratio_set 0.01 0.05 0.1 0.2 0.3 0.5