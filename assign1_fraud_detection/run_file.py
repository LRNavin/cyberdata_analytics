#!/usr/bin/python

import sys
from final_submission import FraudData_Classificaiton
from data_clean_submission import DataCleaner

args = sys.argv

if(str(args[3]) == 'w'):
    param = 1
else:
    param = 2

print 'Data Source:', str(args[2])
print 'Classifier Choice:', str(param)

classif_obj = FraudData_Classificaiton()
cleaner_obj = DataCleaner()


print 'Data Cleaning'
cleaner_obj.trigger_clean(args[2])
print 'Data Cleaning Over !!!!'

print '-----Data Classification-----'
classif_obj.trigger_run(param)