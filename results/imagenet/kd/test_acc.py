import os
import sys
import json

files = os.listdir('.')

epoch = 150

for afile in sorted(files):
    if afile.find('history') >= 0:
        obj = json.load(open(afile, 'r'))

        if len(obj['test_acc']) >= epoch:
            print(afile, obj['test_acc'][epoch - 1])
        else:
            print(afile, 'not finished')
