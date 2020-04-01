import os
import sys
import itertools, functools

distill_type = sys.argv[1]
gpu = int(sys.argv[2])

def hyper_gen():

    configs = [None] * (len(param_names) + 1)
    configs[0] = ['--arc %d' % arc for arc in range(20)]

    for i in range(len(param_names)):
        configs[i+1] = ['--%s %f' % (param_names[i], param_values[i][j]) for j in range(len(param_values[i]))]

    return itertools.product(*configs)

if distill_type == 'crd':
    param_names = ['nce_t', 'beta']
    param_values = [[0.02, 0.05, 0.1, 0.2], [0.5, 1.0, 2.0, 4.0]]
elif distill_type == 'pkt':
    param_names = ['beta']
    param_values = [[0.75e3, 1.5e3, 3e3, 6e3, 12e3]]
elif distill_type == 'similarity':
    param_names = ['beta']
    param_values = [[0.75e4, 1.5e4, 3e4, 6e4, 12e4]]
elif distill_type == 'vid':
    param_names = ['beta']
    param_values = [[0.25, 0.5, 1, 2, 4]]
elif distill_type == 'at':
    param_names = ['beta']
    param_values = [[0.25e3, 0.5e3, 1e3, 2e3, 4e3]]
elif distill_type == 'nst':
    param_names = ['beta']
    param_values = [[12.5, 25, 50, 100, 200]]

if __name__ == '__main__':

    cnt = 0

    A = [len(x) for x in param_values]
    configs_num = functools.reduce(lambda x, y: x*y, A)

    for elem in hyper_gen():
        s = ' '.join(elem)
        print('')
        print(s)
        cmd = 'python train_student.py --gpu %d --part 9 --distill %s --model_s MobileNetV2Trofim -r 1 -a 0 --epochs 100 --nce_k 4096 %s --prefix %s_tune/%s_per11_%d_ ' % (gpu, distill_type, s, distill_type, distill_type, cnt % configs_num)
        print(cmd)
        os.system(cmd)
        cnt += 1
