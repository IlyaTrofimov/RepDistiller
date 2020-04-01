import os
import sys
import itertools, functools

distill_type = sys.argv[1]
gpu = int(sys.argv[2])

def crd_hyper_gen():

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
    param_values = [[0.75e5, 1.5e5, 3e5, 6e5, 12e5]]

if __name__ == '__main__':

    cnt = 0

    A = [len(x) for x in param_values]
    configs_num = functools.reduce(lambda x, y: x*y, A)

    gen = crd_hyper_gen

    for elem in gen():
        s = ' '.join(elem)
        print('')
        print(s)
        cmd = 'python train_student.py --gpu %d --part 9 --distill %s --model_s MobileNetV2Trofim -r 1 -a 0 --epochs 100 --nce_k 4096 %s --prefix %s_tune/%s_per11_%d_ ' % (gpu, distill_type, s, distill_type, distill_type, cnt % configs_num)
        print(cmd)
        os.system(cmd)
        cnt += 1
