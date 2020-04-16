import os
import sys
import itertools, functools

ss = sys.argv[1]
distill_type = sys.argv[2]
gpu = int(sys.argv[3])

if len(sys.argv) >= 5:
    is_dry_run = True
else:
    is_dry_run = False

def hyper_gen():

    configs = [None] * (len(param_names) + 1)
    configs[0] = ['--arc %d' % arc for arc in range(20)]

    for i in range(len(param_names)):
        if param_names[i]:
            configs[i+1] = ['--%s %f' % (param_names[i], param_values[i][j]) for j in range(len(param_values[i]))]
        else:
            configs[i+1] = param_values[i]

    return itertools.product(*configs)

def done_gen():

    configs = [None] * (len(param_names) + 1)
    configs[0] = ['--arc %d' % arc for arc in range(20)]

    for i in range(len(param_names)):
        if param_names[i]:
            configs[i+1] = [param_done[i][j] for j in range(len(param_values[i]))]
        else:
            configs[i+1] = param_values[i]

    return itertools.product(*configs)

param_names = None
param_values = None
param_done = None

if distill_type == 'kd':
    param_names = ['kd_T', None]
    param_values = [[1, 4, 16, 32],
        ['--gamma 0.75 --alpha 0.25',
        '--gamma 0.50 --alpha 0.50',
        '--gamma 0.25 --alpha 0.75',
        '--gamma 0.00 --alpha 1.00']]

elif distill_type == 'crd':
    param_names = ['nce_t', 'beta']
    #param_values = [[0.02, 0.05, 0.1, 0.2, 0.4, 0.8], [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]]
    #param_done =   [[1,       1,   1,   1,   0,   0], [  1,   1,   1,   1,   0,    0]]
    param_values = [[0.02, 0.05, 0.1, 0.2], [0.5, 1.0, 2.0, 4.0]]
    #param_done =   [[1,       1,   1,   1], [  1,   1,   1,   1]]
elif distill_type == 'pkt':
    param_names = ['beta']
    param_values = [[0.75e4, 1.5e4, 3e4, 6e4, 12e4, 24e4, 48e4, 96e4]]
    #param_done = [[1,         1,     1,   1,    1,   0,    0,    0  ]]
elif distill_type == 'similarity':
    param_names = ['beta']
    param_values = [[0.75e3, 1.5e3, 3e3, 6e3, 12e3, 0.37e3, 0.18e3, 0.09e3]]
    #param_done = [[1,         1,     1,   1,    1,   0,    0,       0  ]]
elif distill_type == 'vid':
    param_names = ['beta']
    param_values = [[0.25, 0.5, 1, 2, 4, 8, 16, 32]]
    #param_done = [[1,       1,  1, 1, 1, 0,  0,  0]]
elif distill_type == 'attention':
    param_names = ['beta']
    param_values = [[0.25e3, 0.5e3, 1e3, 2e3, 4e3]]
elif distill_type == 'nst':
    param_names = ['beta']
    param_values = [[12.5, 25, 50, 100, 200, 6, 3, 1.5]]
    #param_done = [[1,      1,   1,   1,   1, 0, 0,  0  ]]
elif distill_type == 'hint':
    param_names = ['beta']
    param_values = [[12.5, 25, 50, 100, 200, 400, 800]]
    #param_done =  [[0,      0,   0,   0,   0,   0,  0]]
elif distill_type == 'correlation':
    param_names = ['beta']
    param_values = [[0.25e-2, 0.5e-2, 1e-2, 2e-2, 4e-2, 8e-2, 16e-2, 32e-2]]
    #param_done =   [[0,            0,    0,    0,    0,    0,     0,     0]]
else:
    param_names = None
    param_values = None
    param_done = None

def mult(arr):
    return functools.reduce(lambda x, y: x*y, arr)

if __name__ == '__main__':

    cnt = 0

    A = [len(x) for x in param_values]
    configs_num = functools.reduce(lambda x, y: x*y, A)

    CONFIGS = [x for x in hyper_gen()]

    if param_done:
        DONE = [x for x in done_gen()]
    else:
        DONE = None

    for i in range(len(CONFIGS)):
        elem = CONFIGS[i]

        if DONE:
            done = mult(DONE[i][1:])
        else:
            done = False
        s = ' '.join(elem)
        print('')
        print(cnt % configs_num)
        print(s)
        print(done)

        common_cmd = 'python train_student.py --gpu %d --part 9 --distill %s --epochs 100 --nce_k 4096 %s' % (gpu, distill_type, s)

        if distill_type == 'kd':
            common_cmd += ' --beta 0'
        else:
            common_cmd += ' -r 1 -a 0'

        if ss == 'mobile':
            cmd = common_cmd + ' --model_s MobileNetV2Trofim --prefix %s_tune/%s_per11_%d_ ' % (distill_type, distill_type, cnt % configs_num)
        elif ss == 'shuffle':
            cmd = common_cmd + ' --model_s ShuffleV2 --path_t shufflev2_teacher.pth --prefix shuffle/%s_tune/%s_per11_%d_' % (distill_type, distill_type, cnt % configs_num)

        print(cmd)
        if not is_dry_run and not done:
            os.system(cmd)
        cnt += 1
