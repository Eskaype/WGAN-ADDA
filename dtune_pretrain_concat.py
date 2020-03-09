## a script for using genetic algorithm for tuning parameters
import sys
import os
import itertools
import pickle
import time
class Tune:
    def __init__(self):
        self.TuneComb = []

    def create_param(self,
        beta=0.0001,
        weight_decay=1e-5,
        num_update = 1,
        lr=1e-4,
        bs = 4,
        backbone='resnet'):
        param = dict()
        param['weight_decay']=weight_decay
        param['lr']=lr
        param['beta'] = beta
        param['num_update'] = num_update
        param['batch_size'] = bs
        param['backbone'] = backbone
        self.TuneComb.append(param)

    def create_population(self,
        Lbeta = [1e-5],
        Lweight_decay=[1e-4, 1e-3],
        Lupdate=[1],
        Llr=[5e-4, 1e-4, 5e-5, 1e-5],
        Lbs = [4,8],
        Lbackbone=['resnet']
        ):
        """
        Lfilter_dim = ["32,32,64,64", "8,8,16,16"]
        TO DO:
        each parameter list has four options to control tuning
        1. the starting value;
        2. tuning type: 0: not tune; 1: additive; 2: multiplicative tune
        3. tuning step: this is useless if type==0; if additive: inital+step; if multiplicative: initial * step
        4. tuning number: number of tuned parameters
        Current:
        In this stage we directly give combinations
        """
        allarg = [Lbeta, Lweight_decay,  Lupdate, Llr, Lbs, Lbackbone]
        initials = list(itertools.product(*allarg))
        for initial in initials:
            self.create_param(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5])

    def create_train(self):
        cmds = []
        for param in self.TuneComb:
            output_dir = "multi_default/"+'concat_output_beta:{}_wd:{}_nup:{}_lr:{}_bs:{}'.format(param['beta'], param['weight_decay'], param['num_update'], param['lr'], param['batch_size'])
            cmd = "CUDA_VISIBLE_DEVICES='0' python3 trainer_dual_source/source_only_concat.py --beta={} --weight-decay={} --meta_update_step={} --lr={}  --batch_size={} --output-dir={}  --backbone={}".format(param['beta'], param['weight_decay'], param['num_update'], param['lr'], param['batch_size'], output_dir, param['backbone'])
            cmds.append(cmd)
        return cmds

if __name__ == "__main__":
    tunes = Tune()
    tunes.create_population()
    cmds = tunes.create_train()
    #print(tunes.TuneComb)
    comb = tunes.TuneComb
    with open('tune.p', 'wb') as pf:
        pickle.dump(comb, pf)
    print(cmds)
    for i, cmd in enumerate(cmds):
        print(cmd)
        if i >= 0:
            os.system(cmd)
            while not os.path.isfile('trainer_dual_source/dual.o'):
                time.sleep(1)
