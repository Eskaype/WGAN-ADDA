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
        gamma=0.2,
        mu=0.1,
        weight_decay=1e-5,
        Lf=2,
        lr=1e-3,
        lr_critic=1e-4,
        backbone='resnet'):
        param = dict()
        param['gamma']=gamma
        param['mu']=mu
        param['weight_decay']=weight_decay
        param['Lf']=Lf
        param['lr']=lr
        param['lr_critic']=lr_critic
        param['backbone']=backbone
        self.TuneComb.append(param)

    def create_population(self,
        Lgamma=[0.2,0.5],
        Lmu=[0.08,0.2],
        Lweight_decay=[1e-4],
        LLf=[1,2,5,10,50,100],
        Llr=[1e-4, 1e-5],
        Llr_critic=[1e-4, 1e-5],
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
        allarg = [Lgamma, Lmu,  Lweight_decay, LLf, Llr, Llr_critic, Lbackbone]
        initials = list(itertools.product(*allarg))
        for initial in initials:
            self.create_param(initial[0],initial[1],initial[2],initial[3],initial[4],initial[5],initial[6])

    def create_train(self):
        cmds = []
        for param in self.TuneComb:
            output_dir = "multi_default/"+'doutput_mu:{}_gamma:{}_wd:{}_Lf:{}_lr:{}_{}'.format(param['mu'], param['gamma'], param['weight_decay'], param['Lf'], param['lr'], param['lr_critic'])
            cmd = "CUDA_VISIBLE_DEVICES='1' python3 trainer_dual_source/mwdan.py --gamma={} --mu={} --weight-decay={} --Lf={} --lr={} --lr_critic={} --output-dir={} --backbone={}".format(param['gamma'], param['mu'], param['weight_decay'], param['Lf'], param['lr'], param['lr_critic'], output_dir, param['backbone'])
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
