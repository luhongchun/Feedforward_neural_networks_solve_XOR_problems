#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as mp
from mpl_toolkits.mplot3d import Axes3D
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import FloatTensor
from torch.autograd import Variable
import time

fig = mp.figure(figsize=(8,8))
ax3d = mp.axes(projection='3d')
mp.ion()


class XORData(object):
    """a class for the generation of XOR validation and training data

    >>> d = XORData()
    >>> d.astype(int)
    array([[0, 0, 0],
           [0, 1, 1],
           [1, 0, 1],
           [1, 1, 0]])

    >>> d = XORData(batchsize=2,delta=0.5)
    >>> len(d)
    8
    >>> np.all(np.rint(d[0:4]) == XORData.TRUTHTABLE)
    True
    >>> np.all(np.rint(d[4:8]) == XORData.TRUTHTABLE)
    True
    >>> np.var(d - np.vstack([XORData.TRUTHTABLE]*2)) > 0
    True

    """

    TRUTHTABLE = np.array([
        #A,B,XOR
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,0],
        ],dtype=float)

    TABLE0 = np.vstack([TRUTHTABLE,[0.5,0.5,0.0]])
    TABLE1 = np.vstack([TRUTHTABLE,[0.5,0.5,1.0]])
    # print("TABLE0",TABLE0)
    # print("TABLE1",TABLE1)
    x = np.array([[0], [0], [1], [1]])
    y = np.array([[0], [1], [0], [1]])
    z = np.array([[0], [1], [1], [0]])

    def __new__(this,batchsize=1,delta=0.0,table=TRUTHTABLE):
        n = len(table)
        assert table.shape == (n,2+1)
        rands = np.random.uniform(-delta,+delta,size=(batchsize,n,2))
        zeros = np.zeros(shape=(batchsize,n,1),dtype=float)
        deltas = np.concatenate((rands,zeros),axis=2)
        assert deltas.shape == (batchsize,n,3)
        dataset = table + deltas
        dataset.shape = (batchsize*n,3)
        return dataset


class XORNet(nn.Module):
    """A classical 2-layer XOR neural network

    >>> net = XORNet()
    >>> net
    XORNet (
      (fc0): Linear (2 -> 2)
      (fc1): Linear (2 -> 1)
    )

    """

    def __init__(self):
        super(XORNet, self).__init__()
        self.fc0 = nn.Linear(2,2)
        self.fc1 = nn.Linear(2,2)
        self.fc2 = nn.Linear(2,2)
        self.fc3 = nn.Linear(2,2)
        self.fc4 = nn.Linear(2,2)
        self.fc5 = nn.Linear(2,2)
        self.fc6 = nn.Linear(2,2)
        self.fc7 = nn.Linear(2,2)
        self.fc8 = nn.Linear(2,1)

    def forward(self,x):
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        x = F.sigmoid(self.fc7(x))
        x = F.sigmoid(self.fc8(x))
        return x

    def setparams_zeros(self):
        for p in self.parameters():
            p.data.zero_()

    def setparams_uniforms(self,delta=1):
        for p in self.parameters():
            p.data.uniform_(-delta,+delta)


class XOR(object):
    """An encapsulation of a neural network, training and testing

    >>> xor = XOR()
    >>> xor
    XOR (
      loss: MSELoss
      optim: Adam
      lr: 0.01
    )

    """

    LEARNING_RATE = 0.01

    def __init__(self,lr=LEARNING_RATE):
        self.net = XORNet()
        self.state_start = self.net.state_dict()
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(),lr)
        self.l = self.training # shorthand

    def training(
            self,nbatch=10,batchsize=10,
            delta=0.2,table=XORData.TRUTHTABLE,
            save=False):
        for ibatch in range(nbatch):
            epsilonsum = 0
            for t in XORData(batchsize,delta,table):
                y = self.net(Variable(FloatTensor(t[0:2])))
                target = Variable(FloatTensor(t[2:]))
                self.optim.zero_grad()
                epsilon = self.loss(y,target)
                #print("epsilon.data",epsilon.data)
                epsilonsum += epsilon.data
                epsilon.backward()
                self.optim.step()
            self.splot()
            if save:
                fmt = save + '{:0' + str(len(str(nbatch))) + '}'
                mp.savefig(fmt.format(ibatch))
            # print('{:<8} {:.4e}'.format(ibatch,epsilonsum/batchsize))

    def test(self):
        """print the truth table evaluated by self.net:"""
        for a,b,xor in XORData():
            y = self.net(Variable(FloatTensor([a,b])))
            target = Variable(FloatTensor([xor]))
            epsilon = self.loss(y,target)
            print('{} {:+.8f}'.format(
                (int(a),int(b)),y.data[0]))
   
    def splot(self,nticks=51):
        
        """surface plot of the xor outputs of
        the self.net for a mesh grid inputs of a and b:"""
        i = np.linspace(-0.5,1.5,nticks)
        a,b = np.meshgrid(i,i)
        ab = np.stack([a,b],axis=-1)
        xor = self.net(Variable(FloatTensor(ab)))
        xor = xor.data.numpy()
        xor.shape = (nticks,nticks)
        ax3d.clear()
        ax3d.scatter(XORData.x[0], XORData.y[0], XORData.z[0], c='r',s=200,alpha=1)  # 绘制数据点,颜色是红色
        ax3d.scatter(XORData.x[1], XORData.y[1], XORData.z[1], c='b',s=200,alpha=1)  # 绘制数据点,颜色是红色
        ax3d.scatter(XORData.x[2], XORData.y[2], XORData.z[2], c='b',s=200,alpha=1)  # 绘制数据点,颜色是红色
        ax3d.scatter(XORData.x[3], XORData.y[3], XORData.z[3], c='r',s=200,alpha=1)  # 绘制数据点,颜色是红色
        ax3d.plot_surface(a,b,xor,edgecolor='none', cmap='rainbow',alpha=0.5)
        ax3d.view_init(elev=30,azim=-60)
        ax3d.set_xticks([0,1]),ax3d.set_xlabel('A')
        ax3d.set_yticks([0,1]),ax3d.set_ylabel('B')
        ax3d.set_zticks([0,1]),ax3d.set_zlabel('XOR')
        mp.draw()
        mp.pause(0.05)
        #ax3d.show()

    def __repr__(self):
        return "\n".join([
            'XOR (',
            '  loss: {}'.format(self.loss.__class__.__name__),
            '  optim: {}'.format(self.optim.__class__.__name__),
            '  lr: {}'.format(self.optim.param_groups[0].get('lr')),
            ')',
            ])


if __name__ == "__main__":

    import sys
    import doctest

    def docscript(obj=None):
        """usage: exec(docscript())"""
        doc = __doc__
        if obj is not None:
            doc = obj.__doc__
        return doctest.script_from_examples(doc)

    if sys.argv[0] == "": # if python is in an emacs buffer:
        print(doctest.testmod(optionflags=doctest.REPORT_ONLY_FIRST_FAILURE))

    state0_dict = OrderedDict((
        ('fc0.weight',FloatTensor([[20,-20],[20,-20]])),
        ('fc0.bias',FloatTensor([-15,15])),
        ('fc1.weight',FloatTensor([[20,-20]])),
        ('fc1.bias',FloatTensor([10])),
        ))

    state1_dict = OrderedDict((
        ('fc0.weight',FloatTensor([[20,20],[20,20]])),
        ('fc0.bias',FloatTensor([-35,-5])),
        ('fc1.weight',FloatTensor([[20,-20]])),
        ('fc1.bias',FloatTensor([10])),
        ))

    state_dict3 = OrderedDict((
        ('fc0.weight',FloatTensor([[0.1, 0.6]])),
        ('fc0.bias',FloatTensor([-0.3])),
        ))
    state1_dict2 = OrderedDict((
        ('fc0.weight',FloatTensor([[1,1],[-1,-1]])),
        ('fc0.bias',FloatTensor([-0.5,0.5])),
        ('fc1.weight',FloatTensor([[1,1]])),
        ('fc1.bias',FloatTensor([-2])),
        ))
    state_dict = OrderedDict((
        ('fc0.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc0.bias',FloatTensor([-0.3, 0.5])),
        ('fc1.weight',FloatTensor([[0.4, 0.0]])),
        ('fc1.bias',FloatTensor([-0.4])),
        ))
    state_dict22 = OrderedDict((
        ('fc0.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc0.bias',FloatTensor([-0.3, 0.5])),
        ('fc1.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc1.bias',FloatTensor([-0.3, 0.5])),
        ('fc2.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc2.bias',FloatTensor([-0.3, 0.5])),
        ('fc3.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc3.bias',FloatTensor([-0.3, 0.5])),
        ('fc4.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc4.bias',FloatTensor([-0.3, 0.5])),
        ('fc5.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc5.bias',FloatTensor([-0.3, 0.5])),
        ('fc6.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc6.bias',FloatTensor([-0.3, 0.5])),
        ('fc7.weight',FloatTensor([[0.1, 0.6],[-0.3, -0.6]])),
        ('fc7.bias',FloatTensor([-0.3, 0.5])),
        ('fc8.weight',FloatTensor([[0.4, 0.0]])),
        ('fc8.bias',FloatTensor([-0.4])),
        ))

    # some shorthands
    t = XORData.TRUTHTABLE
    t0 = XORData.TABLE0
    t1 = XORData.TABLE1
    
    xor = XOR()
    #xor.net.load_state_dict(state_dict)
    # training for solution 0
    #xor.training(nbatch=25,delta=0.2,table=XORData.TABLE0,save='t0')
    xor.net.load_state_dict(state_dict22) # reset the start state
    xor.weights = []
    xor.weights.append(state_dict22)
    # print("w",xor.weights)
    # training for solution 1
    import datetime
    tic = time.time()
    print("开始时间：",datetime.datetime.now())
    xor.training(nbatch=25,delta=0.2,table=XORData.TABLE1,save='t1')
    xor.test()
    # print("w",xor.weights)
    toc = time.time()
    print("所需时间为：{0}秒".format(toc - tic))
    print("结束时间：",datetime.datetime.now())