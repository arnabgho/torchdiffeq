import os
import argparse
import time
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_components', type=int, default=2)
parser.add_argument('--num_hidden_guider', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([0.])
t = torch.linspace(0., 25., args.data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):

    def forward(self, t, y):
        t = t.unsqueeze(0)
        #equation = -1000*y + 3000 - 2000 * torch.exp(-t)
        equation = 10*torch.sin(t) #-1000*y + 3000 - 2000 * torch.exp(-t)
        return equation
        #return torch.mm(y**3, true_A)
        #return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    #true_y = odeint(Lambda(), true_y0, t, method='adams')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png_alternate_mad')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    #ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr, which_one=0):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('True vs Predicted')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0], '--', 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-10, 10)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Predicted')
        ax_phase.set_xlabel('t')
        ax_phase.set_ylabel('y')
        ax_phase.plot(t.numpy(), pred_y.numpy()[:, 0], '--', 'b--')
        ax_phase.set_xlim(t.min(), t.max())
        ax_phase.set_ylim(-10, 10)
        ax_phase.legend()


        fig.tight_layout()
        plt.savefig('png_alternate_mad/{:04d}_{:02d}'.format(itr,which_one))
        plt.draw()
        plt.pause(0.001)



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        t=t.unsqueeze(0)
        t = t.view(1,1)
        y = y.view(y.size(0),1)
        t = t.expand_as(y)
        equation = torch.cat([t,y],1)
        result = self.net(equation)

        if y.size(0)==1:
            result = result.squeeze()
        return result

class Guider(nn.Module):

    def __init__(self):
        super(Guider,self).__init__()

        self.rnn = nn.LSTM(1,args.num_hidden_guider,2)

        self.linear = nn.Linear( args.num_hidden_guider , args.num_components)

    def forward(self,input):
        lstm_out,_ = self.rnn(input)
        batch_size = input.size(1)
        rep = self.linear(lstm_out[0].view(batch_size , args.num_hidden_guider ))

        return rep

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

criterion_guider = nn.CrossEntropyLoss()

if __name__ == '__main__':

    ii = 0

    funcs = []
    func = ODEFunc()
    state = func.state_dict()
    funcs.append(func)

    guide = Guider()

    for i in range(args.num_components-1):
        state_clone = copy.deepcopy(state)
        func = ODEFunc()
        func.load_state_dict(state_clone)
        funcs.append(func)

    optimizers = []
    for i in range(args.num_components):
        optimizers.append(optim.RMSprop(funcs[i].parameters(), lr=1e-3))
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        for i in range(args.num_components):
            optimizers[i].zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_ys=[]
        pred_y = odeint(funcs[0], batch_y0, batch_t)
        pred_ys.append(pred_y)
        for i in range(args.num_components-1):
            pred_y = pred_y + odeint(funcs[i],batch_y0,batch_t)
            pred_ys.append(pred_y)
        # pred_y size : (args.batch_time,args.batch_size,1)

        which_one = np.random.randint(0,args.num_components)

        pred_which_one = guide( pred_ys[which_one]  )
        true_which_one = torch.zeros(pred_which_one.size(0)).long().fill_(which_one)

        loss = torch.mean(torch.abs(pred_y - batch_y)) + criterion_guider(pred_which_one,true_which_one)
        loss.backward()
        for i in range(args.num_components):
            optimizers[i].step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_ys = []
                pred_y = odeint(funcs[0], true_y0, t)
                pred_ys.append(pred_y)

                for i in range(args.num_components-1):
                    pred_ys.append( odeint(funcs[i],true_y0,t) )
                    pred_y = pred_ys[-1]

                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                for i in range(args.num_components):
                    visualize(true_y, pred_y, func, ii,i)
                ii += 1

        end = time.time()