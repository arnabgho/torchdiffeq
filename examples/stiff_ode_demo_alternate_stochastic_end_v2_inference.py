import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import uniform
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--ntest', type=int, default=10)
parser.add_argument('--shrink_std', type=float, default=0.1)
parser.add_argument('--shrink_proportion', type=float, default=0.5)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    from torchdiffeq import odeint_adjoint_stochastic_end_v2 as odeint_stochastic_end_v2
else:
    from torchdiffeq import odeint_stochastic_end_v2
    from torchdiffeq import odeint_stochastic_end_v2_inference
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([0.])
t = torch.linspace(0., 25., args.data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):

    def forward(self, t, y):
        t = t.unsqueeze(0)
        #equation = -1000*y + 3000 - 2000 * torch.exp(-t) + 1000 * torch.sin(t)
        equation = -1000*y + 3000 - 2000 * torch.exp(-t)
        #equation = 10 * torch.sin(t)
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
    makedirs('png_alternate_stochastic_end_v2_inference')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_multiple = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, pred_ys, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('True vs Predicted')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0], '--', 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-100, 100)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Predicted')
        ax_phase.set_xlabel('t')
        ax_phase.set_ylabel('y')
        ax_phase.plot(t.numpy(), pred_y.numpy()[:, 0], '--', 'b--')
        ax_phase.set_xlim(t.min(), t.max())
        ax_phase.set_ylim(-100, 100)
        ax_phase.legend()

        ax_multiple.cla()
        ax_multiple.set_title('Variations')
        ax_multiple.set_xlabel('t')
        ax_multiple.set_ylabel('y')
        for component in pred_ys:
            ax_multiple.plot(t.numpy(), component.numpy()[:, 0], '--')
        ax_multiple.set_xlim(t.min(), t.max())
        ax_multiple.set_ylim(-100, 100)
        ax_multiple.legend()



        fig.tight_layout()
        plt.savefig('png_alternate_stochastic_end_v2_inference/{:04d}'.format(itr))
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


if __name__ == '__main__':

    ii = 0

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        #pred_y = odeint_stochastic_end_v2(func, batch_y0, batch_t,shrink_proportion=args.shrink_proportion,shrink_std=args.shrink_std,mode='train')
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint_stochastic_end_v2(func, true_y0, t,shrink_proportion=args.shrink_proportion,shrink_std=args.shrink_std,mode='test')
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                pred_ys = []

                random_shrink = uniform.Uniform(args.shrink_proportion -args.shrink_std, args.shrink_proportion + args.shrink_std)
                for i in range(args.ntest):
                    pred_ys.append( odeint_stochastic_end_v2_inference(func, true_y0, t,shrink_proportion=random_shrink.sample(),shrink_std=args.shrink_std,mode='test') )

                visualize(true_y, pred_y,pred_ys, func, ii)
                #visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
