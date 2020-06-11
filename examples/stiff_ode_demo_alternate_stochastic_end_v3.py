import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=120) #default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--ntest', type=int, default=10)
parser.add_argument('--n_units', type=int, default=500)
parser.add_argument('--min_length', type=float, default=0.001)
parser.add_argument('--normal_std', type=float, default=0.01)
parser.add_argument('--stiffness_ratio', type=float, default=1000.0)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--version', type=str, choices=['standard','steer','normal'], default='steer')
args = parser.parse_args()
torch.manual_seed(6)
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    from torchdiffeq import odeint_adjoint_stochastic_end_v3 as odeint_stochastic_end_v3
    from torchdiffeq import odeint_adjoint_stochastic_end_normal as odeint_stochastic_end_normal
else:
    from torchdiffeq import odeint_stochastic_end_v3
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([0.])
#true_y0 = torch.tensor([1.])
t = torch.linspace(0., 15., args.data_size)
test_t = torch.linspace(0., 25., args.data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])


class Lambda(nn.Module):

    def forward(self, t, y):
        t = t.unsqueeze(0)
        equation = -1000*y + 3000 - 2000 * torch.exp(-t) + 1000 * torch.sin(t)
        #equation = -1000*y + 3000 - 2000 * torch.exp(-t)
        #equation = -100*y + 300 - 200 * torch.exp(-1*t)
        #equation = -1*y*args.stiffness_ratio + 3*args.stiffness_ratio - 2*args.stiffness_ratio * torch.exp(-1*t)# - 2*args.stiffness_ratio * torch.exp(-10000*t)

        #equation = 0.12*y #-2000*torch.exp(-t)#+ 3000 - 2000 * torch.exp(-t)
        #equation = 10 * torch.sin(t)
        return equation
        #return torch.mm(y**3, true_A)
        #return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_y_test = odeint(Lambda(), true_y0, test_t, method='dopri5')
    #true_y = odeint(Lambda(), true_y0, t, method='adams')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    #print(batch_t)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png_alternate_stochastic_end_v3')
    import matplotlib.pyplot as plt
    #fig = plt.figure(figsize=(8, 4), facecolor='white')
    #ax_traj = fig.add_subplot(121, frameon=True)
    #ax_phase = fig.add_subplot(122, frameon=True)
    ##ax_multiple = fig.add_subplot(133, frameon=False)
    #plt.show(block=False)

def visualize(true_y, pred_y, odefunc, test_t, itr):
    if args.viz:

        plt.clf()
        #plt.set_title('True vs Predicted')
        #plt.set_xlabel('t')
        #plt.set_ylabel('y')
        plt.plot(test_t.numpy(), true_y.numpy()[:, 0], 'g-', label='True')
        plt.plot(test_t.numpy(), pred_y.numpy()[:, 0], 'b--' , label='Predicted' )
        #plt.set_xlim(t.min(), t.max())
        plt.ylim((-1, 25))
        plt.legend(loc="upper right")
        #plt.legend()



        #fig.tight_layout()
        plt.tight_layout()
        plt.savefig('png_alternate_stochastic_end_v3/{:04d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


#def visualize(true_y, pred_y, odefunc, itr):
#
#    if args.viz:
#
#        ax_traj.cla()
#        ax_traj.set_title('True vs Predicted')
#        #ax_traj.set_xlabel('t')
#        #ax_traj.set_ylabel('y')
#        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0], 'g-')
#        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0], '--', 'b--')
#        ax_traj.set_xlim(t.min(), t.max())
#        ax_traj.set_ylim(-10, 25)
#        #ax_traj.legend()
#
#        ax_phase.cla()
#        ax_phase.set_title('Predicted')
#        #ax_phase.set_xlabel('t')
#        #ax_phase.set_ylabel('y')
#        ax_phase.plot(t.numpy(), pred_y.numpy()[:, 0], '--', 'b--')
#        ax_phase.set_xlim(t.min(), t.max())
#        ax_phase.set_ylim(-10, 25)
#        ax_phase.legend()
#
#        #ax_multiple.cla()
#        #ax_multiple.set_title('Variations')
#        #ax_multiple.set_xlabel('t')
#        #ax_multiple.set_ylabel('y')
#        #for component in pred_ys:
#        #    ax_multiple.plot(t.numpy(), component.numpy()[:, 0], '--')
#        #ax_multiple.set_xlim(t.min(), t.max())
#        #ax_multiple.set_ylim(-100, 100)
#        #ax_multiple.legend()
#
#
#
#        fig.tight_layout()
#        plt.savefig('png_alternate_stochastic_end_v3/{:04d}'.format(itr))
#        plt.draw()
#        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, args.n_units),
            nn.Tanh(),
            nn.Linear(args.n_units, args.n_units),
            nn.Tanh(),
            nn.Linear(args.n_units, 1),
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
    optimizer = optim.RMSprop(func.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        if args.version=='standard':
            pred_y = odeint(func, batch_y0, batch_t)
        elif args.version=='steer':
            pred_y = odeint_stochastic_end_v3(func, batch_y0, batch_t,min_length=args.min_length,mode='train')
        elif args.version=='normal':
            pred_y = odeint_stochastic_end_normal(func, batch_y0, batch_t,std=args.normal_std,mode='train')

        #pred_y = odeint_stochastic_end_v2(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, test_t)
                loss = torch.mean(torch.abs(pred_y - true_y_test))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                #pred_ys = []

                #for i in range(args.ntest):
                #    pred_ys.append(  odeint(func, batch_y0, batch_t))

                #visualize(true_y, pred_y,pred_ys, func, ii)
                visualize(true_y_test, pred_y, func, test_t,  ii )
                ii += 1

        end = time.time()
