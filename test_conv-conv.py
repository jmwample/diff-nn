from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time 
import numpy as np
from matplotlib import pyplot as plt
import csv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        interm = self.input_to_interm(x)
        predicted_label = self.interm_to_output(interm)
        return predicted_label

    def input_to_interm(self, x):
        x = F.relu(self.conv1(x))
        return x

    def interm_to_output(self, x):
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x, indices = F.max_pool2d(x, 2, 2, return_indices = True)
        #print(indices.size(), type(indices))
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def split(self):
        return (self.state_dict()['conv1.weight'], self.state_dict()['conv1.bias'])


class RNet(nn.Module):
    """
    Reverse Model -- Simple MLP to do reverse task and re-create the original image
        - could be extended to include more variational imformation 
          (i.e. KL divergence or something)
    """
    def __init__(self):
        super(RNet, self).__init__()
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.deconv2 = nn.ConvTranspose2d(50, 20, 5, 1)
        self.mup1 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv1 = nn.ConvTranspose2d(20, 1, 5, 1)

    #======================[ Reverse Task ]==============================
    def forward(self, x):
        return self.interm_to_input(x)

    def interm_to_input(self, x):
        x, ind = self.mp1(x)
        h1 = F.relu(self.conv2(x))
        h2 = F.relu(self.deconv2(h1))
        h3 = self.mup1(h2, ind)
        h4 = torch.sigmoid(self.deconv1(h3))
        return h4

class CNet(nn.Module):
    """
    Client Model  -- adopts the front end of the fragmented task network
    """
    def __init__(self, weight, bias):
        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1.weight = nn.Parameter(weight)
        self.conv1.bias = nn.Parameter(bias)

        # Lock the client side model as we will not propogate back through this 
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x
    


def train_original(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_original(args, c_model, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    c_times = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            ts = time.time()
            client_out = c_model(data)
            te = time.time()
            c_times.append( te - ts )

            output = model.interm_to_output(client_out)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, c_times

# Reconstruction losses summed over all elements and batch
def loss_function(recon_x, x):
    loss = recon_x - x
    return torch.mean((loss)**2)
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    # return BCE 


def train_reverse(args, r_model, c_model, device, train_loader, optimizer, epoch):
    r_model.train()
    # Don't propogate changes to client model (we're not optimizing for reversibility)
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        interm = c_model(data)
        recon_batch = r_model(interm)

        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('[R] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)

def test_reverse(args, c_model, r_model, device, test_loader, epoch):
    r_model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            
            interm = c_model(data)
            recon_batch = r_model(interm)

            test_loss += loss_function(recon_batch, data)
            if i == 0:
                n = min(data.size(0), 8)
                data_sample = data[:n]
                recon_sample = recon_batch.view(args.test_batch_size, 1, 28, 28)[:n]
                comparison = torch.cat([ data_sample, recon_sample ])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def plot_cdf(x):
    X = np.array( sorted(x) )
    Y = np.exp( -np.power(X,2) )
    CY = np.cumsum(Y) / sum(Y)
    plt.plot(X, CY)
    # plt.plot(X,Y)
    plt.show()


def add_args(parser):
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--epochs-r', type=int, default=10, metavar='N',
                        help='number of epochs to train reverse (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='JTFP PyTorch MNIST Example')
    add_args(parser)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # Download Data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Create model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer_r = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    # Performance Tracking
    epoch_bce=[]
    epoch_d_size=[]
    c_duration=[]

    # RUN
    for epoch in range(1, args.epochs + 1):
        train_original(args, model, device, train_loader, optimizer, epoch)
        w1, b1 = model.split()
        c_model = CNet( w1, b1 ).to(device)
        mse, c_times = test_original(args, c_model, model, device, test_loader)
        c_duration.extend(c_times)

    r_model = RNet().to(device)
    print("\t#===============[ Reverse ]===============#")

    for epoch in range(1, args.epochs_r + 1):
        train_reverse(args, r_model, c_model, device, train_loader, optimizer_r, epoch)
        bce = test_reverse(args, c_model, r_model, device, test_loader, epoch)
        epoch_bce.append(bce)

    print(epoch_bce)
        # # Generate some images from random noise ( Not for this test )
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')

    if (args.save_model):
        #torch.save(model.state_dict(),"mnist_cnn.pt")
        print(model.state_dict()['conv1.weight'].shape)
        print(model.state_dict()['conv1.bias'].shape)
    # dx = 0.01
    # c_duration = np.arange(-2, 2, dx)

    # Evaluate
    with open("stats/client_times_cl", 'w') as csvf:
        csv_writer = csv.writer(csvf)
        csv_writer.writerow(c_duration)

    plot_cdf(c_duration)
        
if __name__ == '__main__':
    main()
