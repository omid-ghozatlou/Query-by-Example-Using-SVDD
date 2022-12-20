from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_MS import CIFAR10_LeNet_MS, CIFAR10_LeNet_MS_Autoencoder
from .scatter_LeNet2 import scatter_LeNet2, scatter_LeNet2_Autoencoder
from .scatter_LeNet3 import scatter_LeNet3, scatter_LeNet3_Autoencoder
from .scatter_LeNet4 import scatter_LeNet4, scatter_LeNet4_Autoencoder

def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'scatter_LeNet2', 'scatter_LeNet3', 'scatter_LeNet4', 'cifar10_LeNet_MS')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()
        
    if net_name == 'scatter_LeNet2':
        net = scatter_LeNet2()
        
    if net_name == 'scatter_LeNet3':
        net = scatter_LeNet3()
        
    if net_name == 'scatter_LeNet4':
        net = scatter_LeNet4()
        
    if net_name == 'cifar10_LeNet_MS':
        net = CIFAR10_LeNet_MS()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'scatter_LeNet2', 'scatter_LeNet3', 'scatter_LeNet4', 'cifar10_LeNet_MS')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'scatter_LeNet2':
        ae_net = scatter_LeNet2_Autoencoder()
        
    if net_name == 'scatter_LeNet3':
        ae_net = scatter_LeNet3_Autoencoder()
        
    if net_name == 'scatter_LeNet4':
        ae_net = scatter_LeNet4_Autoencoder()

    if net_name == 'cifar10_LeNet_MS':
        ae_net = CIFAR10_LeNet_MS_Autoencoder()

    return ae_net
