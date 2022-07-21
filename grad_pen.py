import torch 
from torch import autograd
from torch.autograd import Variable

def gradient_pen(netD, real_data, fake_data, batch_size, lmbda, use_cuda=False):
    alpha = (torch.rand(batch_size, 1, 1)).expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha 

    inter = alpha*real_data + (1-alpha)*fake_data
    if use_cuda:
        inter = inter.cuda()
    inter = autograd.Variable(inter, requires_grad = True)

    inter_output = netD(inter)

    gradients = autograd.grad(outputs = inter_output, inputs = inter,
                              grad_outputs = torch.ones(inter_output.size()).cuda() if use_cuda else
                              torch.ones(inter_output.size()),
                              create_graph = True, retain_graph = True, only_inputs = True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    return lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

