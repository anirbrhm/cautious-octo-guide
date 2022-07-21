from random import sample
import torch
from torch import autograd
from torch import optim
import numpy as np
from model import *
from utils import *
from params import OUTPUT_DIR, LATENT_DIM, BATCH_SIZE
from data_processing import *

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu' 
load_model = True
output_dir = OUTPUT_DIR
latent_dim = LATENT_DIM
batch_size = BATCH_SIZE

# ============= Load Model ============#
if load_model: 
    LOGGER.info('<--------------- Loading Model --------------->')
    netD_path = os.path.join(output_dir, "discriminator.pkl")
    netG_path = os.path.join(output_dir, "generator.pkl")
    netD = torch.load(netD_path)
    netG = torch.load(netG_path) 

# Noise
noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
noise_Var = Variable(noise, requires_grad=False)

if cuda:
    netD = netD.cpu()
    netG = netG.cpu() 
    noise_Var = noise_Var.cpu()

sample_out = netG(noise_Var)

if cuda:
    sample_out = sample_out.cpu() 

save_dir = './'
save_samples(sample_out.data.numpy(), 'sample_outputs', save_dir)
