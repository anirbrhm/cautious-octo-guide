import torch
from torch import autograd
from torch import optim
import numpy as np
from model import *
from params import BATCH_SIZE, BETA1, BETA2, LATENT_DIM, LEARNING_RATE, LMBDA, MODEL_SIZE, NGPUS, OUTPUT_DIR
from utils import *
from logger import *
from grad_pen import *
from data_processing import *

cuda = True if torch.cuda.is_available() else False
device = 'cuda' if cuda else 'cpu' 

# ========Hyperparameters===========
args = parse_arguments()
epochs = args['num_epochs']
load_model = False if args['load_model'] == 'False' else True
batch_size = BATCH_SIZE
latent_dim = LATENT_DIM
ngpus = NGPUS
model_size = MODEL_SIZE
lmbda = LMBDA

audio_dir = AUDIO_DIR
output_dir = OUTPUT_DIR

# =============Network===============
netG = Generator(model_size = model_size, ngpus = ngpus, latent_dim = latent_dim, upsample = True)
netD = Discriminator(model_size = model_size, ngpus = ngpus)

# =============Logger===============
LOGGER = logging.getLogger('WaveGAN')
LOGGER.setLevel(logging.DEBUG)

init_console_logger(LOGGER)

# ============= Load Model ============#
if load_model: 
    LOGGER.info('<--------------- Loading Model --------------->')
    netD_path = os.path.join(output_dir, "discriminator.pkl")
    netG_path = os.path.join(output_dir, "generator.pkl")
    netD = torch.load(netD_path)
    netG = torch.load(netG_path) 

if cuda:
    netG = netG.cuda()
    netD = netD.cuda()

optimizerG = optim.Adam(netG.parameters(), lr = LEARNING_RATE, betas = (BETA1, BETA2))
optimizerD = optim.Adam(netD.parameters(), lr = LEARNING_RATE, betas = (BETA1, BETA2))

LOGGER.info('<--------------- Loading Data --------------->')
audio_paths = get_audio_paths(audio_dir)
train_data, train_size = process_data(audio_paths,batch_size)

BATCH_NUM = train_size // batch_size
train_iter = iter(train_data)
start = time.time()

LOGGER.info('<------ Training started with EPOCHS = {} and BATCH_SIZE = {} ------>'.format(epochs, batch_size))

#============= Tracking Losses ==========# 
D_costs = []
G_costs = []

for epoch in range(1, epochs+1):
    LOGGER.info("{} Epoch: {}/{}".format(time_since(start), epoch, epochs))

    D_cost_epoch = [] 
    G_cost_epoch = [] 
    
    for i in range(1, BATCH_NUM+1):
        for p in netD.parameters():
            p.requires_grad = True
  
        one = torch.tensor(1, dtype=torch.float)
        minus_one = -1 * one 
        if cuda:
            one = one.cuda()
            minus_one = minus_one.cuda()
        
        # (1) Train Discriminator

        for iter_dis in range(5):
            netD.zero_grad()

            noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
            if cuda:
                noise = noise.cuda()
            noise_Var = Variable(noise, requires_grad=False)

            real_data_Var = convert_np(next(train_iter)['X'], cuda)

            D_real = netD(real_data_Var)
            D_real = D_real.mean() 
            D_real.backward(minus_one)  

            fake = autograd.Variable(netG(noise_Var).data)
            D_fake = netD(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            gradient_penalty = gradient_pen(netD, real_data_Var.data,
                                                     fake.data, batch_size, lmbda,
                                                     use_cuda = cuda)
            gradient_penalty.backward(one)

            optimizerD.step()

            D_cost = D_fake - D_real + gradient_penalty
            if cuda:
                D_cost = D_cost.cpu() 

            D_cost_epoch.append(D_cost.data.numpy()) 


        # (2) Train Generator

        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()

        noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
        if cuda:
            noise = noise.cuda()
        noise_Var = Variable(noise, requires_grad = False)

        fake = netG(noise_Var)
        G = netD(fake)
        G = G.mean()
        G.backward(minus_one)

        optimizerG.step()

        G_cost = -G 
        if cuda:
            G_cost = G_cost.cpu() 

        G_cost_epoch.append(G_cost.data.numpy()) 

    D_cost_epoch_avg = sum(D_cost_epoch) / float(len(D_cost_epoch))
    G_cost_epoch_avg = sum(G_cost_epoch) / float(len(G_cost_epoch))

    D_costs.append(D_cost_epoch_avg)
    G_costs.append(G_cost_epoch_avg) 

    LOGGER.info("{} Discriminator Loss: {:.6f} || Generator Loss: {:.6f}".format(time_since(start), D_cost_epoch_avg, G_cost_epoch_avg))

    # Generate audio samples.
    sample_out = netG(noise_Var)
    if cuda:
        sample_out = sample_out.cpu()
    save_samples(sample_out.data.numpy(), epoch, output_dir)
        
    # Save model
    LOGGER.info("Saving models...")
    netD_path = os.path.join(output_dir, "discriminator.pkl")
    netG_path = os.path.join(output_dir, "generator.pkl")

    torch.save(netG, netG_path)
    torch.save(netD, netD_path) 

plot(D_costs, G_costs, output_dir) 
LOGGER.info('<--------------- Training Completed --------------->')