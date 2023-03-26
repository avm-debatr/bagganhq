from lib.__init__ import *

import os
import time

###############################################################################
# EXPERIMENT DETAILS

# data locations --------------------------------------------------------------
# output directory
out_dir = os.path.join(CHECKPT_DIR,
                       'pidray_baggan')

# Logger name and path
baggan_logger_name = 'PIDRay TRAINER'
training_log_path  = os.path.join(out_dir,
                     time.strftime('baggan_train_%m%d%Y_%H%M%S.log',
                                   time.localtime()))

# to save the image + losses per epoch
snap_dir       = os.path.join(out_dir, 'training_snaps')
losses_file    = os.path.join(out_dir, 'training_losses.npz')

# where the generator models are saved
net_version    = 'v1.0.0'
checkpoint_dir = os.path.join(out_dir,
                              'models',
                              'expt_%s'%net_version)

# Experiment parameters -------------------------------------------------------

# for training/testing experiment
is_train       = True           # set for training
ds_type        = 'real'         # default = real - 'simulated' if using DEBISim
mode           = 'bagganhq'     # other option is pix2pix - not used
test_mode      = None           # this is a training script

# preprocessing parameters
image_size      = 256           # for generated images
image_dims      = 384, 384

# training display/save parameters
print_freq       = 400         # set as per dataset size
display_freq     = 2000        # for saving imager snapshots
losses_to_print  = ['g_gan', 'd',  'g_ppl']
save_by_iter     = False        # avoiding using this if ds size is large
save_epoch_freq  = 20           # set as per memory requirements
save_only_latest = False        # if no epochwise saving is required

train_plot_layout = [5, 5]      # no of GAN image to plot
# =============================================================================

###############################################################################
# DATASET DETAILS

# dataset loader parameters ---------------------------------------------------

# See BagGAN-HQ repository for more information about these parameters
ds_dir=os.path.join(DATA_DIR, 'pidray')
subset='easy'

batch_size = 40
serial_batches = False
num_threads = 20

# =============================================================================

###############################################################################
# MODEL PARAMETERS

# normalization + layer options
norm         = 'instance'
init_gain    = 0.02
gpu_ids      = [0]
num_channels = 3 # 2

latent_dim   = 512
z_dim, w_dim = latent_dim, latent_dim

generator_params = dict(latent_dims=(z_dim, w_dim),
                        img_resolution=image_size,
                        mlp_layers=8,
                        mlp_lr=0.01,
                        img_chls=num_channels,
                        fir_filter=[1,3,3,1],
                        res2chlmap=None)

disc_params = dict(img_resolution=image_size,
                   img_chls=num_channels,
                   res2chlmap=None,
                   with_q=False)

start_epoch    = 1
n_epochs       = 100

# for continuing/loading saved experiment
continue_train = False # True
load_epoch     = None # 200
load_net       = False # True
verbose        = True

gan_mode      = 'wgangp' # 'vanilla'  # 'vanilla'

# stylegan2 parameters
chl_multiplier = 2  # channel multiplier
wandb = True
local_rank = 0      #

###############################################################################
# TESTING PARAMETERS

is_train       = False
load_epoch     = 740 # None
load_net       = True # False
max_samples    = 1000

test_size   = 50
test_batch  = 20
test_dir    = os.path.join(out_dir, 'test')

expt_desc = "EXPT. DESCRIPTION: " \
            "BENCHMARK: StyleGAN2 " \
            "Full PIDRay Dataset " \
            "- 256 x 256 res., wgangp loss " \
            "PPL Regularization added + ADA included with random affine"
