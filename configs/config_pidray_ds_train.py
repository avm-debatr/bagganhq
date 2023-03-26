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
display_freq     = 2000         # for saving imager snapshots
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
subset='train'

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

###############################################################################
# TRAINING PARAMETERS

start_epoch    = 1
n_epochs       = 100

# for continuing/loading saved experiment
continue_train = False # True
load_epoch     = None # 200
load_net       = False # True
verbose        = True

gan_mode      = 'wgangp' # 'vanilla'  # 'vanilla'

# stylegan2 parameters
use_ppl      = True
r1_lambda = 10             # R1 regularization
ppl_lambda = 2      # weight of the path length regularization
path_batch_shrink = 2 # batch size reducing factor for ppl
ppl_decay  = 0.01
d_reg_every = 16     # interval of applying r1 reg to D
g_reg_every = 4     # interval of applying r1 reg to G
mixing_prob = 0.9   # probability of mixing latent code
chl_multiplier = 2  # channel multiplier
wandb = True
local_rank = 0      #

g_reg_ratio = g_reg_every / (g_reg_every + 1)
d_reg_ratio = d_reg_every / (d_reg_every + 1)

# adaptive discriminator augmentation
augment = True
augment_p = 0
ada_target = 0.6
ada_length = 500*1000
ada_freq = 256

# optimization/loss parameters
lr    = 0.002
beta1 = 0.0

lr_policy      = 'linear'
lr_params = dict(epoch_count=1,
                 n_epochs=100,
                 n_epochs_decay=100,
                 lr_decay_iters=50)

PLOT_TRAINING_LOSS           = True
DISPLAY_TRAINING_OUTPUT      = True

###############################################################################
# VALIDATION PARAMETERS

valid_flag  = True
valid_size  = 100
valid_batch = 10
valid_dir   = os.path.join(out_dir, 'validation')
valid_tests = ['clutter_stats', 'hist_scores', 'hist_plot']
clutter_valid_file = os.path.join(valid_dir,
                                  'clutter_valid_scores.npz')

valid_clutter_range   = [None]*3 # [0.3, 0.5, 0.7]
num_plot_valid_images = 2

VALIDATE_CGAN_PARAM          = False
VALIDATE_HIST_MATCHING_SCORE = True
