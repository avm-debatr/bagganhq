# Configuration options for training ATP-GAN

import os, time

###############################################################################
# EXPERIMENT DETAILS

# data locations -------------------------------------------------------------
expt_dir     = '/mnt/cloudNAS3/Ankit/baggan_expts_2021/'
expt         = 'v7.5.2_cic_cgan_hm'
out_dir      = os.path.join(expt_dir,
                            'expt_v7.x_cic_unet_cgan/',
                            'expt_%s'%expt)

baggan_logger_name = 'CIC TRAINER'
training_log_path  = os.path.join(out_dir,
                     time.strftime('cic_train_'
                                   '%m%d%Y_%H%M%S.log',
                                   time.localtime()))

snap_dir       = os.path.join(out_dir, 'training_snaps')
losses_file    = os.path.join(out_dir, 'training_losses.npz')

net_version    = 'v7.5.2'
checkpoint_dir = os.path.join(out_dir,
                              'models',
                              'cic_unet_cgan_%s'%net_version)

expt_desc = '\n\nDescription:\n' \
            '* Histogram-based Content Image Creator v7 - BENCHMARKING\n' \
            '* WGAN-GP for 2D Bag Generation\n' \
            '* Trained on primitives_shapes DS_V09 ' \
            '  (Primitive Shapes Dataset), 20 epochs\n' \
            '* 5-layer UNet generator, 4-layer discriminator \n' \
            '* Use a UNet based generator + \n' \
            '* Input is a 256 x 256 noise image sampled from histogram\n' \
            '* (Samples are fixed for each epoch - presaved and loaded)\n' \
            '* Interpolation Layer added to remove checkerboard artifacts\n'\
            '* No HM Loss added\n' \
            '=================\n'

# Experiment parameters --------------------------------------------------------

# for training/testing experiment
is_train       = True
ds_type        = 'sim'
mode           = 'cic' # 'recon' # 'gt' # 'gt_lac'
test_mode      = None

# preprocessing parameters
image_size      = 256
image_dims      = 384, 384

# training display/save parameters
print_freq       = 2000
display_freq     = 10000
losses_to_print  = ['g_l2', 'g_hm', 'g_gan', 'd']
save_by_iter     = False
save_epoch_freq  = 5
save_only_latest = False
# =============================================================================

###############################################################################
# DATASET DETAILS

# dataset loader parameters ---------------------------------------------------

ds_dir       = '/mnt/cloudNAS3/Ankit/1_Datasets/BagGAN_DS/DS_V09'
bag_ds_args = dict(ds_dir=ds_dir,
                   ds_type=ds_type,
                   ds_batch='primitive_shapes',
                   version='7.0',
                   # ds_batch='low_clutter',
                   # version='1.0',
                   logfile=training_log_path)

baggan_args = dict(bag_list=None,
                   rmetal=None,
                   rclutter=None)

baggan_expt = 'e1.0_prim'
# baggan_expt = 'e1.0_low'
baggan_version = '1.0'

baggan_train_args = dict(expt=baggan_expt,
                         train_test_ratio=(0.75, 0.05, 0.2),
                         split_bags=True)

batch_size = 20
serial_batches = False
num_threads = 20
add_noise_vector = False
# =============================================================================

###############################################################################
# MODEL PARAMETERS
is_dcgan = False

# normalization + layer options
norm        = 'batch'
init_gain   = 0.02
use_dropout = False
gpu_ids     = [0]
num_channels  = 1

latent_vector_size = 200
histogram_bins     = 100
hist_vector        = 100
histogram_range    = (-1.0, 1.0)
interp_layer       = True

param_max_vals = {'clutter': 8e4, 'metals': 8e3}

input_params = dict(z_dim=latent_vector_size,
                    hist_size=histogram_bins,
                    c_dim=1,
                    img_dim=image_size)

# input_params['param_map'] = {0:'clutter', 1: 'metals'}
input_params['param_map'] = {0: 'clutter'}

generator_params = dict(
    l1={'down_nc': (64 * 8, 64 * 8),    'up_nc' : (64 * 8, 64 * 8),
        'kspb_d' : (4, 2, 1, False),    'kspb_u': (3, 1, 0, False),
        'norm'   : norm,                'act'   : ('l_relu', 'relu'),
        'dropout': use_dropout},
    l2={'down_nc': (64 * 4, 64 * 8),    'up_nc' : (64 * 16, 64 * 4),
        'kspb_d' : (4, 2, 1, False),    'kspb_u': (3, 1, 0, False),
        'norm'   : norm,                'act'   : ('l_relu', 'relu'),
        'dropout': use_dropout},
    l3={'down_nc': (64 * 2, 64 * 4),    'up_nc' : (64 * 8, 64 * 2),
        'kspb_d' : (4, 2, 1, False),    'kspb_u': (3, 1, 0, False),
        'norm'   : norm,                'act'   : ('l_relu', 'relu'),
        'dropout': use_dropout},
    l4={'down_nc': (64 * 1, 64 * 2),    'up_nc' : (64 * 4, 64 * 1),
        'kspb_d' : (4, 2, 1, False),    'kspb_u': (3, 1, 0, False),
        'norm'   : norm,                'act'   : ('l_relu', 'relu'),
        'dropout': use_dropout},
    l5={'down_nc': (1, 64),             'up_nc' : (64 * 2, 1),
        'kspb_d' : (4, 2, 1, False),    'kspb_u': (3, 1, 0, False),
        'norm'   : norm,                'act'   : ('l_relu', 'relu'),
        'dropout': use_dropout}
)

critic_params = dict(
    l1={'in': 2,       'out': 64,           'kspb': (4, 2, 1, False),
        'norm': None,  'act': 'l_relu'},
    l2={'in': 64 *  1, 'out': 64 *  2,      'kspb': (4, 2, 1, False),
        'norm': norm,  'act': 'l_relu'},
    l3={'in': 64 *  2, 'out': 64 *  4,      'kspb': (4, 2, 1, False),
        'norm': norm,  'act': 'l_relu'},
    l4={'in': 64 *  4, 'out': 64 *  8,      'kspb': (4, 2, 1, False),
        'norm': norm,  'act': 'l_relu'},
    l5={'in': 64 *  8, 'out': 1,            'kspb': (4, 2, 1, False),
        'norm': None,  'act': 'none'}
)

###############################################################################
# TRAINING PARAMETERS

start_epoch    = 1
n_epochs       = 50
max_samples    = None
load_hist_images = True

# for continuing/loading saved experiment
continue_train = False
load_epoch     = None
load_net       = False
verbose        = True

gan_mode      = 'wgangp' # 'vanilla'  # 'vanilla'
lambda_l2     = 1.0 # 0.1
lambda_hm     = 0.0

# optimization/loss parameters
lr    = 0.0002
beta1 = 0.5

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
clutter_valid_file = os.path.join(valid_dir, 'clutter_valid_scores.npz')

valid_clutter_range   = [0.3, 0.5, 0.7]
num_plot_valid_images = 5

VALIDATE_CGAN_PARAM          = False
VALIDATE_HIST_MATCHING_SCORE = True

###############################################################################
# TESTING PARAMETERS

test_size   = 100
test_batch = 50
test_dir    = os.path.join(out_dir, 'test')
test_list   = ['clutter_stats', 'hist_scores', 'hist_plot']
clutter_test_file = os.path.join(valid_dir, 'clutter_test_scores.npz')

test_clutter_range   = [0.3, 0.35, 0.4, 0.45, 0.5,
                        0.55, 0.6, 0.65, 0.7, 0.75,
                        0.8]
num_plot_test_images = 50

TEST_CGAN_PARAM          = False
TEST_HIST_MATCHING_SCORE = True
