"""
-------------------------------------------------------------------------------
General Purpose Training script for BagGAN -
To run training with specified parameters see scripts/directory.

The steps executed by the code are as follows:
- It first creates model + data_loader given the options by reading in the
  configs file.
- Standard network training is then performed based on training parameters
  in configs file.
- If validation params are set in the configs file, validation tests are also
  performed.

- During the training, it also
    - visualizes/saves the images at fixed intervals,
    - print/save the loss plots during the training,
    - saves models at fixed checkpoints.
-------------------------------------------------------------------------------
"""

from lib.util.helper import *
from lib.gan.tests import *
from lib.util.util import load_config

from lib.datasets.pidray_data_loader import PIDRayDataLoader
from models.bagganhq import BagGANHQ

plt.switch_backend('agg')


parser = argparse.ArgumentParser(
    description="Script to train BagGAN framework. "
                "Training configuration is setup using the config.py file which "
                "contains the model, dataset and training parameter values. "
                "Example config files are provided in configs/ directory.")

parser.add_argument('--config',
                    default=os.path.join(CONFIG_DIR,
                                         'config_pidray_ds_train.py'),
                    help='Configuration filepath for training '
                         '- check configs/ directory for examples to create '
                         'your own')

parser.add_argument('--ds',
                    default=os.path.join(DATA_DIR,
                                         'pidray'),
                    help='Path to dataset directory for training')

parser.add_argument('--ds_name',
                    default='pidray',
                    choices=['pidray', 'gdxray', 'debisim', 'custom'],
                    help='Type of dataset used - this open-source package '
                         'only supports PIDRay dataset')

parser.add_argument('--out_dir',
                    default=os.path.join(CHECKPT_DIR,
                                         'pidray_baggan'),
                    help='Output directory')

args = parser.parse_args()

config = load_config(args.config)

config.out_dir = args.out_dir
config.training_log_path  = os.path.join(config.out_dir,
                     time.strftime('baggan_train_%m%d%Y_%H%M%S.log',
                                   time.localtime()))

# to save the image + losses per epoch
config.snap_dir       = os.path.join(config.out_dir, 'training_snaps')
config.losses_file    = os.path.join(config.out_dir, 'training_losses.npz')

# where the generator models are saved
config.checkpoint_dir = os.path.join(config.out_dir,
                                     'models',
                                     'expt_%s'%config.net_version)

config.ds_dir = args.ds

assert args.ds_name=='pidray', 'Open-source package does not support datasets ' \
                               'other than PIDRay'

# Create experiment directory
os.makedirs(config.out_dir, exist_ok=True)

# Set Dataset =================================================================

if not os.path.exists(config.training_log_path):
    lf = open(config.training_log_path, 'w+')
    lf.write("* BagGAN TRAINING =======================\n")
    lf.close()
else:
    open(config.training_log_path, 'w').close()

logger = get_logger('BagGAN',
                    config.training_log_path)

logger.info("-+" * 40)
logger.info("-+" * 40)
logger.info("BagGAN Training Expt. - "
            "Starting at " + time.strftime('%m/%d/%Y %H:%M:%S',
                                           time.localtime()))
logger.info("-+" * 40)
logger.info("-+" * 40)

baggan_ds = PIDRayDataLoader(config.ds_dir,
                               config.subset,
                               is_train=config.is_train,
                               image_size=config.image_size)

os.makedirs(config.snap_dir, exist_ok=True)

data_loader = DataLoader(baggan_ds,
                         batch_size=config.batch_size,
                         shuffle=True,
                         num_workers=config.num_threads,
                         pin_memory=True)

if config.continue_train and config.load_epoch is None:
    config.load_epoch, config.continue_train = \
        get_latest_saved_checkpoint(config)

# Set Model ===================================================================
# create a model with load configs
model = BagGANHQ(config)
model.setup_gan()  # Setup GAN - load if previously and print

# algos; create schedulers
total_iters = 0  # the total number of training iterations

logger.info("-+" * 40)
logger.info("* BagGAN Model Created ...")
logger.info("-+" * 40)

iters_per_epoch = len(data_loader) * config.batch_size

if config.continue_train and os.path.exists(config.losses_file):
    load_iter = config.load_epoch * iters_per_epoch // config.print_freq
    loss_dict = {k: list(v)[:load_iter]
                 for k, v in np.load(config.losses_file).items()}
else:
    loss_dict = {k: [] for k in model.loss_names}

# =============================================================================
# Start training in epochs ====================================================

# outer loop for different epochs:
# we save the model by <start_epoch>, <start_epoch>+<save_latest_freq>

epoch, epoch_iter = 0, 0
sample_latents = torch.randn(config.train_plot_layout[0]
                             * config.train_plot_layout[1],
                             config.z_dim,
                             device=model.device)
bsize = sample_latents.shape[0]
cat_input, cat_labels = None, None

# =============================================================================
# TRAINING EPOCHS

for epoch in range(config.start_epoch,
                   config.n_epochs + 1):

    # Timers and counters -----------------------------------------------------
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in
                                    # current epoch, reset to 0 every epoch
    # -------------------------------------------------------------------------
    t_disp_time = time.time()

    # Training Loop for 1 epoch ===============================================
    # inner loop within one epoch

    for i, data in enumerate(data_loader):

        # Timers and Counters -------------------------------------------------
        iter_start_time = time.time()  # timer for computation per iteration

        if total_iters % config.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += config.batch_size
        epoch_iter  += config.batch_size
        # ---------------------------------------------------------------------

        # Feed input data from dset to network for training -------------------

        model.set_input(data,                # unpack data from data_loader
                        epoch_no=epoch,
                        iter_no=i * config.batch_size)

        # and apply preprocessing
        model.optimize_parameters()  # calculate loss functions,
                                     # get gradients,
                                     # update network weights
        # ---------------------------------------------------------------------

        # Periodically save images and display them ---------------------------

        # print training losses and save logging information to the disk
        if epoch_iter % config.print_freq == 0:
            losses = model.get_current_losses()
            t_comp = time.time() -  t_disp_time
            l_print = ' (%i/%i) ' % (epoch, epoch_iter)

            for loss, val in losses.items():
                l_print = l_print + '%s: %.3f ' % (loss, val)

            l_print = l_print + 'T: %.3f' % t_comp
            logger.info(l_print)
        # ---------------------------------------------------------------------

        # Display Output from trained model at specified intervals ------------
        if total_iters % config.display_freq == 0:

            # Save loss values for plotting
            losses = model.get_current_losses()
            for loss, val in losses.items():
                loss_dict[loss].append(val)

            # vary clutter parameter values and save o/p as pil collage -------
            if config.DISPLAY_TRAINING_OUTPUT:

                fpath = os.path.join(config.snap_dir,
                f'samples_e_{epoch}_i_{total_iters%iters_per_epoch}.png')
                plot_stylegan_rgb_samples(model,
                                               [sample_latents],
                                               fpath,
                                               is_labeled=False,
                                               layout=config.train_plot_layout)

                logger.info("Saved snapshots at E: %i, Iter: %i" % (
                    epoch,
                    total_iters%iters_per_epoch))
            # -----------------------------------------------------------------

            # Plot Training loss curves ---------------------------------------
            if config.PLOT_TRAINING_LOSS:

                plot_training_loss(loss_dict,
                                   epoch,
                                   10,
                                   iters_per_epoch // config.display_freq,
                                   os.path.join(config.out_dir,
                                                'loss_plot_all.png'))

                np.savez_compressed(config.losses_file, **loss_dict)
                logger.info("Losses plotted and saved")
        # ---------------------------------------------------------------------

        iter_data_time = time.time()

    # -------------------------------------------------------------------------
    # End of Epoch
    # -------------------------------------------------------------------------

    # Validate model at the end of every epoch --------------------------------

    # cache our model every <save_epoch_freq> epochs --------------------------
    if epoch % config.save_epoch_freq == 0:
        logger.info(
            'saving the model at the end of epoch %d, iters %d' %
            (epoch, total_iters%iters_per_epoch))

        save_suffix = 'e_%d_i_%d' % (epoch, 0)

        if config.save_only_latest:
            model_files = os.listdir(config.checkpoint_dir)
            for m in model_files:
                os.remove(os.path.join(config.checkpoint_dir, m))

        model.save_networks(save_suffix)
    # -------------------------------------------------------------------------

    logger.info('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch,
                 config.n_epochs,
                 time.time() - epoch_start_time))

    # --------------------------------------------------------------------

model.save_networks('final')
np.savez_compressed(config.losses_file, **loss_dict)
logger.info("Losses saved at %s" % config.losses_file)

plt.figure()
plt.title("GAN Training Loss")

for lname, lcurve in loss_dict.items():
    plt.plot(lcurve, label=lname)
    l = lcurve

plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Losses')

plt.xticks(np.arange(0, len(l), 10*iters_per_epoch//config.display_freq),
           np.arange(0, epoch, 10))
plt.savefig(os.path.join(config.out_dir, 'loss_plot_all.png'))
plt.close()

logger.info("-+" * 40)
logger.info("-+" * 40)
logger.info("Training Completed - "
            "Ending at " + time.strftime('%m/%d/%Y, %H:%M:%S',
                                         time.localtime()))
logger.info("-+" * 40)
logger.info("-+" * 40)