"""
-------------------------------------------------------------------------------
General Purpose Testing script for Content Image Creator in BagGAN -
To test networks with specified parameters see scripts.

The steps executed by the code are as follows:
- It first loads model + dataset as specified in the config file.
- Performance tests are conducted as specified by

-------------------------------------------------------------------------------
"""

from lib.util.helper import *
from lib.gan.tests import *
from lib.util.util import load_config
from lib.metrics.fidelity_tests import *

from lib.datasets.pidray_data_loader import PIDRayDataLoader
from models.bagganhq import BagGANHQ

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(
    description="Script to train BagGAN framework. "
                "Training configuration is setup using the config.py file which "
                "contains the model, dataset and training parameter values. "
                "Example config files are provided in configs/ directory.")

parser.add_argument('--config',
                    default=os.path.join(CONFIG_DIR,
                                         'config_pidray_ds_test.py'),
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
                     time.strftime('baggan_test_%m%d%Y_%H%M%S.log',
                                   time.localtime()))

# to save the image + losses per epoch
config.snap_dir       = os.path.join(config.out_dir, 'testing_snaps')
config.losses_file    = os.path.join(config.out_dir, 'testing_losses.npz')

# where the generator models are saved
config.checkpoint_dir = os.path.join(config.out_dir,
                                     'models',
                                     'expt_%s'%config.net_version)
config.test_dir = os.path.join(config.out_dir, 'test')

config.ds_dir = args.ds

assert args.ds_name=='pidray', 'Open-source package does not support ' \
                               'datasets ' \
                               'other than PIDRay'

# You may set this in the configuration directory itself
config.is_train       = False
config.load_epoch     = 740
config.load_net       = True
config.num_threads    = 1

# Create experiment directory
os.makedirs(config.out_dir, exist_ok=True)

# Set Dataset ============================================================

if not os.path.exists(config.training_log_path):
    lf = open(config.training_log_path, 'w+')
    lf.write("* BagGAN TESTING =======================\n")
    lf.close()
else:
    open(config.training_log_path, 'w').close()

logger = get_logger('BagGAN EXPT', config.training_log_path)

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

# =============================================================================

# Set Model ===================================================================
model = BagGANHQ(config)  # create a model with
                          # default_configuration
model.setup_gan()         # Setup GAN - load if previously
                          # and print

generator = model.generator.module

logger.info("-+" * 40)
logger.info("* BagGAN Model Created ...")
logger.info("-+" * 40)

test_img_dir = os.path.join(config.test_dir, 'images')

os.makedirs(test_img_dir, exist_ok=True)

for n in range(config.test_size):
    sample_latents = torch.randn(config.train_plot_layout[0]
                                 * config.train_plot_layout[1],
                                 config.z_dim,
                                 device=model.device)

    fpath = os.path.join(test_img_dir, f'test_samples_{n}.png')
    plot_stylegan_rgb_samples(model,
                              [sample_latents],
                              fpath,
                              is_labeled=False,
                              layout=config.train_plot_layout)

    logger.info(f"Saved Test Samples for iteration {n} as {fpath}")


logger.info("Calculating Inception Score ...")
mean, cov = calculate_inception_score(model, baggan_ds)
logger.info(f"Inception Score: Mean - {mean}, Cov - {cov}")

logger.info("Calculating FID Scores ...")
fid = calculate_fid_score(model)
logger.info(f"FID Score: {fid}")

