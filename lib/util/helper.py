from lib.util.visualization import *
from torch.utils.data import DataLoader


def create_baggan_dataset_loader(config,
                                 logger,
                                 mode='train',
                                 is_dect=False,
                                 baggan_handler=None,
                                 verbose=True):
    """
    ---------------------------------------------------------------------------
    Creates a PyTorch data loader object for training/testing/validation.
    - specifications for creating the data loader are obtained from the config
      file.
    - A logger can be provided to print the dataset specifications
    - The output is print only if verbose=True
    - A lib.datasets.BagDataHandler object can be provided to set up the
      dataloader. If not provided, a new handler is created and returned.

    :param config:      Config data object
    :param logger:      Python logger object for printing
    :param mode:        {"train" | "test" | "valid"}
    :param verbose:     set to True to print output

    :return: data_loader    - torch.utils.data.DataLoader,
             baggan_ds      - lib.datasets.BagDataLoader,
             baggan_handler - { lib.datasets.BagDataHandler | None }
    ---------------------------------------------------------------------------
    """

    assert mode in ['train', 'test', 'valid'], \
        'Mode not recognized! Choose from {train | test | valid}'

    # Create a new handler if none is provided
    if baggan_handler is None:
        baggan_handler = BagDataPreprocessor(
            config.bag_ds_args,
            config.baggan_args,
            expt=config.baggan_expt,
            baggan_version=config.baggan_version,
            target=config.target,
            new_target=config.new_target,
            aug_dir=config.aug_dir
        )

    if verbose:
        logger.info("* EXPT NAME: %s" % config.expt)
        logger.info("* DESC.: %s" % config.expt_desc)

        logger.info("-+" * 40)
        logger.info("* ContentGAN Dataset Handler Loaded ...")
        logger.info("-+" * 40)

    # DataLoader arguments change depending on the mode
    if mode in ['train', 'test']:

        if config.mode=='atr':
            baggan_ds = ATRDataLoader(
                ds_dir=config.ds_dir,
                ds_handler=baggan_handler,
                baggan_train_args=config.baggan_train_args,
                is_train=config.is_train,
                mode=config.mode,
                image_size=config.image_size,
                sample_list=config.max_samples
            )
        else:
            baggan_ds = BagDataLoader(
                ds_dir=config.ds_dir,
                ds_handler=baggan_handler,
                baggan_train_args=config.baggan_train_args,
                is_train=config.is_train,
                mode=config.mode,
                image_size=config.image_size,
                test_mode=config.test_mode,
                sample_list=config.max_samples,
                load_hist=config.load_hist_images
            )

        if verbose:
            logger.info("-+" * 40)
            logger.info("* ContentGAN Dataset Loader Created ...")
            logger.info("-+" * 40)

        data_loader = DataLoader(
            baggan_ds,
            batch_size=config.batch_size if mode=='train' else config.test_batch,
            shuffle=False if mode=='test' else not config.serial_batches,
            num_workers=config.num_threads,
            pin_memory=True
        )

    elif mode=='valid':
        if not os.path.exists(config.valid_dir):
            os.makedirs(config.valid_dir)

        if config.mode=='atr':
            baggan_ds = ATRDataLoader(
                ds_dir=config.ds_dir,
                ds_handler=baggan_handler,
                baggan_train_args=config.baggan_train_args,
                is_train=False,
                mode=config.mode,
                image_size=config.image_size,
                sample_list=config.valid_size
            )
        else:
            baggan_ds = BagDataLoader(
                ds_dir=config.ds_dir,
                ds_handler=baggan_handler,
                baggan_train_args=config.baggan_train_args,
                is_train=False,
                mode=config.mode,
                image_size=config.image_size,
                test_mode=config.test_mode,
                sample_list=config.valid_size
            )

        if verbose:
            logger.info("-+" * 40)
            logger.info("* Data Loader for Validation Created ...")
            logger.info("-+" * 40)

        data_loader = DataLoader(
            baggan_ds,
            batch_size=config.valid_batch,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
    else:
        raise IOError("mode argument not recognized!")

    if not os.path.exists(config.snap_dir):
        os.makedirs(config.snap_dir)

    if verbose: logger.info(f'Number of {mode} images = {len(baggan_ds)}')

    return data_loader, baggan_ds, baggan_handler
# -----------------------------------------------------------------------------


def plot_training_output(model,
                         config,
                         nepoch=0,
                         niter=0,
                         layout=(3,3)):
    """
    ---------------------------------------------------------------------------
    Plot sample output images for GAN for a given epoch/iteration.
    Plotted images are saved as .png file using PIL and in the path given by
    config.snap_dir

    :param model:  GAN model
    :param config: config data object
    :param nepoch: epoch number
    :param niter:  iteraiton number
    :param layout: (rows, cols) layout for display
    :return:
    ---------------------------------------------------------------------------
    """
    out_images = []
    lyt = layout[0]*layout[1]

    hist_length = model.d_hist.shape[0]

    for i in range(lyt):

        if config.is_dcgan:
            r_vectors = model.get_latent_vector(
                {'clutter': 0.7},
                channels=config.latent_vector_size,
                hist=torch.unsqueeze(
                    model.d_hist[i, :, :, :],
                    0).type(torch.FloatTensor),
                batch=1
            )
            out_image = model.test(input_latent_vector=r_vectors)
        else:
            if config.target is None:
                out_image = model.test(
                    hist_vector=torch.unsqueeze(
                        model.d_hist[lyt%hist_length if lyt>=hist_length
                                                     else hist_length%lyt,
                                     :, :, :],
                        0).type(torch.FloatTensor))
            else:
                out_image, mask = model.test(
                    hist_vector=torch.unsqueeze(
                        model.d_hist[lyt % hist_length if lyt >= hist_length
                                     else hist_length % lyt,
                        :, :, :],
                        0).type(torch.FloatTensor),
                    return_mask=True
                )

        if config.mode=='cic_de':
            out_images.append(out_image[0,:,:].cpu().numpy())
            out_images.append(out_image[1,:,:].cpu().numpy())

            if config.target is not None:
                out_images.append(mask.cpu().numpy()*2-1)

        else:
            out_images.append(out_image.cpu().numpy())

    if config.mode == 'cic_de':
        if config.target is None:
            layout[1] *= 2
        else:
            layout[1] *= 3

    create_pil_collage([(x + 1) for x in out_images],
                       os.path.join(config.snap_dir,
                                    f'snap_epoch_{nepoch}_iter_{niter}.png'),
                       layout)
# -----------------------------------------------------------------------------

def plot_atr_training_output(model,
                         config,
                         data,
                         nepoch=0,
                         niter=0,
                         layout=(4,2)):
    """
    ---------------------------------------------------------------------------
    Plot sample output images for GAN for a given epoch/iteration.
    Plotted images are saved as .png file using PIL and in the path given by
    config.snap_dir

    :param model:  GAN model
    :param config: config data object
    :param nepoch: epoch number
    :param niter:  iteraiton number
    :param layout: (rows, cols) layout for display
    :return:
    ---------------------------------------------------------------------------
    """
    out_images = []
    lyt = layout[0]*layout[1]
    layout[1] *= 3

    ctr = 0

    for i, cdata in enumerate(data):

        model.set_input(cdata)
        out_im = model.test()

        b = cdata['ct'].shape[0]

        for k in range(b):
            if k>=(lyt%b): ctr=1; break
            out_images.append(
                torch.squeeze(cdata['ct'][k,:,:,:]).cpu().numpy()*0.5+0.5)
            out_images.append(torch.squeeze(cdata['gt'][k,:,:,:]).cpu().numpy())
            out_images.append(torch.squeeze(out_im[k,:,:]).cpu().numpy())
        if ctr==1: break

    # print(len(out_images), layout)

    create_pil_collage([x for x in out_images],
                       os.path.join(config.snap_dir,
                                    f'snap_epoch_{nepoch}_iter_{niter}.png'),
                       layout)
# -----------------------------------------------------------------------------


def plot_training_loss(loss_dict,
                       nepoch,
                       epoch_ticks,
                       niter,
                       out_file):
    """
    ---------------------------------------------------------------------------
    Plot and Save Training Loss curves for GAN

    :param loss_dict:       Dictionary containing loss curve values
    :param nepoch:          Current epoch
    :param epoch_ticks:     Epoch Ticks to display on Plot
    :param niter:           No. of iters/epoch for which loss is recorded
    :param out_file:        Path to save the plot
    :return:
    ---------------------------------------------------------------------------
    """
    plt.figure()
    plt.title("GAN Training Loss")

    for lname, lcurve in loss_dict.items():
        plt.plot(lcurve, label=lname)
        l = lcurve

    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Losses')

    if niter > 0:
        plt.xticks(np.arange(0, len(l),
                          epoch_ticks * niter),
                   np.arange(0, nepoch, epoch_ticks))
    else:
        plt.xticks(np.arange(0, len(l), len(l)),
                   np.arange(0, nepoch, nepoch))

    plt.savefig(out_file)
    plt.close()
# -----------------------------------------------------------------------------


def plot_image(ax,
               im=None,
               title=None,
               cmap='gray',
               vmin=None,
               vmax=None):
    """
    ---------------------------------------------------------------------------

    :param ax:
    :param im:
    :param title:
    :param cmap:
    :param colorbar:
    :return:
    ---------------------------------------------------------------------------
    """

    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)

    im_ax = None

    if im is not None: im_ax = ax.imshow(im,
                                         cmap=cmap,
                                         vmin=im.min() if vmin is None else vmin,
                                         vmax=im.max() if vmax is None else vmax,
                                         )
    if title is not None: ax.set_title(title, fontsize=6)
    if colorbar: ax.colorbar()
    return im_ax
# -----------------------------------------------------------------------------


def plot_multiple_images(im_list,
                         layout,
                         titles=None,
                         cmap='gray',
                         vmin=None,
                         vmax=None,
                         suptitle=None,
                         save_as=None
                         ):
    """
    ---------------------------------------------------------------------------


    :param im_list:
    :param layout:
    :param titles:
    :param cmap:
    :param vmin:
    :param vmax:
    :param suptitle:
    :param save_as:
    :return:
    ---------------------------------------------------------------------------
    """

    assert len(im_list)==layout[0]*layout[1]
    assert titles is None or len(im_list)==layout[0]*layout[1]

    fig, ax = plt.subplots(nrows=layout[0], ncols=layout[1])

    for i in range(layout[0]):
        for j in range(layout[1]):
            _ = plot_image(ax[i][j],
                           im_list[i+j*layout[0]],
                           cmap=cmap,
                           vmin=vmin,
                           vmax=vmax
                           )

    if suptitle is not None: fig.suptitle(suptitle)
    if save_as is not None: plt.savefig(save_as)

    return fig, ax

# -----------------------------------------------------------------------------


def get_latest_saved_checkpoint(config):

    if not os.path.exists(config.checkpoint_dir): return None, False

    chk_pts_list = os.listdir(config.checkpoint_dir)

    if not chk_pts_list: return None, False

    chk_pts = [int(x.replace('.pth', '').split('_')[3])
               for x in chk_pts_list].sort()

    return chk_pts[-1], True
# -----------------------------------------------------------------------------
