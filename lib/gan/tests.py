from lib.util.visualization import *


def plot_stylegan_training_samples(model,
                                   latents,
                                   fpath,
                                   is_labeled=True,
                                   gen_args=None,
                                   latent_code=None,
                                   layout=(8,8)):
    """
    ---------------------------------------------------------------------------
    Plot sample output images for GAN for a given epoch/iteration.
    Plotted images are saved as .png file using PIL and in the path given by
    config.snap_dir

    :param model:  GAN model
    :param config: config data object
    :param nepoch: epoch number
    :param niter:  iteration number
    :param layout: (rows, cols) layout for display
    :return:
    ---------------------------------------------------------------------------
    """

    if is_labeled:
        lyt = layout[0] * layout[1]*2

        try:
            model.set_input(latent=latents,
                            disentangled=False,
                            gen_args=gen_args,
                            latent_code=latent_code)
        except:
            model.set_input(latent=latents,
                            disentangled=False,
                            gen_args=gen_args)

        out_ims = model.test()
        out_ims = torch.squeeze(out_ims).cpu().numpy()
        b = out_ims.shape[0]

        ims = []

        for i in range(b):
            ims.append(clip(out_ims[i, 0, :, :], -1, 1))
            ims.append(clip(out_ims[i, 1, :, :], -1, 1))

        assert out_ims.shape[0] == lyt//2

        create_pil_collage([i+1.0 for i in ims],
                           os.path.join(fpath),
                           (layout[0], layout[0]*2))
    else:
        lyt = layout[0]*layout[1]
        model.set_input(latent=latents,
                        disentangled=False,
                        gen_args=gen_args)

        out_ims = model.test()
        out_ims = torch.squeeze(out_ims).cpu().numpy()
        b = out_ims.shape[0]

        assert out_ims.shape[0]==lyt

        create_pil_collage([out_ims[i,:,:]+1.0
                            for i in range(b)],
                           os.path.join(fpath),
                           layout)
# -----------------------------------------------------------------------------


def plot_stylegan_rgb_samples(model,
                              latents,
                              fpath,
                              is_labeled=True,
                              gen_args=None,
                              latent_code=None,
                              layout=(8,8)):
    """
    ---------------------------------------------------------------------------
    Plot sample output images for GAN for a given epoch/iteration.
    Plotted images are saved as .png file using PIL and in the path given by
    config.snap_dir

    :param model:  GAN model
    :param config: config data object
    :param nepoch: epoch number
    :param niter:  iteration number
    :param layout: (rows, cols) layout for display
    :return:
    ---------------------------------------------------------------------------
    """

    if is_labeled:
        lyt = layout[0] * layout[1]*2

        try:
            model.set_input(latent=latents,
                            disentangled=False,
                            gen_args=gen_args,
                            latent_code=latent_code)
        except:
            model.set_input(latent=latents,
                            disentangled=False,
                            gen_args=gen_args)

        out_ims = model.test()
        out_ims = torch.squeeze(out_ims).cpu().numpy()
        b = out_ims.shape[0]

        ims = []

        for i in range(b):
            ims.append(clip(out_ims[i, 0:3, :, :], -1, 1))
            ims.append(clip(out_ims[i, 3:4, :, :], -1, 1).repeat(repeats=3,
                                                               axis=0))

        assert out_ims.shape[0] == lyt//2

        # for o in ims: print(o.shape)

        create_pil_collage([i+1.0 for i in ims],
                           os.path.join(fpath),
                           (layout[0], layout[0]*2),
                           vlims=[0, 2])

    else:
        lyt = layout[0]*layout[1]
        model.set_input(latent=latents,
                        disentangled=False,
                        gen_args=gen_args)

        out_ims = model.test()
        out_ims = torch.squeeze(out_ims).cpu().numpy()
        b = out_ims.shape[0]

        assert out_ims.shape[0]==lyt

        create_pil_collage([out_ims[i,:,:,:]+1.0
                            for i in range(b)],
                           os.path.join(fpath),
                           layout,
                           vlims=[0,2])
# -----------------------------------------------------------------------------

