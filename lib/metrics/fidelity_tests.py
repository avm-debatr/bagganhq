from lib.__init__ import *

import torch.nn as nn
import os
import numpy as np
import pickle
import lib.metrics.inception_score as inception_score
import lib.metrics.fid as fid


INCEPTION_MODEL = os.path.join(INC_DIR, 'inception_pidray.pkl')


def calculate_inception_score(model,
                              data_loader,
                              num_samples=10000):
    """
    ---------------------------------------------------------------------------
    inception score for baggan model
    :param model: BagGAN model
    :param data_loader: data loader for dataset used for training
    :param num_samples: for calculating inception score
    :return:
    ---------------------------------------------------------------------------
    """

    inception = inception_score.load_patched_inception_v3()
    inception = nn.DataParallel(inception).eval().to(model.device)

    features = inception_score.extract_features(data_loader,
                                                inception,
                                                model.device,
                                                n_samples=num_samples).numpy()

    features = features[:num_samples]

    mean = np.mean(features, 0)
    cov = np.cov(features, rowvar=False)

    return mean, cov
# -----------------------------------------------------------------------------


def calculate_fid_score(model,
                        num_samples=10000,
                        truncation=0.5
                        ):
    """
    ---------------------------------------------------------------------------
    Frechet Inception Distance

    :param model: BagGAN model
    :param num_samples: for calculating FID score
    :param truncation:  fr sampling images from model
    :return:
    ---------------------------------------------------------------------------
    """

    inception = nn.DataParallel(inception_score.load_patched_inception_v3()).to(model.device)
    inception.eval()
    features = fid.extract_feature_from_samples(model,
                                                inception,
                                                truncation,
                                                None,
                                                64,
                                                num_samples,
                                                model.device)

    features = features.numpy()
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(INCEPTION_MODEL, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    fid_score = fid.calc_fid(sample_mean,
                             sample_cov,
                             real_mean,
                             real_cov)

    return fid_score
# -----------------------------------------------------------------------------
