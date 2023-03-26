from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy import linalg

from lib.util.helper import *
from lib.gan.tests import *

# from model import Generator
from lib.metrics.inception_score import load_patched_inception_v3

from lib.datasets.pidray_data_loader import PIDRayDataLoader
from scipy.linalg import sqrtm


@torch.no_grad()
def extract_feature_from_samples(model,
                                 inception,
                                 truncation,
                                 truncation_latent,
                                 batch_size,
                                 n_sample,
                                 device):
    """
    ---------------------------------------------------------------------------
    :param generator:
    :param inception:
    :param truncation:
    :param truncation_latent:
    :param batch_size:
    :param n_sample:
    :param device:
    :return:
    ---------------------------------------------------------------------------
    """

    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        model.set_input(latent=[latent],
                        disentangled=False)

        img = model.test()

        # img, _ = g([latent], truncation=truncation,
        #            truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features
# -----------------------------------------------------------------------------


def calc_fid(sample_mean,
             sample_cov,
             real_mean,
             real_cov,
             eps=1e-6):
    """
    ---------------------------------------------------------------------------
    :param sample_mean:
    :param sample_cov:
    :param real_mean:
    :param real_cov:
    :param eps:
    :return:
    ---------------------------------------------------------------------------
    """

    cov_sqrt, _ = sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) \
            + np.trace(real_cov) \
            - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid
# -----------------------------------------------------------------------------
