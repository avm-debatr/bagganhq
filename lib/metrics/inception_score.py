from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import Inception3
from tqdm import tqdm

from lib.metrics.inception import InceptionV3
from lib.datasets.pidray_data_loader import PIDRayDataLoader

from lib.util.util import *


class Inception3Feature(Inception3):
    """
    ---------------------------------------------------------------------------

    ---------------------------------------------------------------------------
    """
    def forward(self, x):
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode="bilinear",
                              align_corners=True)

        x = self.Conv2d_1a_3x3(x)  # 299 x 299 x 3
        x = self.Conv2d_2a_3x3(x)  # 149 x 149 x 32
        x = self.Conv2d_2b_3x3(x)  # 147 x 147 x 32
        x = F.max_pool2d(x,
                         kernel_size=3,
                         stride=2) # 147 x 147 x 64

        x = self.Conv2d_3b_1x1(x)  # 73 x 73 x 64
        x = self.Conv2d_4a_3x3(x)  # 73 x 73 x 80
        x = F.max_pool2d(x,
                         kernel_size=3,
                         stride=2) # 71 x 71 x 192

        x = self.Mixed_5b(x)  # 35 x 35 x 192
        x = self.Mixed_5c(x)  # 35 x 35 x 256
        x = self.Mixed_5d(x)  # 35 x 35 x 288

        x = self.Mixed_6a(x)  # 35 x 35 x 288
        x = self.Mixed_6b(x)  # 17 x 17 x 768
        x = self.Mixed_6c(x)  # 17 x 17 x 768
        x = self.Mixed_6d(x)  # 17 x 17 x 768
        x = self.Mixed_6e(x)  # 17 x 17 x 768

        x = self.Mixed_7a(x)  # 17 x 17 x 768
        x = self.Mixed_7b(x)  # 8 x 8 x 1280
        x = self.Mixed_7c(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)  # 8 x 8 x 2048

        return x.view(x.shape[0], x.shape[1])  # 1 x 1 x 2048
    # -------------------------------------------------------------------------


def load_patched_inception_v3():
    inception_feat = InceptionV3([3], normalize_input=False)
    return inception_feat


@torch.no_grad()
def extract_features(loader,
                     inception,
                     device,
                     n_samples):
    pbar = tqdm(loader)

    feature_list = []

    for i, img in enumerate(pbar):
        img = img['ct'].to(device)
        feature = inception(img.unsqueeze(0))[0].view(-1)
        feature_list.append(feature.to("cpu"))
        if i>n_samples: break

    features = torch.cat(feature_list, 0)

    return features
# -----------------------------------------------------------------------------
