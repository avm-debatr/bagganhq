"""
pidray_data_loader.py: loads data from the PIDRay for
                       2D BagGAN Training / Testing
"""

__author__    = "Ankit Manerikar"
__copyright__ = "Copyright (C) 2022, Purdue University"
__date__      = "March 3rd, 2022"
__credits__   = ["Ankit Manerikar"]
__license__   = "Public Domain"
__version__   = "1.0"
__maintainer__= ["Ankit Manerikar"]
__email__     = ["amanerik@purdue.edu"]
__status__    = "Prototype"
# -----------------------------------------------------------------------------

"""
-------------------------------------------------------------------------------
PIDRay Dataset Loader

Used for training the BagGAN network.

You need to implement the following functions:
    -- <modify_commandline_options>:　Add data-specific options and rewrite 
       default values for existing options.
    -- <__init__>: Initialize this data class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
-------------------------------------------------------------------------------
"""

import torch.utils.data as data
from pycocotools.coco import COCO
from lib.util.util import *
from PIL import Image
import torchvision.transforms as transforms

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

SAMPLES_TO_BE_DISCARDED = load(PIDRAY_SAMPLE_FILE)
DEFAULT_PIDRAY_ARGS = dict(dims=(3, 448, 800),
                           num_channels=3,
                           range=(0,255))


class PIDRayDataLoader(data.Dataset):
    """
    ---------------------------------------------------------------------------
    Dataset for PIDRay
    ---------------------------------------------------------------------------
    """

    def __init__(self,
                 ds_dir,
                 subset,
                 is_train=True,
                 sample_list=None,
                 mode='recon',
                 image_size=None,
                 pidray_args=None,
                 labeled=False
                 ):
        """
        -----------------------------------------------------------------------
        :param ds_dir:            baggage dataset directory
        :param ds_handler:        dataset handler
        :param baggan_train_args: parameters for baggan training
        :param is_train:          training or testing
        :param sample_list:       list of samples to be loaded
        -----------------------------------------------------------------------
        """

        self.ds_dir = ds_dir

        self.batch_dir = os.path.join(self.ds_dir, subset)

        if subset=='train':
            self.ann_path = os.path.join(self.ds_dir,
                                         'annotations',
                                         f'xray_{subset}.json')
        else:
            self.ann_path = os.path.join(self.ds_dir,
                                         'annotations',
                                         f'xray_test_{subset}.json')

        self.annotations = COCO(self.ann_path)

        sample_files = self.annotations.getImgIds()

        sample_files = [s for s in sample_files
                        if s not in SAMPLES_TO_BE_DISCARDED[subset]]

        if is_train:
            self.train_samples = sample_files
        else:
            self.test_samples  = sample_files

        if sample_list is not None and is_train:
            if isinstance(sample_list, list):
                self.train_samples = sample_list
            if isinstance(sample_list, int):
                self.train_samples = self.train_samples[:sample_list]

        if sample_list is not None and (not is_train):
            if isinstance(sample_list, list):
                self.test_samples = sample_list
            elif isinstance(sample_list, int):
                self.test_samples = self.test_samples[:sample_list]

        self.is_train = is_train
        self.mode = mode
        self.image_resize = image_size
        if pidray_args is None:
            self.pidray_args = DEFAULT_PIDRAY_ARGS
        else:
            self.pidray_args = pidray_args

        tx = [transforms.ToTensor(),]

        self.tx_aug = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(degrees=(0, 180),
                                                                    fill=255)
                                    ])

        self.im_transforms = transforms.Compose(tx)
        self.image_thresh  = 512
        self.labeled = labeled
    # -------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        -----------------------------------------------------------------------
        Return a data point and its metadata information.

        :param    index -- a random integer for data indexing

        :return   dictionary of input, original, reference image patches
                  and their metadata
        -----------------------------------------------------------------------
        """

        fpath = self.annotations.loadImgs(self.train_samples[index]
                                          if self.is_train
                                          else self.test_samples[index])[0]['file_name']
        fpath = os.path.join(self.batch_dir, fpath)

        im_sample = Image.open(fpath)
        im_sample = self.im_transforms(im_sample)
        im_sample = (im_sample       - im_sample.min()) / \
                    (im_sample.max() - im_sample.min())

        im_sample = 2*im_sample - 1

        if self.labeled:

            ann = self.annotations.anns[self.train_samples[index]
                                        if self.is_train
                                        else self.test_samples[index]]

            im_mask   = self.annotations.annToMask(ann)
            im_mask   = self.im_transforms(im_mask)

            if im_mask.shape[1]>self.pidray_args['dims'][1]:
                im_mask = im_mask[:,:self.pidray_args['dims'][1], :]

            im_mask   = 2*im_mask - 1

            if im_mask.shape!=im_sample.shape:

                if im_mask.shape[-1]<im_sample.shape[-1]:
                    m_init = (-1)*torch.ones(1, im_sample.shape[1], im_sample.shape[2])
                    m_init[:,:im_mask.shape[-2],:im_mask.shape[-1]] = im_mask
                    im_mask = m_init
                else:
                    m_init = (1)*torch.ones(3, im_mask.shape[1], im_mask.shape[2])
                    m_init[:,:,:im_sample.shape[-1]] = im_sample
                    im_sample = m_init

        im_w = im_sample.shape[-1]
        im_dims = im_sample.shape

        if im_w > self.pidray_args['dims'][2]:
            im_sample = im_sample[:,
                                  :self.pidray_args['dims'][1],
                                  :self.pidray_args['dims'][2]]
            if self.labeled:
                im_mask = im_mask[:,
                                  :self.pidray_args['dims'][1],
                                  :self.pidray_args['dims'][2]]

        elif im_w < self.image_thresh:
            im_init = (1)*torch.ones(3,
                                      self.image_thresh,
                                      self.image_thresh)
            im_init[:,
                    - im_dims[1]//2 + self.image_thresh//2:
                    + im_dims[1]//2 + self.image_thresh//2,
                    - im_dims[2]//2 + self.image_thresh//2:
                    + im_dims[2]//2 + self.image_thresh//2] \
                = im_sample
            im_sample = im_init

            if self.labeled:
                mask_init = (-1) * torch.ones(1,
                                             self.image_thresh,
                                             self.image_thresh)
                mask_init[:,
                - im_dims[1] // 2 + self.image_thresh // 2:
                + im_dims[1] // 2 + self.image_thresh // 2,
                - im_dims[2] // 2 + self.image_thresh // 2:
                + im_dims[2] // 2 + self.image_thresh // 2] \
                    = im_mask
                im_mask = mask_init

        elif self.image_thresh < im_w <= self.pidray_args['dims'][2]:
            im_init = (1)*torch.ones(3,
                                     self.pidray_args['dims'][2],
                                     self.pidray_args['dims'][2])
            im_init[:,
                    - im_dims[1]//2 + self.pidray_args['dims'][2]//2:
                    + im_dims[1]//2 + self.pidray_args['dims'][2]//2,
                    - im_dims[2]//2 + self.pidray_args['dims'][2]//2:
                    + im_dims[2]//2 + self.pidray_args['dims'][2]//2] \
                = im_sample
            im_sample = im_init

            if self.labeled:
                mask_init = (-1) * torch.ones(1,
                                           self.pidray_args['dims'][2],
                                           self.pidray_args['dims'][2])
                mask_init[:,
                - im_dims[1] // 2 + self.pidray_args['dims'][2] // 2:
                + im_dims[1] // 2 + self.pidray_args['dims'][2] // 2,
                - im_dims[2] // 2 + self.pidray_args['dims'][2] // 2:
                + im_dims[2] // 2 + self.pidray_args['dims'][2] // 2] \
                    = im_mask
                im_mask = mask_init

        else:
            im_init = (1)*torch.ones(self.pidray_args['dims'])
            im_init[:,
                    - im_dims[1]//2 + self.pidray_args['dims'][2]//2:
                    + im_dims[1]//2 + self.pidray_args['dims'][2]//2,
                    - im_dims[2]//2 + self.pidray_args['dims'][2]//2:
                    + im_dims[2]//2 + self.pidray_args['dims'][2]//2] \
                = im_sample

            im_sample = im_init

            if self.labeled:
                mask_init = (-1) * torch.ones(1,
                                              self.pidray_args['dims'][1],
                                              self.pidray_args['dims'][2])
                mask_init[:,
                - im_dims[1] // 2 + self.pidray_args['dims'][2] // 2:
                + im_dims[1] // 2 + self.pidray_args['dims'][2] // 2,
                - im_dims[2] // 2 + self.pidray_args['dims'][2] // 2:
                + im_dims[2] // 2 + self.pidray_args['dims'][2] // 2] \
                    = im_sample
                im_mask = mask_init

        im_sample = transforms.Resize((self.image_resize,
                                       self.image_resize))(im_sample)

        cdata = dict(ct=im_sample,
                     dims=im_dims,
                     index=index,
                     fname=fpath)

        if self.labeled:
            im_mask = transforms.Resize((self.image_resize,
                                         self.image_resize),
                                        interpolation=Image.NEAREST)(im_mask)

            nchls = im_sample.shape[1]
            aug_im = self.tx_aug(torch.cat((im_sample, im_mask), 1))
            cdata['ct'], cdata['mask'] = aug_im[:, :nchls, :, :], \
                                         aug_im[:, nchls:, :, :]
        else:
            nchls = cdata['ct'].shape[0]
            f_im = self.tx_aug(torch.cat((cdata['ct'],
                                                 torch.ones_like(cdata['ct'])),
                                                0))
            m = 1-f_im[nchls:,:,:]
            cdata['ct'] = f_im[:nchls,:,:]+m

        return cdata
    # -------------------------------------------------------------------------

    def __len__(self):
        """
        -----------------------------------------------------------------------
        Return the total number of images.

        -----------------------------------------------------------------------
        """
        if self.is_train:
            return len(self.train_samples)
        else:
            return len(self.test_samples)
    # -------------------------------------------------------------------------


if __name__=="__main__":

    ds = PIDRayDataLoader(ds_dir=os.path.join(DATA_DIR, 'pidray'),
                          subset='train',
                          image_size=512,
                          labeled=True)

    for d in ds:
        break

