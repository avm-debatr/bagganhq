import os, sys, time

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
CHECKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')
DATA_DIR      = os.path.join(ROOT_DIR, 'data')
INC_DIR  = os.path.join(ROOT_DIR, 'include')
LIB_DIR  = os.path.join(ROOT_DIR, 'lib')
MODEL_DIR       = os.path.join(ROOT_DIR, 'models')
PIDRAY_DIR    = os.path.join(DATA_DIR, 'pidray')
PIDRAY_SAMPLE_FILE = os.path.join(INC_DIR, 'samples', 'pidray_samples_v1.npz')

