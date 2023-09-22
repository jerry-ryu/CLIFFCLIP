"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = "/mnt/RG/data/h36m/images"
LSP_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/lsp"
LSP_ORIGINAL_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/lsp_ori"
LSPET_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/hr-lspet/hr-lspet"
MPII_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/mpii"
COCO_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/coco"
MPI_INF_3DHP_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/mpi_inf_3dhp/tmp/mpi_inf_3dhp"
PW3D_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/3DPW"
UPI_S1H_ROOT = "/mnt/RG/data/data_ROMP/ROMP_datasets/UPi-S1h"

# Output folder to save test/train npz files
DATASET_NPZ_PATH = "/mnt/RG/data/data_SPIN/data/dataset_extras"

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate
# the .npz files with the annotations.
OPENPOSE_PATH = "datasets/openpose"

# Path to test/train npz files
DATASET_FILES = [
    {
        "h36m-p1": join(DATASET_NPZ_PATH, "h36m_valid_protocol1_made.npz"),
        "h36m-p2": join(DATASET_NPZ_PATH, "h36m_valid_protocol2_made.npz"),
        "lsp": join(DATASET_NPZ_PATH, "lsp_dataset_test.npz"),
        "mpi-inf-3dhp": join(DATASET_NPZ_PATH, "mpi_inf_3dhp_valid.npz"),
        "3dpw": join(DATASET_NPZ_PATH, "3dpw_test.npz"),
    },
    {
        "h36m": join(DATASET_NPZ_PATH, "h36m_train_made.npz"),
        "lsp-orig": join(DATASET_NPZ_PATH, "lsp_dataset_original_train.npz"),
        "mpii": join(DATASET_NPZ_PATH, "mpii_train.npz"),
        "coco": join(DATASET_NPZ_PATH, "coco_2014_train.npz"),
        "lspet": join(DATASET_NPZ_PATH, "hr-lspet_train.npz"),
        "mpi-inf-3dhp": join(DATASET_NPZ_PATH, "mpi_inf_3dhp_train.npz"),
    },
]

DATASET_FOLDERS = {
    "h36m": H36M_ROOT,
    "h36m-p1": H36M_ROOT,
    "h36m-p2": H36M_ROOT,
    "lsp-orig": LSP_ORIGINAL_ROOT,
    "lsp": LSP_ROOT,
    "lspet": LSPET_ROOT,
    "mpi-inf-3dhp": MPI_INF_3DHP_ROOT,
    "mpii": MPII_ROOT,
    "coco": COCO_ROOT,
    "3dpw": PW3D_ROOT,
    "upi-s1h": UPI_S1H_ROOT,
}

CUBE_PARTS_FILE = "/mnt/RG/data/data_SPIN/data/data/cube_parts.npy"
JOINT_REGRESSOR_TRAIN_EXTRA = "/mnt/RG/data/data_SPIN/data/data/J_regressor_extra.npy"
JOINT_REGRESSOR_H36M = "/mnt/RG/data/data_SPIN/data/data/J_regressor_h36m.npy"
VERTEX_TEXTURE_FILE = "/mnt/RG/data/data_SPIN/data/data/vertex_texture.npy"
STATIC_FITS_DIR = "/mnt/RG/data/data_SPIN/data/static_fits"
SMPL_MEAN_PARAMS = "/mnt/RG/CLIFFCLIP/CLIFF/data/smpl_mean_params.npz"
SMPL_MODEL_DIR = "/mnt/RG/CLIFFCLIP/CLIFF/data/smpl"
