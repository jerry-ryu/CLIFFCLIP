import torch
import torch.nn as nn
import numpy as np
import cv2
import smplx
import torchgeometry as tgm
from models import SMPL
from common.renderer_pyrd import Renderer
import tensorboard

from datasets import MixedDataset
from models.cliff_hr48.cliff import CLIFF
from common.utils import cam_crop2full, estimate_focal_length
from utils.geometry import batch_rodrigues, perspective_projection
from common.renderer_pyrd import Renderer
from utils import BaseTrainer
from torchvision.utils import make_grid
import torchvision

import config
import constants
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.yolov3_detector import HumanDetector
from common.mocap_dataset import MocapDataset
from lib.yolov3_dataset import DetectionDataset


class Trainer(BaseTrainer):
    def init_fn(self):
        self.train_ds = MixedDataset(
            self.options, ignore_3d=self.options.ignore_3d, is_train=True
        )

        self.model = CLIFF(config.SMPL_MEAN_PARAMS)
        self.model = torch.nn.DataParallel(self.model).cuda()

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.options.lr, weight_decay=0
        )
        self.smpl = SMPL(
            config.SMPL_MODEL_DIR,
            batch_size=self.options.batch_size,
            create_transl=False,
        ).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction="none").to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {"model": self.model}
        self.optimizers_dict = {"optimizer": self.optimizer}

        # ramdomly select val/train dataset to visualize during training
        img_path_list = [
            "/mnt/RG/data/h36m/images/s_09_act_09_subact_01_ca_02/s_09_act_09_subact_01_ca_02_000911.jpg",
            "/mnt/RG/data/h36m/images/s_09_act_03_subact_01_ca_04/s_09_act_03_subact_01_ca_04_005536.jpg",
            "/mnt/RG/data/h36m/images/s_09_act_15_subact_01_ca_04/s_09_act_15_subact_01_ca_04_001921.jpg",
            "/mnt/RG/data/h36m/images/s_09_act_06_subact_01_ca_04/s_09_act_06_subact_01_ca_04_003711.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/lsp/images/im1099.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/lsp/images/im1228.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/mpi_inf_3dhp/tmp/mpi_inf_3dhp/mpi_inf_3dhp_test_set/TS2/imageSequence/img_002791.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/mpi_inf_3dhp/tmp/mpi_inf_3dhp/mpi_inf_3dhp_test_set/TS5/imageSequence/img_000110.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/3DPW/imageFiles/outdoors_fencing_01/image_00358.jpg",
            "/mnt/RG/data/h36m/images/s_07_act_04_subact_01_ca_03/s_07_act_04_subact_01_ca_03_003316.jpg",
            "/mnt/RG/data/h36m/images/s_06_act_08_subact_02_ca_04/s_06_act_08_subact_02_ca_04_000591.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/mpii/images/076364863.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/hr-lspet/hr-lspet/im03371.png",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/hr-lspet/hr-lspet/im06861.png",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/mpi_inf_3dhp/tmp/mpi_inf_3dhp/S7/Seq2/imageFrames/video_1/frame_003029.jpg",
            "/mnt/RG/data/data_ROMP/ROMP_datasets/mpi_inf_3dhp/tmp/mpi_inf_3dhp/S3/Seq2/imageFrames/video_4/frame_006270.jpg",
        ]
        # for mode in config.DATASET_FILES:
        #     for ds_name, path in mode.items():
        #         data = np.load(path, mmap_mode="r", allow_pickle=True)
        #         samplelist = np.random.choice(data["imgname"], size=2, replace=False)
        #         for i in samplelist:
        #             img_path_list.append(
        #                 os.path.join(config.DATASET_FOLDERS[ds_name], i)
        #             )

        # print(img_path_list)

        self.orig_img_bgr_all = [cv2.imread(img_path) for img_path in img_path_list]
        self.Humandetector = HumanDetector()

    def keypoint_loss(
        self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight
    ):
        """Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (
            conf
            * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
        ).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (
                conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)
            ).mean()
        else:
            return torch.FloatTensor(1).fill_(0.0).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(
                pred_vertices_with_shape, gt_vertices_with_shape
            )
        else:
            return torch.FloatTensor(1).fill_(0.0).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1, 3)).view(-1, 24, 3, 3)[
            has_smpl == 1
        ]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.0).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.0).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        self.model.train()
        # Get data from the batch
        images = input_batch["img"]  # input image
        gt_keypoints_2d = input_batch["keypoints"]  # 2D keypoints
        gt_pose = input_batch["pose"]  # SMPL pose parameters
        gt_betas = input_batch["betas"]  # SMPL beta parameters
        gt_joints = input_batch["pose_3d"]  # 3D pose
        has_smpl = input_batch[
            "has_smpl"
        ].bool()  # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch[
            "has_pose_3d"
        ].bool()  # flag that indicates whether 3D pose is valid
        scale = input_batch["scale"].float()
        center = input_batch["center"]
        img_h = input_batch["img_h"]
        img_w = input_batch["img_w"]
        focal_length = input_batch["focal_length"]

        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(
            betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3]
        )
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = (
            0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)
        )

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1)

        # The constants below are used for normalization, and calculated from H36M data.
        # It should be fine if you use the plain Equation (5) in the paper.
        bbox_info[:, :2] = (
            bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
        )  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (
            0.06 * focal_length
        )  # [-1, 1]

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_cam_crop = self.model(images, bbox_info)

        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)

        pred_cam_full = cam_crop2full(
            pred_cam_crop, center, scale, full_img_shape, focal_length
        )

        pred_output = self.smpl(
            betas=pred_betas,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, [0]],
            pose2rot=False,
            transl=pred_cam_full,
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(
            pred_joints,
            rotation=torch.eye(3, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1),
            translation=pred_cam_full,
            focal_length=focal_length,
            camera_center=camera_center,
        )
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.0)

        # pred_keypoints_2d[:, :, 0] = pred_keypoints_2d[:, :, 0] / (img_w / 2.0).reshape(
        #     -1, 1
        # )
        # pred_keypoints_2d[:, :, 1] = pred_keypoints_2d[:, :, 1] / (img_h / 2.0).reshape(
        #     -1, 1
        # )

        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(
            pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl
        )

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(
            pred_keypoints_2d,
            gt_keypoints_2d,
            self.options.openpose_train_weight,
            self.options.gt_train_weight,
        )

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, gt_vertices, has_smpl)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = (
            self.options.shape_loss_weight * loss_shape
            + self.options.keypoint_loss_weight * loss_keypoints
            + self.options.keypoint_loss_weight * loss_keypoints_3d
            + self.options.pose_loss_weight * loss_regr_pose
            + self.options.beta_loss_weight * loss_regr_betas
            + ((torch.exp(-pred_cam_crop[:, 0] * 10)) ** 2).mean()
        )
        loss *= 60

        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {
            "pred_vertices": pred_vertices.detach(),
            "pred_cam_full": pred_cam_full.detach(),
            "focal_length": focal_length.detach(),
        }
        losses = {
            "loss": loss.detach().item(),
            "loss_keypoints": loss_keypoints.detach().item(),
            "loss_keypoints_3d": loss_keypoints_3d.detach().item(),
            "loss_regr_pose": loss_regr_pose.detach().item(),
            "loss_regr_betas": loss_regr_betas.detach().item(),
            "loss_shape": loss_shape.detach().item(),
        }

        return output, losses

    def train_summaries(self, input_batch, output, losses):
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)

        device = self.device
        orig_img_bgr_all = self.orig_img_bgr_all

        human_detector = self.Humandetector
        det_batch_size = len(orig_img_bgr_all)
        detection_dataset = DetectionDataset(orig_img_bgr_all, human_detector.in_dim)
        detection_data_loader = DataLoader(
            detection_dataset, batch_size=det_batch_size, num_workers=0
        )
        detection_all = []
        for batch_idx, batch in enumerate(tqdm(detection_data_loader)):
            norm_img = batch["norm_img"].to(device).float()
            dim = batch["dim"].to(device).float()

            detection_result = human_detector.detect_batch(norm_img, dim)
            detection_result[:, 0] += batch_idx * det_batch_size
            detection_all.extend(detection_result.cpu().numpy())
        detection_all = np.array(detection_all)

        self.model.eval()

        smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)

        pred_vert_arr = []

        mocap_db = MocapDataset(orig_img_bgr_all, detection_all)
        mocap_data_loader = DataLoader(
            mocap_db, batch_size=len(detection_all), num_workers=0
        )

        for batch in tqdm(mocap_data_loader):
            norm_img = batch["norm_img"].to(device).float()
            center = batch["center"].to(device).float()
            scale = batch["scale"].to(device).float()
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()
            focal_length = batch["focal_length"].to(device).float()

            cx, cy, b = center[:, 0], center[:, 1], scale * 200
            bbox_info = torch.stack([cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1)
            # The constants below are used for normalization, and calculated from H36M data.
            # It should be fine if you use the plain Equation (5) in the paper.
            bbox_info[:, :2] = (
                bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
            )  # [-1, 1]
            bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (
                0.06 * focal_length
            )  # [-1, 1]

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_cam_crop = self.model(norm_img, bbox_info)

            # convert the camera parameters from the crop camera to the full camera
            full_img_shape = torch.stack((img_h, img_w), dim=-1)
            pred_cam_full = cam_crop2full(
                pred_cam_crop, center, scale, full_img_shape, focal_length
            )

            pred_output = smpl_model(
                betas=pred_betas,
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, [0]],
                pose2rot=False,
                transl=pred_cam_full,
            )
            pred_vertices = pred_output.vertices
            pred_vert_arr.extend(pred_vertices.cpu().numpy())

        pred_vert_arr = np.array(pred_vert_arr)

        rend_imgs = []
        for img_idx, orig_img_bgr in enumerate(orig_img_bgr_all):
            # for img_idx, orig_img_bgr in enumerate(tqdm(orig_img_bgr_all)):
            chosen_mask = detection_all[:, 0] == img_idx
            chosen_vert_arr = pred_vert_arr[chosen_mask]

            # setup renderer for visualization
            img_h, img_w, _ = orig_img_bgr.shape
            focal_length = estimate_focal_length(img_h, img_w)
            renderer = Renderer(
                focal_length=focal_length,
                img_w=img_w,
                img_h=img_h,
                faces=smpl_model.faces,
                same_mesh_color=False,
            )

            front_view = renderer.render_front_view(
                chosen_vert_arr, bg_img_rgb=orig_img_bgr[:, :, ::-1].copy()
            )
            renderer.delete()
            front_view = torch.from_numpy(
                np.transpose(front_view[:, :, :], (2, 0, 1)).copy()
            )

            front_view = torchvision.transforms.Resize((224, 224))(front_view)
            orig_img_bgr = torch.from_numpy(
                np.transpose(orig_img_bgr[:, :, ::-1], (2, 0, 1)).copy()
            )
            orig_img_bgr = torchvision.transforms.Resize((224, 224))(orig_img_bgr)
            rend_imgs.append(front_view)
            rend_imgs.append(orig_img_bgr)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        self.summary_writer.add_image("pred_shape", rend_imgs, self.step_count)
