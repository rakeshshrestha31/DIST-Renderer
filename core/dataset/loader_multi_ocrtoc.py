'''
This file is partly adapted from the original PMO repository
[*] https://github.com/chenhsuanlin/photometric-mesh-optim
'''
import numpy as np
import os, sys
import copy
import json
import torch
import trimesh
import open3d as o3d
import torch.utils.data
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.geometry import Camera

class LoaderMultiOCRTOC(torch.utils.data.Dataset):
    def __init__(self, data_dir, class_num, scale=1, num_points=10000, focal=None):
        self.class_num = class_num
        self.data_dir = data_dir
        self.seq_list = [
            ## chairs
            # '1b30b6c7-b465-49d8-87e6-dd2314e53ad2',
            # 'e5fc4a48-5120-48c7-9f72-b59f53a5c34e',
            '70865842-b9d4-4b18-96b0-0cb8776b6f71'
            ## sofas
            # '55f7c741-d263-4049-bb8a-168d9eea1c77'
            # '779cc50f-6bfe-41d9-8c27-7671bf77e450'
        ]

        self.scale = scale
        self.num_points = num_points
        self.focal = focal

    def __len__(self):
        return len(self.seq_list)

    def sample_points_from_ply(self, num_points_all, ply_fname):
        pcd = o3d.io.read_point_cloud(ply_fname)
        points = np.asarray(pcd.points, dtype=np.float32)
        random_indices = np.random.choice(
            range(points.shape[0]), num_points_all, replace=False
        )
        return points[random_indices, :]

    def resize_image(self, image, scale):
        h, w = image.shape[:2]
        new = np.zeros(image.shape)
        ns_h, ns_w = int(h*scale), int(w*scale)
        if scale < 1:
            new[int(h/2 -ns_h/2):int(h/2 + ns_h/2), int(w/2-ns_w/2):int(w/2 + ns_w/2)] = cv2.resize(image, (ns_h, ns_w))
        else:
            new_img = cv2.resize(image, (ns_h, ns_w))
            h_new, w_new = new_img.shape[:2]
            new = new_img[int(h_new/2 - h/2):int(h_new/2 + h/2), int(w_new/2 - w/2):int(w_new/2 + w/2)]
        return new

    def __getitem__(self, idx):
        instance_name = self.seq_list[idx]

        cam = np.load(
            os.path.join(self.data_dir, instance_name, 'dist_camera_data.npz'),
            allow_pickle=True
        )

        # rgb_dir = os.path.join(self.data_dir, instance_name, 'segmented_color')
        rgb_dir = os.path.join(self.data_dir, instance_name, 'rgb_undistort')
        mask_dir = os.path.join(self.data_dir, instance_name, 'mask')

        img_list = []
        mask_list = []
        camera_list = []

        for img_idx, (cam_id, extr) in enumerate(cam['extr'].item().items()):
            if not 0 <= cam_id <= 53:
                continue

            # rgba = cv2.imread(
            #     os.path.join(rgb_dir, f'segmented_color_{cam_id:03}.png'),
            #     cv2.IMREAD_UNCHANGED
            # ).astype(np.float32) / 255.0

            # img_cur = rgba[..., :3]
            # mask_cur = rgba[..., 3]

            img_cur = cv2.imread(
                os.path.join(rgb_dir, f'color_{cam_id:03}.png')
            ).astype(np.float32) / 255.0

            mask_cur = cv2.imread(
                os.path.join(mask_dir, f'mask_{cam_id:03}.png'),
                cv2.IMREAD_UNCHANGED
            ).astype(np.float32) / 255.0

            cam_cur = Camera(cam['intr'], extr)
            if self.focal is not None:
                img_cur = self.resize_image(img_cur, self.focal)
                mask_cur = self.resize_image(mask_cur.astype(np.float), self.focal)
                mask_cur[mask_cur<1] = 0
                mask_cur = mask_cur.astype(np.bool)
                cam_cur.intrinsic[0, 0] = cam_cur.intrinsic[0, 0]*self.focal
                cam_cur.intrinsic[1, 1] = cam_cur.intrinsic[1, 1]*self.focal

            if self.scale != 1:
                mask_cur = cv2.resize(mask_cur.astype(np.float), None, fx=self.scale, fy=self.scale)
                mask_cur[mask_cur<1] = 0
                mask_cur = mask_cur.astype(np.bool)
                img_cur = cv2.resize(img_cur, None, fx=self.scale, fy=self.scale)
                cam_cur.intrinsic[:2] = cam_cur.intrinsic[:2] * self.scale
                cam_cur.intrinsic[0, 2] = img_cur.shape[1] / 2.0
                cam_cur.intrinsic[1, 2] = img_cur.shape[0] / 2.0

            img_list.append(torch.from_numpy(img_cur).float())
            mask_list.append(torch.from_numpy(mask_cur).type(torch.uint8).cuda())
            camera_list.append(cam_cur)

        # get gt point cloud
        ply_fname = os.path.join(
            self.data_dir, instance_name, 'gt_labels_dist.ply'
        )
        points_gt = self.sample_points_from_ply(self.num_points, ply_fname)

        return instance_name, img_list, mask_list, camera_list, points_gt
