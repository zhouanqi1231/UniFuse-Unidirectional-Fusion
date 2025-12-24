import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import torch

def mkdirs(path):
    try:
        os.makedirs(path)
    except:
        pass

class Saver(object):

    def __init__(self, save_dir):
        self.idx = 0
        self.save_dir = os.path.join(save_dir, "results")
        if not os.path.exists(self.save_dir):
            mkdirs(self.save_dir)

    def save_as_point_cloud(self, depth, rgb, path, mask=None):
        if torch.is_tensor(depth): depth = depth.detach().cpu().numpy()
        if torch.is_tensor(rgb): rgb = rgb.detach().cpu().numpy()

        depth = np.squeeze(depth) 
        
        if depth.ndim == 3:
            if depth.shape[2] == 3: # H, W, C
                depth = depth[:, :, 0]
            elif depth.shape[0] == 3: # C, H, W
                depth = depth[0, :, :]
        
        if depth.ndim != 2:
            shape = sorted(depth.shape, reverse=True)
            h, w = shape[1], shape[0] # 通常宽比高大
            depth = depth.reshape(h, w)
        else:
            h, w = depth.shape

        rgb = np.squeeze(rgb)
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h))

        if mask is not None:
            mask = np.squeeze(mask)
            if mask.ndim != 2 or mask.shape != (h, w):
                mask = (depth > 0)
        else:
            mask = (depth > 0)

        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = -np.repeat(Phi, h, axis=0)

        X = depth * np.sin(Theta) * np.sin(Phi)
        Y = depth * np.cos(Theta)
        Z = depth * np.sin(Theta) * np.cos(Phi)

        valid = mask
        X, Y, Z = X[valid], Y[valid], Z[valid]
        R, G, B = rgb[:, :, 0][valid], rgb[:, :, 1][valid], rgb[:, :, 2][valid]

        XYZ = np.stack([X, Y, Z], axis=1)
        RGB = np.stack([R, G, B], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB / 255.0 if RGB.max() > 1.0 else RGB)
        o3d.io.write_point_cloud(path, pcd)

    def save_samples(self, rgbs, gt_depths, pred_depths, depth_masks=None):
        """
        Saves samples
        """
        rgbs = rgbs.cpu().numpy().transpose(0, 2, 3, 1)
        depth_preds = pred_depths.cpu().numpy()
        gt_depths = gt_depths.cpu().numpy()
        
        if depth_masks is not None:
            depth_masks = depth_masks.cpu().numpy()
        else:
            depth_masks = gt_depths > 0

        for i in range(rgbs.shape[0]):
            self.idx = self.idx + 1
            sample_dir = os.path.join(self.save_dir, '%04d' % (self.idx))
            mkdirs(sample_dir)

            cmap = plt.get_cmap("rainbow_r")

            depth_pred_map = cmap(depth_preds[i][0].astype(np.float32) / 10)
            if depth_pred_map.shape[2] > 3:
                depth_pred_map = np.delete(depth_pred_map, 3, 2)
            
            path_pred_img = os.path.join(sample_dir, '_depth_pred.jpg')
            cv2.imwrite(path_pred_img, (depth_pred_map * 255).astype(np.uint8))

            depth_gt_map = cmap(gt_depths[i][0].astype(np.float32) / 10)
            if depth_gt_map.shape[2] > 3:
                depth_gt_map = np.delete(depth_gt_map, 3, 2)
            
            mask_i = np.squeeze(depth_masks[i][0])
            depth_gt_map[~mask_i] = 0
            
            path_gt_img = os.path.join(sample_dir, '_depth_gt.jpg')
            cv2.imwrite(path_gt_img, (depth_gt_map * 255).astype(np.uint8))

            path_pc_pred = os.path.join(sample_dir, '_pc_pred.ply')
            self.save_as_point_cloud(depth_preds[i][0], rgbs[i], path_pc_pred, mask=None)

            path_pc_gt = os.path.join(sample_dir, '_pc_gt.ply')
            self.save_as_point_cloud(gt_depths[i][0], rgbs[i], path_pc_gt, mask=mask_i)

            rgb_img = (rgbs[i] * 255).astype(np.uint8)
            path_rgb = os.path.join(sample_dir, '_rgb.jpg')
            cv2.imwrite(path_rgb, rgb_img[:, :, ::-1])