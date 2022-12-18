import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd
import re
import argparse
def load_point_cloud(points3D_df):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB']) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])  # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # R, G, B
    return axes


def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def load_poses(r,t):

    r = R.from_quat(r).as_matrix()
    axes = o3d.geometry.LineSet()
    points = np.array([[-540,-960,1],[-540,960,1],[540,960,1],[540,-960,1]])
    points = points/10
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])

    worldpoints = np.zeros((4,3))
    # inverse camera intrinsic
    for i in range(4):
        worldpoints[i] = (np.linalg.inv(r)).dot((np.linalg.inv(cameraMatrix)).dot(points[i]))+t

    worldpoints = np.append(worldpoints,t).reshape(5,3)


    axes.points = o3d.utility.Vector3dVector(worldpoints)
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3],[0,3],[0,4],[1,4],[2,4],[3,4]])  # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [1, 0, 0], [1, 0, 0],[1, 0, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0]])  # R, G, B
    return axes

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='draw trajectory')
    parser.add_argument('--r', type=str, help='path of Rotation vector\'s npy file', required=True)
    parser.add_argument('--t', type=str, help='path of Translation vector\'s npy file', required=True)


    args = parser.parse_args()
    path_r = args.r
    path_t = args.t

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    # load axes
    axes = load_axes()
    vis.add_geometry(axes)


    ## load poses calculated before
    rot = np.load(path_r)
    tran = np.load(path_t)

    #draw each pose
    for i in range(len(rot)):
        axes = load_poses(rot[i], tran[i])
        vis.add_geometry(axes)

    #draw trajetory
    for i in range(0,129):
        axes = o3d.geometry.LineSet()
        pointpair = np.zeros((2,3))
        pointpair[0] = tran[i]
        pointpair[1] = tran[i+1]

        pointpair = np.array(pointpair)
        axes.points = o3d.utility.Vector3dVector(pointpair)
        axes.lines = o3d.utility.Vector2iVector([[0, 1]])  # X, Y, Z
        axes.colors = o3d.utility.Vector3dVector(
            [[0, 0, 0]])  # R, G, B
        vis.add_geometry(axes)

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)


    vis.run()
    vis.destroy_window()


