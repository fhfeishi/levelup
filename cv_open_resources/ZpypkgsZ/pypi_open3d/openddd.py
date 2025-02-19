import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
color_raw = o3d.io.read_image(r"D:\ddesktop\draw\color.png")
depth_raw = o3d.io.read_image(r"D:\ddesktop\draw\depth.png")

# 创建 RGBD 图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

# 相机内参 (假设你有这些参数)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

# 根据 RGBD 图像和相机参数生成点云
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, intrinsic)

# 可视化点云
o3d.visualization.draw_geometries([pcd])

# 保存点云
o3d.io.write_point_cloud(r"D:\ddesktop\draw\output_point_cloud.ply", pcd)