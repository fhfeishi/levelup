from collections import defaultdict

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

class DenosingWithOpen3d(object):
    """用于点云降噪和处理的类"""
    def __init__(self, input_ply):
        input_ply = o3d.io.read_point_cloud(input_ply)
        self.pc = input_ply

    @property
    def bbox(self, pc=None):
        """获取点云的边界框及相关尺寸信息"""
        pcd = self.pc if pc is None else pc
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        length = max_bound[0] - min_bound[0]
        width = max_bound[1] - min_bound[1]
        height = max_bound[2] - min_bound[2]
        print(f"长度 (X轴方向): {length:.3f}")
        print(f"宽度 (Y轴方向): {width:.3f}")
        print(f"高度 (Z轴方向): {height:.3f}")
        return min_bound, max_bound, length, width, height

    # # 1.基于拟合平面去掉干扰点， 速度快效果好  good ++
    def mtd_plane_fitting(self, pc=None, max_distance=0.7, plane_out=False):
        """
        使用平面拟合去除不属于平面的干扰点
        Args:
            pc: 点云对象
            max_distance: 设定的距离阈值，远离平面太远的点会被认为是干扰点
        Returns:
            filtered_pc: 过滤后的点云
        """
        pcd = self.pc if pc is None else pc

        # 使用 RANSAC 拟合平面
        plane_model, inliers = pcd.segment_plane(distance_threshold=max_distance,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        # 获取平面法向量和截距 plane_model 是三维平面的方程的四个参数
        a, b, c, d = plane_model  # Ax + By + Cz + D = 0

        # 计算每个点到平面的距离
        points = np.asarray(pcd.points)
        distances = np.abs(np.dot(points, np.array([a, b, c])) + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)

        # 筛选出距离平面小于阈值的点
        filtered_points = points[distances < max_distance]

        # 创建新的点云对象
        filtered_pc = o3d.geometry.PointCloud()
        filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)

        print(f"距离平面太远的点被去除后的点云大小: {len(filtered_pc.points)}")
        if plane_out:
            return plane_model, filtered_pc
        else:
            return filtered_pc

    # # 2.对齐到一个平面
    def align_to_plane(self, pc=None, plane_model=None):
        pcd = self.pc if pc is None else pc

        # 从平面模型中提取参数
        a, b, c, d = plane_model
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)  # 归一化法向量

        # 估算 pcd 的法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        model_normals = np.asarray(pcd.normals)
        # 使用模型法向量的平均值作为主法向量
        model_normal = np.mean(model_normals, axis=0)
        model_normal = model_normal / np.linalg.norm(model_normal)  # 归一化

        # 计算旋转轴和角度
        rotation_axis = np.cross(model_normal, plane_normal)
        rotation_angle = np.arccos(np.dot(model_normal, plane_normal))

        if np.linalg.norm(rotation_axis) > 1e-6:  # 检查旋转轴是否有效
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # 归一化
            # 创建旋转矩阵
            R = self.rotation_matrix(rotation_axis, rotation_angle)
        else:
            print("不需要旋转")
            R = np.eye(3)  # 不需要旋转

        # 应用旋转
        pcd.rotate(R, center=(0, 0, 0))

        # 计算平移到平面
        model_center = np.mean(np.asarray(pcd.points), axis=0)
        distance_to_plane = (a * model_center[0] + b * model_center[1] + c * model_center[2] + d) / np.sqrt(
            a ** 2 + b ** 2 + c ** 2)

        # 平移模型
        translation = -distance_to_plane * plane_normal
        pcd.translate(translation)

        print("模型已对齐到平面。")
        return pcd

    # 计算旋转矩阵
    def rotation_matrix(self, axis, theta):
        """  计算旋转矩阵
        """
        axis = axis / np.linalg.norm(axis)
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        return np.array([[a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
                         [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                         [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]])

    # # 3. 降采样   关键参数voxel_size 越小精度越高
    def mtd_voxel(self, pc=None, voxel_size=1.0):
        """体素降采样"""
        pc = self.pc if pc is None else pc
        voxel_ = pc.voxel_down_sample(voxel_size)
        print("体素滤波后的点云大小:", len(voxel_.points))
        return voxel_
    # # 提取平板面的边沿点云  # 法向量，并不好
    # def get_planeEdgePCD(self, pc=None, alpha=1.):
    #     pcd = self.pc if pc is None else pc
    #     # 使用 Alpha Shape 创建三角网格
    #     alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    #
    #     # 提取边沿线 (boundary edges)
    #     edges = alpha_shape_mesh.get_non_manifold_edges(allow_boundary_edges=True)
    #
    #     # 提取边沿点索引
    #     edge_vertices = np.asarray(edges).flatten()
    #     edge_vertices = np.unique(edge_vertices)
    #
    #     # 创建边沿点云对象
    #     edge_points = np.asarray(alpha_shape_mesh.vertices)[edge_vertices]
    #     edge_pcd = o3d.geometry.PointCloud()
    #     edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    #
    #     print(f"提取边沿点数量：{len(edge_points)}")
    #     return edge_pcd

    # # 4. 提取边缘点云
    def get_edgesPCD(self, pc=None, max_r=2., neighbor_thresh=(6, 80)):
        pcd = self.pc if pc is None else pc

        # 创建 KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        points = np.asarray(pcd.points)
        edge_points = []
        for i, point in enumerate(points):
            # 搜索半径内的邻居
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, max_r)

            # 取边界点  过滤离群点 和 中间点
            if neighbor_thresh[0] < k < neighbor_thresh[1]:
            # if k < neighbor_thresh[1]:
                edge_points.append(point)
        # 创建边沿点云对象
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge_points)

        print(f"根据密度差异提取的边缘点数: {len(edge_points)}")
        return edge_pcd

    # 3.对齐到平面之后再过滤边缘点
    def _filter_plane(self):
        pass

    # 获取孔洞的点云
    def get_holePCD(self):
        pass

    # 拟合孔洞
    def align_holes(self):
        pass

    # 对齐物理世界的坐标
    def align_to_world(self):
        pass


    # 提取孔洞边沿点云
    # # 非常光滑的平面  根据法向量找孔洞、边沿
    # def get_holeEdgePCD(self, pc=None, max_distance=0.1, max_angle=170.):
    #     """提取边缘点云  平板边缘和孔洞边缘
    #     Args:
    #         pc:
    #         max_distance:
    #         max_angle:
    #
    #     Returns:
    #         edge_pcd
    #     """
    #     pcd = self.pc if pc is None else pc
    #
    #     # 获取点云的法向量和点
    #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     points = np.asarray(pcd.points)
    #     normals = np.asarray(pcd.normals)
    #
    #     # 拟合平面，获取平面模型和内点
    #     plane_model, inliers = pcd.segment_plane(distance_threshold=max_distance, ransac_n=3, num_iterations=1000)
    #     a, b, c, d = plane_model  # 平面方程 Ax + By + Cz + D = 0
    #
    #     # 获取拟合平面的法向量
    #     plane_normal = np.array([a, b, c])
    #
    #     # 计算每个点到平面的距离
    #     distances = np.abs(np.dot(points, plane_normal) + d) / np.linalg.norm(plane_normal)
    #
    #     # 计算每个点的法向量与平面法向量的夹角
    #     dot_product = np.sum(normals * plane_normal, axis=1)
    #     angle_cos = dot_product / (np.linalg.norm(normals, axis=1) * np.linalg.norm(plane_normal))
    #     angle_deg = np.arccos(np.clip(angle_cos, -1.0, 1.0)) * 180.0 / np.pi
    #
    #     # 筛选出距离平面较远的点和法向量夹角大于阈值的点
    #     edge_points = points[(distances > max_distance) & (angle_deg > max_angle)]
    #
    #     # 创建一个新的点云对象，包含边缘点
    #     edge_pcd = o3d.geometry.PointCloud()
    #     edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    #
    #     print(f"提取的边缘点云大小: {len(edge_pcd.points)}")
    #     return edge_pcd

    # # # 曲率
    # def get_holeEdgePCD(self, pc=None, voxel_size=0.05, max_angle=160.):
    #     """ 密度方法
    #     Args:
    #         pc:
    #         voxel_size:
    #         max_angle:
    #
    #     Returns:
    #         pcd
    #     """
    #     pcd = self.pc if pc is None else pc
    #
    #     # 将点云转换为体素网格
    #     voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    #
    #     # 计算法线
    #     voxel_down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #
    #     # 获取点云的 numpy 数据
    #     points = np.asarray(voxel_down_pcd.points)
    #     normals = np.asarray(voxel_down_pcd.normals)
    #
    #     # 拟合平面，获取平面模型和内点
    #     plane_model, inliers = voxel_down_pcd.segment_plane(distance_threshold=0.7, ransac_n=3, num_iterations=1000)
    #     plane_normal = plane_model[:3]  # 获取平面法向量
    #
    #     # 计算每个点的法向量与平面法向量的夹角
    #     dot_product = np.dot(normals, plane_normal)
    #     angle_cos = dot_product / (np.linalg.norm(normals, axis=1) * np.linalg.norm(plane_normal))
    #     angle_deg = np.arccos(np.clip(angle_cos, -1.0, 1.0)) * 180.0 / np.pi
    #
    #     # 筛选出法向量角度大于阈值的点
    #     edge_points = points[angle_deg > max_angle]
    #
    #     # 创建一个新的点云对象，包含孔洞边缘点
    #     hole_edge_pcd = o3d.geometry.PointCloud()
    #     hole_edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
    #
    #     print(f"提取的孔洞边缘点云大小: {len(hole_edge_pcd.points)}")
    #     return hole_edge_pcd

    def holePCD_density(self, pc=None, max_distance=0.8, threshold=0.9, max_r=0.7, max_neighbors=15):
        pcd = self.pc if pc is None else pc
        self.pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max_r, max_nn=max_neighbors))
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        # 使用 Open3D 的 KDTree 逐点搜索邻居
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        edge_points = []
        for i in range(len(points)):
            # 使用kd树搜索领域
            k, idx, _ = pcd_tree.search_radius_vector_3d(pcd.points[i], max_distance)
            if k > 1:
                # 检查法线变化
                neighbor_normals = normals[idx[1:]] # 忽略自身
                mean_normal = np.mean(neighbor_normals, axis=0)
                if np.linalg.norm(mean_normal - normals[i]) > threshold:
                    edge_points.append(points[i])
        # 创建edge point cloud
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
        return edge_pcd


    # 使用统计离群点去除算法  # good
    def mtd_outlier(self, pc=None, nb_neighbors=340, std_ratio=0.19):
        """使用统计离群点去除算法
        Args:
            pc: point cloud
            nb_neighbors: 用于计算每个点的平均距离的邻居数
            std_ratio: 标准差倍数，判断离群点的阈值
            如果点的距离大于 平均值 + std_ratio × 标准差，则该点被认为是离群点。
        returns:
            filtered_pcd: filtered point cloud
        """
        pc = self.pc if pc is None else pc
        _, ind = pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        filtered_pcd = pc.select_by_index(ind)
        print("去离群点outlier后的点云大小:", len(filtered_pcd.points))
        return filtered_pcd

    # 聚类方法 k-means
    def mtd_clusteringKMEANS(self, pc=None, n_clusters=50):
        """  K means聚类方法，返回彩色聚类结果
        Args:
            pc:
            n_clusters:
        return:
            filtered_pc:
        """
        pcd = self.pc if pc is None else pc
        # 获取点云的numpy数据
        points = np.asarray(pcd.points)
        # 使用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(points)
        labels = kmeans.fit_predict(points)

        # 为每个簇生成一个不同颜色
        cmap = plt.get_cmap("tab20")
        colors = cmap(labels % 20)[:, :3]  # 模20以避免超过色彩范围

        # 创建带颜色的新点云
        colored_pc = o3d.geometry.PointCloud()
        colored_pc.points = o3d.utility.Vector3dVector(points)
        colored_pc.colors = o3d.utility.Vector3dVector(colors)

        return colored_pc

    # 聚类方法 DBSCAN
    def mtd_clusteringDBSCAN(self, pc=None, max_radius=1.0, min_samples=8):
        """
        DBSCAN 聚类并为每个簇上不同颜色，方便可视化调参
        -----------------------------------------------------
        max_radius  : eps，半径阈值
        min_samples : 核心点最小邻居数
        Returns
        -------
        colored_pc  : open3d.geometry.PointCloud, 已着色
        """
        pcd = self.pc if pc is None else pc
        pts = np.asarray(pcd.points)

        # ---------- DBSCAN ----------
        db = DBSCAN(eps=max_radius, min_samples=min_samples)
        db.fit(pts)
        labels = db.labels_
        uniq = np.unique(labels)  # 包含 -1 (噪声)
        n_clusters = len(uniq) - (1 if -1 in uniq else 0)

        # ---------- 为每个簇生成颜色 ----------
        cmap = plt.get_cmap("tab20")  # 20 种可区分的颜色
        colors = np.zeros((pts.shape[0], 3))

        for lb in uniq:
            if lb == -1:
                # 噪声点用灰色
                colors[labels == lb] = np.array([0.5, 0.5, 0.5])
            else:
                colors[labels == lb] = cmap(lb % 20)[:3]  # 取前 3 个 RGB 分量

        # ---------- 生成彩色点云 ----------
        colored_pc = o3d.geometry.PointCloud()
        colored_pc.points = o3d.utility.Vector3dVector(pts)
        colored_pc.colors = o3d.utility.Vector3dVector(colors)

        print(f"DBSCAN 发现簇数: {n_clusters}（噪声点 {np.sum(labels == -1)} 个）")
        return colored_pc

    def mtd_clustering(self, pc=None, n_clusters=50):
        """  K means聚类方法，返回彩色聚类结果
        Args:
            pc:
            n_clusters:
        return:
            filtered_pc:
        """
        pcd = self.pc if pc is None else pc
        # 获取点云的numpy数据
        points = np.asarray(pcd.points)
        # 使用 KMeans 进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(points)
        labels = kmeans.fit_predict(points)

        # 为每个簇生成一个不同颜色
        cmap = plt.get_cmap("tab20")
        colors = cmap(labels % 20)[:, :3]  # 模20以避免超过色彩范围

        # 创建带颜色的新点云
        colored_pc = o3d.geometry.PointCloud()
        colored_pc.points = o3d.utility.Vector3dVector(points)
        colored_pc.colors = o3d.utility.Vector3dVector(colors)

        return colored_pc

    # 5. 提取中间的孔洞 点云
    def get_circle(self, pc=None, eps=4, min_samples=20,  # DBSCAN 参数
               resid_ratio_thresh=0.5,           # 残差/半径阈值 (3%)
               radius_range=(2., 15.)):         # 允许的半径范围
        """  提取所有圆形（孔洞）点云簇
        --------------------------------------------------
        eps, min_samples : DBSCAN 聚类参数
        resid_ratio_thresh:  圆拟合平均残差 / 半径  上限 (越小越严格)
        radius_range      :  (min_r, max_r)  过滤过大或过小的圆
        """
        # ---------- 0. 准备点云 ----------
        pcd = self.pc if pc is None else pc
        pts = np.asarray(pcd.points)           # N×3

        # ---------- 1. DBSCAN 聚类 ----------
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(pts)
        labels = db.labels_
        unique_labels = [lb for lb in np.unique(labels) if lb != -1]  # -1 是噪声

        circle_points = []

        # ---------- 2. 对每个簇做圆拟合 ----------
        for lb in unique_labels:
            cluster_pts = pts[labels == lb]          # k × 3
            xy = cluster_pts[:, :2]                  # 投影到 XY

            # --- 2a. Kåsa 圆拟合 ---
            # 方程: x² + y² + Ax + By + C = 0
            A_mat = np.hstack((xy, np.ones((xy.shape[0], 1))))
            f_vec = -(xy[:, 0]**2 + xy[:, 1]**2)

            # 最小二乘解
            sol, *_ = np.linalg.lstsq(A_mat, f_vec, rcond=None)
            A_coef, B_coef, C_coef = sol

            # 圆心和半径
            xc, yc = -A_coef/2, -B_coef/2
            radius = np.sqrt((A_coef**2 + B_coef**2)/4 - C_coef)

            # --- 2b. 计算残差 ---
            dists = np.sqrt((xy[:, 0]-xc)**2 + (xy[:, 1]-yc)**2)
            resid = np.mean(np.abs(dists - radius))    # 平均残差
            resid_ratio = resid / radius
            # # for debug --get params:
            # print(f"{radius = }")
            # print(f"{resid_ratio = }")

            # --- 2c. 根据阈值判断是否为圆孔 ---
            if (radius_range[0] <= radius <= radius_range[1]) and (resid_ratio <= resid_ratio_thresh):
                circle_points.append(cluster_pts)

        # ---------- 3. 合并并返回 ----------
        if not circle_points:
            print("未检测到符合条件的圆形簇")
            return None

        hole_pts = np.vstack(circle_points)
        hole_pcd = o3d.geometry.PointCloud()
        hole_pcd.points = o3d.utility.Vector3dVector(hole_pts)

        # 可选：给孔洞染色，观察效果
        hole_pcd.paint_uniform_color([1, 0, 0])  # 红色

        print(f"圆形簇数量: {len(circle_points)},  孔洞点数: {len(hole_pts)}")
        return hole_pcd

    # 6. 找到 仅包含孔洞的点云中 每个孔洞的圆心,  孔洞就是圆形的
    def circle_aligns(self, pc=None,
                  eps=3.0, min_samples=8,         # DBSCAN 参数
                  n_circle_samples=200):          # 每圈采样点数
        """
        找孔洞圆心并绘制拟合圆
        -------------------------------------------------
        eps, min_samples  :   DBSCAN 聚类参数
        n_circle_samples  :   绘制圆时的采样点数
        Returns
        -------
        colored_pcd : open3d.geometry.PointCloud
                      原孔洞点(白) + 拟合圆(绿) + 圆心(黄)
        """
        pcd = self.pc if pc is None else pc
        pts = np.asarray(pcd.points)

        # ---------- 1. DBSCAN 聚类（在 XY 平面聚类即可） ----------
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts[:, :2])
        labels = db.labels_
        clusters = [c for c in np.unique(labels) if c != -1]   # -1 -> 噪声
        if not clusters:
            raise RuntimeError("未检测到可用的孔洞簇，请调整 eps/min_samples")

        # 新建用于存放显示用的数据
        all_points  = [pts]                         # 原始孔洞点
        all_colors  = [np.ones_like(pts) * 0.95]    # 白色 [0.95,0.95,0.95]

        circle_dict = defaultdict(list)

        # ---------- 2. 对每个簇做圆拟合并生成绘制点 ----------
        for idx, lb in enumerate(clusters):
            cluster_pts = pts[labels == lb]         # k × 3
            xy = cluster_pts[:, :2]

            # ---- 2a. 圆拟合（Kåsa） ----
            A = np.hstack((xy, np.ones((xy.shape[0], 1))))
            f = -(xy[:, 0]**2 + xy[:, 1]**2)
            sol, *_ = np.linalg.lstsq(A, f, rcond=None)
            A_, B_, C_ = sol
            xc, yc = -A_/2, -B_/2
            r  = np.sqrt((A_**2 + B_**2)/4 - C_)

            # ---- 2b. 圆心（黄） ----
            zc = np.mean(cluster_pts[:, 2])
            center_pt  = np.array([[xc, yc, zc]])
            center_clr = np.array([[1.0, 1.0, 0.0]])          # Yellow
            all_points.append(center_pt)
            all_colors.append(center_clr)

            circle_dict[r] = [xc, yc, zc]

            # ---- 2c. 拟合圆（绿） ----
            theta = np.linspace(0, 2*np.pi, n_circle_samples, endpoint=False)
            circle_xy = np.vstack((xc + r*np.cos(theta),
                                   yc + r*np.sin(theta))).T
            circle_z  = np.full((n_circle_samples, 1), zc)
            circle_pts = np.hstack((circle_xy, circle_z))
            circle_clr = np.tile(np.array([[0.0, 1.0, 0.0]]), (n_circle_samples, 1))  # Green
            all_points.append(circle_pts)
            all_colors.append(circle_clr)

        # ---------- 3. 合并并返回 ----------
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(all_points)
        colored_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        print(f"检测到孔洞数量: {len(clusters)}")
        return colored_pcd, circle_dict

    # 7.
    def aligns_world(self,pc=None):
        """对齐物理世界的尺度、物理约束（如果跟实际存在偏差就需要）"""
        pcd = self.pc if pc is None else pc

        pass


    # 连通方法
    def mtd_connected(self, pc=None, max_radius=2, min_cluster_size=25):
        pc = self.pc if pc is None else pc
        # 使用 KDTreeFlann 进行邻近点查询
        kdtree = o3d.geometry.KDTreeFlann(pc)
        # 获取点云中的点
        points = np.asarray(pc.points)
        labels = np.full(len(points), -1)  # 初始化标签数组
        cluster_id = 0
        # 遍历每个点进行聚类
        for i, point in enumerate(points):
            if labels[i] != -1:  # 如果点已经被分配了簇标签，则跳过
                continue
            # 查找该点的邻居
            [k, idx, _] = kdtree.search_radius_vector_3d(point, max_radius)
            if k > 0:
                # 将相邻点加入当前簇
                for j in idx:
                    if labels[j] == -1:  # 如果邻居没有被分配簇标签，则分配
                        labels[j] = cluster_id
                cluster_id += 1
        # 获取簇标签大小
        unique_labels = set(labels)
        filtered_points = []
        # 保留大于指定最小簇大小的簇
        for label in unique_labels:
            if label != -1:  # 排除噪声点（标签为-1）
                cluster_points = points[labels == label]
                if len(cluster_points) >= min_cluster_size:  # 筛选掉小簇
                    filtered_points.append(cluster_points)
        # 合并大区域的点
        filtered_points = np.vstack(filtered_points)
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        print(f"连通区域过滤后的点云大小：{len(filtered_points)}")
        return filtered_pcd

    # 光滑平面、法向量平滑方法  # 有点像磨砂材质 表面并不光滑
    def mtd_smoothing(self, pc=None, angle_max=160.0):
        """ 基于法向量一致性检查平滑表面并去除不符合规则的噪声点
        Args:
            pc: point cloud
            angle_threshold: 法向量差异的阈值，单位为度。大于此值的点将被认为是噪声点。
        returns:
            filtered_pc: filtered point cloud
        """
        pc = self.pc if pc is None else pc

        # 使用 KDTree 构建索引，找到每个点的邻居
        kdtree = o3d.geometry.KDTreeFlann(pc)

        # 计算法向量
        if not pc.has_normals():
            pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 获取点云的点和法向量
        points = np.asarray(pc.points)
        normals = np.asarray(pc.normals)

        mask = np.ones(len(points), dtype=bool)  # 初始化所有点都为保留点

        for i, point in enumerate(points):
            # 查找邻居点
            [k, idx, _] = kdtree.search_knn_vector_3d(point, 20)  # 搜索20个邻居
            if k < 3:  # 如果邻居数量小于3，跳过
                continue

            # 获取邻居点的法向量
            neighbor_normals = normals[idx]

            # 计算当前点的法向量与邻居点法向量的角度差异
            angle_diffs = []
            for neighbor_normal in neighbor_normals:
                # 计算法向量之间的夹角，使用点积来计算角度差
                cos_theta = np.clip(np.dot(normals[i], neighbor_normal), -1.0, 1.0)
                angle = np.arccos(cos_theta) * 180.0 / np.pi  # 转换为角度
                angle_diffs.append(angle)

            # 判断当前点的法向量是否与邻居法向量一致
            if np.max(angle_diffs) > angle_max:
                mask[i] = False  # 如果最大角度差异大于阈值，认为是噪声点，去除该点

        # 筛选保留的点
        filtered_points = points[mask]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 如果有颜色，复制颜色属性
        if pc.has_colors():
            colors = np.asarray(pc.colors)
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        # 如果有法向量，复制法向量属性
        if pc.has_normals():
            normals = np.asarray(pc.normals)
            filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

        print(f"法向量平滑后的点云大小: {len(filtered_pcd.points)}")
        return filtered_pcd

    # 密度方法
    def mtd_density(self, pc=None, radius=2.0, min_neighbors=40):
        """   基于密度过滤点云
        Args:
            pc: point cloud
            radius: 搜索半径
            min_neighbors: 最小邻居数量
        returns:
            filtered_pcd: filtered point cloud
        """
        pc = self.pc if pc is None else pc

        # 使用 KDTree 构建索引，找到每个点的邻居
        kdtree = o3d.geometry.KDTreeFlann(pc)
        # 计算每个点的邻居数量
        points = np.asarray(pc.points)
        mask = np.ones(len(points), dtype=bool)
        for i, point in enumerate(points):
            # [k, idx, _] = kdtree.search_knn_vector_3d(point, min_neighbors)
            [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
            if k < min_neighbors:
                mask[i] = False  # 如果邻居数少于指定阈值，标记为离散点
        # 筛选保留的点
        filtered_points = points[mask]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 如果有颜色，复制颜色属性
        if self.pc.has_colors():
            colors = np.asarray(self.pc.colors)
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        # 如果有法向量，复制法向量属性
        if self.pc.has_normals():
            normals = np.asarray(self.pc.normals)
            filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

        print(f"密度过滤后的点云大小: {len(filtered_pcd.points)}")
        return filtered_pcd

    # 可视化方法
    def visualize(self, pc=None):
        pcd = self.pc if pc is None else pc
        o3d.visualization.draw_geometries([pcd])
    # 点云保存
    def _save(self, tgt_path=None, pc=None):
        pc = self.pc if pc is None else pc
        tgt_path = "0_filtered.ply" if tgt_path is None else tgt_path
        o3d.io.write_point_cloud(tgt_path, pc)

    # get np points
    def _get_points(self, pc=None):
        """获取点云的numpy数组"""
        pcd = self.pc if pc is None else pc
        points = np.asarray(pcd.points)
        return points

    # new PointCloud obj
    def _new_pcd(self, points):
        """根据给定的points新建一个PointCloud对象"""
        filtered_pc = o3d.geometry.PointCloud()
        filtered_pc.points = o3d.utility.Vector3dVector(points)
        return filtered_pc

    # 保留点云原属性： 颜色属性  法向量属性
    def _aligns(self, filtered_pc):
        # 如果有颜色，复制颜色属性
        # 如果有法向量，复制法向量属性
        pass

    # z 根据深度范围过滤掉干扰区域
    def filterBybbox(self, pc=None, margin=0.05, mode="z"):
        """
        根据边界框过滤点云
        Args:
            margin: 边界缩小的边距
        """
        pc = self.pc if pc is None else pc
        min_bound, max_bound, *_ = self.bbox
        if mode == "z":
            min_bound_filtered = min_bound + np.array([0, 0,margin])
            max_bound_filtered = max_bound - np.array([0, 0, margin])
        else:
            min_bound_filtered = min_bound + np.array([margin, margin, margin])
            max_bound_filtered = max_bound - np.array([margin, margin, margin])

        # 根据新的边界范围筛选点
        points = np.asarray(pc.points)
        mask = np.all((points >= min_bound_filtered) & (points <= max_bound_filtered), axis=1)

        # 通过筛选保留点云
        filtered_points = points[mask]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 如果有颜色，复制颜色属性
        if self.pc.has_colors():
            colors = np.asarray(self.pc.colors)
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

        # 如果有法向量，复制法向量属性
        if self.pc.has_normals():
            normals = np.asarray(self.pc.normals)
            filtered_pcd.normals = o3d.utility.Vector3dVector(normals[mask])

        print(f"边界框过滤后的点云大小: {len(filtered_pcd.points)}")
        return filtered_pcd

    def process_pipeline(self, pc=None):
        pcd = self.pc if pc is None else pc

        # # filter by bbox     # useful  margin手工调整(特定情况下也有用)
        # pcd = self.filterBybbox(margin=110)

        # # filter method voxel    # 不精确
        # pcd = self.mtd_voxel(pc=pcd,voxel_size=1.0)

        # 过滤离散点      # useful
        pcd = self.mtd_outlier(pc=pcd)

        # # # filter by density  # useful
        # pcd = self.mtd_density(pc=pcd)

        # # filter connected
        pcd = denos.mtd_connected(pc=pcd, max_radius=1, min_cluster_size=20)

        # # filter smooth   # 难以保留边缘部分
        pcd = self.mtd_smoothing(pc=pcd, angle_max=180)

        # visual  and save
        self.visualize(pcd)
        self._save(pc=pcd)

    def pipeline(self, pc=None):
        pcd = self.pc if pc is None else pc

        # # 1. filter by plane  过滤
        plane_model, pcd = self.mtd_plane_fitting(pc=pcd, max_distance=0.7, plane_out=True)

        # # 2. align to plane (plane_model)
        pcd = self.align_to_plane(pc=pcd, plane_model=plane_model)

        # # 3. voxel dowansample  # 需要降采样，不然点云密度分布不均
        pcd = self.mtd_voxel(pc=pcd, voxel_size=0.44)

        # # 如果不降采样呢

        # # 4. extract edges point cloud， 这里使用的是密度方法，边沿的密度小（搜索半径设置合适的话）
        pcd = self.get_edgesPCD(pc=pcd)

        # # 5. 通过粗略的圆圈拟合 提取出孔洞的点云
        pcd = self.get_circle(pc=pcd)

        # # 6. 精确的圆圈拟合  circle_dict = {r:[cx,cy,cz], ...}
        pcd, circle_dict = self.circle_aligns(pc=pcd)

        # # 7. 对齐物理尺度 以及平面约束等
        # # todo ...

        # visual  and save
        self.visualize(pcd)
        self._save(pc=pcd)




if __name__ == '__main__':

    ply_path = "24_pipe12345.ply"
    # ply_path = "24/output.ply"
    # ply_path = "24_raw.ply"
    # ply_path = "0/output.ply"
    # ply_path = "14/output.ply"
    denos = DenosingWithOpen3d(ply_path)

    # 聚类方法测试  # 一般
    ## kmeans   # 速度快 不好用
    # pcd_cleaned = denos.mtd_clusteringKMEANS(n_clusters=30)
    # pcd_cleaned = denos.mtd_clusteringKMEANS(pc=pcd_cleaned, n_clusters=10)
    # pcd_cleaned = denos.mtd_clusteringDBSCAN()  # 还是会有一些噪声点

    # # 离散点过滤方法测试  # good  # 缺陷之处在于： 孔洞边沿的点应该也会被认为是离散点（这样的点倒是不多）
    # # pcd_cleaned = denos.mtd_outlier()
    # pcd_cleaned = denos.mtd_outlier()

    # # 密度方法测试  # --
    # pcd_cleaned = denos.mtd_density(radius=0.08, min_neighbors=9)

    # 连通方法测试 # --
    # pcd_cleaned = denos.mtd_connected(max_radius=0.2, )

    # # # 法向量平滑方法测试
    # pcd_cleaned = denos.mtd_smoothing(angle_max=150.)  # slow

    # # 继续整理去掉 离群点 效果不好
    # pcd_cleaned = denos.mtd_outlier(nb_neighbors=800, std_ratio=1.5)

    # # #1 拟合平面  去掉干扰的噪声点簇  # good ++  # 又快又好
    # pcd_cleaned = denos.mtd_plane_fitting()

    # # # #2 提取边缘点
    # pcd_cleaned = denos.holePCD_density()

    # # #  重心、质心 ++
    # pcd_cleaned = denos.mtd_clusteringKMEANS(n_clusters=18)
    # pcd_cleaned = denos.mtd_clusteringDBSCAN(max_radius=3, min_samples=10)
    # pcd_cleaned = denos.mtd_clusteringKMEANS(n_clusters=40)
    # pcd_cleaned = denos.get_circle()
    pcd_cleaned, circle_dict = denos.circle_aligns()
    print(circle_dict)
    denos.visualize(pcd_cleaned)
    denos._save(pc=pcd_cleaned)

    # # # proprecess pipeline
    # pcd_cleaned = denos.pipeline()
