import numpy as np
from collections import defaultdict
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans

class det3d_holes():
    def __init__(self, input_ply):
        self.pc = o3d.io.read_point_cloud(input_ply)

    # 1. 粗拟合一个平面，过滤掉干扰噪声点
    def plane_filter(self, pc=None, max_distance=0.7):
        """  """
        pcd = self.pc if pc is None else pc

        plane_model, inliers = pcd.segment_plane(distance_threshold=max_distance,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        print("plane model:", plane_model)
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
        return filtered_pc

    # 2. 然后将平面上的点拟和到这个平面上 aligns_to_plane
    def align_to_plane(self, pc=None, max_distance=0.05):
        """
        将点云投影到给定平面 Ax + By + Cz + D = 0 上
        -------------------------------------------------
        Parameters
        ----------
        pc : open3d.geometry.PointCloud | None
             要处理的点云；None → 使用 self.pc
        plane_model : (a, b, c, d)
             平面方程 4 参数；必须提供

        Returns
        -------
        planed_pcd : open3d.geometry.PointCloud
                     已经严格落在平面上的点云
        """
        pcd = self.pc if pc is None else pc
        # target plane, max_distance越小越好
        plane_model, inliers = pcd.segment_plane(distance_threshold=max_distance,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        print("target plane model:", plane_model)
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise ValueError("非法的平面法向量 (0,0,0)")

        n_unit = n / n_norm                     # 单位法向
        pts = np.asarray(pcd.points)            # N × 3

        # 每个点到平面的有符号距离
        signed_dist = (pts @ n + d) / n_norm    # N
        # 正交投影
        proj_pts = pts - signed_dist[:, None] * n_unit

        planed_pcd = o3d.geometry.PointCloud()
        planed_pcd.points = o3d.utility.Vector3dVector(proj_pts)

        # 复制颜色 / 法向量（若有）
        if pcd.has_colors():
            planed_pcd.colors = pcd.colors
        if pcd.has_normals():
            planed_pcd.normals = o3d.utility.Vector3dVector(
                np.tile(n_unit, (len(planed_pcd.points), 1)))

        return planed_pcd, plane_model

    # 3. 降采样 让点云的密度均匀一些
    def mtd_voxel(self, pc=None, voxel_size=0.4):
        """体素降采样"""
        pcd = self.pc if pc is None else pc
        voxel_pcd = pcd.voxel_down_sample(voxel_size)
        print("体素滤波后的点云大小:", len(voxel_pcd.points))
        return voxel_pcd

    # 4. 根据密度区间，找到边沿
    def get_edgesPCD(self, pc=None, max_r=2., neighbor_thresh=(6, 80)):
        pcd = self.pc if pc is None else pc

        # 创建 KDTree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        min_n, max_n = neighbor_thresh

        mask = np.zeros(len(points), dtype=bool)
        for i, point in enumerate(points):
            # 搜索半径内的邻居
            k, *_ = pcd_tree.search_radius_vector_3d(point, max_r)
            nbrs = k - 1    # 排除自身
            mask[i] = (min_n <= nbrs <= max_n)
        edge_points = points[mask]
        # 创建边沿点云对象
        edge_pcd = o3d.geometry.PointCloud()
        edge_pcd.points = o3d.utility.Vector3dVector(edge_points)

        print(f"根据密度差异提取的边缘点数: {len(edge_points)}")
        return edge_pcd

    # 5. 提取中间的孔洞 点云
    def get_circle(self, pc=None, eps=4, min_samples=20,  # DBSCAN 参数
                   resid_ratio_thresh=0.5,  # 残差/半径阈值 (3%)
                   radius_range=(2., 15.)):  # 允许的半径范围
        """  提取所有圆形（孔洞）点云簇
        --------------------------------------------------
        eps, min_samples : DBSCAN 聚类参数
        resid_ratio_thresh:  圆拟合平均残差 / 半径  上限 (越小越严格)
        radius_range      :  (min_r, max_r)  过滤过大或过小的圆
        """
        # ---------- 0. 准备点云 ----------
        pcd = self.pc if pc is None else pc
        pts = np.asarray(pcd.points)  # N×3

        # ---------- 1. DBSCAN 聚类 ----------
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(pts)
        labels = db.labels_
        unique_labels = [lb for lb in np.unique(labels) if lb != -1]  # -1 是噪声

        circle_points = []

        # ---------- 2. 对每个簇做圆拟合 ----------
        for lb in unique_labels:
            cluster_pts = pts[labels == lb]  # k × 3
            xy = cluster_pts[:, :2]  # 投影到 XY

            # --- 2a. Kåsa 圆拟合 ---
            # 方程: x² + y² + Ax + By + C = 0
            A_mat = np.hstack((xy, np.ones((xy.shape[0], 1))))
            f_vec = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)

            # 最小二乘解
            sol, *_ = np.linalg.lstsq(A_mat, f_vec, rcond=None)
            A_coef, B_coef, C_coef = sol

            # 圆心和半径
            xc, yc = -A_coef / 2, -B_coef / 2
            radius = np.sqrt((A_coef ** 2 + B_coef ** 2) / 4 - C_coef)

            # --- 2b. 计算残差 ---
            dists = np.sqrt((xy[:, 0] - xc) ** 2 + (xy[:, 1] - yc) ** 2)
            resid = np.mean(np.abs(dists - radius))  # 平均残差
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

    # 6. 找到 仅包含孔洞的点云中 每个孔洞的圆心,  孔洞就是圆形的  + 平面约束
    def circle_aligns(self, pc=None, plane_model=None, eps=4.0, min_samples=8,  # DBSCAN 参数
                      n_circle_samples=200):  # 每圈采样点数
        """
        在倾斜平面内聚类孔洞、拟合圆并绘制:
            • 原孔洞点  : 白
            • 圆心      : 黄
            • 拟合圆    : 绿
        Returns
        -------
        colored_pcd : open3d.geometry.PointCloud
        circle_dict : {radius: [xc, yc, zc]}  (3‑D 圆心坐标)
        """
        # 0. 取点云
        pcd = self.pc if pc is None else pc
        pts = np.asarray(pcd.points)

        # --------------------------------------------------------------
        # 1. 平面模型 (ax + by + cz + d = 0)
        # --------------------------------------------------------------
        if plane_model is None:
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.15,
                                                     ransac_n=3,
                                                     num_iterations=1000)
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=float)
        n /= np.linalg.norm(n)

        # 取平面上一点 p0  (用几何中心即可)
        p0 = pts.mean(axis=0)
        # --------------------------------------------------------------
        # 2. 建立平面内正交基 u, v
        # --------------------------------------------------------------
        # 先挑一个不与 n 平行的辅助向量
        aux = np.array([0, 0, 1]) if abs(n[2]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(n, aux)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)  # 已归一

        # --------------------------------------------------------------
        # 3. 所有点投影到平面 (x, y)
        # --------------------------------------------------------------
        p_rel = pts - p0  # N×3
        xy = np.stack([p_rel.dot(u),  # N
                       p_rel.dot(v)], axis=1)  # N×2

        # --------------------------------------------------------------
        # 4. DBSCAN 聚类 (平面 2‑D)
        # --------------------------------------------------------------
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
        labels = db.labels_
        clusters = [c for c in np.unique(labels) if c != -1]
        if not clusters:
            raise RuntimeError("未检测到可用的孔洞簇，请调整 eps/min_samples")

        # --------------------------------------------------------------
        # 5. 拟合圆 + 可视化数据
        # --------------------------------------------------------------
        all_points = [pts]  # 原孔洞点
        all_colors = [np.ones_like(pts) * 0.95]  # 白
        circle_dict = defaultdict(list)

        for lb in clusters:
            cluster_xy = xy[labels == lb]  # k×2
            cluster_pts = pts[labels == lb]  # k×3

            # ---------- (a) Kåsa 圆拟合 ----------
            A = np.hstack((cluster_xy, np.ones((cluster_xy.shape[0], 1))))
            f = -(cluster_xy[:, 0] ** 2 + cluster_xy[:, 1] ** 2)
            sol, *_ = np.linalg.lstsq(A, f, rcond=None)
            A_, B_, C_ = sol
            xc2d, yc2d = -A_ / 2, -B_ / 2
            r = np.sqrt((A_ ** 2 + B_ ** 2) / 4 - C_)

            # ---------- (b) 圆心 (3‑D) ----------
            center3d = p0 + xc2d * u + yc2d * v
            center_col = np.array([[1.0, 1.0, 0.0]])  # 黄
            all_points.append(center3d[None, :])
            all_colors.append(center_col)

            circle_dict[str(r)] = center3d.tolist()

            # ---------- (c) 拟合圆 (3‑D) ---------
            theta = np.linspace(0, 2 * np.pi, n_circle_samples, endpoint=False)
            circle_xy = np.vstack((xc2d + r * np.cos(theta),
                                   yc2d + r * np.sin(theta))).T  # m×2
            circle_3d = p0 + circle_xy[:, 0][:, None] * u + circle_xy[:, 1][:, None] * v
            circle_clr = np.tile(np.array([[0.0, 1.0, 0.0]]), (n_circle_samples, 1))  # 绿
            all_points.append(circle_3d)
            all_colors.append(circle_clr)

        # --------------------------------------------------------------
        # 6. 合并
        # --------------------------------------------------------------
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(all_points)
        colored_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        print(f"检测到孔洞数量: {len(clusters)}")
        return colored_pcd, circle_dict

    def pipeline(self, tgt_path="24_detpipe.ply"):

        # 1. 粗拟合平面去噪 0.5
        pcd= self.plane_filter(max_distance=0.7)
        pcd = self.plane_filter(pc=pcd, max_distance=0.2)
        # # 2. 精确拟合平面, 并且拟合到这个平面 plane_model
        pcd, plane_model = self.align_to_plane(pc=pcd, max_distance=0.05)
        # # 3. 体素降采样
        pcd = self.mtd_voxel(pc=pcd, voxel_size=0.24)
        # # 4. 根据密度找到边沿
        pcd = self.get_edgesPCD(pc=pcd, max_r=2.0, neighbor_thresh=(115, 160))
        # # 5. 提取出目标孔洞
        pcd = self.get_circle(pc=pcd)
        # # 6. 拟合圆、找圆心， + 平面约束
        pcd, circle_dict = self.circle_aligns(pc=pcd, plane_model=plane_model)
        print(circle_dict)
        self._save(pc=pcd, tgt_path=tgt_path)
        self._visualize(pc=pcd)


    def _visualize(self, pc=None):
        pcd = self.pc if pc is None else pc
        o3d.visualization.draw_geometries([pcd])

    def _save(self, tgt_path=None, pc=None):
        pc = self.pc if pc is None else pc
        tgt_path = "0_filtered.ply" if tgt_path is None else tgt_path
        o3d.io.write_point_cloud(tgt_path, pc)



if __name__ == "__main__":
    # 平板材料表面是磨砂材质，平板，上有多个孔洞，孔洞口非垂直 "v"不是垂直的，
    # 现在应该是 平板材料的最外层表面
    # -- todo 或许目标应该是 孔洞内径的那层表面？ 如果是一个孔洞有多个内径（类似多层结构） ？？
    pcpath = r"24_raw.ply"
    holes3d = det3d_holes(pcpath)
    holes3d.pipeline(tgt_path="24_detpipe6.ply")

