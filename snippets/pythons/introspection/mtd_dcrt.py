import open3d as o3d 
import numpy as np 

"""
class det3d_holes():
    def __init__(self, input_ply):
        self.pc = o3d.io.read_point_cloud(input_ply)
    # 1. 粗拟合一个平面，过滤掉干扰噪声点
    def plane_filter(self, pc=None, max_distance=0.7):
        pcd = self.pc if pc is None else pc
        plane_model, inliers = pcd.segment_plane(distance_threshold=max_distance,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        # print(plane_model)
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
        return filtered_pc, plane_model
"""


# 为类方法添加一个 有默认值的pc参数   # 使用functools.partial创建带默认pc参数的方法
#                                   xx = functools.partial(self.funName, pc=None)




# 为方法添加一个默认参数pc=None, 并且pc初始化： pcd = self.pc if pc is None else pc
import functools

def with_pc_param(cls):
    """ 类装饰器：自动为所有的方法添加pc参数处理"""
    
    # 创建pc参数处理函数
    def pc_handler(method):
        @functools.wraps(method)
        def wrapper(self, *args, pc=None, **kwargs):
            actual_pc = self.pc if pc is None else pc 
            return method(self, actual_pc, *args, **kwargs)
        return wrapper 
    
    # 查找类中所有需要处理的方法
    for name, attr in list(cls.__dict__.items()):
        # 跳过特殊方法和非方法属性
        if not name.startswith('__') and callable(attr):
            # 包装方法
            setattr(cls, name, pc_handler(attr))
    return cls 






@with_pc_param
class det3d_holes():
    def __init__(self, ply_path):
        self.pc = o3d.io.read_point_cloud(ply_path)
    def plane_filter(self, pc=None, max_distance=0.7):
        plane_model, _ =pc.segment_plane(distance_threshold=max_distance, ransac_n=3, num_iterations=1000)
        # print(plane_model) 
        a,b,c,d = plane_model
        
        points = np.asarray(pc.points)
        distances = np.abs(np.dot(points, np.array([a,b,c]))+d)/np.sqrt(a**2+b**2+c**2)
        
        filtered_points = points[distances < max_distance]
        
        filtered_pc = o3d.geometry.PointCloud()
        filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
        
        print(f"距离拟合平面太远的点去除后，点云大小：{len(filtered_pc.points)}")
        return filtered_pc, plane_model
    
    def _visual(self, pc=None):
        o3d.visualization.draw_geometries([pc])
        
if __name__ == '__main__':
    
    inp = r"D:\ddesktop\3ds\24_raw.ply"
    det3d = det3d_holes(inp)
    pcd, _ = det3d.plane_filter()
    pcd, _ = det3d.plane_filter(pc=pcd,max_distance=0.3)
    det3d._visual(pc=pcd)
    
    
    
    