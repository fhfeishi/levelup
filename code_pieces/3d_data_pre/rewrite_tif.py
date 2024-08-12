# 3D tif data
import os
# from PIL import Image
import tifffile as tiff
import open3d as o3d
import numpy as np

# gdal读取tiff光栅化图像数据的cppp库
from osgeo import osr, gdal as gd
import matplotlib.pyplot as plt

from skimage.metrics import mean_squared_error, structural_similarity as ssim

def load_point_cloud_from_tif(file_path):
    # 从tif文件读取数据
    img = tiff.imread(file_path)
    # 获取图像的高度和宽度
    height, width = img.shape
    # 生成x, y坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    # 将x, y坐标和图像高度值合并成三维坐标数组
    points = np.stack((x.flatten(), y.flatten(), img.flatten()), axis=-1)
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    """
    pcd 是一个 Open3D 点云对象 (o3d.geometry.PointCloud)，用于表示三维点云数据。
    代码中将图像数据 img 转换为三维点云数据，生成对应的 x 和 y 坐标网格，
    并将这些坐标与高度值合并成一个三维坐标数组 points。
    points 是一个形状为 (12800000, 3) 的 NumPy 数组，
    其中每行表示一个三维点的坐标 (x, y, z)，即 (水平坐标, 垂直坐标, 高度值)。
    这些三维点随后被转换为 Open3D 点云对象 pcd
    """
    return pcd, img

def save_3d_point_cloud_to_tif(pcd, file_path, img_shape):
    points = np.asarray(pcd.points)
    x = points[:, 0].reshape(img_shape)
    y = points[:, 1].reshape(img_shape)
    z = points[:, 2].reshape(img_shape)

    driver = gd.GetDriverByName('GTiff')
    outdata = driver.Create(file_path, img_shape[1], img_shape[0], 3, gd.GDT_Float64)

    outdata.GetRasterBand(1).WriteArray(x)
    outdata.GetRasterBand(2).WriteArray(y)
    outdata.GetRasterBand(3).WriteArray(z)
    
    outdata.FlushCache()
    outdata = None

def save_height_point_cloud_to_tif(height_data, file_path, dpi=(300, 300)):
    # 获取数据维度
    height, width = height_data.shape
    
    # 创建新的tif文件
    driver = gd.GetDriverByName('GTiff')
    outdata = driver.Create(file_path, width, height, 1, gd.GDT_Float32)
    
    if outdata is None:
        print("Failed to create the file.")
        return
    
    # 写入高度数据到新的tif文件
    outdata.GetRasterBand(1).WriteArray(height_data)
    
    # # 设置投影和地理转换
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(4326)  # 假设使用WGS84投影，可以根据需要更改
    # outdata.SetProjection(srs.ExportToWkt())
    # outdata.SetGeoTransform([0, 1, 0, 0, 0, -1])  # 可以根据需要设置
    
    # 设置DPI元数据
    metadata = {
        'TIFFTAG_XRESOLUTION': dpi[0],
        'TIFFTAG_YRESOLUTION': dpi[1],
        'TIFFTAG_RESOLUTIONUNIT': 2  # 2表示英寸
    }
    for key, value in metadata.items():
        outdata.SetMetadataItem(key, str(value))

    # 保存并关闭文件
    outdata.FlushCache()
    outdata = None

# # 计算a.tif   aa.tif的相似性
# def caculate_mse(tif1, tif2):
#     return mean_squared_error(tif1, tif2)
# def calculate_ssim(tif1, tif2):
#     return ssim(tif1, tif2, data_range=tif2.max()-tif2.min())

def get_tif_info(file_path):
    dataset = gd.Open(file_path)
    if not dataset:
        print("Failed to open file.")
        return

    print(f"Driver: {dataset.GetDriver().ShortName}/{dataset.GetDriver().LongName}")
    print(f"Size: {dataset.RasterXSize} x {dataset.RasterYSize} x {dataset.RasterCount}")
    print(f"Projection: {dataset.GetProjection()}")
    
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        print(f"Origin: ({geotransform[0]}, {geotransform[3]})")
        print(f"Pixel Size: ({geotransform[1]}, {geotransform[5]})")

    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        print(f"Band {i}:")
        print(f"  Data Type: {gd.GetDataTypeName(band.DataType)}")
        print(f"  Min: {band.GetMinimum()}")
        print(f"  Max: {band.GetMaximum()}")
        print(f"  NoData Value: {band.GetNoDataValue()}")
        print(f"  Scale: {band.GetScale()}")
        print(f"  Offset: {band.GetOffset()}")

        # 获取波段数据并打印统计信息
        band_data = band.ReadAsArray()
        print(f"Band {i} Data: {band_data}")

        # 可视化波段数据
        plt.imshow(band_data, cmap='gray')
        plt.title(f'Band {i} Visualization')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    data_root = r"D:\Ddesktop\ppt\work\aokeng3d"

    tif_folder = os.path.join(data_root, "tif")
    tif_folder = os.path.normpath(tif_folder)

    png_folder = os.path.join(data_root, "png")
    png_folder = os.path.normpath(png_folder)


    tif_name_list = [name for name in os.listdir(tif_folder) if name.endswith('.tif')]


    # tif_img = Image.open(os.path.join(tif_folder, tif_name_list[1]))
    # tif_img.show()   # 2d png图片  灰度图

    # data_set = gd.Open(os.path.join(tif_folder, tif_name_list[1]))
    # print(data_set.RasterCount)  # 1
    # band_1 = data_set.GetRasterBand(1)
    # b1 = band_1.ReadAsArray()
    # img = np.array(b1)
    # f = plt.figure()
    # plt.imshow(img)
    # plt.show()
    # # 这个可视化就很呆

    tiff_path = r"D:\Ddesktop\ppt\work\aokeng3d\tif\030.tif"
    # get_tif_info(tiff_path)
    pcd, img = load_point_cloud_from_tif(tiff_path)
    # # print(pcd.shape) # error
    # # print(img.shape)
    # # # 可视化点云
    # o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")

    # 将点云数据保存为新的tif文件
    new_tif_file_path = os.path.join(tif_folder, "aa.tif")
    save_height_point_cloud_to_tif(img, new_tif_file_path, (300,300))
    # get_tif_info(new_tif_file_path)

    # save_point_cloud_to_tif(pcd, new_tif_file_path, img.shape)    
    # print(f"Saved new .tif file as {new_tif_file_path}")



