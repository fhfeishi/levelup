from PIL import Image
from PIL import PngImagePlugin

# 打开图像
image_path = r'D:\ddesktop\sucai\头像\Qin Zou-\QinZou.jpg'
image = Image.open(image_path).convert("RGB")

# 创建一个新的图像元数据对象
metadata = PngImagePlugin.PngInfo()
# 设置程序名称为空
metadata.add_text("Software", "")

# 保存新的图像，使用新的元数据
new_image_path = r'D:\ddesktop\sucai\头像\Qin Zou-\a_new.jpg'
image.save(new_image_path, "JPEG", pnginfo=metadata)

print("图像已保存，程序名称已更改。")