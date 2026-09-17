from PIL import Image, ImageOps, ImageFont, ImageDraw
from collections import OrderedDict, defaultdict
import numpy as np
import PIL, random 
from importlib.metadata import version, PackageNotFoundError

# python > 3.8 
# 版本号字符串比大小，这里用官方的工具转一下再比较
def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    try:
        # 使用 importlib.metadata 来获取当前版本
        current_version = version(name)
        current = current_version if current == "0.0.0" else current
    except PackageNotFoundError:
        # 如果找不到包，说明可能未安装
        print(f"{name} package not found")
        return False
    # 版本号比较
    result = (current == minimum) if pinned else (current >= minimum)
    return result



class slideimg(object):
    def __init__(self):
        
        self.sldim_dict = dict()  # key: (row_index, col_index), value:image_block

    def slide(self, pillowimg, crop_size, rows_num=None, cols_num=None,
              stride=None, fill=128, padding='ltrb',
              drawBoxForTest=False,show_paddedimg=True):
        
        iw, ih = pillowimg.size
        assert crop_size is not None, "crop_size is None !!"
        cw, ch = crop_size
        assert iw >= cw and ih >= ch, "image size is too small "
        
        if isinstance(rows_num, int) and isinstance(cols_num, int):
            assert cols_num >1 and rows_num >1, f"{rows_num} and {cols_num} must > 1."
            # 将图片分成rows_num x cols_num 块，可能会有重叠， 一定没有padding
            stride_w = (iw-cw) // (cols_num-1)
            stride_h = (ih-ch) // (rows_num-1)
            padding = None
        elif stride is not None:
            stride_w, stride_h = stride 
            # 将图片分成rows_num x cols_num 块, 有padding
            # rows_num
            rest_h = ih - ch
            if rest_h % ih != 0:
                rows_num = int(rest_h//stride_h) + 1
            else:
                rows_num = int(rest_h / stride_h)
            
            # cols_num
            rest_w = iw-cw
            if rest_w % stride_w != 0:
                cols_num = int(rest_w // stride_w) + 1
            else:
                cols_num = int(rest_w // stride_w)
        else:
            print("wrong params input!")
            print(f"{stride=}, {rows_num=}, {cols_num=}, {crop_size=}") 
        
        assert padding in ['ltrb', 'lt', 'rb', None], f"{padding} must in ['ltrb', 'lt', 'rb']"
        # padding
        padding_h = (rows_num-1) * stride_h + ch - ih  # padding_h可能不是偶数
        padding_w = (cols_num-1) * stride_w + cw - iw  # padding_w可能不是偶数
        # padding
        if padding == 'ltrb':  # left top right bottom
            padding_left   = int(padding_w/2)
            padding_top    = int(padding_h/2)
            padding_right  = padding_w - int(padding_w/2)
            padding_bottom = padding_h - int(padding_h/2)
            padded_img = ImageOps.expand(pillowimg, 
                                         border=(padding_left, padding_top, padding_right, padding_bottom),
                                         fill=(fill, fill, fill))
        elif padding == 'lt':  # left top 
            padded_img = ImageOps.expand(pillowimg, 
                                         border=(padding_w, padding_h, 0, 0),
                                         fill=(fill, fill, fill))
        elif padding == 'rb':  # right bottom
            padded_img = ImageOps.expand(pillowimg, 
                                         border=(0, 0, padding_w, padding_h),
                                         fill=(fill, fill, fill)) 
        else:
            # no padding
            padded_img = pillowimg.copy()
            
        # get sldim_dict: (r,c): img_block 
        for r in range(rows_num):
            for c in range(cols_num):
                xmin = r * stride_h  
                xmax = xmin + ch 
                ymin = c * stride_w
                ymax = ymin + cw 
                img_block = padded_img.crop((xmin, ymin, xmax, ymax)) 
                if img_block.size == crop_size:
                    self.sldim_dict[(r,c)] = img_block
                else:
                    print(f"{(img_block.size)=} but {crop_size=} why?")
        
        # 是否测试一下slide逻辑
        if drawBoxForTest:
            draw = ImageDraw.Draw(padded_img)
            font = ImageFont.truetype('arial.ttf', size=np.floor(3e-2 * padded_img.size[1] + 0.5).astype('int32'))
            
            for thick, ((row_, col_), _) in enumerate(self.sldim_dict.items()):
                x1 = int(col_*stride_w)
                y1 = int(row_*stride_h)
                x2 = x1 + cw 
                y2 = y1 + ch
                
                text = f"row:{row_} col:{col_}"
                text_size = (
                    draw.textbbox((0, 0), text, font=font)[2] - draw.textbbox((0, 0), text, font=font)[0], 
                    draw.textbbox((0, 0), text, font=font)[3] - draw.textbbox((0, 0), text, font=font)[1]) \
                    if check_version(PIL.__version__, '9.2.0') else draw.textsize(text, font)
                color = tuple(random.randint(0,255) for _ in range(3))
                # rect bbox
                draw.rectangle([x1,y1,x2,y2], outline=color, width=thick)
                
                # text
                draw.rectangle([x1-1, y1-1, x2+text_size[0]+1, y2+text_size[1]+1], fill=color)
                draw.text((x1,y1), str(text), fill=(0,0,0), font=font)
        if show_paddedimg:
            padded_img.show()
        
        return self.sldim_dict

def cutimg():
    pass


def draw_bbox(pillowimg, newimg=True):
    if newimg:
        im_copy = pillowimg.copy()
        draw = ImageDraw.Draw(im_copy)
    else:    
        draw = ImageDraw.Draw(pillowimg)
    
    font = ImageFont.truetype(font=)
    
    
    pass







