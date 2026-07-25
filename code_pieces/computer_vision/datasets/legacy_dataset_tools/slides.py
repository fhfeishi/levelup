from PIL import Image, ImageDraw
import os
from collections import defaultdict
class slideSeg:
    def __init__(self, image):
        # 存储 image 的  blocks
        self.imBlocks = defaultdict(list)  # (row_index, col_index): [image_block, (x1, y1, x2, y2)]
        self.image = image

    def cropWithNoPad_2x2(self, crop_size=(1200,1200), zrows=2, zcols=2, stamp_=True, show_=True):
        """ 
        将输入的图片分成 2x2=4块，每一块的大小是1200x1200，没有padding
        show_: 可视化每块的边缘
        """
        iw, ih = self.image.size
        cw, ch = crop_size
        assert iw >= cw and ih >= ch
        
        # stride 
        sw = iw-cw
        sh = ih-ch 
        assert (cw >= sw >= 0)  and (ch >= sh >= 0)   # 保证原图的每一部分都利用上
            
        if stamp_:
            # 打印看看，
            print("图像宽度高度：", self.image.size)
            print("crop_size:", crop_size)
            print("移动步长宽度高度：", (sw,sh))
            
        for i in range(zrows):
            for j in range(zcols):
                x1 = int(j * sw)
                y1 = int(i * sh)
                x2 = x1 + cw
                y2 = y1 + ch
                self.imBlocks[(i+1,j+1)] = [self.image.crop((x1, y1, x2, y2)), ((x1, y1, x2, y2))]
        if show_:
            # 可视化看看效果
            drawim = self.image.copy()
            draw1 = ImageDraw.Draw(drawim)
            for k in self.imBlocks.keys():
                r = k[0] - 1
                c = k[1] - 1
                thick = r+c+1  # 增加厚度以便更好地区分
                x1 = int(c * sw)
                y1 = int(r * sh)
                x2 = x1 + cw
                y2 = y1 + ch
                draw1.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=thick)
            drawim.show()

    def cropWithNoPad(self, crop_size, zrows, zcols, stamp_=True, show_=True):
        """ 
        将输入的图片分成 zrows x zcols块，每一块的大小是crop_size，没有padding
        show_: 可视化每块的边缘
        
        分块超过4（2x2）的话 可能没法保证步长宽度高度  都是整数， 这可能会造成误差 ？
        """
        iw, ih = self.image.size
        cw, ch = crop_size
        assert iw >= cw and ih >= ch
        
        # stride 
        assert zcols > 1 and zrows > 1   
        sw = (iw-cw) / (zcols-1)
        sh = (ih-ch) / (zrows-1)
        if stamp_:
            # 打印看看，
            print("图像宽度高度：", self.image.size)
            print("crop_size:", crop_size)
            print("移动步长宽度高度：", (sw,sh))
        
        assert (cw >= sw >= 0)  and (ch >= sh >= 0)   # 保证原图的每一部分都利用上
            
        
            
        for i in range(zrows):
            for j in range(zcols):
                x1 = int(j * sw)
                y1 = int(i * sh)
                x2 = x1 + cw
                y2 = y1 + ch
                self.imBlocks[(i+1,j+1)] = [self.image.crop((x1, y1, x2, y2)), ((x1, y1, x2, y2))]
        if show_:
            # 可视化看看效果
            drawim = self.image.copy()
            draw1 = ImageDraw.Draw(drawim)
            for k in self.imBlocks.keys():
                r = k[0] - 1
                c = k[1] - 1
                thick = r+c+1  # 增加厚度以便更好地区分
                x1 = int(c * sw)
                y1 = int(r * sh)
                x2 = x1 + cw
                y2 = y1 + ch
                draw1.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=thick)
            drawim.show()
        
    def saveBlocks(self,strImPath,num_blocks,tgtd):
        imblocks    =   [imb[0] for imb in  self.imBlocks.values()]
        for x in  range(num_blocks):
            strStem, suffix = os.path.basename(strImPath).rsplit('.', 1)
            tgt_stem = f"{strStem}_{x}"
            tgtPath = os.path.join(tgtd, tgt_stem+suffix)
            imblocks[x].save(tgtPath)
    
    @property
    def getCropBox(self):
        cropBoxs =  [imb[1] for imb in  self.imBlocks.values()]
        return  cropBoxs
    
    def cropBoxAligns(self, srcImage, strImPath, num_blocks, tgtd):
        # 比如说 需要crop其他其他图片
        cropBoxs = [imb[1] for imb in self.imBlocks.values()]
        for x in range(num_blocks):
            strStem, suffix = os.path.basename(strImPath).rsplit('.', 1)
            tgt_stem = f"{strStem}_{x}"
            tgtPath = os.path.join(tgtd, tgt_stem+suffix)
            srcImage.crop(cropBoxs[x]).save(tgtPath)
        


if __name__ == '__main__':
    imp = r"D:\chyCodespace\project\jueyuanziboom\deeplabv3-plus-pytorch\VOCdevkit\VOC2007\JPEGImages\image(4).jpg"
    im = Image.open(imp)
    slide = slideSeg(im)
    imblocks = slide.cropWithNoPad(crop_size=(1000,1000), zrows=3, zcols=4)
    # slide.saveBlocks(strImPath=imp, num_blocks=4, tgtd=r"xxx")   
    
    


