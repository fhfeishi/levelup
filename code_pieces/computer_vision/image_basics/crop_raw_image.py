# 如果现在能确定图片中只有一堆主要目标（相对集中），将其裁剪出来
# opencv 默认 前景是白色，背景是黑色。
import cv2 as cv
import numpy as np
from PIL import Image
#传入一个轮廓
def myArea(cnt): 
    rect = cv.minAreaRect(cnt)  
    box = cv.boxPoints(rect)
    box = np.int0(box)
    return cv.contourArea(box)

def get_mainObj_bbox(opencvImage):
    
    # 转换为灰度图像
    gray = cv.cvtColor(opencvImage, cv.COLOR_BGR2GRAY)

    # 应用高斯模糊，减少噪声
    blurred = cv.GaussianBlur(gray, (7, 7), 0)

    # 使用Otsu's方法自动阈值化
    _, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 找到阈值图像中的轮廓
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓是主要目标
    if contours:
        main_contour = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(main_contour)
        # 裁剪图像
        cropped_img = opencvImage[y:y+h, x:x+w]
        return cropped_img
    else:
        return opencvImage  # 如果没有找到轮廓，返回原图
    
    
def get_mainObj_bbox2(opencvImage):
    # 转换为灰度图并进行阈值处理
    imgray = cv.cvtColor(opencvImage, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, bin_threshold, 255, cv.THRESH_BINARY)

    # 使用Canny边缘检测并应用高斯模糊
    edges = cv.Canny(thresh, 100, 150)
    blurred_edges = cv.GaussianBlur(edges, (7, 7), 0)

    # 寻找轮廓
    contours, _ = cv.findContours(blurred_edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未找到要检测的轮廓")
        return None

    # 选择最大轮廓
    contour = max(contours, key=cv.contourArea)
    if myArea(contour) * 0.2 > cv.contourArea(contour):
        print("轮廓面积过小")
        return None

    # 计算轮廓的边界框
    x, y, w, h = cv.boundingRect(contour)
    
    
       
# 裁剪目标，返回一个包含目标的 w=h //64 的crop 
def cut_squareWHx(opencvImage, xmin, ymin, xmax, ymax, x_2x=64, half_out = True):
    ih, iw, _ = opencvImage.shape
    
    # square crop  ->  square_crop_x1, square_c1rop_y1, square_crop_x2, square_crop_y2
    top_rest = ymin
    left_rest = xmin
    bot_rest = ih - ymax
    right_rest = iw - xmax 
    if ymax-ymin > xmax-xmin:
        # 宽度方向
        expand_x_ = (ymax-ymin) - (xmax-xmin)
        # print("expand x:", expand_x_)
        if left_rest + right_rest < expand_x_:
            print("没有成功crop, obj is too big, image no enough margin --square --x")
        else:
            xmin_s = max(0, xmin-(expand_x_//2))
            xmax_s = min(iw, xmax+(expand_x_-(xmin-xmin_s)))
            xmin = xmin_s
            xmax = xmax_s
    else:
        # ymax-ymin <= xmax-xmin
        # 高度方向
        expand_y_ = (xmax-xmin) - (ymax-ymin)
        # print("expand y:", expand_y_)
        if top_rest + bot_rest < expand_y_:
            print("没有成功crop, obj is too big, image no enough margin --square --y")
        else:
            ymin_s = max(0, ymin-(expand_y_//2))
            ymax_s = min(ih, ymax+(expand_y_-(ymin-ymin_s)))
            ymin = ymin_s
            ymax = ymax_s
    
    # print("square:", xmax-xmin, ymax-ymin)
    # expand ->   final_crop_x1, final_c1rop_y1, final_crop_x2, final_crop_y2  
    expand = x_2x   # expand % 64 = 0
    top_rest_n = ymin
    left_rest_n = xmin
    bot_rest_n = ih - ymax
    right_rest_n = iw - xmax

    expand_xn_ = ((((xmax-xmin)//expand)+2) * expand) - (xmax-xmin)
    expand_yn_ = ((((ymax-ymin)//expand)+2) * expand) - (ymax-ymin)
    assert expand_xn_ == expand_yn_, f"square not work, why?"
    if expand_xn_ > (left_rest_n+right_rest_n):
        print("没有成功crop, obj is too big, image no enough margin --expand --x")
    else:
        n_x1 = max(0, xmin-(expand_xn_//2))
        n_x2 = min(iw, xmax+(expand_xn_-(xmin-n_x1)))
        
    if expand_yn_ > (top_rest_n+bot_rest_n):
        print("没有成功crop, obj is too big, image no enough margin --expand --y")
    else:
        n_y1 = max(0, ymin-(expand_yn_//2))
        n_y2 = min(ih, ymax+(expand_yn_-(ymin-n_y1)))
    # print("expand x y:", expand_xn_, expand_yn_)
    # print("final:", n_x2-n_x1, n_y2-n_y1)
    imgray = opencvImage[n_y1:n_y2, n_x1:n_x2]
    # ----------------------------------------------------------------------------------------

    p_ = Image.fromarray(imgray)
    p_.show()
    # # ----0809-----------show imgray----------------
    # cv.namedWindow("cut-out img", cv.WINDOW_NORMAL)
    # cv.resizeWindow("cut-out img", 1024, 800)
    # cv.imshow("cut-out img", imgray)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # # ----------------------------------------------
    if half_out:
        imgray = cv.resize(imgray, (imgray.shape[1]//2, imgray.shape[0]//2), interpolation=cv.INTER_AREA)
    return imgray


# 
def bin_thre(imgpath):
    g_ = cv.imread(imgpath, cv.IMREAD_GRAYSCALE)
    bimg, bin_thre = cv.threshold(g_, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)
    return bin_thre

def cutimg(inputpic, half_out=True):  #从大图中截出小图
    # 从大图中截出小图, 小图要求：w=h, w%64=0 and h%64=0
    img0=cv.imread(inputpic)
    imgray = cv.imread(inputpic,0)
    ret, imgray = cv.threshold(imgray, bin_thre(inputpic), 255, 0)
    imgc = cv.Canny(imgray, 100, 150)
    imgc = cv.GaussianBlur(imgc, (7, 7), 0)
    contours, _ = cv.findContours(imgc,  cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    if len(contours)<1:
        print("未找到要检测的圆环")
        return 0
    contours=list(contours)
    contours.sort(key=lambda x: cv.contourArea(x), reverse=True)
    if myArea(contours[0]) * 0.2 > cv.contourArea(contours[0]):
        print("未找到要检测的圆环")
        return 0
    #cv.drawContours(imgc, contours, 0, (255, 255, 0), 100)
    listx1 = contours[0].reshape(-1, 2)[:, 0]
    listy1 = contours[0].reshape(-1, 2)[:, 1]

    ih, iw, _ = img0.shape
    xmin, ymin, xmax, ymax = min(listx1), min(listy1), max(listx1), max(listy1)










