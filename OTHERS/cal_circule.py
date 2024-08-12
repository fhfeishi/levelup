# coding=utf-8
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 多个圆返回 -1 没有圆返回 0 其他返回圆半径
def calurate_circle_diameter2(img_path, need_show=False):
    img = cv.imread(img_path)
    # cv.imshow("input img", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # epf边缘滤波
    temp = cv.pyrMeanShiftFiltering(img, 2, 30)
    # cv.imshow("epf", temp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 灰度处理
    img_gray = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)

    # gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # thresh,result1=cv.threshold(gray,40,255,cv.THRESH_BINARY)
    # img_gray = result1
    # cv.imshow("gray", img_gray)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 霍夫梯度法识别图像中的圆形
    # @param dp 图像分辨率与累加器分辨率之比. 举个例子, 若`dp=1`, 则累加器与输入图片有相同的分辨率, 若`dp=2`，累加器的宽度和高度只有其一半.
    # @param minDist 被检测到的圆心的最小距离. 如果该参数很小, 除了一个正确的圆之外, 该圆的邻居也可能被错误地检测出来. 如果该参数很大, 一些圆将可能被错过
    # @param param1 第一个指定方法参数的参数. 在HOUGH_GRADIENT的情况下，其为传递给Canny边缘检测的两个阈值中 较高的阈值(较小的阈值为1/2)
    # @param param2 第二个指定方法参数的参数. 在HOUGH_GRADIENT的情况下, 它是检测阶段圆心的累加器阈值. 该值越小, 越多错误的圆将被检测出来. 在投票中获得高票的圆将被先返回.
    # @param minRadius 最小圆圈半径. 
    # @param maxRadius 最大的圆圈半径. 如果该值等于0, 则使用图像的最大尺寸. 如果小于0, 则返回圆心位置，而不去返回找到的半径

    circles = cv.HoughCircles(img_gray,
                              cv.HOUGH_GRADIENT,
                              1.2,
                              100,
                              param1=120,
                              param2=120,
                              minRadius=100,
                              maxRadius=3000)


    if circles is None :
        print(" no circles ")
        if need_show:
            img = cv.resize(img, (0,0), fx = 0.25, fy = 0.25)
            cv.imshow("res", img)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return 0,0
    circles = circles.reshape(-1, 3)                          
    circles = np.uint16(np.around(circles))  

    print(img_gray.shape, circles.shape, circles)
    # circles[i] 表示一个圆的参数 (x, y, radius)
    for i in circles:
        print(i[2])
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 10)  # 圆
        cv.circle(img, (i[0], i[1]), 10, (255, 255, 0), 10)  # 圆心

    if need_show:
        img = cv.resize(img, (0,0), fx = 0.25, fy = 0.25)
        cv.imshow("res", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    radius = 0
    if len(circles) >= 1:
        radius = circles[0][2]
    
    return len(circles),radius
    
def myArea(cnt): #传入一个轮廓
    rect = cv.minAreaRect(cnt)  #最小外接矩形
    box = cv.boxPoints(rect)
    box = np.int0(box)
    return cv.contourArea(box)

# 多个圆返回 -1 没有圆返回 0 其他返回圆半径
def calurate_circle_diameter(img_path, need_show=False):
    '''
    retval, dst = cv2.threshold( src, thresh, maxval, type[, dst] )
    retval：返回计算后的阈值
    src：原图像，单通道矩阵，CV_8U&CV_32F
    dst：结果图像
    thresh：当前阈值
    maxVal：最大阈值，一般为255
    TRIANGLE阈值处理
    '''
    src = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    triThe = 0
    maxval = 255
    #triThe, dst_tri = cv2.threshold(src, triThe, maxval, cv2.THRESH_TRIANGLE + cv2.THRESH_BINARY)
    bin, imgray = cv.threshold(src, triThe, maxval, cv.THRESH_OTSU + cv.THRESH_BINARY)

    img = cv.imread(img_path)
    #imgray = cv.imread(img_path,0)
    #ret, imgray = cv.threshold(imgray, bin, 255, 0)
    imgc = cv.Canny(imgray, 100, 150)
    imgc = cv.GaussianBlur(imgc, (3, 3), 0)
    contours, _ = cv.findContours(imgc,  cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    if len(contours)<1:
        print("未找到要检测的圆环")
        return 0,0
    contours=list(contours)
    contours.sort(key=lambda x: cv.contourArea(x), reverse=True)
    if myArea(contours[0]) * 0.2 > cv.contourArea(contours[0]):
        print("未找到要检测的圆环")
        return 0,0
    _,(d_x,d_y),_ = cv.minAreaRect(contours[0])
    # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    d_xy=(d_x+d_y)/2.0

    if need_show:
        cv.drawContours(img, contours, 0, (255, 255, 0), 5)
        plt.imsave("./calibrated.jpg", img)
    
    return 1,d_xy

import glob
if __name__ == "__main__":
    # f = "/home/nvidia/new/projects/build-MeasureTool-Desktop-Release/calibrate.jpg"
    f = "/home/nvidia/new/projects/python_generate_document/calibrated.jpg"
    d = calurate_circle_radius(f, True) 

    file_list = glob.glob("./examples/*.jpg") 
    for f in file_list:
        print("img path ", f)
        # r = calurate_circle_diameter(f, True)   
        print("diameter is ", d)









