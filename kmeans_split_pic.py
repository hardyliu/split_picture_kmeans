# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 18:39:33 2019

@author: hardyliu
"""

# -*- coding: utf-8 -*-
# 使用 K-means 对图像进行聚类，并显示聚类压缩后的图像
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

# 加载图像，并对数据进行规范化
def load_data(filePath):
    # 读文件
    f = open(filePath,'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    #压缩图片，超过1000会很慢
    if height>1000:
        width=int(width/2)
        height=int(height/2)
        print(width,height)
        img = img.resize((width,height))
        
    for x in range(width):
        for y in range(height):
            # 得到点 (x,y) 的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            #把范围为 0-255 的数值投射到 1-256 数值之间
            data.append([(c1+1)/256.0, (c2+1)/256.0, (c3+1)/256.0])
    f.close()
    return np.mat(data), width, height
# 加载图像，得到规范化的结果 imgData，以及图像尺寸
img, width, height = load_data('./meizi.jpg')
# 用 K-Means 对图像进行 16 聚类
kmeans =KMeans(n_clusters=16)
label = kmeans.fit_predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像 img，用来保存图像聚类压缩后的结果
img=image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        c1 = kmeans.cluster_centers_[label[x, y], 0]
        c2 = kmeans.cluster_centers_[label[x, y], 1]
        c3 = kmeans.cluster_centers_[label[x, y], 2]
        #对其进行反变换，还原出原图对应的通道值
        img.putpixel((x, y), (int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
img.save('meizi_new.jpg')
