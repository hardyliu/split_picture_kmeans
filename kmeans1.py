# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:45:31 2019

@author: claireliu
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:40:38 2019

@author: hardyiu
"""
import PIL.Image as image
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
            data.append([c1, c2, c3])
    f.close()
    # 采用 Min-Max 规范化
    mm = MinMaxScaler()
  
    data = mm.fit_transform(data)
    
    
    return np.mat(data), width, height


# 加载图像，得到规范化的结果 img，以及图像尺寸
img, width, height = load_data('./meizi.jpg')

# 用 K-Means 对图像进行 2 聚类
kmeans =KMeans(n_clusters=16)
kmeans.fit(img)
label = kmeans.predict(img)
print(label.shape)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像 pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置图像灰度, 类别 0 灰度值为 255， 类别 1 灰度值为 127
        pic_mark.putpixel((x, y), int(256/(label[x][y]+1))-1)
pic_mark.save("weixin_mark.jpg", "JPEG")


from skimage import color
# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label)*255).astype(np.uint8)
label_color = label_color.transpose(1,0,2)
images = image.fromarray(label_color)
images.save('weixin_mark_color.jpg')



