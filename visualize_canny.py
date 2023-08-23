import cv2
import numpy as np


threshold1 = 50
threshold2 = 550
alpha = 0.3

path = "3.jpg"
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1, threshold2)

# 将灰度图像转换为具有三个通道的图像
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# 归一化映射到0-1之间
normalized_image = image.astype(float) / 255
normalized_edges = edges_rgb.astype(float) / 255

# 进行叠加
result = alpha * normalized_edges + (1 - alpha) * normalized_image


# result = normalized_edges


# 还原到0-255之间
img_merge = (result * 255).astype(np.uint8)

# 调整图像大小
resized_img_merge = cv2.resize(img_merge, (1024, 768))

# 显示结果图像
cv2.imshow('Result', resized_img_merge)
cv2.waitKey(0)
cv2.destroyAllWindows()