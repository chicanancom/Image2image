import cv2
import numpy as np
import glob
def get_images(path, cell_size):
    file_paths = glob.glob("*{}\*.jpg".format(path))
    result = []
    images = []
    for img_path in file_paths:
        image = cv2.imread(img_path)
        image = cv2.resize(image, (cell_size, cell_size))
        avg_color = np.mean(image, axis=(0,1))
        result.append(avg_color)
        images.append(image)
    return images, result
cell_size = 5
image = cv2.imread("aaa.jpg")
height, width, _ = image.shape
num_cols = width // cell_size
num_rows = height // cell_size
image_avg_colors = np.mean(image, axis=(0,1))
pool_images, sub_image_avg_colors = get_images("image_pool", cell_size)
output_image = np.zeros((height,width,3))
for i in range(num_cols):
    for j in range(num_rows):
        sub_image = image[j*cell_size:(j+1)*cell_size,i*cell_size:(i+1)*cell_size,:]
        avg_color = np.mean(sub_image, axis=(0,1))
        result = np.sqrt(np.sum((avg_color - sub_image_avg_colors)**2,axis=1))
        idx = np.argmin(result)
        output_image[j*cell_size:(j+1)*cell_size,i*cell_size:(i+1)*cell_size,:]=pool_images[idx]
cv2.imwrite("image.jpg", output_image)
