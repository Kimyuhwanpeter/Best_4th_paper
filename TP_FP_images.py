# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


color_map = np.array([[255, 0, 0], 
                      [0, 0, 255], 
                      [0,0,0], 
                      [255, 212, 0], 
                      [0, 255, 0],
                      [139, 0, 255],
                      [255, 127, 0],
                      [255, 255, 255]], dtype=np.uint8)
def main():
    #Crop에 대한 TP - 빨강 0
    #Weed에 대한 TP - 파랑 1
    #background 에 대한 TP - 검은색 2

    #Crop 에 대한 FP (background인데 crop으로 잘못판단)- 노랑 3
    #Crop 에 대한 FP (weed인데 crop로 잘못판단) - 초록색 4
    #Weed에 대한 FP (background인데 weed로 잘못판단)- 보라색 5
    #Weed에 대한 FP (crop인데 weed으로 잘못판단) -주황색 6

    #background에 대한 FP (crop 혹은 weed 인데 background로 잘못판단) - 흰색 7

    lab_img = tf.io.read_file("C:/Users/Yuhwan/Downloads/test_images/label.png")
    lab_img = tf.image.decode_png(lab_img, 3)

    img = tf.io.read_file("C:/Users/Yuhwan/Downloads/test_images/predict.png")
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize(img, (lab_img.shape[0], lab_img.shape[1]), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    func1 = lambda r: r[:, :, 0] == 255
    func2 = lambda g: g[:, :, 1] == 0
    func3 = lambda b: b[:, :, 2] == 0

    func4 = lambda r: r[:, :, 0] == 0
    func5 = lambda g: g[:, :, 1] == 0
    func6 = lambda b: b[:, :, 2] == 255

    func7 = lambda r: r[:, :, 0] == 0
    func8 = lambda g: g[:, :, 1] == 0
    func9 = lambda b: b[:, :, 2] == 0

    gray_lab_img = tf.where(func1(lab_img) & func2(lab_img) & func3(lab_img), 255, 0)
    gray_lab_img = tf.where(func4(lab_img) & func5(lab_img) & func6(lab_img), 128, gray_lab_img)
    gray_lab_img = np.where(gray_lab_img == 0, 2, gray_lab_img)
    gray_lab_img = np.where(gray_lab_img == 255, 0, gray_lab_img)
    gray_lab_img = np.where(gray_lab_img == 128, 1, gray_lab_img)

    gray_img = tf.where(func1(img) & func2(img) & func3(img), 255, 0)
    gray_img = tf.where(func4(img) & func5(img) & func6(img), 128, gray_img)
    gray_img = np.where(gray_img == 0, 2, gray_img)
    gray_img = np.where(gray_img == 255, 0, gray_img)
    gray_img = np.where(gray_img == 128, 1, gray_img)

    temp_img = gray_img

    func_crop = lambda x: x[:] == 0
    func_weed = lambda x: x[:] == 1
    func_back = lambda x: x[:] == 2

    temp_img = tf.where(func_back(gray_lab_img) & func_crop(gray_img), 3, gray_img)
    temp_img = tf.where(func_weed(gray_lab_img) & func_crop(gray_img), 4, temp_img)
    temp_img = tf.where(func_back(gray_lab_img) & func_weed(gray_img), 5, temp_img)
    temp_img = tf.where(func_crop(gray_lab_img) & func_weed(gray_img), 6, temp_img)

    temp_img = tf.where(func_crop(gray_lab_img) & func_back(gray_img), 7, temp_img)
    temp_img = tf.where(func_weed(gray_lab_img) & func_back(gray_img), 7, temp_img)

    RGB_temp_img = color_map[temp_img]
    plt.imsave("C:/Users/Yuhwan/Downloads/test_images/CED_Net.png", RGB_temp_img / 255)
    plt.imshow(RGB_temp_img / 255)
    plt.show()

    #plt.imshow(gray_lab_img.numpy() / 255, cmap="gray")
    #plt.show()
    #plt.imshow(gray_img.numpy() / 255, cmap="gray")
    #plt.show()


if __name__ == "__main__":
    main()
