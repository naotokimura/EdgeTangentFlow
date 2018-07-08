#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

M_PI = 3.14159265358979323846
KERNEL = 5
COLOUR_OR_GRAY = 0
input_img = "{./data/input.jpg}"
SIZE = (1000, 1000, 3)

flowField = np.zeros(SIZE, dtype = np.float32)
refinedETF = np.zeros(SIZE, dtype = np.float32)
gradientMag = np.zeros(SIZE, dtype = np.float32)

####################
# Generate ETF 
####################
#memo
 #cv2.normalize(src[, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]]])：正規化
 #cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])：微分
 #cv2.magnitude(x, y[, magnitude])：2次元ベクトルの大きさ

def initial_ETF(input_img, size):
    global flowField
    global refinedETF
    global gradientMag
    
    src = cv2.imread(input_img, COLOUR_OR_GRAY)
    src_n = np.zeros(size, dtype = np.float32)
    src_n = cv2.normalize(src.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    #Generate grad_x and grad_y
    grad_x = []
    grad_y = []
    grad_x = cv2.Sobel(src_n, cv2.CV_32FC1, 1, 0, ksize=5)
    grad_y = cv2.Sobel(src_n, cv2.CV_32FC1, 0, 1, ksize=5)
    
    #Compute gradient
    gradientMag = cv2.sqrt(grad_x**2.0 + grad_y**2.0) 
    gradientMag = cv2.normalize(gradientMag.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    h,w = src.shape[0], src.shape[1]
    for i in range(h):
        for j in range(w):
            u = grad_x[i][j]
            v = grad_y[i][j]
            n = np.array([v, u, 0.0])
            cv2.normalize(np.array([v, u, 0.0]).astype('float32'), n)
            flowField[i][j] = n
    rotateFlow(flowField, flowField, 90.0)

def rotateFlow(src, dst, theta):
    theta = theta / 180.0 * M_PI;
    h,w = src.shape[0], src.shape[1]
    for i in range(h):
        for j in range(w):
            v = src[i][j]
            rx = v[0] * np.cos(theta) - v[1] * np.sin(theta)
            ry = v[1] * np.cos(theta) + v[0] * np.sin(theta)
            flowField[i][j] = [rx, ry, 0.0]

def refine_ETF(kernel):
    global flowField
    global refinedETF
    global gradientMag
    h_f,w_f = flowField.shape[0], flowField.shape[1]
    for r in range(h_f):
        for c in range(w_f):
            computeNewVector(c, r, kernel)
    flowField = copy.deepcopy(refinedETF)
    
#Paper's Eq(1)
def computeNewVector(x, y, kernel):
    global flowField
    global refinedETF
    global gradientMag
    t_cur_x = flowField[y][x]
    t_new = (0, 0, 0)
    h_r,w_r = refinedETF.shape[0], refinedETF.shape[1]
    for r in range(y - kernel, y + kernel + 1):
        for c in range(x - kernel, x + kernel + 1):
            if (r < 0 or r >= h_r or c < 0 or c >= w_r): 
                continue
            t_cur_y = flowField[r][c]
            a = np.array([x, y])
            b = np.array([c, r])
            phi = computePhi(t_cur_x, t_cur_y);
            w_s = computeWs(a, b, kernel);
            w_m = computeWm(gradientMag[y][x], gradientMag[r][c])
            w_d = computeWd(t_cur_x, t_cur_y)
            t_new += phi * t_cur_y * w_s * w_m * w_d
    n = t_new
    cv2.normalize(t_new, n)
    refinedETF[y][x] = n

#Paper's Eq(5)
def computePhi(x, y):
    if np.dot(x,y) > 0:
        return 1
    else:
        return -1
    
#Paper's Eq(2)
def computeWs(x, y, r):
    if np.linalg.norm(x-y) < r:
        return 1
    else:
        return 0

#Paper's Eq(3)
def computeWm(gradmag_x, gradmag_y):
    wm = (1 + np.tanh(gradmag_y - gradmag_x)) / 2
    return wm

#Paper's Eq(4)
def computeWd(x, y):
    return abs(x.dot(y))

#plot arrowline and save image.
def draw_arrowline(count,KERNEL):
    global flowField
    dis = cv2.imread(input_img, COLOUR_OR_GRAY)
    resolution = 10;
    h,w = dis.shape[0], dis.shape[1]
    for i in range(0,h,resolution):
        for j in range(0,w,resolution):
            v = flowField[i][j]
            p = (j, i)
            p2 = (int(j+v[1]*5), int(i+v[0]*5))
            dis = cv2.arrowedLine(dis, p, p2, (255, 0, 0), 1, 8, 0, 0.3)
    cv2.imwrite('etf_kernel' + str(KERNEL) + '_' + str(count) +'.png',dis)
    np.save('np_etf_kernel' + str(KERNEL) + '_' + str(count) +'.npy', flowField)

if __name__ == '__main__':
    initial_ETF(input_img, SIZE)
    for i in range(10):
        refine_ETF(KERNEL)
        draw_arrowline(i,KERNEL)

