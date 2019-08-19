#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility function for rendering PLY objects into videos for model.
Author: Ruochenliu
Date: June 2019
"""

import os
from os import listdir
from os.path import isfile, join
import math
import psutil
import numpy as np
import open3d
import cv2

def get_one_hot(targets, n_classes):
    res = np.eye(n_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[n_classes])

def get_memory_used():
    process = psutil.Process(os.getpid())
    util = np.round(process.memory_info().rss/1024/1024/1024, 2)
    return util

def cos(a):
    return math.cos(math.radians(a))
def sin(a):
    return math.sin(math.radians(a))

def stand(img):
    return (img - np.mean(img))/np.std(img)

def get_rotation_matrix(a, k):
    if k == 2:
        t = np.asarray([[cos(a), sin(a), 0, 0], [-sin(a), cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif k == 1:
        t = np.asarray([[-sin(a), cos(a), 0, 0], [0, 0, 1, 0], [cos(a), sin(a), 0, 0], [0, 0, 0, 1]])
    else:
        t = np.asarray([[0, 0, 1, 0], [cos(a), sin(a), 0, 0], [-sin(a), cos(a), 0, 0], [0, 0, 0, 1]])
    return t

def get_random_rotation():
    x = np.random.randint(0, 360, 1)[0]
    y = np.random.randint(0, 360, 1)[0]
    z = np.random.randint(0, 360, 1)[0]
    x_m = get_rotation_matrix(x, 0)
    y_m = get_rotation_matrix(y, 1)
    z_m = get_rotation_matrix(z, 2)
    t = x_m.dot(y_m.dot(z_m))
    return t

def get_3d(mesh_dir, size=(128, 128), axis=1, rand=None):
    m = open3d.io.read_triangle_mesh(mesh_dir)
    m.compute_vertex_normals()
    
    images = []
    
    for x in range(0, 360, 30):
        m2 = open3d.geometry.TriangleMesh(m)
        t_matrix = get_rotation_matrix(x, axis)
        if len(rand) == 4:
            m2.transform(rand)
        m2.transform(t_matrix)

        vis = open3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=640, height=640, left=5, top=5, visible=False)
        vis.add_geometry(m2)
        #vis.run()
        #vis.update_geometry()
        vis.poll_events()
        #vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer())
        image = cv2.resize(image, size)
        gray = np.reshape(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (size[0], size[1], 1))
        images.append(gray)
        vis.destroy_window()
    
    return images

# 0 for train, 1 for test; 0 for x axis, 1 for y axis, 2 for z axis
def prepare_data(train_test=2, axis=0, rand=False):
    cat_list = [cat for cat in listdir("../data/ModelNet10/PLY/") if "." not in cat]
    data_type = ["/train/", "/test/"]
    if train_test == 2:
        types = data_type
    else:
        types = [data_type[train_test]]
    
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    size = (128, 128)
    n_conv = 0
    n_all = [3991, 908, 4899]
    mm_all = 10
    
    if rand:
        rand_matrix = get_random_rotation()
    else:
        rand_matrix = [1]

    for i in range(len(cat_list)):
        for use in types:
            data_dir = "../data/ModelNet10/PLY/" + cat_list[i] + use
            file_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and ".ply" in f]
            for file_name in file_list:
                file = data_dir + file_name
                new = get_3d(file, size, axis, rand_matrix)
                if use == "/train/":
                    train_X.append(new)
                    train_y.append(i)
                else:
                    test_X.append(new)
                    test_y.append(i)
                mm = get_memory_used()
                if mm >= 10:
                    print("Not Enough Memory")
                    return 1
                n_conv += 1
                print("3D Polygon processed: {0:.2f}% == Memory used: {1:.2f}% == N{2}".format(n_conv/n_all[train_test]*100, mm/mm_all*100, n_conv), end = "\r")
                
    if train_test == 0: 
        np.save("../data/new/X_train", np.asarray(train_X))
        np.save("../data/new/y_train", get_one_hot(np.asarray(train_y), 10))
    elif train_test == 1:
        np.save("../data/new/X_test", np.asarray(test_X))    
        np.save("../data/new/y_test", get_one_hot(np.asarray(test_y), 10))
    else:
        np.save("../data/new/X_train", np.asarray(train_X))
        np.save("../data/new/y_train", get_one_hot(np.asarray(train_y), 10))
        np.save("../data/new/X_test", np.asarray(test_X))    
        np.save("../data/new/y_test", get_one_hot(np.asarray(test_y), 10))
    
    print("Finished")
    return 0

def get_video(img, pathOut="video.mp4", size=(128, 128), fps=12):
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img)):
        out.write(np.uint8(img[i]*255))
    out.release()

def main():
    prepare_data(2, 1, False)

if __name__ == "__main__":
    main()