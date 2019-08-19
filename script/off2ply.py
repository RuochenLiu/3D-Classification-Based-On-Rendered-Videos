#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility function for converting OFF obejct to PLY format.
Author: Ruochenliu
Date: June 2019
"""

from os import listdir
from os.path import isfile, join

def off2ply(data):
    new_data = data[:-3] + "ply"
    with open(data, "r") as content_file:
        data_content = content_file.read()

    num_vertex, num_face, _ = (data_content.split("\n")[1]).split(" ")
    data_append = "\n".join(data_content.split("\n")[2:])

    header = "../data/header.txt"
    with open(header, "r") as content_file:
        header_content = content_file.read()
    header_append = header_content.replace("num_vertex", num_vertex).replace("num_face", num_face)

    ply_data = header_append + data_append
    with open(new_data, "w+") as text_file:
        text_file.write(ply_data)

def main():
    cat_list = [cat for cat in listdir("../data/ModelNet10/OFF/") if "." not in cat]
    data_type = ["/train/", "/test/"]
    n_conv = 0
    for cat in cat_list:
        for use in data_type:
            data_dir = "../data/ModelNet10/OFF/" + cat + use
            file_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and ".off" in f]
            for file_name in file_list:
                file = data_dir + file_name
                off2ply(file)
                n_conv += 1
                print("Converting OFF to PLY: {}%".format(round(n_conv/4901*100, 2)), end="\r")

if __name__ == "__main__":
    main()
