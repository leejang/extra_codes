#!/usr/bin/env python
import os, glob, sys
from random import sample

# importing shutil module
import shutil

# txt file name to read
# train and val set
#txt_fname = "images_manufacturer_trainval.txt"
# test set
txt_fname = "images_manufacturer_test.txt"

# data path
src_path = "images"
# train
#dst_path = "temp/train/planes"
dst_path = "temp/test/planes"

with open(txt_fname) as f:
    lines = f.read().splitlines()
    #print(type(lines))
    #print(len(lines))

    for x in sample(lines, 1000):
        #print(x.split())
        src_f = os.path.join(src_path, x.split()[0]) + '.jpg'
        dst_f = os.path.join(dst_path, x.split()[0]) + '.jpg'
        print(src_f, dst_f)
        # cope files
        shutil.copyfile(src_f, dst_f) 

print ("done!")

