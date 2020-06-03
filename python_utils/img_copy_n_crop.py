#!/usr/bin/env python
import os, glob, sys
from random import sample

# importing shutil module
import shutil

# Improting Image class from PIL module
from PIL import Image

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

# bbx info
bbx_info_txt = "images_box.txt"

with open(txt_fname) as f:
    lines = f.read().splitlines()
    #print(type(lines))
    #print(len(lines))

    # train
    #for x in sample(lines, 1000):
    # test
    for x in sample(lines, 500):
        #print(x.split())
        src_f = os.path.join(src_path, x.split()[0]) + '.jpg'
        dst_f = os.path.join(dst_path, x.split()[0]) + '.jpg'
        print(src_f, dst_f)
        # cope files
        #shutil.copyfile(src_f, dst_f)

        with open(bbx_info_txt) as bbx_f:
            bbx_lines = bbx_f.read().splitlines()
            for y in bbx_lines:
                bbx_info = y.split()
                if (x.split()[0] == bbx_info[0]):
                    #print(bbx_info)
                    img_bbx = list(map(int, bbx_info[1:5]))
                    src_img = Image.open(src_f)
                    width, height = src_img.size
                    #print(width, height)
                    #print (img_bbx)
                    dst_img = src_img.crop((img_bbx[0], img_bbx[1], img_bbx[2], img_bbx[3]))
                    #dst_img.save(dst_f, "JPEG")

                    # resize
                    dst_width = 300
                    wpercent = (dst_width/float(dst_img.size[0]))
                    hsize = int((float(dst_img.size[1])*float(wpercent)))
                    rsz_img = dst_img.resize((dst_width, hsize), Image.ANTIALIAS)
                    rsz_img.save(dst_f, "JPEG")

print ("done!")

