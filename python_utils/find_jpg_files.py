import os, fnmatch

# importing shutil module
import shutil

from random import sample

def find_files(directory, pattern):
    file_list = []
    for root, dirs, files in os.walk(directory):
        #print (root)
        for basename in files:
            #print (basename)
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                #print (filename)
                file_list.append(filename)
                #yield filename

    return file_list


# image directory path
img_dir = '101_ObjectCategories' 

#for filename in find_files(img_dir, '*.jpg'):
#    print ('Found JPG file:', filename)

# find img files
img_files = find_files(img_dir, '*.jpg')
print (len(img_files))

sample_list = sample(img_files, 1500)
#print(sample_list)

src_path = img_dir
train_dst_path = "kau_aircraft/temp/train/no_planes"
test_dst_path = "kau_aircraft/temp/test/no_planes"

idx = 0
for x in sample_list:

    # cope files
    src_f = x
    y = x.split('/')
    dst_f = y[-2] + '_' + y[-1]
    if (idx < 1000):
        dst_f = os.path.join(train_dst_path, dst_f)
    else:
        dst_f = os.path.join(test_dst_path, dst_f)

    print(idx, ":", src_f, dst_f)
    shutil.copyfile(src_f, dst_f) 
    idx += 1

print ("done!")
