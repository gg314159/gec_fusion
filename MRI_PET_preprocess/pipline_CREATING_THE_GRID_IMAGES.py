import os
import numpy as np
import SimpleITK as sitk
from nipype.interfaces import fsl
from datetime import datetime
from skimage.io import imread,imsave
import pandas as pd
import cv2 as cv
import cv2
from scipy.stats import entropy
from dltk.io.preprocessing import whitening



SKS_MASK_DIR = '/home/data/gecpet/SKULL_STRIPPING_MASK_1'
AFFINE_REG_SKS_MASK_2D_IMAGES_DIR = '/home/data/gecpet/AFFINE_REG_SKS_MASK_2D_1'


start = 50
end = 150
nb_img = 16


for indx , file in enumerate(os.listdir(SKS_MASK_DIR)):
    
    image = sitk.ReadImage(SKS_MASK_DIR+'/'+file)

    array = sitk.GetArrayFromImage(sitk.ReadImage(SKS_MASK_DIR+'/'+file))
    array = np.interp(array, (array.min(), array.max()), (0, 255))

    graid_image = np.array([])
    data = np.array([])
    entpy_data = {}

    for i in range(start,end):
        value,counts = np.unique(array[i,:,:], return_counts=True)
        entpy_data[i] = entropy(counts, base=2)
    entpy_data = {k: v for k, v in sorted(entpy_data.items(),reverse=True, key=lambda item: item[1])}
    index_of_slices = list(entpy_data.keys())[0:nb_img]

    
    for i , max_indx in enumerate(index_of_slices):
        if (i+1) % 4 == 0:
            data = np.hstack((data,array[max_indx,:,:]))
            if graid_image.size < 1:
                graid_image = data.copy()
            else:
                graid_image = np.vstack((graid_image,data))
            data = np.array([])
            
        else:
            if data.size < 1:
                data = array[max_indx,:,:]
            else:
                data = np.hstack((data,array[max_indx,:,:]))
    imsave(f'{AFFINE_REG_SKS_MASK_2D_IMAGES_DIR}/{file.replace(".nii.gz","")}.png',cv.equalizeHist(np.uint8(graid_image)))
    #imsave(f'{AFFINE_REG_SKS_MASK_2D_IMAGES_DIR}/{file.replace(".nii.gz","")}.png',np.uint8(graid_image))
    print(indx,"已完成")





# #单张图片

# SKS_MASK_DIR = '/home/data/gec_1045mri_affine/SKULL_STRIPPING_MASK'
# AFFINE_REG_SKS_MASK_2D_IMAGES_DIR = '/home/data/gec_1045mri_affine/AFFINE_REG_SKS_MASK_2D'

# padd = 5
# start = 25
# end = 70
# nb_img = 16



    
# image = sitk.ReadImage('/home/data/gec_1045mri_affine/SKULL_STRIPPING_MASK/I119262.nii.gz')

# array = sitk.GetArrayFromImage(sitk.ReadImage('/home/data/gec_1045mri_affine/SKULL_STRIPPING_MASK/I119262.nii.gz'))
# array = np.interp(array, (array.min(), array.max()), (0, 255))

# graid_image = np.array([])
# data = np.array([])
# entpy_data = {}

# for i in range(start,end):
#     value,counts = np.unique(array[i,:,:], return_counts=True)
#     entpy_data[i] = entropy(counts, base=2)
# entpy_data = {k: v for k, v in sorted(entpy_data.items(),reverse=True, key=lambda item: item[1])}
# index_of_slices = list(entpy_data.keys())[0:nb_img]


# for i , max_indx in enumerate(index_of_slices):
#     if (i+1) % 4 == 0:
#         data = np.hstack((data,array[max_indx,:,:]))
#         if graid_image.size < 1:
#             graid_image = data.copy()
#         else:
#             graid_image = np.vstack((graid_image,data))
#         data = np.array([])
        
#     else:
#         if data.size < 1:
#             data = array[max_indx,:,:]
#         else:
#             data = np.hstack((data,array[max_indx,:,:]))
# imsave(f'/home/data/gec_1045mri_affine/AFFINE_REG_SKS_MASK_2D/I119262.png',cv.equalizeHist(np.uint8(graid_image)))
# print("已完成")