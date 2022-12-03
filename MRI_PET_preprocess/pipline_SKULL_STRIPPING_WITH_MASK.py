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

REG_DIR = '/home/data/gecpet/AFFINE_REGISTRATION_1'
SKS_MASK_DIR = '/home/data/gecpet/SKULL_STRIPPING_MASK_1'
MNI152_MASK_PATH = '/home/gc/fsl/data/standard/MNI152_T1_1mm_brain_mask.nii.gz'




size = len(os.listdir(SKS_MASK_DIR))

mask = fsl.ApplyMask()
mask.inputs.output_type = "NIFTI_GZ"
mask.inputs.mask_file = MNI152_MASK_PATH




for index,file in enumerate(os.listdir(REG_DIR)):
    # init path varibles
    print(file)
    sk_input_file_path = f'{REG_DIR}/{file}'
    sk_output_file_path = f'{SKS_MASK_DIR}/{file}'
    

    start = datetime.now()
    mask.inputs.in_file = sk_input_file_path
    mask.inputs.out_file = sk_output_file_path
    mask.run()
    end = datetime.now()-start

    print(f'{index+1} out of {size} took {end}')
    
    
    
    
# # 单个图片
# REG_DIR = '/home/data/gec_1045mri_affine/AFFINE_REGISTRATION'
# SKS_MASK_DIR = '/home/data/gec_1045mri_affine/SKULL_STRIPPING_MASK'
# MNI152_MASK_PATH = '/home/gc/gechang/PFE-alzheimer-disease-classification/Application/atlas/MNI_152/MNI152lin_T1_2mm_brain_mask.nii.gz'




# mask = fsl.ApplyMask()
# mask.inputs.output_type = "NIFTI_GZ"
# mask.inputs.mask_file = MNI152_MASK_PATH




# sk_input_file_path = f'/home/data/gec_1045mri_affine/AFFINE_REGISTRATION/I119262.nii.gz'
# sk_output_file_path = f'/home/data/gec_1045mri_affine/SKULL_STRIPPING_MASK/I119262.nii.gz'


# # start = datetime.now()
# mask.inputs.in_file = sk_input_file_path
# mask.inputs.out_file = sk_output_file_path
# mask.run()
# # end = datetime.now()-start

# # print(f'{index+1} out of {size} took {end}')