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

#MNI152_PATH = '/home/gc/gechang/PFE-alzheimer-disease-classification/Application/atlas/MNI_152/MNI152lin_T1_2mm.nii.gz'
MNI152_PATH = '/home/gc/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
ADNI_DIR = '/home/data/gecpet/rowdata'
REG_DIR = '/home/data/gecpet/AFFINE_REGISTRATION_1'
MAT_DIR = '/home/data/gecpet/MAT'

flt = fsl.FLIRT()    
flt.inputs.output_type = "NIFTI_GZ"
flt.inputs.reference = MNI152_PATH
flt.inputs.dof = 12
size = len(os.listdir(ADNI_DIR))

for index,file in enumerate(os.listdir(ADNI_DIR)):
    # init path varibles
    reg_input_file_path = f'{ADNI_DIR}/{file}'
    reg_output_file_path = f'{REG_DIR}/{file}'
    mat_out_file_path = f'{MAT_DIR}/{file.replace(".nii","")}.mat' 
    start=datetime.now()
    flt.inputs.in_file = reg_input_file_path
    flt.inputs.out_file = reg_output_file_path
    flt.inputs.out_matrix_file = mat_out_file_path    
    res = flt.run()
    end = datetime.now()-start  
    print(f'{index+1} out of {size} took {end}')


# #单个图片
# MNI152_PATH = '/home/gc/gechang/PFE-alzheimer-disease-classification/Application/atlas/MNI_152/MNI152lin_T1_2mm.nii.gz'
# ADNI_DIR = '/home/data/ADNI1_Screening_1.5T_rename'
# REG_DIR = '/home/data/gec_1045mri_affine/AFFINE_REGISTRATION'
# MAT_DIR = '/home/data/gec_1045mri_affine/MAT'

# flt = fsl.FLIRT()    
# flt.inputs.output_type = "NIFTI_GZ"
# flt.inputs.reference = MNI152_PATH
# flt.inputs.dof = 12



# reg_input_file_path = f'/home/data/ADNI1_Screening_1.5T_rename/I119262.nii'
# reg_output_file_path = f'/home/data/gec_1045mri_affine/AFFINE_REGISTRATION/I119262.nii.gz'
# mat_out_file_path = f'/home/data/gec_1045mri_affine/MAT/I119262.mat' 
# start=datetime.now()
# flt.inputs.in_file = reg_input_file_path
# flt.inputs.out_file = reg_output_file_path
# flt.inputs.out_matrix_file = mat_out_file_path    
# res = flt.run()
# end = datetime.now()-start  
# print(f' took {end}')
