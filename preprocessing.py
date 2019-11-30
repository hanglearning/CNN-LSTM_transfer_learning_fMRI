# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:04:04 2019

@author: Syed Ali Asif
"""

import numpy as np
import nibabel as nib
import nilearn

import warnings
warnings.filterwarnings("ignore")

seq = '001'
n_frames = 370 # some 375. some 370
n_persons = 30
frames = np.empty(( n_persons,n_frames, 80, 80, 35))
print(frames.shape)

for person in range(1, n_persons + 1):
   print(seq, person)
   if person < 10:
       # print('hi', person)
       person = '0' + str(person)
   img_path = (f'D:\\Hang_Folder\\ds\\001\\sub-{person}\\func\\sub-{person}_task-passiveimageviewing_bold.nii.gz')
   img = nilearn.image.load_img(img_path)
   for frame in range(n_frames):
       rsn=nilearn.image.index_img(img, frame)
       # rescale image to 80, 80, 35
       reshaped_img = nilearn.image.resample_img(rsn, target_affine=np.eye(4), target_shape=(80, 80, 35))
       reshaped_img.to_filename('D:\\Hang_Folder\\temp.nii.gz')
       #reshaped_img = nib.Nifti1Image(reshaped_img, affine=np.eye(4))
       reshaped_img = nib.load('D:\\Hang_Folder\\temp.nii.gz')
       frames[int(person)-1, frame, :,:,:] = np.array(reshaped_img.dataobj)

np.save(f'D:\\Hang_Folder\\ds\\001\\calorie_data_{seq}', frames)
print(frames.shape)

seq = '002'
n_frames = 180
n_persons = 17
frames = np.empty(( n_persons,n_frames * 6, 80, 80, 35))

for person in range(1, n_persons + 1):
    print(seq, person)
    if person < 10:
        person = '0' + str(person)
    img_path_run1 = (f'D:\\Hang_Folder\\ds\\002\\sub-{person}\\func\\sub-{person}_task-deterministicclassification_run-01_bold.nii.gz')
    img_path_run2 = (f'D:\\Hang_Folder\\ds\\002\\sub-{person}\\func\\sub-{person}_task-deterministicclassification_run-02_bold.nii.gz')
    img_path_run3 = (f'D:\\Hang_Folder\\ds\\002\\sub-{person}\\func\\sub-{person}_task-mixedeventrelatedprobe_run-01_bold.nii.gz')
    img_path_run4 = (f'D:\\Hang_Folder\\ds\\002\\sub-{person}\\func\\sub-{person}_task-mixedeventrelatedprobe_run-02_bold.nii.gz')
    img_path_run5 = (f'D:\\Hang_Folder\\ds\\002\\sub-{person}\\func\\sub-{person}_task-probabilisticclassification_run-01_bold.nii.gz')
    img_path_run6 = (f'D:\\Hang_Folder\\ds\\002\\sub-{person}\\func\\sub-{person}_task-probabilisticclassification_run-02_bold.nii.gz')
    imgs = []
    for i in range(1, 7):
        imgs.append(vars()[f'img_path_run{i}'])
    run = 0
    for img in imgs:
        for frame in range(n_frames):
            rsn=nilearn.image.index_img(img, frame)
            # rescale image to 80, 80, 35
            reshaped_img = nilearn.image.resample_img(rsn, target_affine=np.eye(4), target_shape=(80, 80, 35))
            reshaped_img.to_filename('D:\\Hang_Folder\\temp.nii.gz')
            #reshaped_img = nib.Nifti1Image(reshaped_img, affine=np.eye(4))
            reshaped_img = nib.load('D:\\Hang_Folder\\temp.nii.gz')
            frames[int(person)-1, run * n_frames + frame, :,:,:] = np.array(reshaped_img.dataobj)
        run += 1

np.save(f'D:\\Hang_Folder\\ds\\002\\calorie_data_{seq}', frames)
print(frames.shape)

del frames

seq = '003'
n_frames = 170
n_persons = 17
frames = np.empty(( n_persons,n_frames * 6, 80, 80, 35))

for person in range(1, n_persons + 1):
    print(seq, person)
    if person < 10:
        person = '0' + str(person)
    img_path_run1 = (f'D:\\Hang_Folder\\ds\\003\\sub-{person}\\func\\sub-{person}_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-01_bold.nii.gz')
    img_path_run2 = (f'D:\\Hang_Folder\\ds\\003\\sub-{person}\\func\\sub-{person}_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-02_bold.nii.gz')
    img_path_run3 = (f'D:\\Hang_Folder\\ds\\003\\sub-{person}\\func\\sub-{person}_task-auditoryoddballwithbuttonresponsetotargetstimuli_run-03_bold.nii.gz')
    img_path_run4 = (f'D:\\Hang_Folder\\ds\\003\\sub-{person}\\func\\sub-{person}_task-visualoddballwithbuttonresponsetotargetstimuli_run-01_bold.nii.gz')
    img_path_run5 = (f'D:\\Hang_Folder\\ds\\003\\sub-{person}\\func\\sub-{person}_task-visualoddballwithbuttonresponsetotargetstimuli_run-02_bold.nii.gz')
    img_path_run6 = (f'D:\\Hang_Folder\\ds\\003\\sub-{person}\\func\\sub-{person}_task-visualoddballwithbuttonresponsetotargetstimuli_run-03_bold.nii.gz')
    imgs = []
    for i in range(1, 7):
        imgs.append(vars()[f'img_path_run{i}'])
    run = 0
    for img in imgs:
        for frame in range(n_frames):
            rsn=nilearn.image.index_img(img, frame)
            # rescale image to 80, 80, 35
            reshaped_img = nilearn.image.resample_img(rsn, target_affine=np.eye(4), target_shape=(80, 80, 35))
            reshaped_img.to_filename('D:\\Hang_Folder\\temp.nii.gz')
            #reshaped_img = nib.Nifti1Image(reshaped_img, affine=np.eye(4))
            reshaped_img = nib.load('D:\\Hang_Folder\\temp.nii.gz')
            frames[int(person)-1, run * n_frames + frame, :,:,:] = np.array(reshaped_img.dataobj)
        run += 1

np.save(f'D:\\Hang_Folder\\ds\\003\\calorie_data_{seq}', frames)
print(frames.shape)

