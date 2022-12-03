# this file take the path the ADNI dataset then change the rename every file to it image id 
import os

path = "/home/data/nomask_addiction"
for parent, dirnames, filenames in os.walk(path):
    print(filenames)
    for filename in filenames:
        if len(filenames)==1:
            filename0 = filenames[0]
            print(filename0)
            try:
                if filename0.endswith('.nii'):
                    old_name = filename0.replace('.nii','')
                    new_name = old_name.split('_')[-1]+'.nii'
                    print(new_name)
                    print("####")
                    os.rename(os.path.join(parent, filename0), os.path.join("/home/data/nomask_addiction_rename", new_name))
            except:
                pass


#改完名字的数据都到/home/data/743snpwithmri_rename