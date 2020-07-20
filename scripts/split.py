import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data/for_users', help='path to the data(aim_inpainting_task1.zip)')
parser.add_argument('--img_dir', type=str, default='./datasets/test_img', help='path to the img dictionatry')
parser.add_argument('--mask_dir', type=str, default='./datasets/test_mask', help='path to the mask dictionatry')
args = parser.parse_args()


img = os.listdir(args.data_dir)

for fileNum in img:
    if fileNum.find('with_holes') != -1:
        imgPath = os.path.join(args.data_dir,fileNum)
        shutil.copy(imgPath,args.img_dir)
print('finish image dir')

for fileNum in img:
    if fileNum.find('mask') != -1:
        imgPath = os.path.join(args.data_dir,fileNum)
        shutil.copy(imgPath,args.mask_dir)
print('finish mask dir')

