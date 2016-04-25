import glob,os
import SIFT_match2
import cv2
from PIL import Image


if __name__=="__main__":

    img1 = cv2.imread('kyt-gsi-4.png')
    path = r'./satellite/'
    fp = open('./matchresult.txt','w')
    for files in glob.glob('./satellite/*.png'):
        filepath,filename = os.path.split(files)
        filterame,exts = os.path.splitext(filename)

        opfile = './'

        if (os.path.isdir(opfile)==False):
            os.mkdir(opfile)
        img2 = cv2.imread(files)
        #for filename in os.listdir(path):
        print path+filename
        match_time,match_score,correctness = SIFT_match2.matchfunction(img1,img2)
        fp.write(filename+'   '+str(match_time)+'    '+str(match_score)+'    '+str(correctness)+'%'+'\n')

    fp.close()