import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from functools import wraps
import os

'''def fn_timer(function):
  @wraps(function)

  def function_timer(*args, **kwargs):
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    print ("Total time running %s: %s seconds" %
        (function.func_name, str(t1-t0))
        )
    return result

  return function_timer,m

@fn_timer'''
def sift_detect(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
 # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print "min(n1,n2):%d" % min(len(kp1),len(kp2))
# BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    return kp1,kp2,matches

def good_matches(matches,img1,img2,kp1,kp2):
    MIN_MATCH_COUNT = 5
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)
    print "correspondences:%d" % len(good)
    if  len(good)<MIN_MATCH_COUNT:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        img3 = img2
        matchesMask = None
        flag = 0
        tmp = 0
    else:
        flag = 1
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,4.0)
        tmp = 0
        for i in mask:
            if i[0]==1:
                tmp+=1
        print "correct matches:%d" % tmp
        matchesMask = mask.ravel().tolist()
        h,w, = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img3 = cv2.polylines(img2,[np.int32(dst)],True,255,10, cv2.LINE_AA)
        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    return good,matchesMask,img3,tmp,flag

def drawmatches(matchesMask,img1,img2,kp1,kp2,good):
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    cv2.imwrite('diffrrrsift1817.png',img3)
    #plt.imshow(img3, 'gray'),plt.show()

def matchfunction(img,img2):
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1,kp2,matches = sift_detect(gray,gray2)
    good,matchesMask,img3,tmp,flag = good_matches(matches,gray,img2,kp1,kp2)
    t1 = time.time()
    if flag == 1:
        #drawmatches(matchesMask,img,img3,kp1,kp2,good)
        match_score = float(tmp)/float(len(matches))
        correctness = float(tmp)/float(len(good))*100
        match_time = str(t1-t0)
    else:
        match_score = 0
        correctness = 0
        match_time = 0
    return match_time,match_score,correctness




'''img = cv2.imread('kyt-gsi-4.png')
img2 = cv2.imread('./satellite/35.014-135.75-17-satellite.png')
out1 = matchfunction(img,img2)'''


'''img3 = cv2.imread('./image/35.6954424-139.7009008-17-satellite-180.png')
img4 = cv2.imread('./image/shinjuku3.jpg')
matchfunction(img3,img4)'''


















