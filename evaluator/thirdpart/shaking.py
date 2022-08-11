from header import *

import cv2
import numpy as np
import math
import sys
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

def get_penalty(kp1, des1, kp2, des2, half_diag, h, w, d):

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1,des2,k=2)
    except cv2.error as e:
        return -1 #Max penalty if could not estimate homography (will be applied at the end)

    # bfmatcher = cv2.BFMatcher()
    # matches = bfmatcher.knnMatch(des1,des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        pts = np.float32([ [0,0],[0,h-1],[w/2-1,h/2-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        if M is not None:
            dst = cv2.perspectiveTransform(pts,M)
            penalty = math.sqrt(math.pow(dst[2,0,0]-pts[2,0,0],2)+math.pow(dst[2,0,1]-pts[2,0,1],2))/half_diag #Amount of center point shift
        else:
            penalty = -1 #Max penalty if cannot estimate homography (will be applied at the end)
    else:
        #print( 'Not enough matches are found - {}/{}'.format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
        penalty = -1 #Max penalty if could not find enough matches (will be applied at the end)

    return min(penalty, 1)

def calc_video_shaking_p(i,frame_list_i):

    i1,i2,k,n = i[0],i[1],i[2],i[3]
    
    print("  Init part "+str(k)+"/"+str(n))

    frame_list = frame_list_i[i1:min(i2+1,len(frame_list_i))]
    
    #video = cv2.VideoCapture(video_fnam)

    #if not video.isOpened():
    #    print ('Could not open the video {}'.format(video_fnam))
    #    exit()
    
    #num_frames = int(video.get(7))

    #print('Video loaded! It contains {} frames'.format(num_frames))

    penalties = []

    num_frames = len(frame_list)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    prev_frame = cv2.imread(frame_list[0])
    h,w,d = prev_frame.shape
    half_diag = math.sqrt(h**2+w**2)/2
    kp1, des1 = sift.detectAndCompute(prev_frame,None)
    #print ('{} Calculating stability penalties...'.format(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')))
    if num_frames: #Video has frames!
        for frame_idx in tqdm(range(1,num_frames), desc='Calculating Shaking Ratio'):
            curr_frame = cv2.imread(frame_list[frame_idx])
               
            # find the keypoints and descriptors with SIFT
            kp2, des2 = sift.detectAndCompute(curr_frame,None)

            if des1 is None or des2 is None:
                penalty = -1 #Max penalty since it could not find descriptors (will be applied at the end)
            else:
                penalty = get_penalty(kp1, des1, kp2, des2, half_diag, h, w, d)

            penalties.append(penalty)

            #Updating
            kp1, des1 = kp2, des2
            prev_frame = curr_frame

    else: #Let's try another path (Counter could not get :/)
        for frame_idx in tqdm(range(len(frame_list)), desc='Calculating Shaking Ratio'):
            curr_frame = cv2.imread(frame_list[frame_idx])

            # find the keypoints and descriptors with SIFT
            kp2, des2 = sift.detectAndCompute(curr_frame,None)

            if des1 is None or des2 is None:
                penalty = -1 #Max penalty since it could not find descriptors (will be applied at the end)
            else:
                penalty = get_penalty(kp1, des1, kp2, des2, half_diag, h, w, d)

            penalties.append(penalty)

            #Updating
            kp1, des1 = kp2, des2
            prev_frame = curr_frame
            frame_idx += 1

    #video.release()
    
    penalties = np.array(penalties)
    penalties[penalties==-1] = np.max(penalties) #Apply max penalty to failure estimations
    num_max_penalties = len(np.where(penalties == np.max(penalties))[0])
    avg_penalty = np.sum(penalties)/float(len(penalties))

    #np.savetxt('last_penalties.txt', penalties)
    
    #print ('\n===================== SUMMARY =====================')
    #print ('\t Video Filename: {}'.format(video_fnam))
    #print ('\t Average Stability Penalty: {} '.format(avg_penalty))
    #print ('\t Number of Max Penalties Applied: {} - Max Penalty: {}'.format(num_max_penalties, np.max(penalties)))
    #print ('===================================================')
    #print ('{} Done!'.format(datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')))

    print("  Done part "+str(k)+"/"+str(n))

    return avg_penalty

def calc_video_shaking(frame_list):

    njobs = num_cores
    ljobs = utils.get_ljobs(0,len(frame_list),njobs)
    louts = Parallel(n_jobs=njobs)(
            delayed(calc_video_shaking_p)(i,frame_list)
                for i in ljobs)
    
    avg_penalties = []
    for lout in louts:
        avg_penalties.append(lout)
    
    avg_penalty = np.mean(avg_penalties)

    return avg_penalty
    
