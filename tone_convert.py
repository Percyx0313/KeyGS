import cv2
import numpy as np

def match_color(source_img, target_img):
    source_img = cv2.imread(source_img)
    target_img = cv2.imread(target_img)
    
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab).astype(np.float32)
    target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # Compute mean and standard deviation for each channel
    src_mean, src_std = cv2.meanStdDev(source_lab)
    tgt_mean, tgt_std = cv2.meanStdDev(target_lab)
    src_mean, src_std = src_mean.astype(np.float32), src_std.astype(np.float32)
    tgt_mean, tgt_std = tgt_mean.astype(np.float32), tgt_std.astype(np.float32) 
    
    # Adjust the color of the source image to match the target image
    # matched_img = (source_lab - src_mean.reshape(1,1,3)) * (tgt_std.reshape(1,1,3) / (src_std.reshape(1,1,3))) + tgt_mean.reshape(1,1,3)
    matched_img = ((source_lab - src_mean.reshape(1,1,3))  / (src_std.reshape(1,1,3))+1)/2*255
    matched_img = np.clip(matched_img, 0, 255)
    # Convert back to BGR color space
    matched_img = cv2.cvtColor(np.uint8(matched_img), cv2.COLOR_Lab2BGR)
    
    return matched_img

for i in range(1, 202-1):
    result = match_color('dataset/CFGS_CO3D_KEY10/bench/415_57112_110099/raw_images/frame{:06d}.jpg'.format(i+1), 'dataset/CFGS_CO3D_KEY10/bench/415_57112_110099/tone_images/frame{:06d}.jpg'.format(i))
    cv2.imwrite('dataset/CFGS_CO3D_KEY10/bench/415_57112_110099/tone_images/frame{:06d}.jpg'.format(i+1), result)
