
import sys

import cv2
import numpy

cap = cv2.VideoCapture('/Users/jack/Desktop/2.MP4') #/Volumes/drive12/34.MP4

sum_matrix = None
n_frames = 0
i = 0
every = 125
max_frames = 2100
#backsub = cv2.BackgroundSubtractorMOG(500, 6, 0.9, 1)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#fgbgKNN = cv2.createBackgroundSubtractorKNN(detectShadows=False)
#fgmask = None
#gray = None
#avg1 = None
#avg2 = None
#avg3 = None
#avg4 = None
#avg5 = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if i % every == 0:
        sys.stderr.write('[INFO] Processing frame %d / %d / %d\n' % (i, n_frames, max_frames))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        #kernel = numpy.ones((9,9),numpy.uint8)
        #thresh = cv2.erode(thresh,kernel,iterations = 1)
        #gray = cv2.bitwise_and(gray, gray, mask=thresh)
        gray[gray > 80 ] = 20
        #gray = cv2.equalizeHist(gray)
        #gray = cv2.Canny(gray,30,30)
        #gray = cv2.Laplacian(gray, cv2.CV_64F, ksize = 31)
        #gray = cv2.bilateralFilter(gray, 5, 255, 5)
        #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            #cv2.THRESH_BINARY, 171, 59)
        #bilateralFilter(gray_frame,filtered_frame, 5, 255, 5);
        #gray = gray.astype(numpy.uint8)
        #sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=21)  # x
        #sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=21)  # y
        #cv2.equalizeHist(gray)
        #cv2.imwrite('high.png', gray)
        #break
        if n_frames == 0:
            #avg1 = np.float64(gray)
            #avg2 = np.float64(gray)
            #avg3 = np.float64(gray)
            #avg4 = np.float64(gray)
            #avg5 = np.float64(gray)
            #cv2.accumulateWeighted(gray,avg1,0.1)
            #cv2.accumulateWeighted(gray,avg2,0.01)
            #cv2.accumulateWeighted(gray,avg3,0.001)
            #cv2.accumulateWeighted(gray,avg4,0.0001)
            #cv2.accumulateWeighted(gray,avg5,0.5)
            #fgmask = backsub.apply(gray, None, 0.01)
            #fgmask = fgbg.apply(gray)
            #fgmaskKNN = fgbgKNN.apply(gray)
            #sum_matrix.append(gray) # .astype(numpy.float64)
            sum_matrix = gray.astype(numpy.float64)
            #cv2.imwrite('bit.png', gray)
        else:
            #cv2.accumulateWeighted(gray,avg1,0.1)
            #cv2.accumulateWeighted(gray,avg2,0.01)
            #cv2.accumulateWeighted(gray,avg3,0.001)
            #cv2.accumulateWeighted(gray,avg4,0.0001)
            #cv2.accumulateWeighted(gray,avg5,0.5)
            #fgmask = fgbg.apply(gray)
            #fgmaskKNN = fgbgKNN.apply(gray)
            #sum_matrix.append(gray)
            sum_matrix += gray.astype(numpy.float64)
        n_frames += 1
        if n_frames >= max_frames:
            sys.stderr.write('[INFO] Got enough frames (%d/%d), exiting\n' % (n_frames, max_frames))
            break
    i += 1


sum_matrix /= n_frames
#sum_matrix = numpy.array(sum_matrix)
#img = numpy.median(sum_matrix, axis=0).astype(numpy.uint8)
#img = numpy.percentile(sum_matrix, 75, axis=0).astype(numpy.uint8) #10
#print(img.dtype)
#print(img.shape)
img = sum_matrix.astype(numpy.uint8)
#img = cv2.GaussianBlur(img,(3,3),0)
'''
avg1 = avg1.astype(np.uint8)
avg2 = avg2.astype(np.uint8)
avg3 = avg3.astype(np.uint8)
avg4 = avg4.astype(np.uint8)
avg5 = avg5.astype(np.uint8)

#avg.astype(numpy.uint8)

#fgbg.getBackgroundImage(gray)
#gray = cv2.equalizeHist(gray)
avg1 = cv2.equalizeHist(avg1)
avg2 = cv2.equalizeHist(avg2)
avg3 = cv2.equalizeHist(avg3)
avg4 = cv2.equalizeHist(avg4)
avg5 = cv2.equalizeHist(avg5)

cv2.imwrite('run1.png', avg1)
cv2.imwrite('run2.png', avg2)
cv2.imwrite('run3.png', avg3)
cv2.imwrite('run4.png', avg4)
cv2.imwrite('run5.png', avg5)
'''
#img = cv2.equalizeHist(img)

clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(9,9))
clahe_img = clahe.apply(img)
cv2.imwrite('nighttime10.png', clahe_img)


# When everything done, release the capture
cap.release()
