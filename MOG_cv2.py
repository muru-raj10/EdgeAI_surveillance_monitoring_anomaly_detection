#read video file
import cv2

#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  #install opencv-contrib-python
backSub = cv2.createBackgroundSubtractorMOG2()  #best one
#fgbgBayesianSegmentation = cv2.bgsegm.createBackgroundSubtractorGMG()
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


capture = cv2.VideoCapture(cv2.samples.findFileOrKeep('video.avi'))

maskedframe = []
while True:
    ret, frame = capture.read()  #reads frames in sequence
    if frame is None:
        break

    #fgmogMask = fgbg.apply(frame)
    fgMask = backSub.apply(frame) #this is where everything happens!
    #value 127 is shadows, 0 is background, 255 is foreground
    #bayesianMask = fgbgBayesianSegmentation.apply(frame)
    #fgbgBayesianSegmentationmask = cv2.morphologyEx(bayesianMask, cv2.MORPH_OPEN, kernel)

    backtorgb = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2RGB)
    maskedframe.append(backtorgb)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)
    #cv2.imshow('Background?', backgrndimage)
    #cv2.imshow('MoG', fgmogMask)
    cv2.imshow('FG Mask', fgMask)
    #cv2.imshow('Bayesian Mask', fgbgBayesianSegmentationmask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

capture.release()
cv2.destroyAllWindows()

#gives one image of the background.
#backgrnd = backSub.getBackgroundImage()
#cv2.imshow('backgrnd',backgrnd)
#cv2.waitKey(2)


