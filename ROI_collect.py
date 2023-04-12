import cv2
import os
import ROI
import time
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def IncreaseContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    #result = np.hstack((img, enhanced_img))
    return enhanced_img
vid = cv2.VideoCapture(0)  
def getROI(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
                     
                imgaeResize = IncreaseContrast(frame)
                imgaeRGB = imgaeResize
                imgaeResize.flags.writeable = False
                imgaeRGB.flags.writeable = False
                imgaeRGB = imgaeResize
                results = hands.process(imgaeResize)
                # cv2.imshow("RESIZE ", imgaeResize)
                cropped_image = cv2.cvtColor(imgaeResize, cv2.COLOR_BGR2GRAY)
                h = cropped_image.shape[0]
                w = cropped_image.shape[1]
                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                        pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                        pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                        pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                        x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                        y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                        x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                        y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50
                        theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 
                        R = cv2.getRotationMatrix2D(
                            (int(x2), int(y2)), theta, 1)
                        align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                        imgaeRGB = cv2.warpAffine(imgaeRGB, R, (w, h)) 
                results = hands.process(imgaeRGB)
                cropped_image = cv2.cvtColor(imgaeRGB, cv2.COLOR_BGR2GRAY)
                h = cropped_image.shape[0]
                w = cropped_image.shape[1]
                print("Đưa tay vào đi bạn....!!!!!!!!!")
                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                        pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                        pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                        pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                        x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 
                        y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 
                        x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 
                        y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 
                        theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 
                        R = cv2.getRotationMatrix2D(
                            (int(x2), int(y2)), theta, 1)
                        align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                        roi_zone_img = cv2.cvtColor(align_img, cv2.COLOR_GRAY2BGR)
                        point_1 = [x1, y1]
                        point_2 = [x2, y2]
                        point_1 = (R[:, :2] @ point_1 + R[:, -1]).astype(np.int)
                        point_2 = (R[:, :2] @ point_2 + R[:, -1]).astype(np.int)
                        landmarks_selected_align = {
                            "x": [point_1[0], point_2[0]], "y": [point_1[1], point_2[1]]}
                        point_1 = np.array([landmarks_selected_align["x"]
                                            [0], landmarks_selected_align["y"][0]])
                        point_2 = np.array([landmarks_selected_align["x"]
                                            [1], landmarks_selected_align["y"][1]])
                        uxROI = pixelCoordinatesLandmarkPoint17[0]
                        uyROI = pixelCoordinatesLandmarkPoint17[1]
                        lxROI = pixelCoordinatesLandmarkPoint5[0]
                        lyROI = point_2[1] + 4*(point_2-point_1)[0]//3 
                        
                        roi_img = align_img[uyROI:lyROI, uxROI:lxROI]
                        roi_img = cv2.resize(roi_img, (128,128))
                        # roi_img = check_and_convert_to_rgb(roi_img)
                        
                        cv2.rectangle(imgaeResize, (uxROI, uyROI),
                            (lxROI, lyROI), (10, 255, 15), 2)

                        # cv2.imshow("FaceDetection", roi_img)
                        # key = cv2.waitKey(1) & 0xFF
                return roi_img
