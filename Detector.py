import cv2
from time import sleep
from PIL import Image 
import ROI
from ctypes import sizeof
from tkinter import W
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import os


import numpy as np

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
def main_app(name):
        
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")
        cap = cv2.VideoCapture(0)
        pred = 0
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            while True:
                ret, frame = cap.read()
                #default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                print("Dua tay vao di ban")
                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        pixelCoordinatesLandmarkPoint5 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[5].x, hand_landmark.landmark[5].y, w, h)
                        pixelCoordinatesLandmarkPoint9 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[9].x, hand_landmark.landmark[9].y, w, h)
                        pixelCoordinatesLandmarkPoint13 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[13].x, hand_landmark.landmark[13].y, w, h)
                        pixelCoordinatesLandmarkPoint17 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[17].x, hand_landmark.landmark[17].y, w, h)
                        pixelCoordinatesLandmarkPoint0 = mp_drawing._normalized_to_pixel_coordinates(hand_landmark.landmark[0].x, hand_landmark.landmark[0].y, w, h)
                        
                        # print(pixelCoordinatesLandmarkPoint5)
                        # print(pixelCoordinatesLandmarkPoint17)
                        center5 = np.array(
                            [np.mean(hand_landmark.landmark[5].x)*w, np.mean(hand_landmark.landmark[5].y)*h]).astype('int32')
                        center9 = np.array(
                            [np.mean(hand_landmark.landmark[9].x)*w, np.mean(hand_landmark.landmark[9].y)*h]).astype('int32')
                        center13 = np.array(
                            [np.mean(hand_landmark.landmark[13].x)*w, np.mean(hand_landmark.landmark[13].y)*h]).astype('int32')
                        center17 = np.array(
                            [np.mean(hand_landmark.landmark[17].x)*w, np.mean(hand_landmark.landmark[17].y)*h]).astype('int32')
                        cropped_image = cropped_image[0:pixelCoordinatesLandmarkPoint0[1] + 50, 0:pixelCoordinatesLandmarkPoint5[0] + 100]
                        x1 = (pixelCoordinatesLandmarkPoint17[0] +  pixelCoordinatesLandmarkPoint13[0]) / 2 + (pixelCoordinatesLandmarkPoint17[0] - pixelCoordinatesLandmarkPoint13[0])
                        y1 = (pixelCoordinatesLandmarkPoint17[1] + pixelCoordinatesLandmarkPoint13[1]) / 2 - 50
                        x2 = (pixelCoordinatesLandmarkPoint5[0] + pixelCoordinatesLandmarkPoint9[0]) / 2 - 50
                        y2 = (pixelCoordinatesLandmarkPoint5[1] + pixelCoordinatesLandmarkPoint9[1]) / 2 - 50

                        theta = np.arctan2((y2 - y1), (x2 - x1))*180/np.pi 

                        if (theta >= -15 and theta < 0):
                            # print("theta", theta)
                            R = cv2.getRotationMatrix2D(
                                (int(x2), int(y2)), theta, 1)
                            align_img = cv2.warpAffine(cropped_image, R, (w, h)) 
                            # imgaeRGB = cv2.warpAffine(imgaeRGB, R, (w, h)) 
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

                            ux = point_1[0]
                            uy = point_1[1] + (point_2-point_1)[0]//3
                            lx = point_2[0]
                            ly = point_2[1] + 4*(point_2-point_1)[0]//3
                            roi_zone_img = cv2.cvtColor(align_img, cv2.COLOR_GRAY2BGR)

                            
                            roi_img = align_img[uy:ly + 85, ux:lx + 85]
                            roi_img = cv2.resize(roi_img, (128,128))
                            # roi_img = check_and_convert_to_rgb(roi_img)
                            roi_img=cv2.cvtColor(roi_img,cv2.COLOR_RGB2BGR)
                            cv2.rectangle(imgaeResize, (ux, uy),
                                    (lx+85, ly + 85), (10, 255, 15), 2)
           
              
                            

                            id,confidence = recognizer.predict(roi_img)
                            confidence = 100 - int(confidence)
                            pred = 0
                            if confidence > 50:
                                #if u want to print confidence level
                                        #confidence = 100 - int(confidence)
                                        pred += +1
                                        text = name.upper()
                                        font = cv2.FONT_HERSHEY_PLAIN
                                        frame = cv2.rectangle(frame, (ux, uy), (lx, ly), (0, 255, 0), 2)
                                        frame = cv2.putText(frame, text, (ux, uy-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                            else:   
                                        pred += -1
                                        text = "UnknownFace"
                                        font = cv2.FONT_HERSHEY_PLAIN
                                        frame = cv2.rectangle(frame, (ux, uy), (lx, ly), (0, 0, 255), 2)
                                        frame = cv2.putText(frame, text, (ux, uy-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

                        cv2.imshow("image", frame)


                        if cv2.waitKey(20) & 0xFF == ord('q'):
                            print(pred)
                            if pred > 0 : 
                                dim =(124,124)
                                img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                                cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                                Image1 = Image.open(f".\\2.png") 
                                
                                # make a copy the image so that the  
                                # original image does not get affected 
                                Image1copy = Image1.copy() 
                                Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
                                Image2copy = Image2.copy() 
                                
                                # paste image giving dimensions 
                                Image1copy.paste(Image2copy, (195, 114)) 
                                
                                # save the image  
                                Image1copy.save("end.png") 
                                frame = cv2.imread("end.png", 1)

                                cv2.imshow("Result",frame)
                                cv2.waitKey(5000)
                            break


                    cap.release()
                    cv2.destroyAllWindows()
                
