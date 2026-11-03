from ultralytics import YOLO
import cv2 as cv

import util
from sort.sort import *
from util import get_car, read_license_plate,write_csv



results ={}


mot_tracker=Sort()
#load models
license_plate_model = YOLO("C:/Users/fth/Documents/AI PROJECTS/LicensePlateProject/runs/runs/detect/train-3/weights/best.pt")
vehicle_model= YOLO('yolo26n.pt')

#load video
cap=cv.VideoCapture("C:/Users/fth/Documents/AI PROJECTS/LicensePlateProject/data/AlgerianLicencePlates_Yolo5/test/Videos/algerian_car2.mp4")

vehicles=[2,3,5,7]
#read frames
frame_nmr=-1
ret= True

while ret:
    frame_nmr+=1
    ret, frame = cap.read()
    if ret and frame_nmr<250 :
        results[frame_nmr]={}
        #detect vehicles
        detections= vehicle_model(frame)[0]
        detections_= []
        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id =detection
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score])

        #track vehicles
        detections_array = np.asarray(detections_, dtype=float).reshape(-1, 5)
        track_ids = mot_tracker.update(detections_array)


        #detect license plates
        license_plates = license_plate_model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id = license_plate

            #assign license plate to a car
            x1car, y1car, x2car , y2car, car_id = get_car(license_plate,track_ids)

            #crop license plate
            license_plate_crop=frame[int(y1):int(y2),int(x1):int(x2),:]

            #process license plate
            license_plate_crop_gray=cv.cvtColor(license_plate_crop, cv.COLOR_BGR2GRAY)
            _,license_plate_crop_thresh=cv.threshold(license_plate_crop_gray,64,255,cv.THRESH_BINARY_INV,0)


            #read license plate number
            license_plate_text, license_plate_text_score=read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None :
                results[frame_nmr][car_id]={
                    'car': {'bbox':[x1car, y1car, x2car , y2car]},
                    'license_plate':{'bbox':[x1,y1,x2,y2,score],
                                     'text':license_plate_text,
                                     'bbox_score':score,
                                     'text_score':license_plate_text_score}
                }
#write results

write_csv(results, 'C:/Users/fth/Documents/AI PROJECTS/LicensePlateProject/new_test.csv')