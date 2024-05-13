import cv2
import cvzone
from ultralytics import YOLO
from Sort import *
model = YOLO("C:\\Users\\intel\\PycharmProjects\\pythonProject\\AI\\INTERNSHIP\\yoloweights\\yolov8n.pt")

 #                   detection object in the image

# img = cv2.imread(r"C:\Users\intel\PycharmProjects\pythonProject\AI\INTERNSHIP\istockphoto-518590341-612x612.jpg")
# result = model(img,show = True)
# cv2.waitKey(0)

#                       detection object in the webcam
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# Classname = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# while True:
#      ret,frame = cap.read()
#
#      result = model(frame, stream=True)
#      for r in result:
#          Boxes = r.boxes
#          for box in Boxes:
#            x1, y1, x2, y2 = box.xyxy[0]
#
#            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
#            w, h = x2 - x1, y2 - y1
#            confidence = (int(box.conf[0]*100))/100
#            cls = int(box.cls[0])
#            print(confidence)
#            print(cls)
#            cvzone.cornerRect(frame, (x1,y1,w,h))
#            cvzone.putTextRect(frame, f'{Classname[cls]},{confidence}',(max(0,x1),max(35,y1)),scale=1.5,thickness=1)
#
#      cv2.imshow("Detected Objects", frame)
#      if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
#                           car detection  in video
cap = cv2.VideoCapture("C:\\Users\\intel\\PycharmProjects\\pythonProject\\AI\\INTERNSHIP\\WhatsApp Video 2024-04-03 at 2.16.39 PM.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('outputcountervideo.mp4', fourcc, 20.0, (frame_width, frame_height))
mask = cv2.imread("C:\\Users\\intel\\PycharmProjects\\pythonProject\\AI\\INTERNSHIP\\Screenshot 2024-05-05 202654.jpg")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
newsize = (1280 ,720)
maskedimage = cv2.resize(mask ,newsize)
limit = [360 ,279,930,279]
totalcount = []
Classname = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
while True:
     ret,frame = cap.read()
     imagemask = cv2.bitwise_and(frame ,maskedimage)
     result = model(frame,stream = True)
     detection = np.empty((0,5))
     for r in result:
         Boxes = r.boxes
         for box in Boxes:
           x1, y1, x2, y2 = box.xyxy[0]

           x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
           w, h = x2 - x1, y2 - y1
           confidence = (int(box.conf[0]*100))/100
           cls = int(box.cls[0])
           print(confidence)
           print(cls)
           if confidence > 0.4 and cls == 2 :
               #cvzone.cornerRect(frame, (x1,y1,w,h), l = 9 )
               #cvzone.putTextRect(frame, f'{Classname[cls]},{confidence}',(max(0,x1),max(35,y1)),offset=9,scale=1.5,thickness=1)
               currentarray = np.array([x1,y1,x2,y2,confidence])
               detection = np.vstack((detection,currentarray))

     cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 5)
     trackingresult = tracker.update(detection)
     for  result in trackingresult:
         x1,y1,x2,y2,id = result
         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
         w, h = x2 - x1, y2 - y1
         #cvzone.cornerRect(frame, (x1, y1, w, h), l=9)
         #cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1)), offset=9, scale=1.5,thickness=1)

         cx,cy = x1+w//2, y1+h//2
         cv2.circle(frame,(int(cx),int(cy)),5,(255,0,255),cv2.FILLED)
         if limit[0] < cx < limit[2] and limit[1]-9 < cy < limit[1]+9:
           if totalcount.count(id)== 0:
             totalcount.append(id)
             cv2.line(frame, (limit[0], limit[1]), (limit[2], limit[3]), (0, 0, 255), 5)


     cvzone.putTextRect(frame, f'count : {len(totalcount)}', (50, 50))
     out.write(frame)
     cv2.imshow("Detected Objects", frame)
     #cv2.imshow("Masked region ", imagemask )
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
