import cv2 
import dlib  
import imutils 
from scipy.spatial import distance as dist 
from imutils import face_utils 

cam = cv2.VideoCapture(0) 


def calculate_EAR(eye): 
    y1 = dist.euclidean(eye[1], eye[5]) 
    y2 = dist.euclidean(eye[2], eye[4]) 

    x1 = dist.euclidean(eye[0], eye[3]) 

    EAR = (y1+y2) / x1 
    return EAR 

blink_thresh = 0.45
succ_frame = 2
count_frame = 0
total_blinks = 0
blink_state = False

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

detector = dlib.get_frontal_face_detector() 
landmark_predict = dlib.shape_predictor('../model/shape_predictor_68_face_landmarks.dat') 

while True: 
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    frame = imutils.resize(frame, width=640) 

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    faces = detector(img_gray) 
    for face in faces: 
        shape = landmark_predict(img_gray, face) 

        shape = face_utils.shape_to_np(shape) 

        lefteye = shape[L_start: L_end] 
        righteye = shape[R_start:R_end] 

        left_EAR = calculate_EAR(lefteye) 
        right_EAR = calculate_EAR(righteye) 

        avg = (left_EAR+right_EAR)/2

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        if avg < blink_thresh: 
            count_frame += 1  
        else: 
            if count_frame >= succ_frame and not blink_state: 
                total_blinks += 1
                blink_state = True
                cv2.putText(frame, 'Blink Detected', (30, 30), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2) 
            count_frame = 0
            blink_state = False

        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(frame, f'Blinks: {total_blinks}', (frame.shape[1] - 200, 30), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Blink Detection", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cam.release() 
cv2.destroyAllWindows()