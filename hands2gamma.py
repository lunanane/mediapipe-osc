import argparse

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

from utils import add_default_args, get_video_input

OSC_ADDRESS = "/mediapipe/hands"


def send_hands(client: udp_client,
               detections: [landmark_pb2.NormalizedLandmarkList]):
    if detections is None:
        client.send_message(OSC_ADDRESS, 0)
        return

    # create message and send
    builder = OscMessageBuilder(address=OSC_ADDRESS)
    builder.add_arg(len(detections))
    for detection in detections:
        for landmark in detection.landmark:
            builder.add_arg(landmark.x)
            builder.add_arg(landmark.y)
            builder.add_arg(landmark.z)
            builder.add_arg(landmark.visibility)
    msg = builder.build()
    print(msg)
    client.send(msg)


def main():
    # read arguments
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    args = parser.parse_args()
    
    
# create osc client
client = udp_client.SimpleUDPClient("127.0.0.1", 4444)




mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:   
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        #send_hands(client, results.right_hand_landmarks)
        
        if results.right_hand_landmarks is None:
            client.send_message(OSC_ADDRESS, 0)
        else:
            builder = OscMessageBuilder(address=OSC_ADDRESS)    
            if results.right_hand_landmarks:
                #builder.add_arg(84)
                print(results.right_hand_landmarks)

                for landmark in results.right_hand_landmarks.landmark:
                    builder.add_arg(landmark.x)
                    builder.add_arg(landmark.y)
                    builder.add_arg(landmark.z)
                    builder.add_arg(landmark.visibility)
                    print(landmark.visibility)

                msg = builder.build()
                client.send(msg)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()
