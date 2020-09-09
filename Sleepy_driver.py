#!/usr/bin/env python3
# import necessary modules from kivy
import os
import sys
import cv2
import dlib
import time
import playsound
import numpy as np
from gtts import gTTS
from kivy.app import App
from threading import Thread
from kivy.config import Config
from kivy.uix.button import Button
from scipy.spatial import distance
from kivy.uix.floatlayout import FloatLayout

Config.set('graphics','width','563')
Config.set('graphics','height','842')
###############################################################################
def get_ear_value(eye):
    """To find the eye_aspect_ratio of an eye"""
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    ear = (a + b) / (2.0 * c)
    return ear

def get_lip_distance(top_lip, bottom_lip):
    """Returns the lip distance"""
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    distance = abs(top_mean[1] - bottom_mean[1])
    return distance

def alarm(path):
    """Alarm for drowsy and yawn detection."""
    playsound.playsound(path)

def speak(text):
    """converts text to speech by using google text to speech module(gtts)."""
    tts = gTTS(text=text, slow=False, lang='en')
    file_name = "voice.mp3"
    tts.save(file_name)

    playsound.playsound(file_name)
###############################################################################
# create a background class which inherits the boxlayout class
class Background(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eye_ear_threshold = 0.24  # may vary upon distance and quality of camera
        self.yawn_threshold = 22
        self.flag=0
        self.modify_threshold=0


    def drowsy(self):
        def show_plot(label):
            plot_canvas_graph = np.ones((height_graph, width_graph, 3)) * 255
            cv2.line(plot_canvas_graph,
                     (0, int(height_graph / 2)),
                     (width_graph, int(height_graph / 2)), (0, 255, 0), 1)
            for i in range(len(val_graph) - 1):
                cv2.line(plot_canvas_graph, (i, int(height_graph / 2) - val_graph[i]),
                         (i + 1, int(height_graph / 2) - val_graph[i + 1]), color_graph, 1)
                cv2.putText(plot_canvas_graph, "Plot of eye_ear value", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i in range(len(val2_graph)-1):
                cv2.line(plot_canvas_graph, (i, int(height_graph) - val2_graph[i]),
                        (i + 1, int(height_graph) - val2_graph[i + 1]), color_graph, 1)
                cv2.putText(plot_canvas_graph, "Plot of Yawn threshold",
                            (10,int(height_graph/2)+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.resize(plot_canvas_graph, (width_graph,height_graph))
            cv2.imshow(label,plot_canvas_graph)
            cv2.waitKey(10)

        # Update new values in plot
        def plot(value,value2,label="Plot"):
            value=int(value)
            value2=int(value2)
            val_graph.append(value)
            val2_graph.append(value2)
            while len(val_graph) > width_graph or len(val2_graph) > width_graph:
                val_graph.pop(0)
                val2_graph.pop(0)
            show_plot(label=label)
        def drawcontours(img, lower, upper):
            """To draw contour when called and return array of landmarks of eye point"""
            for n in range(lower, upper + 1):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                next_point = n + 1
                if n == upper:
                    # To come back to the starting point
                    next_point = lower
                x2 = landmarks.part(next_point).x
                y2 = landmarks.part(next_point).y
                cv2.line(img, (x, y), (x2, y2), (0, 255, 0), 1)

        def getcontour_points(lower, upper, array):
            """Get the contour points x and y and appen them as tuple in eye array"""
            for n in range(lower, upper + 1):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                array.append((x, y))
            # to return the eye array
            return array

        ###############################################################################

        eye_ear = 0
        lip_distance = 0
        print("-> Loading the predictor and detector...")
        playsound.playsound("predictor.mp3")
        facecascade = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("-> Starting Video Stream")
        playsound.playsound("cap.mp3")
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        cap.set(10, 100)
        count = 0
        count_threshold = 30  # for the frequency of eye closure time and can adjust it
        time.sleep(1.0)
        # speak("Wake up sir")   #can change the alarm for DROWSINESS from here
        alarm_on1 = False  # for drowsy alarm
        alarm_on2 = False  # for yawn alarm
        # graph values
        color_graph=(255,0,0)
        val_graph=[]
        val2_graph=[]
        width_graph=480
        height_graph=280
        plot_canvas_graph=np.ones((width_graph,height_graph,3))*255
        ###############################################################################

        while True:
            success, img = cap.read()
            #print(img.shape)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = facecascade(img_gray)
            for face in faces:
                # cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
                landmarks = predictor(img_gray, face)
                # for left eye detection
                left_eye = drawcontours(img, 36, 41)
                # for right eye detection
                right_eye = drawcontours(img, 42, 47)
                # for external lip detection
                drawcontours(img, 48, 60)
                # for internal lip detection
                # drawcontours(img,61,67)

                left_eye = []
                right_eye = []
                # TO get the ear value of left eye
                left_eye = getcontour_points(36, 41, left_eye)
                left_eye_ear = get_ear_value(left_eye)
                # To get the ear value of th right eye
                right_eye = getcontour_points(42, 47, right_eye)
                right_eye_ear = get_ear_value(right_eye)
                eye_ear = (left_eye_ear + right_eye_ear) / (2.0)
                # print(eye_ear)  #trail and error method for
                # adjusting the eye_ear_threshold

                top_lip = []
                bottom_lip = []
                # To get the distance between top lip and low lip
                top_lip = getcontour_points(48, 54, top_lip)
                bottom_lip = getcontour_points(55, 60, bottom_lip)
                lip_distance = get_lip_distance(top_lip, bottom_lip)
                # print(lip_distance)    #trail and error method for
                # adjusting yawn_threshold
                if self.flag == 1:
                    if self.modify_threshold == 1:
                        print("You have modified eye_ear to",self.eye_ear_threshold)
                    self.threshold_changer()
                x=lip_distance
                y=eye_ear*100
                plot(y,x)

                if eye_ear < self.eye_ear_threshold:
                    count += 1
                    if count >= count_threshold:
                        if not alarm_on1:
                            alarm_on1 = True
                            t = Thread(target=alarm("voice.mp3"), args=("voice.mp3"))
                            t.deamon = True
                            t.start()
                        cv2.putText(img, "DROWSINESS ALERT", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    count = 0
                    alarm_on1 = False

                if lip_distance > self.yawn_threshold:
                    if not alarm_on2:
                        alarm_on2 = True
                        t = Thread(target=alarm("yawn.mp3"), args=("yawn.mp3"))
                        t.deamon = True
                        t.start()
                        cv2.putText(img, "YAWN ALERT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)
                else:
                    alarm_on2 = False
            cv2.putText(img, "EAR: {:.2f}".format(eye_ear), (500, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(img, "YAWN: {:.2f}".format(lip_distance), (500, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
            cv2.putText(img, "Press and Hold Q to Exit", (20, 470), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Cap", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(1)

    def threshold_changer(self):
        """Changes the threshold value if the detection is wrong"""
        def empty(a):
            """An empty function which does nothing and for on_change function
                in  threshold window"""
            pass
        self.flag=1
        self.modify_threshold=1
        cv2.namedWindow("Thresholds")
        cv2.resizeWindow("Thresholds",640,240)
        cv2.createTrackbar("Eye ear(0.01)","Thresholds",24,50,empty)
        cv2.createTrackbar("Yawn threshold","Thresholds",22,50,empty)
        self.eye_ear_threshold = cv2.getTrackbarPos("Eye ear(0.01)","Thresholds")
        self.eye_ear_threshold/=100  # may vary upon distance and quality of camera
        # adjust it by trail error on line
        self.yawn_threshold = cv2.getTrackbarPos("Yawn threshold","Thresholds")
        cv2.waitKey(1)

    def end(self):
        """Exits the code if choosen to exit"""
        print("THANK YOU!")
        sys.exit(1)

# Create App class with name of your app
class AndroidApp(App):

    # return the Window having the background template.
    def build(self):
        return Background()

if __name__ == '__main__':
    AndroidApp().run()
