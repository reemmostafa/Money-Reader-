import cv2
import numpy as np
import operator
from gtts import gTTS
import Tkinter as tk
from win32com.client import Dispatch
from time import sleep
import imutils
import threading
import os


def Sound(filename):
    mp = Dispatch("WMPlayer.OCX")
    tune = mp.newMedia(filename)
    mp.currentPlaylist.appendItem(tune)
    mp.controls.play()
    sleep(1)
    mp.controls.playItem(tune)
    raw_input("Press Enter to stop playing")
    #sleep(5)
    mp.controls.stop()
    
def start_main_thread():
    t = threading.Thread(target = main)
    t.start()
    t.join

frame = 0
collect={}

def speak(money):
    new_money=money[0:2]
    if (money == "one" ):
        speech = 'one pound'
    elif (money == "ten"):
        speech = 'ten pounds'
    new_money=money[0:3]     
    if (money == "half" ):
        speech = '50 piasters'
    elif (money == "five"):
        speech = 'five pounds'
    new_money=money[0:6]   
    if (money == "qaurter"):
        speech = 'twenty five piasters'
    elif(money == "hundred"):
        speech = 'hundred pounds'
    new_money=money[0:5]    
    if (money == "twenty"):
        speech = 'twenty pounds' 
    new_money=money[0:4]   
    if (money == "fifty"):
        speech = 'fifty pounds'
    new_money=money[0:9]          
    if (money == "two hundred"):
        speech = 'two hundred pounds'
    
    tts = gTTS(text = speech, lang = 'en')
    tts.save("good.mp3")
    os.system("good.mp3")
    

def recognize_bill(descriptor, bill_name, kp, knn):
    count = 0
    bill_center = (0, 0)
    for h, des in enumerate(descriptor):
        des = np.array(des, np.float32).reshape(1, len(des))
        retval, results, neigh_resp, dists = knn.findNearest(des, 1)
        res, distance = int(results[0][0]), dists[0][0]
        x, y = kp[res].pt
        center = (int(x), int(y))
        if distance < 0.1:
            bill_center = center
            color = (0, 0, 255)
            count += 1
        else:
            color = (255, 0, 0)

        cv2.circle(frame, center, 2, color, -1)

    print float(count) / len(descriptor)
    collect.update({bill_name: float(count) / len(descriptor)})

    print(bill_name)
    if float(count) / len(descriptor) >= 0.5:
        cv2.putText(frame, ">>" + bill_name + "<<", bill_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

def main():
    # SURF extraction
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(500)
    
    print "opening templates images..."
    # get the templates
    temp_10_front = cv2.imread("templates\egypt_10.jpg",0)
    temp_5_front = cv2.imread("templates\Egypt 5.jpg",0)
    temp_05_front = cv2.imread("templates\E05.jpg",0)
    temp_100_front = cv2.imread("templates\E100.jpg",0)
    temp_1_front = cv2.imread("templates\E1.jpg",0)
    temp_200_front = cv2.imread("templates\E200.jpg",0)
    
    temp_10_back = cv2.imread("templates\E10_2.jpg",0)
    temp_5_back = cv2.imread("templates\E5_2.jpg",0)
    temp_05_back = cv2.imread("templates\E05_2.jpg",0)
    temp_100_back = cv2.imread("templates\E100_2.jpg",0)
    temp_1_back = cv2.imread("templates\E1_2.jpg",0)
    temp_200_back = cv2.imread("templates\E200_2.jpg",0)
    print "calculating template descriptors..."
    key1, desc_10_front = surf.detectAndCompute(temp_10_front, None)
    key2, desc_5_front = surf.detectAndCompute(temp_5_front, None)
    key3, desc_05_front = surf.detectAndCompute(temp_05_front, None)
    key4, desc_100_front = surf.detectAndCompute(temp_100_front, None)
    key5, desc_1_front = surf.detectAndCompute(temp_1_front, None)
    key6, desc_200_front = surf.detectAndCompute(temp_200_front, None)
    key7, desc_10_back = surf.detectAndCompute(temp_10_back, None)
    key8, desc_5_back = surf.detectAndCompute(temp_5_back, None)
    key9, desc_05_back = surf.detectAndCompute(temp_05_back, None)
    key0, desc_100_back = surf.detectAndCompute(temp_100_back, None)
    key11, desc_1_back = surf.detectAndCompute(temp_1_back, None)
    key12, desc_200_back = surf.detectAndCompute(temp_200_back, None)
    print "starting video capture..."
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imwrite('5.png',frame)
    # When everything done, release the capture
    cap.release()
    cv2.imshow('d',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
         
    
    kp, desc = surf.detectAndCompute(gray, None)
    
    samples = np.array(desc)
    responses = np.arange(len(kp), dtype=np.float32)
    knn = cv2.ml.KNearest_create()
    
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)
    
    recognize_bill(desc_100_back, "ten", kp, knn)
    recognize_bill(desc_5_back, "five",kp, knn)
    recognize_bill(desc_05_back, "half",kp, knn)
    recognize_bill(desc_100_back, "hundred",kp, knn)
    recognize_bill(desc_1_back, "one",kp, knn)
    recognize_bill(desc_200_back, "two hundred",kp, knn)
    recognize_bill(desc_10_front, "ten",kp, knn)
    recognize_bill(desc_5_front, "five",kp, knn)
    recognize_bill(desc_05_front, "half",kp, knn)
    recognize_bill(desc_100_front, "hundred",kp, knn)
    recognize_bill(desc_1_front, "one",kp, knn)
    recognize_bill(desc_200_front, "two hundred",kp, knn)
    sorted_x = sorted(collect.items(), key=operator.itemgetter(1),reverse=True)
    print(sorted_x)
    k, v =sorted_x[0]
    print(k)
    speak(k)
    print "\n"
    


top = tk.Tk()
top.geometry("400x400")
b = tk.Button(top, text = "Start",fg='black',bd=0,bg='white', width = 40,height = 20, command = start_main_thread)
b.place(x = 50, y = 40)
Sound('g.mp3')

top.mainloop()
