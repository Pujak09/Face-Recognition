from ast import Div
from cgitb import html
from difflib import Match
from email.mime import image
from email.quoprimime import body_check
from json import detect_encoding
from charset_normalizer import detect
from click import style
import cv2
from cv2 import FILLED
import numpy as np
import os
from datetime import datetime
import face_recognition
import streamlit as st

path = 'images'
images = []
personName = []
myList = os.listdir(path)
print(myList)
for cv_img in myList:
    current_img = cv2.imread(f'{path}/{cv_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cv_img)[0])
print(personName)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList




def attendance(name):
    with open('AttendanceML.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(' , ')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name}, {tStr}, {dStr}')
            

encodeListKnown = faceEncodings(images)
print("All Encodings complete!!")

cap = cv2.VideoCapture(0)





while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(name)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0, 255,0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            attendance(name)
            

    cv2.imshow("webcam", frame)
    if cv2.waitKey(1) == 13:
        break


cap.release()
cv2.destroyAllWindows()


#streamlit

def main():
    "Face Recognition web"

    st.title("Face Recognition")
    html_temp = """
    <bodystyle="background-color:red;">
    <divstyle="background-color:teal; padding:10px">
    <h2style="color:white;text-align:center;">Face Recognition webApp</h2>
    </div>
    </body>
    
    
    """
    

    st.markdown(html_temp, unsafe_allow_html= True)

    image_file = st.file_uploader("upload images", ['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = image.open(image_file)
        st.text("original image")
        st.image(our_image)

    if st.button("Recognise"):
        result_img= detect(our_image)
        st.image(result_img)


if __name__ == '__main__':
    
    main()





