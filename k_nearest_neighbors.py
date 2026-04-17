import cv2, os, numpy as np, pickle, time, csv
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

os.makedirs('data', exist_ok=True); os.makedirs('Attendance', exist_ok=True)
v = cv2.VideoCapture(0)
fd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

name = input("Enter your name to capture data (or leave blank to skip capture): ").strip()
if name:
    face_data = []
    print("Capturing 100 face samples... Please look at the camera.")
    while len(face_data) < 100:
        ret, fr = v.read()
        for x, y, w, h in fd.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.3, 5):
            try: face_data.append(cv2.resize(fr[y:y+h, x:x+w], (50, 50)))
            except Exception: pass
            cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Face Capture', fr); cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    face_data = np.array(face_data).reshape(100, -1)
    
    nm = pickle.load(open('data/names.pkl', 'rb')) + [name]*100 if os.path.exists('data/names.pkl') else [name]*100
    fc = np.append(pickle.load(open('data/face_data.pkl', 'rb')), face_data, axis=0) if os.path.exists('data/face_data.pkl') else face_data
    pickle.dump(nm, open('data/names.pkl', 'wb')); pickle.dump(fc, open('data/face_data.pkl', 'wb'))

try: 
    LABELS, FACES = pickle.load(open('data/names.pkl', 'rb')), pickle.load(open('data/face_data.pkl', 'rb'))
except FileNotFoundError: 
    print("No data found to train KNN."); exit()

knn = KNeighborsClassifier(n_neighbors=5).fit(FACES, LABELS)
print("Press 'q' to quit, 'o' to log attendance.")

while True:
    ret, fr = v.read()
    for x, y, w, h in fd.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.3, 5):
        try:
            out = knn.predict(cv2.resize(fr[y:y+h, x:x+w], (50, 50)).flatten().reshape(1, -1))[0]
            cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 255, 0), 2); cv2.putText(fr, out, (x, y-10), 0, 0.7, (255, 255, 255), 2)
            if cv2.waitKey(1) & 0xFF == ord('o'):
                fn, now = f"Attendance/attendance_{datetime.now().strftime('%d-%m-%Y')}.csv", datetime.now()
                fe = os.path.isfile(fn)
                with open(fn, 'a', newline='') as f:
                    if not fe: csv.writer(f).writerow(['Name', 'Time', 'Date'])
                    csv.writer(f).writerow([out, now.strftime('%H:%M:%S'), now.strftime('%d-%m-%Y')])
                print(f"Logged {out}.")
        except Exception: pass
    cv2.imshow('Attendance System', fr)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
