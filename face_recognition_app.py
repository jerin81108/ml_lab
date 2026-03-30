import urllib.request
import face_recognition
import imutils
import pickle
import cv2
import os

def setup_environment():
    """Downloads Messi image and OpenCV cascades"""
    os.makedirs('input', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    messi_url = 'https://upload.wikimedia.org/wikipedia/commons/b/b4/Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg'
    messi_path = 'input/messi.jpg'
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(messi_url, messi_path)
    
    cascade_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml'
    cascade_path = 'data/haarcascade_frontalface_alt2.xml'
    if not os.path.exists(cascade_path):
        urllib.request.urlretrieve(cascade_url, cascade_path)
        
    return messi_path, cascade_path

def create_encodings(image_path, model='hog'):
    """Generates the facial encodings for Messi (hog_face_encodings.py logic)"""
    print("[INFO] quantifying faces...")
    import numpy as np
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV could not read the image at {image_path}. Check if it's a valid image file.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    
    boxes = face_recognition.face_locations(rgb, model=model)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    knownEncodings = []
    knownNames = []
    
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append("Lionel Messi")
        
    data = {"encodings": knownEncodings, "names": knownNames}
    with open("face_enc", "wb") as f:
        f.write(pickle.dumps(data))
    return data

def recognize_faces(image_path, cascade_path, encoding_data):
    """Detects and predicts the face in the image using our encodings (face_reco.py logic)"""
    print("[INFO] recognizing faces...")
    import numpy as np
    faceCascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"OpenCV could not read the image at {image_path}. Check if it's a valid image file.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    
    encodings = face_recognition.face_encodings(rgb)
    names = []
    
    for encoding in encodings:
        matches = face_recognition.compare_faces(encoding_data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = encoding_data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        
    for ((x, y, w, h), name) in zip(faces, names):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
    print("[INFO] display output...")
    cv2.imshow("Messi Face Recognition", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    messi_path, cascade_path = setup_environment()
    encoding_data = create_encodings(messi_path)
    recognize_faces(messi_path, cascade_path, encoding_data)
