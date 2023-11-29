#import numpy as np
import cv2
import time
import datetime
from pymongo import MongoClient
from ultralytics import YOLO
from timeit import default_timer as timer


client = MongoClient("mongodb://localhost:27017/")
db = client["demos"]
collection = db["create"]

model = YOLO("yolo weights/lastt.pt")
classNames = ["needle"]
mycolor = (0, 0, 255)
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Open the video file
cap = cv2.VideoCapture('videos/13.mp4')
start = timer()

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Get the frame rate of the video
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print("frame_rate---", frame_rate)
c = time.time()

needle_start_time = None
is_needle_moving = False
needle_count = 0
count = 0
pount = 0
hount = 0
last_status = None

# Number of frames to skip
frame_skip = 10  # Adjust as needed

frame_count = 0  # To keep track of frames

while True:
    success, img = cap.read()
    if not success:
        break

    # Skip frames according to frame_skip value
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    frame_count += 1

    # Resize the frame
    img = cv2.resize(img, (640, 360))

    # Calculate the time in seconds (elapsed_time) using the frame rate
    elapsed_time = c / frame_rate

    # Run object detection
    result = model(img, stream=True)

    # Initialize variables
    is_running = False

    for r in result:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), mycolor, 3)

    # Check needle movement
    if is_running:
        if needle_start_time is None:
            needle_start_time = elapsed_time
        is_needle_moving = True
    else:
        is_needle_moving = False
        needle_start_time = None

    # Update the status
    if is_needle_moving:
        status = "running"
    else:
        status = "not running"


    if status != last_status:
        dii = {"seconds": elapsed_time, "status": status}
        value = collection.insert_one(dii)
        last_status = status

    # Display the frame
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break

# Calculate needle runtime
needle_runtime = 0
if needle_count > 0:
    needle_end_time = time.time()
    needle_runtime = needle_end_time - needle_start_time

# Calculate total runtime
total = time.time() - c
non = total - needle_runtime

# Format and store runtime data
t_minutes, t_seconds = divmod(int(total), 60)
d_minutes, d_seconds = divmod(int(needle_runtime), 60)
n_minutes, n_seconds = divmod(int(non), 60)

t = f"{t_minutes} minutes, {t_seconds} seconds"
d = f"{d_minutes} minutes, {d_seconds} seconds"
n = f"{n_minutes} minutes, {n_seconds} seconds"

total_runtime_data = {
    "video": "5",
    "path": "D/videos/6video.mp4",
    "uploaded": datetime.datetime.now(),
    "userid": 5,
    "total_runtime": t,
    "needle_runtime": d,
    "non_runtime": n,
    "harold_das": pount,
    "antony_das": hount,
    "dinosaur": count,
}

collection_runtime = db["create"]
result_runtime = collection_runtime.insert_one(total_runtime_data)
print("Stored total runtime data in MongoDB:", result_runtime)

cap.release()
cv2.destroyAllWindows()
nd = timer()

print(nd - start)

client.close()


