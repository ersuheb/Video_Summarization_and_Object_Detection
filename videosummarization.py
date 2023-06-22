import cv2 # pip install opencv-python
import numpy as np # pip install numpy

# Load YOLOv3 object detector
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open video file
video = cv2.VideoCapture('input_video.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
threshold = 10.

# Define minimum number of frames between key frames
min_frames_between_key_frames = 10

# Initialize video writer
writer = cv2.VideoWriter('test1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Initialize list of tracked objects
tracked_objects = []

# # Define output video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

# Read first frame and set as previous frame
ret, frame1 = video.read()
prev_frame = frame1

# Initialize counters for unique and common frames
a = 0
b = 0
c = 0

# Initialize variable to control playback speed
playback_speed = 1

# Define different colors for different classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

key_frames = []


# Loop through video frames
while True:
    ret, frame = video.read()
    if ret is True:
        # Calculate absolute difference between current frame and previous frame
        diff = np.sum(np.absolute(frame - prev_frame)) / np.size(frame)
        # Detect objects only if there is a significant change in the current frame compared to the previous frame
        if diff > threshold:
            key_frames.append(frame)
        if len(key_frames) >= min_frames_between_key_frames or c == total_frames - 1:
            # Detect objects in frame using YOLOv3
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Initialize lists for bounding boxes, confidences and class IDs
            boxes = []
            confidences = []
            class_ids = []

            # Loop through each detection in outs
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Get bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        # Append bounding box coordinates, confidence and class ID to lists
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to remove redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Draw bounding boxes and labels for detected objects
            # Draw bounding boxes and labels for detected objects
            for i, idx in enumerate(indices):
                box = boxes[idx]
                x, y, w, h = box
                label = str(classes[class_ids[idx]])
                confidence = confidences[idx]
    
                # Draw bounding box and label
                color = colors[class_ids[idx]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw label with confidence
                label_text = f"{label} {round(confidence*100, 2)}%"
                cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            
        # Display the resulting frame
        cv2.putText(frame, "Press spacebar to play/pause, and ESC to close", (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow('frame', frame)
        c += 1
    
        # Write frame to output video
        writer.write(frame)


        

        # check for spacebar or ESC key press
        key = cv2.waitKey(1)
        if key == 32:  # spacebar key code
            while True:
                key2 = cv2.waitKey(10)
                cv2.imshow('frame', frame)
                if key2 == 32:
                    break
        elif key == 27:  # ESC key code
            break

    else:
        break





print("Total frames: ", c)

video.release()
writer.release()
cv2.destroyAllWindows()