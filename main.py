import cv2
import numpy as np

# Load YOLO model and weights
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")  # Make sure to adjust the paths to your files
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Load class labels for YOLO
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (use a video file or webcam)
cap = cv2.VideoCapture(0)  # Change to a file path for a video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables to track success/failure
frames_processed = 0
frames_failed = 0
objects_detected = 0
objects_not_detected = 0

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Check if frame is successfully captured
    if not ret:
        frames_failed += 1
        print("Failed to capture frame. Exiting...")
        break
    else:
        frames_processed += 1

    # Prepare the image for YOLO model (Resize, normalize, convert to blob)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    yolo_net.setInput(blob)
    outputs = yolo_net.forward(output_layers)

    # Post-process the detected objects
    class_ids = []
    confidences = []
    boxes = []
    height, width, _ = frame.shape
    detected_in_frame = False

    detected_objects = []  # To store detected object names for the current frame

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates for object detection box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_in_frame = True

                # Store object name for this detection
                detected_objects.append(classes[class_id])

    # If objects were detected, increment detected counter
    if detected_in_frame:
        objects_detected += 1
    else:
        objects_not_detected += 1

    # Non-maxima suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Frame", frame)

    # Print the detected object names in the current frame
    if detected_objects:
        print(f"Frame {frames_processed}: Detected objects - {', '.join(detected_objects)}")
    else:
        print(f"Frame {frames_processed}: No objects detected.")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Print summary at the end
print("\n--- Summary ---")
print(f"Frames Processed: {frames_processed}")
print(f"Frames Failed: {frames_failed}")
print(f"Objects Detected: {objects_detected}")
print(f"Objects Not Detected: {objects_not_detected}")
