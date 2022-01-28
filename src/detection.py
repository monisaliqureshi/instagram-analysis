import numpy as np
import cv2
import os
import gdown


if not os.path.exists('yolov3.weights'):
    url = "https://drive.google.com/u/0/uc?id=1yowMbozC0yozGnq5-ouTG-5UfSWFu7V7"
    output = 'yolov3.weights'
    gdown.download(url, output, quiet=False)
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolov3.weights"])
configPath = os.path.sep.join(["yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def apply_filter(frame):
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
		# loop over each of the detections
        for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
            if confidence > 0.5:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    all_objects = list()
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            all_objects.append(LABELS[classIDs[i]])
            
    
    return all_objects

def make_desc(objects):
    string = ", ".join(list(set(objects)))
    return string