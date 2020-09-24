import getopt
import sys

import cv2
import numpy as np

class_ids = []
confidences = []
boxes = []
min_confidence = 0.5
target_name = "truck"


def main(argv):
    input_image = ''
    try:
        opts, args = getopt.getopt(argv, 'i:', ["input-file"])
    except getopt.GetoptError:
        print('python3 objects_detector.py -i <input-file>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-i", "--input-file"):
            input_image = arg

    net, output_layers, classes = load_net()
    image, blob, height, width, channels = load_image(input_image)
    detect_objects(net, blob, output_layers, height, width)
    show_detected_objects(image, classes)


# Load YOLO network into CV2 with COCO names
def load_net():
    # Weights are available in YOLO website, please check README.md for more information
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, classes


# Loads input image, resize them and generate Blob
def load_image(input_file):
    img = cv2.imread(input_file)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    return img, blob, height, width, channels


# Performs object detection based on Blob
def detect_objects(net, blob, output_layers, height, width):
    for b in blob:
        for n, img_blob in enumerate(b):
            cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                # Object detected position and size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle object delimiter coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


# Show obtained results in input image
def show_detected_objects(image, classes):
    # Performs non maximum suppression given boxes and corresponding scores
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(boxes)):
        if i in indexes:
            if str(classes[class_ids[i]]) == target_name:
                target_label = "{0} ({1} %)".format("Carga pesada", round(confidences[i] * 100, 2))

                x, y, w, h = boxes[i]
                label = target_label
                color = (255, 0, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y + 30), font, 1, color, 1)
                print(label)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
