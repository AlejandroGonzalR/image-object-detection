# Heavy duty vehicles detection based image recognition

<img src="https://user-images.githubusercontent.com/40376561/94055186-38575c00-fda2-11ea-89e4-ee634c32b99c.png" width="1000">
&nbsp

Example of image-based pattern recognition by implementing a pre-trained YOLOv3 model with COCO and OpenCV.

## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/AlejandroGonzalR/image-object-detection/blob/master/requirements.txt) dependencies installed. To install run:

```bash
$ pip install -r requirements.txt
```

Additionally, the weights built for YOLOv3 are required, which can be found by following this [link](https://pjreddie.com/media/files/yolov3.weights). Or run:

```bash
$ wget https://pjreddie.com/media/files/yolov3.weights
```

## Getting Started

Run command below. You can choose between the images in this repository or another you want.

```bash
$ python3 objects_detector.py -i <input-image>
```

For more information please visit Ultralytics [repo](https://github.com/ultralytics/yolov3) for YOLOv3 in PyTorch. Or Joseph Redmon page for YOLO  https://pjreddie.com/darknet/yolo/.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
