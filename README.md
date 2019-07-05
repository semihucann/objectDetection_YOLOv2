# ObjectDetection with YOLOv2

## Usage
`$ git clone https://github.com/thtrieu/darkflow.git` 

`$ cd darkflow`

`$ wget https://pjreddie.com/media/files/yolov2-tiny.weights`

`$ wget https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg`

`$ flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights`

Change codes as your file paths on CFG_FILE_EXTENSION, WEIGHT_FILE_EXTENSION and IMAGE_PATH  in objectDetection_yolo_v2
##### For example
options = {"model": "CFG_FILE_EXTENSION", "load": "WEIGHT_FILE_EXTENSION", "threshold": 0.1}

`$ flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights`

### Tiny Yolo
-Dataset:	COCO trainval	
-mAp    :23.7	 Bn	
-FLOPS  :5.41Bn	

Link for pretrained weight 

https://pjreddie.com/media/files/yolov2-tiny.weights

Link for cfg file

https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg

## Requirement Libraries
tensorflow (1.10.0)


