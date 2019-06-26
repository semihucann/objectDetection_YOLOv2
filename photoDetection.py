from darkflow.net.build import TFNet
import cv2
options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1}
tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/1.jpg")
result = tfnet.return_predict(imgcv)

font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(result)):

	x = result[i]['topleft']['x']
	y = result[i]['topleft']['y']

	x2 = result[i]['bottomright']['x']
	y2 = result[i]['bottomright']['y']

	if(result[i]['confidence']>0.24):
		cv2.putText(imgcv, result[i]['label'], (x2,y),font, 1,(0,100,0),2,cv2.LINE_AA)
		cv2.putText(imgcv, str(result[i]['confidence']), (x2,y+5),font, 1,(0,0,250),1,cv2.LINE_AA)
		image2 = cv2.rectangle(imgcv, (x,y),(x2,y2),(120,200,0),3)
		imgcv = image2


cv2.imshow("cezeri", imgcv)

cv2.waitKey(0)
cv2.destroyAllWindows()
