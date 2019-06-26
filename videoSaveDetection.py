from darkflow.net.build import TFNet
import cv2

cap = cv2.VideoCapture('1.mp4')

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.1}
tfnet = TFNet(options)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))

while(True):
	ret,imgcv = cap.read()

	result = tfnet.return_predict(imgcv)

	font = cv2.FONT_HERSHEY_SIMPLEX

	for i in range(len(result)):

		x = result[i]['topleft']['x']
		y = result[i]['topleft']['y']

		x2 = result[i]['bottomright']['x']
		y2 = result[i]['bottomright']['y']

		if(result[i]['confidence']>0.2):
			cv2.putText(imgcv, result[i]['label'], (x2,y),font, 1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.putText(imgcv, str(round(result[i]['confidence'],2)), (x,y-7),font, 0.5,(0,0,255),1,cv2.LINE_AA)
			image2 = cv2.rectangle(imgcv, (x,y),(x2,y2),(120,200,0),3)
			imgcv = image2


	cv2.imshow("cezeri", imgcv)
	# Write the frame into the file 'output.avi'
	out.write(imgcv)
 
	
	
	
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
