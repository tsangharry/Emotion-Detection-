import numpy as np
import imutils
import cv2

from tensorflow.keras.preprocessing.image import img_to_array

detections = None 
def detect_and_predict_emotions(frame, faceNet, emotionsNet,threshold=0.5):
	"""Detects and predicts emotions.

	Args:
		frame : Calls frame of the video.
		faceNet : Pretrained DNN.
		emotionsNet : Our model trained on our images.
		threshold (float, optional): Defaults to 0.5.

	Returns:
		Pred : Returns prediction.
	"""    
	# grab the dimensions of the frame and then construct a blob
	# from it
	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >threshold:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 24x24, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (48, 48)) # the input size for our model
			face = img_to_array(face)
			# face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
            
			# add the face and bounding boxes to their respective
			# lists
			locs.append((startX, startY, endX, endY))
			#print(maskNet.predict(face)[0].tolist())
			preds.append(emotionsNet.predict(face)[0].tolist())
	return (locs, preds)

def return_annotated_images(frame, faceNet, emotionsNet):
	"""Returns an annotated image.

	Args:
		frame : Frame of video.
		faceNet : Pretrained DNN.
		emotionsNet : Our model trained on our images.

	Returns:
		frame : An annotated frame with bounding box, emotion and probability.
	"""    
	labels=["angry", "happy", "neutral", "sad", "surprise"]

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = imutils.resize(frame, width=400)
		original_frame = frame.copy()
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
		
		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_emotions(frame, faceNet, emotionsNet,0.5)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			# include the probability in the label
			argmaxIdx = np.argmax(pred)
			# some fishy engineering here to increase threshold for neutral 
			# and lower the thresholds for angry and sad
			if pred[0] > 0.10 and str(labels[argmaxIdx]) == 'neutral':	# angry
				label = str(labels[0])
				prob  = pred[0]
			elif pred[3] > 0.10 and str(labels[argmaxIdx]) == 'neutral':	# sad
				label = str(labels[3])
				prob = pred[3]
			else: 
				label = str(labels[argmaxIdx])
				prob = pred[argmaxIdx]
			# display the label and bounding box rectangle on the output
			# frame
			if label == "happy":
				cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,200,50), 2)
				cv2.rectangle(original_frame, (startX, startY), (endX, endY),(0,200,50), 2)
			elif label == "neutral":
				cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255), 2)
				cv2.rectangle(original_frame, (startX, startY), (endX, endY),(255,255,255), 2)
			elif label == "sad":
				cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200,50,0), 2)
				cv2.rectangle(original_frame, (startX, startY), (endX, endY),(200,50,0), 2)
			elif label == "angry":
				cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0, 255), 2)
				cv2.rectangle(original_frame, (startX, startY), (endX, endY),(0,0, 255), 2)
			elif label == "surprise":
				cv2.putText(original_frame, f'{label}: {round(prob, 3)}', (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,0), 2)
				cv2.rectangle(original_frame, (startX, startY), (endX, endY),(255,255,0), 2)
			frame = cv2.resize(original_frame,(860,490))
		return frame
