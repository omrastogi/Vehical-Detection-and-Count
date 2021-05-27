import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
	scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

conf_thres = 0.25
iou_thres = 0.45
DEVICE = ''
view_img = False
save_txt = False
save_conf = False
nosave = False
classes=None 
agnostic_nms=False
augment=False
update=False,
project='runs/detect' 
name='exp' 
exist_ok=False
weights = ['yolov3.pt']
webcam = False
classify = False


def initialize_tracker(frame, bounding_box):
	tracker = cv2.TrackerKCF_create()
	bounding_box[2] = bounding_box[2] - bounding_box[0]
	bounding_box[3] = bounding_box[3] - bounding_box[1]

	tracker.init(frame, tuple(bounding_box))
	return tracker


	
def _remove_stray_blobs(blobs, matched_blob_ids, mcdf):
	'''
	Remove blobs that "hang" after a tracked object has left the frame.
	'''
	for blob_id, blob in list(blobs.items()):
		if blob_id not in matched_blob_ids:
			blob.num_consecutive_detection_failures += 1
		if blob.num_consecutive_detection_failures > mcdf:
			del blobs[blob_id]
	return blobs

def editimage(img0, img_size, stride=32):
	img = letterbox(img0, img_size, stride=stride)[0]
	img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	img = np.ascontiguousarray(img)
	return img

def view(img):
	cv2.imshow("frame", img)
	cv2.waitKey(5000) 




def loadModel(imgsz):
	# Initialize
	set_logging()
	device = select_device(DEVICE)
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	stride = int(model.stride.max())  # model stride
	imgsz = check_img_size(imgsz, s=stride)  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	# if classify:
	#     modelc = load_classifier(name='resnet101', n=2)  # initialize
	#     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names

	# Run inference
	if device.type != 'cpu':
		model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


	return model, stride, names, device
	


def detect(image, imgsz, model, stride, device):
	t0 = time.time()
	# model, stride, names, device = loadModel(imgsz)
	half = device.type != 'cpu' 

	img = editimage(image, img_size=imgsz, stride=stride)
	img = torch.from_numpy(img).to(device)
	img = img.half() if half else img.float()  # uint8 to fp16/32
	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	# Inference
	t1 = time_synchronized()
	pred = model(img, augment=augment)[0]
	# Apply NMS
	pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
	t2 = time_synchronized()

	# Apply Classifier
	# if classify:
	#     pred = apply_classifier(pred, modelc, img, im0s)

	gn = torch.tensor(image.shape)[[1, 0, 1, 0]]     
	det = pred[0]
	if len(det):
		# Rescale boxes from img_size to im0 size
		det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
		return np.asarray(det)
	else:
		return []

def dist(pt1, pt2):
	func = (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2
	return func*0.5

def boxcheck(frame,tboxes, box):
	for tbox in tboxes:
		distance =  dist([(tbox[0]+tbox[2])/2,(tbox[1]+tbox[3])/2],[(box[0]+box[2])/2,(box[1]+box[3])/2])
		print ("DD-",distance,"boxe",tbox,list(box))
		if distance<500:
			return False
	return True

# def plot(frame, box)

def main(video):
	# param to save video 
	fourcc = cv2.VideoWriter_fourcc(*'XVID')	
	out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (550,  450))

	# params for model
	vehical_label = [1,2,3,5,7]
	img_size=320
	model, stride, names, device = loadModel(img_size)
	trackers = []

	   
	cap = cv2.VideoCapture(video)



	# Take first frame and find corners in it
	# ret, old_frame = cap.read()
	# old_frame = cv2.resize(old_frame, (640, 640), interpolation=cv2.INTER_AREA)

	
	
	# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

	flag = 0 
	cnt = 0
	while(1):
		if flag == 1:
			flag = 0
			continue
		flag = 1 

		ret,frame = cap.read()
		frame = cv2.resize(frame, (550, 640), interpolation=cv2.INTER_AREA)
		frame = frame[:450, :]
		boxes = detect(frame, img_size, model, stride, device)
		# print ((boxes == None), boxes)

		if len(boxes):
			tboxes = []
			for tracker in trackers:
				_,tbox = tracker.update(frame[:350, :])
				tbox= (tbox[0],tbox[1],tbox[0]+tbox[2],tbox[1]+tbox[3])
				tboxes.append(tbox)

			for box in boxes:
				if box[3]>350:
					continue
				if box[-1] in vehical_label: 
					if boxcheck(frame, tboxes, box):
						trackers.append(initialize_tracker(frame[:350, :], box[:4].astype(np.int32)))
						cnt+=1

		else:
			tracker = []
		# view(frame, trackers)
		# tboxes to check the trackers and boxes to check detection

		for bbox in boxes:
			p1 = (int(bbox[0]), int(bbox[1]))
			p2 = (int(bbox[2]), int(bbox[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
		cv2.putText(frame, "Vehical Count : " + str(cnt), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,10,200), 2);
		print (frame.shape)
		out.write(frame)
		cv2.imshow("Tracking", frame)
		k = cv2.waitKey(1) & 0xff
		if k == 27 : break

		# print (cnt)

		


if __name__ == '__main__':
	# imggg = cv2.imread('data/images/chow.jpg')
	# model, stride, names, device = loadModel(img_size)
	# print (detect(imggg, img_size, model, stride, device))
	main('dataset/video.mp4')