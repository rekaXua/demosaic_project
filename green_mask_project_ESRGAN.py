import cv2
import numpy as np
import os
import glob
import torch
import math
from scipy.signal import argrelextrema
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help='Use CPU instead of CUDA')
args = parser.parse_args()

#You can change those folder paths
rootdir = "./decensor_input"
outdir = "./decensor_output"
os.makedirs(rootdir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)

files = glob.glob(rootdir + '/**/*.png', recursive=True)
files_jpg = glob.glob(rootdir + '/**/*.jpg', recursive=True)
files.extend(files_jpg)

#-----------------------ESRGAN-Init-----------------------
import architecture as arch
GPUmem = torch.cuda.get_device_properties(0).total_memory

model_path = "models/4x_FatalPixels_340000_G.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu' if args.cpu else 'cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
						mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
	v.requires_grad = False
model = model.to(device)

#-----------------------Logic-----------------------
# GBlur = 5
# CannyTr1 = 20
# CannyTr2 = 100
# LowRange = 2
# HighRange = 20
# DetectionTr = 0.32

GBlur = 5
CannyTr1 = 15
CannyTr2 = 130
LowRange = 2
HighRange = 20
DetectionTr = 0.3

pattern = [None] * (HighRange+2)
for masksize in range(HighRange+2, LowRange+1, -1):
	maskimg = 2+masksize+masksize-1+2
	screen = (maskimg, maskimg)
	img = Image.new('RGB', screen, (255,255,255))
	pix = img.load()
	for i in range(2,maskimg,masksize-1):
		for j in range(2,maskimg,masksize-1):
			for k in range(0,maskimg):
				pix[i, k] = (0,0,0)
				pix[k, j] = (0,0,0)
	pattern[masksize-2] = img

#Working with files
for f in files:
	#-----------------------Files-----------------------
	img_C = Image.open(f).convert("RGBA")
	x, y = img_C.size
	card = Image.new("RGBA", (x, y), (255, 255, 255, 0))
	cvI = Image.alpha_composite(card, img_C)
	card = cv2.cvtColor(np.array(card), cv2.COLOR_BGRA2RGBA)
	cvI = np.array(cvI)
	img_rgb = cv2.cvtColor(cvI, cv2.COLOR_BGRA2RGBA)
	img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.Canny(img_gray,CannyTr1,CannyTr2)
	img_gray = 255-img_gray
	img_gray = cv2.GaussianBlur(img_gray,(GBlur,GBlur),0)
	
	#-----------------------Detection-----------------------
	resolutions = [-1] * (HighRange+2)
	for masksize in range(HighRange+2, LowRange+1, -1):
		template = cv2.cvtColor(np.array(pattern[masksize-2]), cv2.COLOR_BGR2GRAY)
		w, h = pattern[masksize-2].size[::-1]
	
		img_detection = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
		loc = np.where(img_detection >= DetectionTr)
		rects = 0
		for pt in zip(*loc[::-1]):
			rects += 1    #increase rectangle count of single resolution
			cv2.rectangle(card, pt, (pt[0] + w, pt[1] + h), (0,0,0,255), -1)
		resolutions[masksize-1] = rects

	resolutions.append(0)
#	print(resolutions)    #DEBUG Resolutions array
	extremaMIN = argrelextrema(np.array(resolutions), np.less, axis=0)[0]
	extremaMIN = np.insert(extremaMIN,0,LowRange)
	extremaMIN = np.append(extremaMIN,HighRange+2)

	Extremas = []
	for i, ExtGroup in enumerate(extremaMIN[:-1]):
		Extremas.append((ExtGroup, resolutions[extremaMIN[i]:extremaMIN[i+1]+1]))

	ExtremasSum = []
	BigExtrema = [0,0,[0,0]]
	for i, _ in enumerate(Extremas):
		ExtremasSum.append(sum(Extremas[i][1]))
		if BigExtrema[0] <= sum(Extremas[i][1])+int(sum(Extremas[i][1])*0.05):    #5% precedency for smaller resolution
			if max(BigExtrema[2]) < max(Extremas[i][1])+max(Extremas[i][1])*0.15:
				BigExtrema = [sum(Extremas[i][1]),Extremas[i][0],Extremas[i][1]]	
	MosaicResolutionOfImage = BigExtrema[1]+BigExtrema[2].index(max(BigExtrema[2]))    #Output
	if MosaicResolutionOfImage == 0:    #If nothing found - set resolution as smallest
		MosaicResolutionOfImage = HighRange+1
	print('Mosaic Resolution of "' + os.path.basename(f) + '" is: ' + str(MosaicResolutionOfImage))    #The Resolution of Mosaiced Image
	
	#DEBUG Show image
	#cv2.imshow('image',card)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	#-----------------------ESRGAN-Processing-----------------------
	Sx = int(1.2*x/MosaicResolutionOfImage)
	Sy = int(1.2*y/MosaicResolutionOfImage)
	#print(Sx, Sy)
	shrinkedI = cv2.resize(img_rgb, (Sx, Sy))
	maxres = math.sqrt((Sx*Sy)/(GPUmem*0.00008))
	if maxres > 1:
		shrinkedI = cv2.resize(shrinkedI, (int(Sx/maxres),int(Sy/maxres)))
	#print(maxres)
	while True:
		
		imgESR = cv2.cvtColor(shrinkedI, cv2.COLOR_RGBA2RGB)
		imgESR = imgESR * 1.0 / 255
		
		imgESR = torch.from_numpy(np.transpose(imgESR[:, :, [2, 1, 0]], (2, 0, 1))).float()
		img_LR = imgESR.unsqueeze(0)
		img_LR = img_LR.to(device)

		output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
		output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
		output = (output * 255.0).round()
		if MosaicResolutionOfImage > 7:
			MosaicResolutionOfImage = MosaicResolutionOfImage*0.25
			#print("iter")
			height, width, _ = output.shape
			#print(width, height)
			maxres = math.sqrt((width*height)/(GPUmem*0.00008))
			if maxres > 1:
				shrinkedI = cv2.resize(output, (int(width/maxres), int(height/maxres)))
			else:
				shrinkedI = output
			#print(maxres)
			continue
		break

	#-----------------------Unification-and-Saving-----------------------
	imgESRbig = cv2.resize(output, (x, y), cv2.INTER_AREA)
	imgESRbig = imgESRbig.astype(np.uint8)
	imgESRbig = cv2.cvtColor(imgESRbig, cv2.COLOR_RGB2RGBA)
	img2gray = cv2.cvtColor(imgESRbig,cv2.COLOR_RGBA2GRAY)
	ret, mask = cv2.threshold(img2gray, 245, 255, cv2.THRESH_BINARY)
	imgESRbig = cv2.bitwise_not(imgESRbig,imgESRbig,mask = mask)
	card = cv2.cvtColor(card, cv2.COLOR_RGBA2GRAY)
	card_inv = cv2.bitwise_not(card)
	mosaic_reg = cv2.bitwise_and(imgESRbig,imgESRbig,mask = card_inv)
	img_C = cv2.cvtColor(np.array(img_C), cv2.COLOR_BGRA2RGBA)
	out = np.array(Image.alpha_composite(Image.fromarray(img_C), Image.fromarray(mosaic_reg)))
	f=f.replace(rootdir, outdir, 1) 
	os.makedirs(os.path.dirname(f), exist_ok=True)
	
	res, im = cv2.imencode(os.path.basename(f), out)
	with open(f, 'wb') as fout:
		fout.write(im.tobytes())