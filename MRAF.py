# -*- coding: utf-8 -*-
"""
Stella Harrison | sh16g14@soton.ac.uk | 4th October 2017

Mixed Region Amplitude Freeddom (MRAF) Algorithm

"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import cmath as c
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import LogNorm
from MRAF_functions import Ein_0, Eout, newFieldInput, newFieldOutput, merit, rgb2gray, target_image
SLM_width=1920
SLM_height=1920
#refPt = []
roi=[]
central_point=[]
refPt=[]

# Non-importable functions
def ROI(event, x, y, flags, param):
    #global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        roi.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        roi.append((x, y)) 
        cv2.rectangle(img, roi[-2], roi[-1], (225, 0, 0), 1)
        cv2.imshow("Put a Box round your ROI, then hit spacebar", img)
        cv2.resizeWindow("Put a Box round your ROI, then hit spacebar", SLM_width/3,SLM_height/3)

        
    
def centre(event, x, y, flags, param):
    #global refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        central_point.append((x, y)) 
        cv2.circle(img, central_point[-1], 5, (0, 0, 225), 5)
        cv2.imshow("Click at the centre of your ROI, then hit spacebar", img)
        cv2.resizeWindow("Click at the centre of your ROI, then hit spacebar", SLM_width/3,SLM_height/3)
        
def signal(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y)) 
        cv2.rectangle(target, refPt[-2], refPt[-1], (0, 255, 0), 10)
        cv2.imshow("Put a Box round your Signal Region, then hit spacebar", img)
        cv2.resizeWindow("Put a Box round your Signal Region, then hit spacebar", SLM_width/3,SLM_height/3)
        
#-----------------------------------------------------------------------------
# User Inputs
#----------------------------------------------------------------------------- 
path = raw_input('What is the image path? e.g. target_image.jpg\n')
temp = np.array(Image.open(path))

blackIsBright = raw_input('Does black in your target image represent bright spots in the output image? [y/n]\n')
while blackIsBright != 'y' and blackIsBright != 'n':
    blackIsBright = raw_input('Does black in your target image represent bright spots in the output image? [y/n]\n')
    
m = float(raw_input('Choose a mixing number between 0 and 1\n'))
while m < 0.0 or m > 1.0:
    m = float(raw_input('Choose a mixing number between 0 and 1\n'))


reshape= raw_input('Do you want to crop or rescale your image to fit on the SLM? [c/r]\n')
while reshape != 'r' and reshape != 'c':
    reshape= raw_input('Do you want to crop or rescale your image to fit on the SLM? [c/r]\n')
#-----------------------------------------------------------------------------
# Select Region of Interest and Signal Region
#-----------------------------------------------------------------------------
#N = temp.shape[1] #width
#M = temp.shape[0] #height

#Region of Interest             
img = cv2.imread(path)

if reshape == 'r':
    clone1 = img.copy()
    cv2.namedWindow("Put a Box round your ROI, then hit spacebar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Put a Box round your ROI, then hit spacebar", int((SLM_height/3.)*(float(img.shape[1])/float(img.shape[0]))),int(SLM_width/3.))
    cv2.setMouseCallback("Put a Box round your ROI, then hit spacebar", ROI)
    while True:
        cv2.imshow("Put a Box round your ROI, then hit spacebar", img)
        cv2.resizeWindow("Put a Box round your ROI, then hit spacebar",int((SLM_height/3.)*(float(img.shape[1])/float(img.shape[0]))),int(SLM_width/3.))
        key = cv2.waitKey(1) & 0xFF       
        if key == ord("r"):
            img = clone1.copy()
        elif key == ord(" "):
            img = clone1.copy()
            cv2.destroyAllWindows()
            break
    #roi = [roi[-2][1], roi[-1][1], roi[-2][0], roi[-1][0]]
    roi = [roi[-2][1], roi[-1][1], roi[-2][0], roi[-1][0]]
    pts1 = np.float32([[roi[2],roi[0]],[roi[3],roi[0]], [roi[3],roi[1]], [roi[2],roi[1]]])
    pts2 = np.float32([[0,0],[SLM_height,0],[SLM_height,SLM_width],[0,SLM_width]])
    trans = cv2.getPerspectiveTransform(pts1,pts2)
    target = cv2.warpPerspective(img,trans,(1920,1920))
elif reshape == 'c':
    clone1 = img.copy()
    cv2.namedWindow("Click at the centre of your ROI, then hit spacebar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Click at the centre of your ROI, then hit spacebar", int((SLM_height/3.)*(float(img.shape[1])/float(img.shape[0]))),int(SLM_width/3.))
    cv2.setMouseCallback("Click at the centre of your ROI, then hit spacebar", centre)
    while True:
        cv2.imshow("Click at the centre of your ROI, then hit spacebar", img)
        cv2.resizeWindow("Click at the centre of your ROI, then hit spacebar", int((SLM_height/3.)*(float(img.shape[1])/float(img.shape[0]))),int(SLM_width/3.))
        key = cv2.waitKey(1) & 0xFF       
        if key == ord("r"):
            img = clone1.copy()
        elif key == ord(" "):
            img = clone1
            cv2.destroyAllWindows()
            break
    squareLength=np.min([central_point[-1][1], SLM_width - central_point[-1][0],SLM_height - central_point[-1][1],central_point[-1][0]])
    roi= [central_point[-1][1]-squareLength,central_point[-1][1]+squareLength , central_point[-1][0]-squareLength, central_point[-1][0]+squareLength]
    
    pts1 = np.float32([[roi[2],roi[0]],[roi[3],roi[0]], [roi[3],roi[1]], [roi[2],roi[1]]])
    pts2 = np.float32([[0,0],[SLM_height,0],[SLM_height,SLM_width],[0,SLM_width]])
    trans = cv2.getPerspectiveTransform(pts1,pts2)
    target = cv2.warpPerspective(img,trans,(1920,1920))
  
    # Siganl Region
    cv2.namedWindow("Put a Box round your Signal Region, then hit spacebar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Put a Box round your Signal Region, then hit spacebar",int((SLM_height/3.)*(float(target.shape[1])/float(target.shape[0]))),int(SLM_width/3.))
    cv2.setMouseCallback("Put a Box round your Signal Region, then hit spacebar", signal)
while True:
    cv2.imshow("Put a Box round your Signal Region, then hit spacebar", target)
    cv2.resizeWindow("Put a Box round your Signal Region, then hit spacebar",int((SLM_height/3.)*(float(target.shape[1])/float(target.shape[0]))),int(SLM_width/3.))
    key = cv2.waitKey(1) & 0xFF 
    if key == ord("r"):
        target = clone1.copy()
    elif key == ord(" "):
        target = clone1.copy()
        cv2.destroyAllWindows()
        break
sig = [refPt[-2][1], refPt[-1][1], refPt[-2][0], refPt[-1][0]]    
#-----------------------------------------------------------------------------
# Normalised Target Image & Source Wave
#-----------------------------------------------------------------------------

print '\n*WARNING* If you over-write your target image file whilst running this script, the target image will change to the new file saved.\n'

x, y = np.meshgrid(np.linspace(-1,1,SLM_width), np.linspace(-1,1,SLM_height))
d = np.sqrt(x*x + y*y)
sigma, mu = 200.0, 2.0
source = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ))
for i in range(SLM_height):
    for j in range(SLM_width):
        source[i,j] = np.sqrt(1 - source[i,j])

Ein = Ein_0(source,SLM_height,SLM_width)

#----------------------------------------------------------------------------- 
# Start Iterations
#-----------------------------------------------------------------------------                        
iteration = 0
p=100 #old merit factor
deltaMerit=100 #change in merit factor
kinoform=np.zeros((SLM_height,SLM_width))
meritThreshold=0.0001

print 'The iterations will stop once the change in the merit value is less than %s.\n\nMerit Values:\n' %meritThreshold

while deltaMerit > meritThreshold: #change as required
    target = target_image(path, roi, sig,SLM_width, SLM_height, blackIsBright)
    iteration += 1
    
    #Normalised Output
    out = Eout(Ein,sig,SLM_width,SLM_height,m)
            
    #Generated New Output
    G = newFieldOutput(target, out,m ,sig)
        
    #New Input
    Ein = newFieldInput(source, G)
               
    #Figure of Merit
    r = merit(target, out) 
    deltaMerit = abs(r-p)
    p = r #replace old merit with new merit
    print p

#Kinoform
for i in range(SLM_height):
    for j in range(SLM_width):
        kinoform[i,j] = c.phase(Ein[i,j])
print iteration
SR = np.zeros((sig[1]-sig[0], sig[3] - sig[2]))
for i in range(sig[0],sig[1]):
    for j in range(sig[2], sig[3]):
        SR[(i - sig[0]),(j - sig[3])] = abs(out[i,j]) 
#-----------------------------------------------------------------------------
# Plot Target Image, Output Image and Kinoform
#----------------------------------------------------------------------------- 
plt.subplot(1,3,1)
plt.imshow(target, aspect = 'auto')
plt.title('RS: Target Image')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(abs(out),vmin = np.min(abs(SR)), norm=LogNorm(),vmax = np.max(abs(SR)), aspect = 'auto')
plt.title('RS: Output Image\n%s Iterations' %iteration)
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(kinoform, vmin=-(np.pi), vmax=np.pi, aspect='auto', cmap='Greys')
plt.title('FS: SLM Phase Pattern\n%s Iterations' %iteration)
plt.colorbar()
plt.show()