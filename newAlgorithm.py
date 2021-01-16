
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randrange
import operator


def getShape(image):
    return image.shape

def region(image,max_height,maw_width):

    #isolate the gradients that correspond to the lane lines
    #polynomPoints = np.array([[(0, height), (427, 300), (853, 300), (width, height)]])
    polynomPoints = np.array([[(0, max_height),(0, max_height-100), (500, 300), (maw_width-500, 300), (maw_width, max_height-100),(maw_width, max_height)]])

    #create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    #create a mask (poly that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, polynomPoints, color=[255,255,255])
    mask = cv2.bitwise_and(image, mask)
    return mask


def canny(image):
    edges = cv2.Canny(image,60,120)
    return edges

def showImage(image):
    plt.figure(figsize=(15,50))
    plt.imshow(image,cmap = 'gray')
    plt.show()

# learn more about GaussianBlur ==> done (learn about the parameters)

    
#initImage=cv2.imread('images/6.jpg')
#initImage=cv2.imread('images/9.jpg')
#initImage=cv2.imread('images/20.jpg')
#initImage=cv2.imread('images/42.jpg')
#initImage=cv2.imread('images/25.jpg')
#initImage=cv2.imread('images/32.jpg')
#initImage=cv2.imread('images/79.jpg')
#initImage=cv2.imread('images/46.jpg')
initImage=cv2.imread('images/81.jpg')



#===========================================================================================
# Region Of Interest
#===========================================================================================

gaussianBlur = cv2.GaussianBlur(initImage,(3,3),0)
canny = canny(gaussianBlur)
height,width,channel=getShape(initImage)
roi=region(canny,height,width)



showImage(initImage)
showImage(gaussianBlur)
showImage(canny)
showImage(roi)


#===========================================================================================
# Hough Transform
#===========================================================================================


rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 4  # minimum number of pixels making up a line
max_line_gap = 20 # maximum gap in pixels between connectable line segments  ==> Maximum allowed gap between line segments to treat them as single line.


line_image = np.copy(initImage) * 0  # creating a blank to draw lines on


lines = cv2.HoughLinesP(roi, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)


#Show all the lines in the image
showImage(line_image)
dest=cv2.addWeighted(initImage,1,line_image,1,0)


#===========================================================================================
# Filer
#===========================================================================================

#===========================================================================================
# Find all the line that will be removed
#===========================================================================================

def detecteLine(lines):
    height=getShape(initImage)[0]
    width=getShape(initImage)[1]
    for line in lines:
        x1, y1, x2, y2 = line[0]

        #Horizonatl lines
        if(y1 >= y2-6 and y1<=y2+6):
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)

        #Verticales Lines
        if(x1 >= x2-5 and x1<=x2+5):
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)
            

        x=x2-x1
        y=y2-y1

        if( x1<x2 and y1>y2 and y>0 and abs(x-y) >= 10 and  abs(x-y) <= 30 ):   
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)
      
        if(abs(x) > 2*abs(y)):  
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)
            


#===========================================================================================
# Remove lines
#===========================================================================================

def clearLines(lines):
    height=getShape(initImage)[0]
    width=getShape(initImage)[1]
    for line in lines:
        x1, y1, x2, y2 = line[0]

        #Horizonatl lines
        if(y1 >= y2-6 and y1<=y2+6):
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0

        #Verticales Lines
        if(x1 >= x2-5 and x1<=x2+5):
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0
    
        x=x2-x1
        y=y2-y1

        if( x1<x2 and y1>y2 and y>0 and abs(x-y) >= 10 and  abs(x-y) <= 30 ): 
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0    
      
        if(abs(x) > 2*abs(y)):  
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0    

    return lines




#===========================================================================================
# Delete null corrdination from the array (Lines)
#===========================================================================================
def removeNullPoints(lines):
    array=np.array(lines)   
    newArray=np.delete(array, np.where(array == [0,0,0,0])[0], axis=0)
    return newArray

#===========================================================================================
# Apply the functions
#===========================================================================================
detecteLine(lines)
showImage(line_image)

clearLines=clearLines(lines)
newLines=removeNullPoints(clearLines)
showImage(line_image)


#===========================================================================================
# Select the lines
#===========================================================================================

#===========================================================================================
# Insert the right lines in one list
#===========================================================================================
def selectRightLine(lines):
    rightLines=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(x1<x2 and y1<y2):
            rightLines.append(line)
    return rightLines        

#===========================================================================================
# Insert the left lines in other list
#===========================================================================================
def selectLeftLine(lines):
    leftLines=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(x1<x2 and y1>y2):
            leftLines.append(line)
    return leftLines        

#===========================================================================================
rightLines=selectRightLine(newLines)
leftLines=selectLeftLine(newLines)
#===========================================================================================


#===========================================================================================
# Select the left line from the list that contains the left lines and select the right line from the list that contains the right lines
#===========================================================================================
def selectTwoLines(leftLines,rightLines):
    randNumber1=randrange(len(leftLines)-1)
    randNumber2=randrange(len(rightLines)-1)

    while randNumber2==randNumber1 :
        randNumber2=randrange(len(rightLines)-1)

    leftLine = leftLines[randNumber1][0]
    rightLine = rightLines[randNumber2][0]

    cv2.line(line_image, (leftLine[0], leftLine[1]), (leftLine[2], leftLine[3]), (0, 0,250), 5)
    cv2.line(line_image, (rightLine[0], rightLine[1]), (rightLine[2], rightLine[3]), (0, 0,250), 5)

    return [leftLine,rightLine]

#===========================================================================================
# Get the a, b ,c from a line equation 
#===========================================================================================
def getLineParam(line):
    """
    For the line equation a*x+b*y+c=0, if we know two points(x1, y1)(x2,y2) in line, we can get
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1
    """
    x1, y1, x2, y2 = line

    a = y1 - y2 
    b = x2 - x1
    c = x1 * y2 - x2 * y1
   
    return a,b,c
 
#===========================================================================================
# Find the intersaction point between these two lines
#===========================================================================================
def getCrossPoint(line1,line2):
    """
    if we have two lines: a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0,
    when d= (a1 * b2 - a2 * b1) is zero, then the two lines are coincident or parallel.
    The cross point is :
        x = (b1 * c2 - b2 * c1) / d
        y = (a2 * c1 - a1 * c2) / d
    """

    a1, b1, c1 = getLineParam(line1)
    a2, b2, c2 = getLineParam(line2)
    d = a1 * b2 - a2 * b1

    if d == 0:
        return np.inf, np.inf
    
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d
    return abs(int(x)), abs(int(y))



#===========================================================================================
# Draw a point 
#===========================================================================================
def insertCircle(x,y, image):
    if x != float('inf'):
        cv2.circle(image, (int(x),int(y)), radius=5, color=(0, 250, 0), thickness=-1)


#===========================================================================================
# Return the distance between the vanishing point and each line (the result it is an array)
#===========================================================================================
def dist(point, lines):
    dists = []
    for line in lines:

        x1, y1, x2, y2 = line[0]
        x, y = point 
        if x==float('inf') or y==float('inf'):
             dists.append(int(0))
             
        else :
            a, b, c = getLineParam(line[0])
            dist = abs((a*x + b*y + c) / np.sqrt(a*a + b*b))    
            dists.append(int(dist))
    return np.asarray(dists)


#===========================================================================================
# Calculate the number of inliers lines
#===========================================================================================
def getInliers(distinations,lines):
    
    theshold=5
    inlier_nbr=0
    i=0

    for distination in distinations:
  
        if 0 < distination < theshold :
            #print(distination)
            inlier_nbr=inlier_nbr+1
            x1,y1,x2,y2=lines[i][0]
            print(lines[i][0])
            cv2.line(line_image, (x1,y1),  (x2,y2), (150, 200, 100), 5)
        i=i+1    
    return inlier_nbr


#===========================================================================================
# Calculate the number of inliers lines
#===========================================================================================
twoLines=selectTwoLines(leftLines,rightLines)

line1=twoLines[0]
line2=twoLines[1]

x1a, y1a, x2a, y2a = line1
x1b, y1b, x2b, y2b = line2

insertCircle(getCrossPoint(line1,line2)[0],getCrossPoint(line1,line2)[1],line_image)
showImage(line_image)


getInliers(dist(getCrossPoint(line1,line2), newLines), newLines)
showImage(line_image)
