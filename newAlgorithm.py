
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randrange
import operator
import math 


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
    edges = cv2.Canny(image,50,120)
    return edges

def showImage(image):
    plt.figure(figsize=(15,50))
    plt.imshow(image,cmap = 'gray')
    plt.show()


# learn more about GaussianBlur ==> done (learn about the parameters)
    
#initImage=cv2.imread('images/15.jpg')
#initImage=cv2.imread('images/13.jpg')
#
# initImage=cv2.imread('images/14.jpg')

#initImage=cv2.imread('images/29.jpg')
#initImage=cv2.imread('images/6.jpg')

#initImage=cv2.imread('images/20.jpg')
#initImage=cv2.imread('images/21.jpg')


#initImage=cv2.imread('images/32.jpg')
#initImage=cv2.imread('images/79.jpg')
#initImage=cv2.imread('images/111.jpg')


#initImage=cv2.imread('images/0.jpg')
# initImage=cv2.imread('images/81.jpg')
#initImage=cv2.imread('images/8.jpg')

#initImage=cv2.imread('images/1.jpg')
#initImage=cv2.imread('images/9.jpg')


#initImage=cv2.imread('images/100.jpg')
#initImage=cv2.imread('images/m3.jpg')
#initImage=cv2.imread('images/46.jpg')


initImage=cv2.imread('images/25.jpg')
#initImage=cv2.imread('images/m2.jpg')
#initImage=cv2.imread('images/m.jpg')
#initImage=cv2.imread('images/133.jpg')


#===========================================================================================
# Region Of Interest
#===========================================================================================
height,width,channel=getShape(initImage)

gaussianBlur = cv2.GaussianBlur(initImage,(7,7),0)
#roiV2=region(gaussianBlur,height,width)
edge = canny(gaussianBlur)
roi=region(edge,height,width)



#showImage(initImage)
#showImage(gaussianBlur)
#showImage(canny)
#showImage(roi)


#===========================================================================================
# Hough Transform
#===========================================================================================
line_image = np.copy(initImage) * 0  # creating a blank to draw lines on
def HoughLines(image,rho, theta, threshold,min_line_length , max_line_gap):
    
    #line_image = np.copy(image) * 0  # creating a blank to draw lines on


    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)# Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return lines



rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 90  # angular resolution in radians of the Hough grid
threshold = 20  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 20  # minimum number of pixels making up a line
max_line_gap = 100 # maximum gap in pixels between connectable line segments  ==> Maximum allowed gap between line segments to treat them as single line.


lines=HoughLines(roi,rho,theta, threshold,min_line_length , max_line_gap)

#Show all the lines in the image
showImage(line_image)
#dest=cv2.addWeighted(initImage,1,line_image,1,0)



#===========================================================================================
# Filer
#===========================================================================================

#===========================================================================================
# Find all the line that will be removed
#===========================================================================================
def getGradianLine(line):
    #(y2-y1)/(x2-x1)
    return (line[3]-line[1])/(line[2]-line[0])



def detecteLine(lines):
    buttomLine=0
    height=getShape(initImage)[0]
    width=getShape(initImage)[1]

    for line in lines:
        x1, y1, x2, y2 = line[0]

        #Horizonatl lines
        if(y1 >= y2-10 and y1<=y2+10):
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)

        #Verticales Lines
        if(x1 >= x2-10 and x1<=x2+10):
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)
            
        #center bottum
        if(y1 >= height-300 and y1 <= height and y2 >= height-300 and y2 <= height):
            buttomLine=buttomLine+1



            

        x=abs(abs(x2)-abs(x1))
        y=abs(abs(y2)-abs(y1))



        if(getGradianLine(line[0])>0 and getGradianLine(line[0])<0.3):            
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)

        if(getGradianLine(line[0])<0 and getGradianLine(line[0])>-0.3):            
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 250, 0), 2)
        """
        #left lines
        if(getGradianLine(line[0])<0):
            cv2.line(line_image, (x1, y1), (x2, y2), (250, 250, 0), 2)  
        """

    print(buttomLine)
    return round(buttomLine)
 



def distance(line):
    dist = np.sqrt(((line[0] - line[2]) ** 2) + ((line[1] - line[3]) ** 2))
    return dist


#===========================================================================================
# Remove lines
#===========================================================================================

def clearLines(lines,buttomLineNumber):
    height=getShape(initImage)[0]
    width=getShape(initImage)[1]

    
    for line in lines:
        x1, y1, x2, y2 = line[0]

        #Horizonatl lines
        if(y1 >= y2-10 and y1<=y2+10):
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0

        #Verticales Lines
        if(x1 >= x2-10 and x1<=x2+10):
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0
       

        x=abs(abs(x2)-abs(x1))
        y=abs(abs(y2)-abs(y1))

        """
        if( x1<x2 and y1>y2 and y>0 and abs(x-y) >= 10 and  abs(x-y) <= 30 ): 
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0    
      
        if(abs(x) > 2*abs(y)):  
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0   
        """

        if(getGradianLine(line[0])>0 and getGradianLine(line[0])<0.3):            
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0   
            

        if(getGradianLine(line[0])<0 and getGradianLine(line[0])>-0.3):            
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)
            line[0]=0,0,0,0   


    if(buttomLineNumber>50):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if(y1 >= height-300 and y1 <= height and y2 >= height-300 and y2 <= height):
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
detecteLineV1=detecteLine(lines)
showImage(line_image)
clearLinesV1=clearLines(lines,detecteLineV1)

newLines=removeNullPoints(clearLinesV1)
showImage(line_image)


def seperateLine(lines):
    rightLines=[]
    leftLines=[]

    for line in lines:
        if(getGradianLine(line[0])>0):            
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (100, 20,250), 2)
            rightLines.append(line)
            

        if(getGradianLine(line[0])<0):            
            cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (100, 250,250), 2)
            leftLines.append(line)



    
    return [rightLines,leftLines]




rightLinesV1,leftLinesV1=seperateLine(newLines)
showImage(line_image)

"""
def detectBadSidelines(lines):
 
    rightSideArray=[]
    leftSideArray=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]    
        print(x1,"====",x2)
        if(x1>width/2 and x2>width/2):
            print("____________________RIGHT__________________________")
            rightSideArray.append(line)            
            
        if(x1<width/2 and x2<width/2):
            print("______________________LEFT_____________________________")        
            leftSideArray.append(line)            

    
    i=0
    j=0
    deleteArray=[]
    for line in leftSideArray:
        if(getGradianLine(line[0])>0):
            i=i+1
        if(getGradianLine(line[0])<0):
            j=j+1
            deleteArray.append

    if(i>j):

            
"""


def changeColorLines(lines):
    for line in lines:
        cv2.line(line_image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 0), 2)




def verificationLines(rightLines,leftLines,lines,initImagege):
    linesV2=[]
    i=140
    j=100
    threshold=10
    rho=1

    
    #detectBadSidelines(lines)
    while(len(rightLines)==0 or len(leftLines)==0): 
        
        
        
        roiV2=region(canny(cv2.GaussianBlur(initImage,(3,3),0)),height,width)
        changeColorLines(lines)
        
        lines=np.delete(np.array(lines) , np.s_[::] )
        linesV2=HoughLines(roiV2,rho,np.pi / 90 , threshold ,j , i)     
        #showImage(line_image)
        newLines=removeNullPoints(clearLines(linesV2,detecteLine(linesV2)))
        rightLines,leftLines=seperateLine(newLines)
        i=i+40 
        j=j-20
        threshold=threshold-2
    return [rightLines,leftLines]


rightLines,leftLines=verificationLines(rightLinesV1,leftLinesV1,newLines,initImage)
#print(rightLines[0][0])
#print(len(leftLines))
#print("===========")
showImage(line_image)

#===========================================================================================
# Select the left line from the list that contains the left lines and select the right line from the list that contains the right lines
#===========================================================================================
def selectTwoLines(leftLines,rightLines):
    randNumber1=0
    randNumber2=0
    if len(leftLines)>1:
        randNumber1=randrange(len(leftLines)-1)

    if len(rightLines)>1:
        randNumber2=randrange(len(rightLines)-1)


    while randNumber2==randNumber1 and len(leftLines)>1 and len(rightLines)>1 :
        randNumber2=randrange(len(rightLines)-1)
        randNumber1=randrange(len(leftLines)-1)


    leftLine = leftLines[randNumber1][0]
    rightLine = rightLines[randNumber2][0]

    #cv2.line(line_image, (leftLine[0], leftLine[1]), (leftLine[2], leftLine[3]), (0, 0,250), 5)
    #cv2.line(line_image, (rightLine[0], rightLine[1]), (rightLine[2], rightLine[3]), (0, 0,250), 5)

    return [leftLine,rightLine]


#===========================================================================================
# Select two random lines
#===========================================================================================
def selectTwoRandomLines(lines):
    randNumber1=randrange(len(lines)-1)
    randNumber2=randrange(len(lines)-1)

    while randNumber2==randNumber1 :
        randNumber2=randrange(len(lines)-1)

    leftLine = lines[randNumber1][0]
    rightLine = lines[randNumber2][0]

    #cv2.line(line_image, (leftLine[0], leftLine[1]), (leftLine[2], leftLine[3]), (0, 0,250), 5)
    #cv2.line(line_image, (rightLine[0], rightLine[1]), (rightLine[2], rightLine[3]), (0, 0,250), 5)

    return [leftLine,rightLine]
#===========================================================================================
# Get the a, b ,c from a line equation 
#===========================================================================================
def getLineParam(line):


#    For the line equation a*x+b*y+c=0, if we know two points(x1, y1)(x2,y2) in line, we can get
#        a = y1 - y2
#        b = x2 - x1
#        c = x1*y2 - x2*y1


    x1, y1, x2, y2 = line

    a = y1 - y2 
    b = x2 - x1
    c = x1 * y2 - x2 * y1
   
    return a,b,c
 
#===========================================================================================
# Find the intersaction point between these two lines
#===========================================================================================
def getCrossPoint(line1,line2):

#     if we have two lines: a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0,
#    when d= (a1 * b2 - a2 * b1) is zero, then the two lines are coincident or parallel.
#     The cross point is :
#         x = (b1 * c2 - b2 * c1) / d
#         y = (a2 * c1 - a1 * c2) / d
  

    a1, b1, c1 = getLineParam(line1)
    a2, b2, c2 = getLineParam(line2)
    d = a1 * b2 - a2 * b1

    if d == 0:
        cv2.line(line_image, (line1[0],line1[1]),  (line1[2],line1[3]), (250, 0, 0), 2)        
        cv2.line(line_image, (line2[0],line2[1]),  (line2[2],line2[3]), (250, 0, 0), 2)        
        return np.inf, np.inf
    
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d

    if(y>300):
        return np.inf, np.inf
    return abs(int(x)), abs(int(y))



#===========================================================================================
# Draw a point 
#===========================================================================================
def insertCircle(x,y, image):
    if x != float('inf'):
        cv2.circle(image, (int(x),int(y)), radius=5, color=(0, 250, 0), thickness=-1)


#===========================================================================================
# Return the distance between the intersaction point and each line (the result it is an array)
#===========================================================================================
def dist(point, lines):
    dists = []
    for line in lines:

        x1, y1, x2, y2 = line[0]
        x, y = point 
        if x==float('inf') or y==float('inf'):
             dists.append(int(-1))
             
        else :
            a, b, c = getLineParam(line[0])
            dist = abs((a*x + b*y + c) / np.sqrt(a*a + b*b))    
            dists.append(int(dist))
    return np.asarray(dists)


#===========================================================================================
# Calculate the number of inliers lines
#===========================================================================================
def getInliers(distinations,lines):
    
    theshold=10
    inlier_nbr=0

    for distination in distinations:
      
        if (7 < distination<23) :
            inlier_nbr=inlier_nbr+1

            #cv2.line(line_image, (x1,y1),  (x2,y2), (250, 200, 250), 5)


    """
    print(inlier_nbr2,"====",inlier_nbr)
    if inlier_nbr2 > inlier_nbr:
        return inlier_nbr2
    """

    return inlier_nbr


#===========================================================================================
# Calculate the number of inliers lines
#===========================================================================================
twoLines=selectTwoLines(leftLines,rightLines)

line1=twoLines[0]
line2=twoLines[1]


insertCircle(getCrossPoint(line1,line2)[0],getCrossPoint(line1,line2)[1],line_image)
#showImage(line_image)


#getInliers(dist(getCrossPoint(line1,line2), newLines), newLines)
#showImage(line_image)


#===========================================================================================
# Iterate the process N time
#===========================================================================================
def iteration(lines, N):
    inliersWithPoint=[]
    iniersArray=[]

    res={}
    inliers_nbr=0
    rightLines,leftLines=seperateLine(lines)
    rightLines,leftLines=verificationLines(rightLines,leftLines,lines,initImage)

    for i in range(N):
        twoLines=selectTwoLines(leftLines,rightLines)

        line1=twoLines[0]
        line2=twoLines[1]

        distination = dist(getCrossPoint(line1,line2), lines)
        inliers_nbr=getInliers(distination, lines)

        insertCircle(getCrossPoint(line1,line2)[0],getCrossPoint(line1,line2)[1],line_image)
        res[getCrossPoint(line1,line2)]=inliers_nbr
 
    
    return res

#===========================================================================================
# Select the vanishing point 
#===========================================================================================

def SelectVanishingPoint(iteration, blackImage, initImage):
    vanishing_point = max(iteration.items(), key=operator.itemgetter(1))[0]

    print("===========================")
    print(vanishing_point)
    print("===========================")

    x,y = vanishing_point
   
    cv2.circle(blackImage, (int(x),int(y)), radius=9, color=(250, 0, 0), thickness=-1)
    cv2.circle(initImage, (int(x),int(y)), radius=9, color=(20, 250, 20), thickness=-1)
    
    #first line from point to left side 
    cv2.line(initImage, (200, 720), (int(x),int(y)), (0, 255, 255), 4)

    #second line from point to right side 
    cv2.line(initImage, (950, 720), (int(x),int(y)), (0, 255, 255), 4)


iteration=iteration(newLines,32)
showImage(line_image)



def showImage(imageUrl):
    cv2.imshow('image', imageUrl)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()


SelectVanishingPoint(iteration,line_image,initImage)
showImage(initImage)