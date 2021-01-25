import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randrange
import operator

def showImage(image):
    plt.figure(figsize=(15,50))
    plt.imshow(image,cmap = 'gray')
    plt.show()


def canny(image):
    edges = cv2.Canny(image,60,120)
    return edges




#===========================================================================================
# Select the photo
#===========================================================================================
initImage=cv2.imread('images/13.jpg')







#===========================================================================================
# Region Of Interest
#===========================================================================================
gray = cv2.cvtColor(initImage, cv2.COLOR_BGR2GRAY)
showImage(initImage)
blured = cv2.GaussianBlur(initImage, (5, 5), 0)

cannyImg = canny(blured)
showImage(cannyImg)






#===========================================================================================
# Hough Transform
#===========================================================================================


blackImage=np.copy(initImage) * 0 


lines = cv2.HoughLinesP(cannyImg, 1, np.pi/180, 100, minLineLength=60, maxLineGap=10)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(blackImage, (x1, y1), (x2, y2), (250, 0,0), 2)


showImage(blackImage)


def drawLine(imageUrl,x1,y1,x2,y2):    
    cv2.line(imageUrl, (x1, y1), (x2, y2), (0,25, 0), 2)




def selectTwoLines(lines):
    randNumber1=randrange(len(lines)-1)
    randNumber2=randrange(len(lines)-1)

    while randNumber2==randNumber1 :
        randNumber2=randrange(len(lines)-1)

    firstLine = lines[randNumber1][0]
    secondeLine = lines[randNumber2][0]

    return [firstLine,secondeLine]



def getlineparam(line):
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

def getcrosspoint(line1,line2):
    """
    if we have two lines: a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0,
    when d(= a1 * b2 - a2 * b1) is zero, then the two lines are coincident or parallel.
    The cross point is :
        x = (b1 * c2 - b2 * c1) / d
        y = (a2 * c1 - a1 * c2) / d
    """

    a1, b1, c1 = getlineparam(line1)
    a2, b2, c2 = getlineparam(line2)
    d = a1 * b2 - a2 * b1

    if d == 0:
        return np.inf, np.inf
    
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d
    return abs(int(x)), abs(int(y))



def insertCircle(x,y, image):
    if x != float('inf'):
        cv2.circle(image, (int(x),int(y)), radius=5, color=(0, 250, 0), thickness=-1)
    #showImage(blackImage)

    

line1=selectTwoLines(lines)[0]
line2=selectTwoLines(lines)[1]

x1a, y1a, x2a, y2a = line1
x1b, y1b, x2b, y2b = line2
"""
drawLine(blackImage,x1a, y1a, x2a, y2a)
drawLine(blackImage,x1b, y1b, x2b, y2b)

showImage(blackImage)
"""
print(getcrosspoint(line1,line2))


insertCircle(getcrosspoint(line1,line2)[0],getcrosspoint(line1,line2)[1],blackImage)


#return the distance between the vanishing point and each line (the result it is an array)
def dist(point, lines):
    dists = []
    for line in lines:

        x1, y1, x2, y2 = line[0]
        x, y = point 
        if x==float('inf') or y==float('inf'):
             dists.append(int(0))
             
        else :
            a, b, c = getlineparam(line[0])
            dist = abs((a*x + b*y + c) / np.sqrt(a*a + b*b))    
            dists.append(int(dist))
    return np.asarray(dists)



def getinliers(distinations,lines):

    theshold=5
    inlier_nbr=0
    i=0

    for distination in distinations:
  
        if 0 < distination < theshold :
            #print(distination)
            inlier_nbr=inlier_nbr+1
            x1,y1,x2,y2=lines[i][0]
            #cv2.line(blackImage, (x1, y1), (x2, y2), (255,0, 0), 2)
        i=i+1    
    return inlier_nbr
   

    """

    print('===============================')
    print(inlier_nbr)        
    print('===============================')
    """

        
#getinliers(dist(getcrosspoint(line1,line2), lines), lines)
#showImage(blackImage)

#repeat N=32 time
#return res that contains a map of inliers number with the point (x,y)
def iteration(lines, N):
    inliersWithPoint=[]
    iniersArray=[]

    res={}
    inliers_nbr=0

    for i in range(N):
        line1=selectTwoLines(lines)[0]
        line2=selectTwoLines(lines)[1]

        distination = dist(getcrosspoint(line1,line2), lines)
        inliers_nbr=getinliers(distination, lines)
        print('=================================================================')
        print(inliers_nbr)


        insertCircle(getcrosspoint(line1,line2)[0],getcrosspoint(line1,line2)[1],blackImage)
        res[getcrosspoint(line1,line2)]=inliers_nbr
    
    print(res)
    return res





map = iteration(lines, 32)
showImage(blackImage)

#select the vanidhing point from the map that contains (point):{number inliers}
def SelectVanishingPoint(map, image, initImage):
    vanishing_point = max(map.items(), key=operator.itemgetter(1))[0]
    print("===========================")
    print(vanishing_point)
    print("===========================")

    x,y = vanishing_point
    print(x)
    print(y)
    cv2.circle(image, (int(x),int(y)), radius=9, color=(250, 0, 0), thickness=-1)
    cv2.circle(initImage, (int(x),int(y)), radius=9, color=(20, 250, 20), thickness=-1)
    
    #first line from point to left side 
    cv2.line(initImage, (220, 720), (int(x),int(y)), (0, 255, 255), 4)

    #second line from point to right side 
    cv2.line(initImage, (1050, 720), (int(x),int(y)), (0, 255, 255), 4)



def showImage(imageUrl):
    cv2.imshow('image', imageUrl)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()





SelectVanishingPoint(map,blackImage,initImage)
#showImage(blackImage)
showImage(initImage)
    
    

#showImage(initImage)