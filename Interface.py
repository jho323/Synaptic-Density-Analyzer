import cv2 as cv
import numpy as np 
from tkinter import *
from PIL import ImageTk, Image
from tkinter import simpledialog
import ntpath
import os

class SpineCounter(object):

    """
    Identifies the spines in an image of a dendrite and labels according to morphological type
    
    """

    def __init__(self,array,filename,folder="",spineDict=""): # work that would 
        self.array = array
        self.avgKP =[]
        self.filename = str(filename)
        self.spineDict = spineDict
        self.folder = folder

    def crop(self,img,lst,filename):
        """
        crops images at each key point
        """
        spineDict = {}
        count = 0
        filename = filename.replace('.tif','')
        folder = filename+"_singleSpines"
        if not os.path.exists(folder):
            os.mkdir(str(folder))
            for coord in lst:
                h=30
                x,y = coord
                x = int(x)
                y = int(y)
                cropImg = img[y-h:y+h, x-h:x+h]
                count+=1
                if os.path.exists(folder):
                    cv.imwrite(os.path.join(os.path.abspath(folder),"singleSpine_"+str(count)+".tif"), cropImg)
                spineDict["singleSpine_"+str(count)+".tif"] = coord
        else:
            for coord in lst:
                h=30
                x,y = coord
                x = int(x)
                y = int(y)
                cropImg = img[y-h:y+h, x-h:x+h]
                count+=1
                spineDict["singleSpine_"+str(count)+".tif"] = coord

        return spineDict,folder

    @staticmethod    
    def findAverage(lst):
        """
        averages key-points found my counter

        """
    
        seen = []
        avgLst = []
        skipInd = []
        count = 0
        for i in range(len(lst)):
            if i not in skipInd:
                x1,y1  = lst[i]
                xValues = []
                yValues = []
                count+=1
            else:
                continue
            for t in range(i,len(lst)):
                x2,y2 = lst[t]

                if x1-15<=x2<=x1+15 and y1-15<=y2<=y1+15:
                    xValues.append(x2)
                    yValues.append(y2)
                    skipInd.append(t)

            if len(xValues) >= 5:
                avgX = (sum(xValues)/len(xValues))
                avgY = (sum(yValues)/len(xValues))
                avgLst.append((avgX,avgY))

        return avgLst

    def counter(self):
        """
        finds the features of image that are spines
         code taken and modified from:
         https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html

         """
        kpLst = []
        img = self.array

        dilatation_type = cv.MORPH_ELLIPSE
        dilatation_size = 1
        element = cv.getStructuringElement(dilatation_type, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
        img = cv.dilate(img,element)

        morph_elem = cv.MORPH_ELLIPSE #CLOSE
        morph_size = 1

        element = cv.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
        operation = cv.MORPH_CLOSE
        img = cv.morphologyEx(img, operation, element)

        morph_elem = cv.MORPH_RECT #OPEN
        morph_size = 2

        element = cv.getStructuringElement(morph_elem, (2*morph_size + 1, 2*morph_size+1), (morph_size, morph_size))
        operation = cv.MORPH_OPEN
        img = cv.morphologyEx(img, operation, element)

        nfeatures = 2500 #500
        scaleFactor = 1.1 #1.2
        nlevels = 40 #8
        edgeThreshold = 2 #31
        firstLevel = 0 #0
        WTA_K = 3 #2
        scoreType = 0 #0
        patchSize = 2 #31
        fastThreshold = 20 #20

        orb = cv.ORB_create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K, scoreType,patchSize,fastThreshold)

        # orb = cv.ORB_create()
       
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)

        for point in kp:
            x,y=point.pt

            kpLst.append((x,y))

        self.avgKP = SpineCounter(self.array,self.filename).findAverage(kpLst)
        spineDict,folder = SpineCounter(self.array,self.filename).crop(img,self.avgKP,self.filename)
        return self.avgKP,spineDict,folder

    def featureMatching(self,spineDict,folder):

        """
        Compares cropped images of spines from dendrite and compares them with model images of specific types of spines
         code taken and modified from:
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

         """  
        path = os.path.join(os.path.abspath(folder),folder)
        spineType = {}
        spineType["long thin"] = []
        spineType["filopodia"] = []
        spineType["mushroom"] = []
        spineType["stubby"] = []
        longThinCount = 0
        filopodiaCount = 0
        mushroomCount = 0
        stubbyCount = 0

        for img in os.listdir(path):

            spine = os.path.join(path,img)

# parameters for ORB detector
            nfeatures = 1500 #500
            scaleFactor = 1.1 #1.2
            nlevels = 40 #8
            edgeThreshold = 10 #31
            firstLevel = 0 #0
            WTA_K = 3 #2
            scoreType = 0 #0
            patchSize = 10 #31
            fastThreshold = 20 #20

#model images to be compared against
            longThin = os.path.join(os.path.abspath("Models"),"longThin_sample.tif")
            filopodia = os.path.join(os.path.abspath("Models"),"filopodia_sample.tif")
            mushroom = os.path.join(os.path.abspath("Models"),"mushroom_sample.tif")
            stubby = os.path.join(os.path.abspath("Models"),"stubby_sample.tif")

            modelsDict = {"long thin":longThin,"filopodia":filopodia,"mushroom":mushroom,"stubby":stubby}
            distances = {}
            for key in modelsDict:
                model = modelsDict[key]
                img1 = cv.imread(spine,0) # queryImage
                img2 = cv.imread(model,0) # trainImage
                # Initiate ORB detector
                orb = cv.ORB_create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K, scoreType,patchSize,fastThreshold)
                # find the keypoints and descriptors with ORB

                kp1, des1 = orb.detectAndCompute(img1,None)
                kp2, des2 = orb.detectAndCompute(img2,None)

                # create BFMatcher object
                bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
                # Match descriptors.
                matches = bf.match(des1,des2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key = lambda x:x.distance)

                if len(matches) != 0: #averages the distances found between matches of the cropped image and the model image
                    sumDst = 0
                    for i in range(0,len(matches)):
                        sumDst += matches[i].distance
                    avgDst = sumDst/len(matches)

                # print(key,avgDst,"avgDst")
                    distances[key] = avgDst #finds which type best associates with cropped image by finding the lowest average distance
                    guess = min(distances, key=distances.get)
                else:
                    break

            if guess == "long thin":
                spineType["long thin"].append(spineDict[str(img)])
                longThinCount+=1

            elif guess == "filopodia":
                spineType["filopodia"].append(spineDict[str(img)])
                filopodiaCount+=1
            elif guess == "mushroom":
                spineType["mushroom"].append(spineDict[str(img)])
                mushroomCount+=1
            elif guess == "stubby":
                spineType["stubby"].append(spineDict[str(img)])
                stubbyCount+=1

        return longThinCount,filopodiaCount, mushroomCount,stubbyCount, spineType

class Interface(object):

    def __init__(self,width,height,numLst):
        self.width = width
        self.height = height
        self.numLst = numLst

    def draw(self,canvas):
        newX = self.width-self.width//3

        canvas.create_rectangle(newX,0,self.width,self.height,fill="white",outline="gray",width=10)

class spineCountInterface(Interface):

    def __init__(self,width,height,numLst):
        super().__init__(width,height,numLst)

    def draw(self,canvas):
        newX = self.width-self.width//3
        leftBorder = newX+25
        #leftBorder = 825

        canvas.create_text(newX+90,20,font="arial 15",fill="black",text="Spine Count")

        spineTypes = ["Total","Long\nThin","Mushroom","Filopoida","Stubby"]
        colorTypes = ["red","cyan","green","purple","yellow"]

        for row in range(2): #creates table to display spine numbers
            for col in range(5):
                cellWidth = 70
                cellHeight = 40
                left = leftBorder+col*cellWidth
                top = 40+row*cellHeight

                if row == 0:
                    canvas.create_rectangle(left,top,left+cellWidth,top+cellHeight,width=1,fill="gray")
                    canvas.create_text(left+cellWidth//2,top+cellHeight//2,text=spineTypes[col],font="arial 10",fill=colorTypes[col])
                    
                else:
                    canvas.create_text(left+cellWidth//2,top+1.5*cellHeight//2,text=str(self.numLst[col]),font="arial 10",fill=colorTypes[col])
                    canvas.create_rectangle(left,top,left+cellWidth,top+cellHeight,width=1)

        canvas.create_rectangle(leftBorder,130,leftBorder+100,160,width=1) #Auto-counter box
        canvas.create_text(leftBorder+50,145,text="Auto counter",font="arial 10",fill="blue")
        canvas.create_text(leftBorder+39,180,text="Show", font="arial 10", fill="black")
        canvas.create_text(leftBorder+110,180,text="Hide", font="arial 10", fill="black")
        canvas.create_rectangle(leftBorder+10, 170, leftBorder+70, 190,width=1 ) #show generic markers
        canvas.create_rectangle(leftBorder+80, 170, leftBorder+140, 190,width=1 ) #hid generic markers

        canvas.create_rectangle(leftBorder,200,leftBorder+100,230,width=1) #Spine-counter box
        canvas.create_text(leftBorder+50,215,text="Spine counter",font="arial 10",fill="blue")
        canvas.create_text(leftBorder+39,250,text="Show", font="arial 10", fill="black")
        canvas.create_text(leftBorder+110,250,text="Hide", font="arial 10", fill="black")
        canvas.create_rectangle(leftBorder+10, 240, leftBorder+70, 260,width=1 ) #show spine markers
        canvas.create_rectangle(leftBorder+80, 240, leftBorder+140, 260,width=1 ) #hide spine markers

        canvas.create_rectangle(newX+300,10,newX+350,30,width=1) #help function box
        canvas.create_text(newX+325,20,text="Help",font="arial 10",fill="green")
        canvas.create_rectangle(newX+240,10,newX+290,30,width=1) #Reset button
        canvas.create_text(newX+265,20,text="Reset",font="arial 10",fill="green")
        canvas.create_rectangle(leftBorder+150, 130, leftBorder+200, 160,width=1 ) #undo box
        canvas.create_text(leftBorder+175,145,text="undo", font="arial 10", fill="black")
        canvas.create_rectangle(leftBorder+210, 130, leftBorder+260, 160,width=1 ) #clear box
        canvas.create_text(leftBorder+235,145,text="clear", font="arial 10", fill="black")
        canvas.create_rectangle(leftBorder+270, 130, leftBorder+320, 160,width=1 ) #select box
        canvas.create_text(leftBorder+295,145,text="select", font="arial 10", fill="black")
        canvas.create_line(leftBorder-15, 270, self.width-10, 270, width=2)

        #brightness and contrast buttons
        canvas.create_text(leftBorder+20,290,text="Brightness", font="arial 10 bold", fill="black")
        canvas.create_text(leftBorder+29,315,text="Decrease", font="arial 10", fill="black")
        canvas.create_text(leftBorder+100,315,text="Increase", font="arial 10", fill="black")
        canvas.create_rectangle(leftBorder, 305, leftBorder+60, 325,width=1 ) #Decrease Brightness
        canvas.create_rectangle(leftBorder+70, 305, leftBorder+130, 325,width=1 ) #Increase Brightness
        canvas.create_text(leftBorder+20,360,text="Contrast", font="arial 10 bold", fill="black")
        canvas.create_rectangle(leftBorder, 370, leftBorder+60, 390,width=1 ) #Decrease Contrast
        canvas.create_rectangle(leftBorder+70, 370, leftBorder+130, 390,width=1 ) #Increase Contrast
        canvas.create_text(leftBorder+29,380,text="Decrease", font="arial 10", fill="black")
        canvas.create_text(leftBorder+100,380,text="Increase", font="arial 10", fill="black")
        canvas.create_line(leftBorder-15, 420, self.width-10, 420, width=2)

        #grayscale, RGB
        canvas.create_text(leftBorder+29,440,text="Grayscale", font="arial 10 bold", fill="black")
        canvas.create_text(leftBorder+29,465,text="Gray", font="arial 10", fill="black")
        canvas.create_text(leftBorder+100,465,text="Original", font="arial 10", fill="black")
        canvas.create_rectangle(leftBorder, 455, leftBorder+60, 475,width=1 ) #Gray
        canvas.create_rectangle(leftBorder+70, 455, leftBorder+130, 475,width=1 ) #Original
        canvas.create_text(leftBorder+45,500,text="RGB Adjustment", font="arial 10 bold", fill="black")
        canvas.create_text(leftBorder+50,520,text="Red", font="arial 10", fill="Red")
        canvas.create_text(leftBorder+170,520,text="Green", font="arial 10", fill="Green")
        canvas.create_text(leftBorder+290,520,text="Blue", font="arial 10", fill="Cyan")
        canvas.create_text(leftBorder+30,550,text="-", font="arial 10", fill="black") #Red-
        canvas.create_text(leftBorder+70,550,text="+", font="arial 10", fill="black") #Red+
        canvas.create_text(leftBorder+150,550,text="-", font="arial 10", fill="black") #Green-
        canvas.create_text(leftBorder+190,550,text="+", font="arial 10", fill="black") #Green+
        canvas.create_text(leftBorder+270,550,text="-", font="arial 10", fill="black") #Blue-
        canvas.create_text(leftBorder+310,550,text="+", font="arial 10", fill="black") #Blue+

        canvas.create_rectangle(leftBorder+20, 540, leftBorder+40, 560,width=1 ) #Red+
        canvas.create_rectangle(leftBorder+60, 540, leftBorder+80, 560,width=1 ) #Red-
        canvas.create_rectangle(leftBorder+140, 540, leftBorder+160, 560,width=1 ) #Green+
        canvas.create_rectangle(leftBorder+180, 540, leftBorder+200, 560,width=1 ) #Green-
        canvas.create_rectangle(leftBorder+260, 540, leftBorder+280, 560,width=1 ) #BLue+
        canvas.create_rectangle(leftBorder+300, 540, leftBorder+320, 560,width=1 ) #Blue-

        #open,save,save as, directory, croppedSpines

        canvas.create_line(leftBorder-15, 630, self.width-10, 630, width=2)
        canvas.create_rectangle(leftBorder+190,650, leftBorder+240,670,width=1 )
        canvas.create_text(leftBorder+215,660,text="open",font="arial 10",fill="purple")
        canvas.create_rectangle(leftBorder+250,650, leftBorder+300,670,width=1 )
        canvas.create_text(leftBorder+275,660,text="save",font="arial 10",fill="purple")
        canvas.create_rectangle(leftBorder+310,650, leftBorder+360,670,width=1 )
        canvas.create_text(leftBorder+335,660,text="save as",font="arial 10",fill="purple")
        canvas.create_rectangle(leftBorder+130,650, leftBorder+180,670,width=1 )
        canvas.create_text(leftBorder+155,660,text="back",font="arial 10",fill="purple")
        canvas.create_rectangle(leftBorder+70,650, leftBorder+120,670,width=1 )
        canvas.create_text(leftBorder+95,660,text="directory",font="arial 10",fill="purple")
        canvas.create_rectangle(leftBorder-20,650, leftBorder+60,670,width=1 )
        canvas.create_text(leftBorder+20,660,text="Single Spines",font="arial 10",fill="purple")



    def drawHelpBox(self,canvas):
        cx = self.width//2
        cy = self.height//2

        canvas.create_rectangle(cx-100,cy-100,cx+100,cy+100,fill="white",width=5,outline="green")
        canvas.create_text(cx,cy-85,text="Press T to count any spine",font="arial 10",fill="black")
        canvas.create_text(cx,cy-65,text="Press L to count Long-Thin spine",font="arial 10",fill="black")
        canvas.create_text(cx,cy-45,text="Press M to count Mushroom spine",font="arial 10",fill="black")
        canvas.create_text(cx,cy-25,text="Press F to count Filopodia spine",font="arial 10",fill="black")
        canvas.create_text(cx,cy-5,text="Press S to count Stubby spine",font="arial 10",fill="black")
        canvas.create_text(cx,cy+15,text="Use arrow keys to move image",font="arial 10",fill="black")
        canvas.create_text(cx,cy+35,text="Press - to shrink image",font="arial 10",fill="black")
        canvas.create_text(cx,cy+55,text="Press + to enlarge image",font="arial 10",fill="black")
        canvas.create_text(cx,cy+80,text="Backspace then select marker \n              to delete",font="arial 10",fill="black")

class countTool(object):

    def __repr__(self):
        return ("x:"+str(self.x)+" y:"+str(self.y)+" "+str(self.color))

    def __init__(self,x, y, color,num):
        self.x=x
        self.y=y
        self.color=color
        self.num=num

    def position(self):
        return (self.x,self.y)

    def info(self):
        return (self.x,self.y,self.color,self.num)

    def drawCountMarkers(self,canvas):
        # canvas.create_text(self.x,self.y,text="."+str(self.num),fill=self.color,font="arial 10")
        canvas.create_oval(self.x-5,self.y-5,self.x+5,self.y+5,width=1,outline=self.color)

    def __eq__(self,other):
        return (isinstance(other,countTool)) and ((self.color == other.color) or ((self.x==other.x) and (self.y==other.y)))

class ImageDisplay(object):

    def __init__(self,left,top,imageWidth,imageHeight,preImg):
        self.left=left
        self.top=top
        # self.filename=filename
        self.preImg=preImg
        self.imageWidth=imageWidth
        self.imageHeight=imageHeight
        self.img = ImageTk.PhotoImage(self.preImg)
        
    def scale(self): #scales image 
        self.preImg = self.preImg.resize((self.imageWidth,self.imageHeight),Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.preImg)
        return self.preImg

    def contrastAndBrightness(self,alpha,beta):
        #alpha is contrast control
        #beta is brightness control
        array = np.asarray(self.preImg)
        modImg = np.zeros(array.shape,array.dtype) #modifies intensity of RGB of each pixel
        for r in range(array.shape[0]):
            for g in range(array.shape[1]):
                for b in range(array.shape[2]):
                    modImg[r,g,b] = np.clip(alpha*array[r,g,b]+beta,0,255)
        self.preImg = Image.fromarray(modImg)
        self.img = ImageTk.PhotoImage(self.preImg)
        return modImg,self.preImg
 
    def gray(self): #turns image into gray scale
        array = np.asarray(self.preImg)
        gray = cv.cvtColor(array,cv.COLOR_RGB2GRAY)
        self.preImg = Image.fromarray(gray)
        self.img = ImageTk.PhotoImage(self.preImg)
        return gray,self.preImg

    def red(self,red): #modifies the red of each pixel
        array = np.asarray(self.preImg)
        array.setflags(write=1)
        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                imgRed = array.item(x,y,0)
                if imgRed >= 8:
                    array.itemset((x,y,0),red+imgRed)
        

        self.preImg = Image.fromarray(array)
        self.img = ImageTk.PhotoImage(self.preImg)
        return array,self.preImg

    def blue(self,blue): #modifies the blue of each pixel
        array = np.asarray(self.preImg)
        array.setflags(write=1)

        for x in range(array.shape[0]):
            for y in range(array.shape[1]):
                imgBlue = array.item(x,y,1)
                imgBlue = imgBlue + blue
                array.itemset((x,y,1),imgBlue)
        

        self.preImg = Image.fromarray(array)
        self.img = ImageTk.PhotoImage(self.preImg)
        return array,self.preImg

    def green(self,green): #modifies the green of each pixel
        array = np.asarray(self.preImg)
        array.setflags(write=1)

        for x in range(array.shape[0]):
            for y in range(array.shape[2]):
                imgGreen = array.item(x,y,2)
                imgGreen = imgGreen + green
                array.itemset((x,y,2),imgGreen)
        

        self.preImg = Image.fromarray(array)
        self.img = ImageTk.PhotoImage(self.preImg)
        return array,self.preImg


    def importImage(self,canvas): 
        canvas.create_image(self.left+20, self.top+20, anchor=NW, image=self.img)


class fileExplorer(object):

    def __init__(self,width,height,filename):
        self.filename=filename
        self.width=width
        self.height=height
        self.saveLst = []
        self.empty = True
        self.imgLst = []
        self.windowHeight = 0
        self.entry = None

    def Open(self,path):
        fileLst=[] # returns a list of all the files in a folder
        for file in os.listdir(path):
            fileLst+=[path+"/"+str(file)]            
        return fileLst

    def drawOpen(self,canvas,fileLst): #draws the open window to display files
        height=len(fileLst)
        up = 15*height//2
        down = 15*height//2
        canvas.create_rectangle(self.width//2-150,self.height//2-up,self.width//2+150,self.height//2+down+20,fill="white")
        top = self.height//2-up+10
        left = self.width//2
        for i in range(len(fileLst)):
            path = fileLst[i]
            file = ntpath.basename(path)
            row = 15*i
            canvas.create_text(left,top+row,text=file,font="Arial 8")


    def Save(self,array,path,filename): #saves the file
        file = filename
        # file = str(ntpath.basename(file))
        newName = simpledialog.askstring("Input","Enter Filename:")
        print(newName)

        if file != None:
            cv.imwrite(os.path.join(path, newName+".tif"), array)
            return (path+"/"+newName+".tif")
        # cv2.imwrite(os.path.join("C:/Users/Jonathan's PC/Documents/112/termProject/Directory", file+'_newImage.tif'), array)

    def returnEntry(self,entry):
        return entry.get()


    def drawSave(self,canvas,fileLst): #draws save window
        height=len(fileLst)
        up = 15*height//2
        down = 15*height//2
        canvas.create_rectangle(self.width//2-150,self.height//2-up,self.width//2+150,self.height//2+down+20,fill="white")
        top = self.height//2-up+10
        left = self.width//2
        for i in range(len(fileLst)):
            path = fileLst[i]
            file = ntpath.basename(path)
            row = 15*i
            canvas.create_text(left,top+row,text=file,font="Arial 8")


    def drawDirectory(self,canvas,saveLst): #draws directory window
        self.imgLst = []
        self.saveLst = saveLst
        if self.saveLst == [] and self.empty:
            canvas.create_rectangle(self.width//2-300,self.height//2-100,self.width//2+100,self.height//2+100,fill="white",width = 10, outline="yellow")
            canvas.create_text(self.width//2-100,self.height//2,text="No Images in Directory",font="Arial 20 bold",fill="black")

        else:
            self.windowHeight=len(self.saveLst)
            up = 30*self.windowHeight//2
            down = 30*self.windowHeight//2
            canvas.create_rectangle(self.width//2-200,self.height//2-up,self.width//2+150,self.height//2+down+20,fill="white")
            top = self.height//2-up+10
            left = self.width//2
            for i in range(len(self.saveLst)):
                path = str(self.saveLst[i])
                file = ntpath.basename(path)
                preImg=Image.open(os.path.join(path), "r")
                preImg = preImg.resize((25,25),Image.ANTIALIAS)
                img = ImageTk.PhotoImage(preImg)

                row = 30*i
                self.imgLst.append(img)
                canvas.create_text(left,top+row,text=file,font="Arial 8")

    def drawDirectoryImage(self,canvas):
        up = 30*self.windowHeight//2
        top = self.height//2-up
        left = self.width//2

        for i in range(len(self.imgLst)):
            img = self.imgLst[i]
            row=30*i
            canvas.create_image(left-150,top+row, anchor=NW, image=img)

##################################################################################################################################################
#________________________________________________________________________________________________________________________________________________#

def init(data):
    data.filename = ""
    data.printHelpBox = False
    data.drawTotal = False
    data.drawLongThin = False
    data.drawMushroom = False
    data.drawFilopodia = False
    data.drawStubby = False
    data.numTotal = 0
    data.numLongThin = 0
    data.numMushroom = 0
    data.numFilopodia = 0
    data.numStubby = 0
    data.x = 0
    data.y = 0
    data.colorTypes = ["red","cyan","green","purple","yellow"]
    data.countLstTotal = []
    data.countLstType = []
    data.numLst = [data.numTotal,data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby]
    data.autoCount = False
    data.undo = False
    data.clear = False
    data.summedCount = 0
    data.imageLeft = 0
    data.imageTop = 0
    data.selectedDelete=False
    data.selected = False
    data.selectedMark = None
    data.alpha = 1
    data.beta = 0
    data.display = False
    data.osPaths = os.path.dirname(os.path.abspath("Interface.py"))
    data.displayError = False
    data.red = 0
    data.green = 0
    data.blue = 0
    data.fileLst = []
    data.drawWindow = False
    data.drawSaveWindow = False
    data.toBeSaved = False
    data.savePath = ''
    data.drawDirectory = False
    data.directoryLst = []
    data.filePath = ""
    data.drawMarkTotal = False
    data.drawMarkSpine = False
    data.spineDict = {}
    data.cropFolder = ""
    data.canCount = False
    data.total = 0
    data.savedDendrite = {}
    data.valuesToSave = [data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby,data.total,data.countLstTotal,data.countLstType]

    data.preImg = None
    data.array = None
    data.imageWidth = None
    data.imageHeight = None
    data.imageDisplay = None
    data.spineCounter = None

    data.interface = Interface(data.width,data.height,data.numLst)
    data.spineCountInterface = spineCountInterface(data.width,data.height,data.numLst)
    data.countTool = countTool(0,0,"black",1) #draws markers on spines
    data.fileExplorer = fileExplorer(data.width,data.height,data.filename)

#####################################################################################################

def mousePressed(event, data):
    if 1100<=event.x<=1150 and 10<=event.y<=30:
        if data.printHelpBox:
            data.printHelpBox = False
        else:
            data.printHelpBox = True

    data.x = event.x
    data.y = event.y

#_______________________________________________________________________________________________
#places markers according to mouse click

    if data.drawTotal and 0<=event.x<=800 and 0<=event.y<=data.height: 
        data.numTotal +=1
        color = data.colorTypes[0]
        markerT = countTool(data.x,data.y,color,data.summedCount)
        data.countLstType.append(markerT)
        
    elif data.drawLongThin and 0<=event.x<=800 and 0<=event.y<=data.height: 
        data.numLongThin +=1
        color = data.colorTypes[1]
        markerL = countTool(data.x,data.y,color,data.numLongThin)
        data.countLstType.append(markerL)
        
    elif data.drawMushroom and 0<=event.x<=800 and 0<=event.y<=data.height:
        data.numMushroom +=1
        color = data.colorTypes[2]
        markerM = countTool(data.x,data.y,color,data.numMushroom)
        data.countLstType.append(markerM)
        
    elif data.drawFilopodia and 0<=event.x<=800 and 0<=event.y<=data.height:
        data.numFilopodia +=1
        color = data.colorTypes[3]
        markerF = countTool(data.x,data.y,color,data.numFilopodia)
        data.countLstType.append(markerF)

    elif data.drawStubby and 0<=event.x<=800 and 0<=event.y<=data.height:
        data.numStubby +=1
        color = data.colorTypes[4]
        markerS = countTool(data.x,data.y,color,data.numStubby)
        data.countLstType.append(markerS)

# Autocounter Buttons____________________________________________________________________________________________________________________________________

    if 825<=event.x<=925 and 130<=event.y<=160: #Total auto-counter
        autoLst,data.spineDict,data.cropFolder = data.spineCounter.counter()
        for coor in autoLst:
            x,y=coor
            color=data.colorTypes[0]
            markerAuto = countTool(x+20,y+20,color,data.numTotal)
            data.countLstTotal.append(markerAuto)
            data.total +=1
        data.canCount = True

    if data.canCount:
        if 825<=event.x<=925 and 200<=event.y<=230: #Specific spine-counter
            data.numLongThin,data.numFilopodia,data.numMushroom,data.numStubby, autoDict = data.spineCounter.featureMatching(data.spineDict,data.cropFolder)
            for key in autoDict:
                coorLst = autoDict[key]

                if key == "long thin":
                    color = data.colorTypes[1]
                elif key == "mushroom":
                    color = data.colorTypes[2]
                elif key == "filopodia":
                    color = data.colorTypes[3]
                elif key == "stubby":
                    color = data.colorTypes[4]

                for coor in coorLst:
                    x,y=coor
                    markerAuto = countTool(x+20,y+20,color,data.numTotal)
                    data.countLstType.append(markerAuto)
                    # data.numTotal+=1


    if 835<=event.x<=895 and 170<=event.y<=190:
        data.drawMarkTotal = True

    elif 905<=event.x<=965 and 170<=event.y<=190:
        data.drawMarkTotal = False

    if 835<=event.x<=895 and 240<=event.y<=260:
        data.drawMarkSpine = True

    elif 905<=event.x<=965 and 240<=event.y<=260:
        data.drawMarkSpine = False

#_______________________________________________________________________________________________________________________________________________________________
    
    if 975<=event.x<=1025 and 130 <= event.y<= 160: # Undo a specific spine marker
        if len(data.countLstType) >= 1:
            for marker in data.countLstType:
                if data.drawTotal and marker == countTool(0,0,"red",0):
                    data.countLstType.remove(marker)
                    data.numTotal-=1
                    return
                elif data.drawLongThin and marker == countTool(0,0,"cyan",0):
                    data.countLstType.remove(marker)
                    data.numLongThin-=1
                    return
                elif data.drawMushroom and marker == countTool(0,0,"green",0):
                    data.countLstType.remove(marker)
                    data.numMushroom-=1
                    return
                elif data.drawFilopodia and marker == countTool(0,0,"purple",0):
                    data.countLstType.remove(marker)
                    data.numFilopodia-=1
                    return
                elif data.drawStubby and marker == countTool(0,0,"yellow",0):
                    data.countLstType.remove(marker)
                    data.numStubby-=1
                    return
    
    if 1035<=event.x<=1085 and 130 <= event.y<= 160: #clear all the markers of a spine type
        if len(data.countLstType) >= 1:
            for marker in data.countLstType:
                if data.drawTotal and marker == countTool(-1,-1,"red",-1):
                    data.countLstType.remove(marker)
                    data.numTotal-=1
                elif data.drawLongThin and marker == countTool(-1,-1,"cyan",-1):
                    data.countLstType.remove(marker)
                    data.numLongThin-=1
                elif data.drawMushroom and marker == countTool(-1,-1,"green",-1):
                    data.countLstType.remove(marker)
                    data.numMushroom-=1
                elif data.drawFilopodia and marker == countTool(-1,-1,"purple",-1):
                    data.countLstType.remove(marker)
                    data.numFilopodia-=1
                elif data.drawStubby and marker == countTool(-1,-1,"yellow",-1):
                    data.countLstType.remove(marker)
                    data.numStubby-=1

    if 1095<=event.x<=1155 and 130 <= event.y<= 160: #selected button
        if data.selected:
            data.selected = False
        else:
            data.selected = True

    if data.selected:
        data.drawTotal = False
        data.drawLongThin = False
        data.drawMushroom = False
        data.drawFilopodia = False
        data.drawStubby = False

        for i in range(len(data.countLstType)): #associates mouse click with a marker in the window
            x,y=data.countLstType[i].position()
            if x+5>=event.x>=x-5 and y+5>=event.y>=y-5:
                data.selectedMark = i
                data.selectedDelete=True
                return
    
    if 825<=event.x<=885 and 305<=event.y<=325: #brightness decrease
        data.beta-=1
        data.array,data.preImg = data.imageDisplay.contrastAndBrightness(data.alpha,data.beta)
        data.spineCounter = SpineCounter(data.array,data.filename)
        data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)

    elif 895<=event.x<=945 and 305<=event.y<=325: #brightness increase
        data.beta+=1
        data.array,data.preImg = data.imageDisplay.contrastAndBrightness(data.alpha,data.beta)
        data.spineCounter = SpineCounter(data.array,data.filename)
        data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)

    elif 825<=event.x<=885 and 370<=event.y<=390: #contrast decrease
        data.alpha-=.05
        data.array,data.preImg = data.imageDisplay.contrastAndBrightness(data.alpha,data.beta)
        data.spineCounter = SpineCounter(data.array,data.filename)
        data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)

    elif 895<=event.x<=945 and 370<=event.y<=390:#contrast increase
        data.alpha+=.05
        data.array,data.preImg = data.imageDisplay.contrastAndBrightness(data.alpha,data.beta)
        data.spineCounter = SpineCounter(data.array,data.filename)
        data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
    
   
    elif 1040<=event.x<=1090 and 10<=event.y<=30: #reset button
        if data.display:
            data.preImg=Image.open(data.filename)
            data.array = np.asarray(data.preImg)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(0,0,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)
            data.summedCount = 0
            data.numTotal = 0
            data.numLongThin = 0
            data.numMushroom = 0
            data.numFilopodia = 0
            data.numStubby = 0
            data.total = 0
            data.countLstTotal = []
            data.countLstType = []
            data.summedCount = data.numLongThin+data.numMushroom+data.numFilopodia+data.numStubby+data.numTotal
            data.numLst = [data.summedCount,data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby]
            data.spineCountInterface = spineCountInterface(data.width,data.height,data.numLst)

    if 995<=event.x<=1045 and 650<=event.y<=670: #open window
        if data.drawWindow:
            data.drawWindow=False
        else:

            data.fileLst = data.fileExplorer.Open(data.osPaths)
            data.drawWindow = True 

    elif 935<=event.x<=985 and 650<=event.y<=670: #back
        data.osPaths = os.path.dirname(os.path.abspath(data.osPaths))
        data.fileLst = []
        data.fileLst = data.fileExplorer.Open(data.osPaths)
     
        
    if data.drawWindow: #creates open window
        height = len(data.fileLst)
        up = 15*height//2
        if data.width//2-150<=event.x<=data.width//2+150 and data.height//2-up<=event.y<=data.height//2+up+20:
            ind = (event.y-data.height//2-up)//15 #allows mouse click to open a file
            path = data.fileLst[ind]
            if os.path.isdir(path):
                data.osPaths = path
                data.fileLst = []
                data.fileLst = data.fileExplorer.Open(data.osPaths)

            else:
              
                if ".tif" in path:
                    data.osPaths = path 
                    data.filename = os.path.join(data.osPaths)
                    data.filePath = (os.path.join(data.osPaths) , "r")
                    data.preImg=Image.open(os.path.join(data.osPaths), "r")
                    data.array = np.asarray(data.preImg)
                    data.img = ImageTk.PhotoImage(data.preImg)
                    data.imageWidth,data.imageHeight = data.preImg.size
                    data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
                    data.spineCounter = SpineCounter(data.array,data.filename)
                    data.display = True
                    data.osPaths = os.path.dirname(os.path.abspath(data.osPaths))
                    if path not in data.directoryLst:
                        data.directoryLst.append(path)
                    data.summedCount = 0
                    data.numTotal = 0
                    data.numLongThin = 0
                    data.numMushroom = 0
                    data.numFilopodia = 0
                    data.numStubby = 0
                    data.total = 0
                    data.countLstTotal = []
                    data.countLstType = []
                    data.summedCount = data.numLongThin+data.numMushroom+data.numFilopodia+data.numStubby+data.numTotal
                    data.numLst = [data.summedCount,data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby]
                    data.spineCountInterface = spineCountInterface(data.width,data.height,data.numLst)

                    # if data.filename not in data.savedDendrite:
                    #     data.savedDendrite[data.filename] = []
                    #     data.summedCount = 0
                    #     data.numTotal = 0
                    #     data.numLongThin = 0
                    #     data.numMushroom = 0
                    #     data.numFilopodia = 0
                    #     data.numStubby = 0
                    #     data.total = 0
                    #     data.countLstTotal = []
                    #     data.countLstType = []

                    # else:
                    #     values = data.savedDendrite[data.filename] 
                    #     print(values)
                    #     data.numLongThin = data.savedDendrite[data.filename][0]
                    #     data.numMushroom = data.savedDendrite[data.filename][1]
                    #     data.numFilopodia = data.savedDendrite[data.filename][2]
                    #     data.numStubby = data.savedDendrite[data.filename][3]
                    #     data.total = data.savedDendrite[data.filename][4]
                    #     data.countLstTotal = data.savedDendrite[data.filename][5]
                    #     data.countLstType = data.savedDendrite[data.filename][6]
                    #     data.summedCount = data.numLongThin+data.numMushroom+data.numFilopodia+data.numStubby+data.numTotal
                    #     data.numLst = [data.summedCount,data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby]
                    #     data.spineCountInterface = spineCountInterface(data.width,data.height,data.numLst)
                else:
                    data.displayError=True
                    data.drawWindow = False

    elif 1115<=event.x<=1165 and 650<=event.y<=670: #saveas
        if data.drawSaveWindow:
            data.drawSaveWindow = False
        else:

            data.fileLst = data.fileExplorer.Open(data.osPaths)
            data.drawSaveWindow = True

    if data.drawSaveWindow:
        height = len(data.fileLst)
        up = 15*height//2
        if data.width//2-150<=event.x<=data.width//2+150 and data.height//2-up<=event.y<=data.height//2+up+20:
            ind = (event.y-data.height//2-up)//15
            path = data.fileLst[ind]
            if os.path.isdir(path):
                print('hi')
                data.savePath = str(path)
            else:
                print("no")

    if 1075<=event.x<=1125 and 650<=event.y<=670: #save
        print("save")

        newFile = data.fileExplorer.Save(data.array,data.savePath,data.filename)
        data.toBeSaved = False
        data.directoryLst.append(newFile)

    
    elif data.displayError and data.width//2+20<=event.x<=data.width//2+70 and data.height//2+50<=event.y<=data.height//2+70:
        data.displayError = False
        data.drawWindow = True

    elif 825<=event.x<=885 and 455<=event.y<=475: #grayscale
        if data.display:
            data.array,data.preImg=data.imageDisplay.gray()
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    elif 895<=event.x<=955 and 455<=event.y<=475: #reset grayscale
        if data.display:
            data.preImg=Image.open(data.filename)
            data.array = np.asarray(data.preImg)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)
            

    elif 845<=event.x<=865 and 540<=event.y<=560: #red decrease button to decrease red of image
        if data.display:
            data.red-=5
            data.array,data.preImg=data.imageDisplay.red(data.red)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    elif 885<=event.x<=1005 and 540<event.y<=560: #red increase button to increase red of image
        if data.display:
            data.red+=5
            data.array,data.preImg=data.imageDisplay.red(data.red)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    elif 965<=event.x<=985 and 540<=event.y<=560: #green decrease button to decrease green of image
        if data.display:
            data.green-=5
            data.array,data.preImg=data.imageDisplay.green(data.blue)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    elif 1005<=event.x<=1025 and 540<event.y<=560: #green increase button to increase green of image
        if data.display:
            data.green+=5
            data.array,data.preImg=data.imageDisplay.green(data.green)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    elif 1085<=event.x<=1105 and 540<=event.y<=560: #blue decrease button to decrease blue of image
        if data.display:
            data.blue-=5
            data.array,data.preImg=data.imageDisplay.blue(data.blue)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    elif 1125<=event.x<=1145 and 540<event.y<=560: #blue increase button to increase blue of image
        if data.display:
            data.blue+=5
            data.array,data.preImg=data.imageDisplay.blue(data.blue)
            data.img = ImageTk.PhotoImage(data.preImg)
            data.imageWidth,data.imageHeight = data.preImg.size
            data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
            data.spineCounter = SpineCounter(data.array,data.filename)

    # canvas.create_rectangle(leftBorder+50,650, leftBorder+100,670,width=1 )

    if 875<=event.x<=925 and 650<=event.y<=670: #directory button
        if data.drawDirectory:
            data.drawDirectory = False
        else:
            data.drawDirectory = True

    if data.drawDirectory: #draws the directory window
        height = len(data.directoryLst)
        up = 30*height//2
        if data.width//2-150<=event.x<=data.width//2+150 and data.height//2-up<=event.y<=data.height//2+up+20:
            ind = (event.y-data.height//2-up)//30
            path = data.directoryLst[ind]
            if os.path.isdir(path):
                data.osPaths = path
                data.fileLst = []
                data.fileLst = data.fileExplorer.Open(data.osPaths)

            else:
                data.osPaths = path                    
                data.filename = os.path.join(data.osPaths)
                data.filePath = (os.path.join(data.osPaths),"r")
                data.preImg=Image.open(os.path.join(data.osPaths), "r")
                data.array = np.asarray(data.preImg)
                data.img = ImageTk.PhotoImage(data.preImg)
                data.imageWidth,data.imageHeight = data.preImg.size
                data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
                data.spineCounter = SpineCounter(data.array,data.filename)
                data.display = True
                data.osPaths = os.path.dirname(os.path.abspath(data.osPaths))
                if path not in data.directoryLst:
                    data.directoryLst.append(path)

    data.summedCount = data.numLongThin+data.numMushroom+data.numFilopodia+data.numStubby+data.numTotal
    data.numLst = [data.summedCount,data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby]
    data.spineCountInterface = spineCountInterface(data.width,data.height,data.numLst)

    # if data.filename != "":
    #     if data.savedDendrite[data.filename] == []:
    #         data.savedDendrite[data.filename].append(data.numLongThin)
    #         data.savedDendrite[data.filename].append(data.numMushroom)
    #         data.savedDendrite[data.filename].append(data.numFilopodia)
    #         data.savedDendrite[data.filename].append(data.numStubby)
    #         data.savedDendrite[data.filename].append(data.total)
    #         data.savedDendrite[data.filename].append(data.countLstTotal)
    #         data.savedDendrite[data.filename].append(data.countLstType)

###################################################################################################################################3


def keyPressed(event, data):
    if event.keysym == "t": #key press allows to manually place markers
        if data.drawTotal:
            data.drawTotal = False
        else:
            data.drawLongThin = False
            data.drawMushroom = False
            data.drawFilopodia = False
            data.drawStubby = False
            data.drawTotal = True
            data.selected = False
    elif event.keysym == "l":
        if data.drawLongThin:
            data.drawLongThin = False
        else:
            data.drawLongThin = True
            data.drawTotal = False
            data.drawMushroom = False
            data.drawFilopodia = False
            data.drawStubby = False
            data.selected = False
    elif event.keysym == "m":
        if data.drawMushroom:
            data.drawMushroom = False
        else:
            data.drawMushroom = True
            data.drawTotal = False
            data.drawLongThin = False
            data.drawFilopodia = False
            data.drawStubby = False
            data.selected = False
    elif event.keysym == "f":
        if data.drawFilopodia:
            data.drawFilopodia = False
        else:
            data.drawFilopodia = True
            data.drawTotal = False
            data.drawLongThin = False
            data.drawMushroom = False
            data.drawStubby = False
            data.selected = False
    elif event.keysym == "s":
        if data.drawStubby:
            data.drawStubby = False
        else:
            data.drawStubby = True
            data.drawTotal = False
            data.drawLongThin = False
            data.drawMushroom = False
            data.drawFilopodia = False
            data.selected = False

    elif event.keysym == "Up": #shifts image and the markers placed on the screen
        data.imageTop-=20
        for i in range(len(data.countLstType)):
            x,y,color,num = data.countLstType[i].info()
            newMarker = countTool(x,y-20,color,num)
            data.countLstType.pop(i)
            data.countLstType.append(newMarker)
    elif event.keysym == "Down": 
        data.imageTop+=20
        for i in range(len(data.countLstType)):
            x,y,color,num = data.countLstType[i].info()
            newMarker = countTool(x,y+20,color,num)
            data.countLstType.pop(i)
            data.countLstType.append(newMarker)
    elif event.keysym == "Left":
        data.imageLeft-=20
        for i in range(len(data.countLstType)):
            x,y,color,num = data.countLstType[i].info()
            newMarker = countTool(x-20,y,color,num)
            data.countLstType.pop(i)
            data.countLstType.append(newMarker)
    elif event.keysym == "Right":
        data.imageLeft+=20
        for i in range(len(data.countLstType)):
            x,y,color,num = data.countLstType[i].info()
            newMarker = countTool(x+20,y,color,num)
            data.countLstType.pop(i)
            data.countLstType.append(newMarker)
    data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)

    if event.keysym == "equal": #enlarges images
        data.imageWidth+=20
        data.imageHeight+=20
        data.preImg = data.imageDisplay.scale()
        data.array = np.asarray(data.preImg)
        data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
        data.spineCounter = SpineCounter(data.array,data.filename)
    
    elif event.keysym == "minus": #shrinks image
        data.imageWidth-=20
        data.imageHeight-=20
        data.preImg = data.imageDisplay.scale()
        data.array = np.asarray(data.preImg)
        data.imageDisplay = ImageDisplay(data.imageLeft,data.imageTop,data.imageWidth,data.imageHeight,data.preImg)
        data.spineCounter = SpineCounter(data.array,data.filename)
    if data.selectedDelete: #to delete specific spine markers
        if event.keysym == "BackSpace":
            if data.countLstTotal == countTool(-1,-1,"red",-1): #matches according to marker type and then subtracts the count of that type
                data.numTotal-=1
                data.countLstTotal.pop(data.selectedMark)
            elif data.countLstType[data.selectedMark] == countTool(-1,-1,"cyan",-1):
                data.numLongThin-=1
            elif data.countLstType[data.selectedMark] == countTool(-1,-1,"green",-1):
                data.numMushroom-=1
            elif data.countLstType[data.selectedMark] == countTool(-1,-1,"purple",-1):
                data.numFilopodia-=1
            elif data.countLstType[data.selectedMark] == countTool(-1,-1,"yellow",-1):
                data.numStubby-=1
            data.countLstType.pop(data.selectedMark)
            data.selectedDelete=False


    if event.keysym =="r": #resets the window to blank
        init(data)

    data.summedCount = data.numLongThin+data.numMushroom+data.numFilopodia+data.numStubby+data.numTotal
    data.numLst = [data.summedCount,data.numLongThin,data.numMushroom,data.numFilopodia,data.numStubby]
    data.spineCountInterface = spineCountInterface(data.width,data.height,data.numLst)

    # if data.filename != "":
    #     if data.savedDendrite[data.filename] == []:
    #         data.savedDendrite[data.filename].append(data.numLongThin)
    #         data.savedDendrite[data.filename].append(data.numMushroom)
    #         data.savedDendrite[data.filename].append(data.numFilopodia)
    #         data.savedDendrite[data.filename].append(data.numStubby)
    #         data.savedDendrite[data.filename].append(data.total)
    #         data.savedDendrite[data.filename].append(data.countLstTotal)
    #         data.savedDendrite[data.filename].append(data.countLstType)

######################################################################################################################################

def timerFired(data):
    pass

def redrawAll(canvas, data):

    canvas.create_rectangle(0,0,data.width,data.height,fill="black")

    if data.display:
        data.imageDisplay.importImage(canvas)
    data.interface.draw(canvas)
    data.spineCountInterface.draw(canvas)
    
    if data.printHelpBox:
        data.spineCountInterface.drawHelpBox(canvas)

    if data.drawMarkTotal:
        for mark in data.countLstTotal:
            mark.drawCountMarkers(canvas)

    if data.drawMarkSpine:
        for mark in data.countLstType:
            mark.drawCountMarkers(canvas)
    
    if data.drawWindow:
        data.fileExplorer.drawOpen(canvas,data.fileLst)

    if data.drawSaveWindow:
        data.fileExplorer.drawSave(canvas,data.fileLst)

    if data.drawDirectory:
        data.fileExplorer.drawDirectory(canvas,data.directoryLst)
        data.fileExplorer.drawDirectoryImage(canvas)

    canvas.create_rectangle(930,135,950,155,fill="",width=2)
    canvas.create_text(940,145,text=str(data.total),font="Arial 10",fill="red")


    if data.displayError:
        canvas.create_rectangle(data.width//2-300,data.height//2-100,data.width//2+100,data.height//2+100,fill="white",width = 10, outline="yellow")
        canvas.create_text(data.width//2-100,data.height//2,text="Please select image file",font="Arial 20 bold",fill="black")
        canvas.create_rectangle(data.width//2+20,data.height//2+50,data.width//2+70,data.height//2+70,fill="black")
        canvas.create_text(data.width//2+45,data.height//2+60,text="close",font="Arial 10",fill="white")
        
####################################
# use the run function as-is
####################################


# code copied from 112 website
def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()    

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 250 # milliseconds
    root = Tk()
    root.resizable(width=False, height=False) # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

run(1200, 700)