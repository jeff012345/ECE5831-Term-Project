import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import os
import shutil
import ntpath
import functools
import math

##
##
##  Config Values
##
##

## source images
image_path = "F:/ece5831/windows/Screens_256Pix/"

## destination to copy
save_image_path = os.path.join(image_path, "groupings")

## image preview size
image_size = (512, 512)

## groups
groups = ['paved_urban', 'paved_rural', 'unpaved_urban', 'unpaved_rural', 'unknown_urban', 'unkown_rural', 'unknown']

##
##
##  App Code
##
##

#create output folders
if os.path.exists(save_image_path) is False:
    os.mkdir(save_image_path)

groupPaths = []
for group in groups:
    path = os.path.join(save_image_path, group)
    groupPaths.append(path)

    try:
        os.mkdir(path)
    except OSError:
        print ("Output already exists: %s" % path)


class App:
    def __init__(self, window):
        self.window = window

        #self.createOutputFolders()
        self.loadImageList()
        self.createGui()

    def loadImageList(self):
        self.findClassifiedImages()
        
        self.images = []

        for root, dirs, files in os.walk(image_path):            
            if root.startswith(save_image_path):
                continue
            
            for file in files:
                if file.endswith(".jpg") and file not in self.classified:
                    self.images.append(os.path.join(root, file))

    def createGui(self):
        # Create a window
        self.window.title("OpenCV and Tkinter")

        colSpan = math.floor(len(groups)/2) + (len(groups) % 2)
        
        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(self.window, width = image_size[0], height = image_size[1])
        self.canvas.grid(row=0, column=0, columnspan=colSpan, pady = 2)

        self.canvas2 = tkinter.Canvas(self.window, width = image_size[0], height = image_size[1])
        self.canvas2.grid(row=0, column=colSpan, columnspan=colSpan, pady = 2)
        
        self.nextImage()

        self.setupButtons()

        # Run the window loop
        self.window.mainloop()

    def getImage(self, path):
        print("getImage", path)
        # Load an image using OpenCV
        cv_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return cv2.resize(cv_img, image_size, interpolation = cv2.INTER_AREA)

    def newImage(self, path):
        self.cv_img = self.getImage(path)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        (_, maskFilePath) = self.getMaskPath()
        self.cv_img2 = self.getImage(maskFilePath)
        self.photo2 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img2))
        self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

    def setupButtons(self):
        i = 0
        for group in groups:
            btn = tkinter.Button(self.window, text=group, command=functools.partial(self.groupBtnClick, group))
            btn.grid(row=1, column=i, pady = 2)
            i+=1

    def groupBtnClick(self, group):
        print(group)

        path = groupPaths[groups.index(group)]
        
        self.copyImageAndMask(path)
        self.nextImage()

    def nextImage(self):
        print("next image")
        self.currentPath = self.images.pop()
    
        if self.currentPath is None:
            print("all done")
            quit()
        
        print(self.currentPath)
        self.newImage(os.path.join(self.currentPath))

    def findClassifiedImages(self):
        self.classified = {}
        
        for root, dirs, files in os.walk(save_image_path):                      
            for file in files:
                if file.endswith(".jpg"):
                    self.classified[file] = True        
        
    def copyImageAndMask(self, folder):
        # copy image
        filename = ntpath.basename(self.currentPath)
        shutil.copyfile(self.currentPath, os.path.join(folder, filename))

        # find and copy mask
        (maskFilename, maskFilePath) = self.getMaskPath()

        shutil.copyfile(maskFilePath, os.path.join(folder, maskFilename))

    def getMaskPath(self):
        filename = ntpath.basename(self.currentPath)
        directory = os.path.dirname(self.currentPath)    
        maskFilename = filename.replace("_sat_", "_mask_").replace(".jpg", ".png")
        maskFilePath = os.path.join(directory, maskFilename)

        return (maskFilename, maskFilePath)
    
if __name__ == "__main__":
    App(tkinter.Tk())
