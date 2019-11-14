import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import os
import shutil
import ntpath
import functools

##
##
##  Config Values
##
##

## source images
image_path = "F:/ece5831/windows/Sample/"

## destination to copy
save_image_path = "F:/ece5831/windows/groupings/"

## image preview size
image_size = (300, 300)

## groups
groups = ['paved', 'gravel', 'unpaved', 'unknown']

##
##
##  App Code
##
##

#create output folders
groupPaths = []

for group in groups:
    path = os.path.join(save_image_path, group)
    groupPaths.append(path)

    try:
        os.mkdir(path)
    except OSError:
        print ("Output already exists or folder does not exist: %s\%s" % (save_image_path, group))
        #quit()

class App:
    def __init__(self, window):
        self.window = window

        #self.createOutputFolders()
        self.loadImageList()
        self.createGui()

    def loadImageList(self):
        self.images = []

        for root, dirs, files in os.walk(image_path):
            for file in files:
                if file.endswith(".jpg"):
                    self.images.append(os.path.join(root, file))

    def createGui(self):
        # Create a window
        self.window.title("OpenCV and Tkinter")

        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(self.window, width = image_size[0], height = image_size[1])
        self.canvas.grid(row=0, column=0, columnspan=len(groups))

        self.nextImage()

        self.setupButtons()

        # Run the window loop
        self.window.mainloop()

    def getImage(self, path):
        # Load an image using OpenCV
        cv_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return cv2.resize(cv_img, image_size, interpolation = cv2.INTER_AREA)

    def showImage(self, cv_img):
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))

        # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

    def newImage(self, path):
        self.cv_img = self.getImage(path)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def setupButtons(self):
        i = 0
        for group in groups:
            btn = tkinter.Button(self.window, text=group, command=functools.partial(self.groupBtnClick, group))
            btn.grid(row=1, column=i)
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

    def copyImageAndMask(self, folder):
        # copy image
        filename = ntpath.basename(self.currentPath)
        shutil.copyfile(self.currentPath, os.path.join(folder, filename))

        # find and copy mask
        directory = os.path.dirname(self.currentPath)    
        maskFilename = filename.replace("_sat_", "_mask_").replace(".jpg", ".png")
        maskFilePath = os.path.join(directory, maskFilename)

        shutil.copyfile(maskFilePath, os.path.join(folder, maskFilename))

if __name__ == "__main__":
    App(tkinter.Tk())
