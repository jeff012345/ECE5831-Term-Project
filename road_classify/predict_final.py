#Import Libraries 
import csv
import cv2
import ntpath
import numpy as np
import os
import shutil
import tensorflow as tf
import util
from util import create_dir_or_is_empty
from model import create_model
from PIL import Image

############################################################################
################################## Config ##################################
############################################################################

# To run this program, make sure that the file locations that are created exist and that Predict2, predict2/RoP and 
# Predicted_Classes are all empty.  Predict should have the full size images and their masks.  Direct to a base 
# location and make sure the correct models are pointed to 
# NOTE: There may be a way to get the predicted class name files to be better than class_0, class_1 ect. 

#direct to the h5 file for paved vs unpaved classification
#direct to the h5 file for urban vs rural classification
#direct to the base folder where the rest of the files are

checkpoint_dir        = None 
model_PaveVUnpaved    = 'C:/Users/RMYERS83/Desktop/Project/Completed_Models/Pave_Unpave/Paved_Unpaved.h5'
model_UrbanVRural     = 'C:/Users/RMYERS83/Desktop/Project/Completed_Models/Urban_Rural/Urban_Rural.h5'
base_location         = 'C:/Users/RMYERS83/Desktop/Project/Screens_256Pix/'

## Possibly add something to check if these exist/ are empty
Full_Image_Predict    = base_location + 'Predict/' 				#file location of full sized images AND their masks
predict_data_root_dir = base_location + 'Predict2/'				#Output of full images, of 256x256 pixel and the masks
RPoI_Out              = base_location + 'Predict2/RoP'			#folder which has only road pixels present
predections_output    = base_location + 'Predicted_Classes/'	#folder with output files in their apropriate folders 
output_folders        = {}


#########################################################################################
################################## All Functions (def) ##################################
#########################################################################################

# Function Name: ClassName
# Function Desc: This seems pretty useless tbh
#       
def className(pred_class):
    return "class_" + str(pred_class)

# Function Name: copy_predicted_images
# Function Desc: copy the source images into their prediction folder
def copy_predicted_images(predictions, filepaths):
    make_class_folders(predictions)

    i = 0
    prec = []
    for p in predictions:
        temp1 = p[0:2].argmax() #find the max of the first two (Model 1)
        temp2 = p[2:4].argmax() #find the max of the first two (Model 2)
        
        #Assign to a class based on the four possible combos
        if (temp1 == 0 ) & (temp2 == 0):
            prec = 0
        elif (temp1 == 0) & (temp2 == 1):
            prec = 1
        elif (temp1 == 1) & (temp2 == 0):
            prec = 2
        elif (temp1 == 1) & (temp2 == 1):
            prec = 3		
        name     = className(prec)
        filename = ntpath.basename(filepaths[i])

        shutil.copyfile(filepaths[i], os.path.join(output_folders[name], filename))
        i += 1

# Function Name: create_output_folder
# Function Desc: duh       	
def create_output_folder(image_path):
    src_directory = os.path.dirname(image_path)
    (head, tail) = os.path.split(src_directory)
        
    output_sub_dir = os.path.join(RPoI_Out)
    
    if os.path.isdir(output_sub_dir) is False:
        os.mkdir(output_sub_dir)

    return output_sub_dir		
		
# Function Name: entropy
# Function Desc: Calculate a screen's entropy 		
def entropy(signal):
    lensig = signal.size;
    symset = list(set(signal));
    propab = [np.size(signal[signal==i])/(1.0*lensig) for i in symset];
    try:
        ent    = np.sum([p*np.log2(1.0/p) for p in propab]);
    except:
        print("ent is zero, assign to -1")
        ent = np.sum([1000*p for p in propab]);
    
    return ent;
	
# Function Name: find_images
# Function Desc: Gives lists of the satellite images and the masks in a directory 
def find_images(inFile):
    images = []
    masks  = []

    for root, dirs, files in os.walk(inFile):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))
            
            if file.endswith(".png"):
                masks.append(os.path.join(root, file))

    return (images, masks)	
	
# Function Name: gather_image_filepaths
# Function Desc: Gets all the jpg images in the directory passed to it
def gather_image_filepaths(file_path):
    predict_images = []
    A = os.listdir(file_path)
    for items in A:
        if items.endswith(".jpg"):
            predict_images.append(os.path.join(file_path, items))

    return predict_images
	
# Function Name: GetImageLists
# Function Desc: Accepts full path to image data and returns image number/id
#                list, image file full pathlist, and image mask full path list.
def GetImageLists(FullPathName):
    # Initialize lists
    ImageNum   = [];
    ImageFile  = [];
    ImageMask  = [];
    
    with os.scandir(FullPathName) as listOfEntries:    #Get all items in directory
        for entry in listOfEntries:                    #Address an individual image
            if entry.is_file():
                if "_sat" in entry.name:               
                    ImageFile.append(entry.path);
                    ImageNum.append(entry.name[:entry.name.find("_")]);
                elif "_mask" in entry.name:
                    ImageMask.append(entry.path);
                else:
                    # Error handling here
                    pass;
                    
    ImageNum.sort();
    ImageFile.sort();
    ImageMask.sort();
    
    return ImageNum, ImageFile, ImageMask;
	
# Function Name: GetImageWindows
# Function Desc: Accepts image file full path name and window size and returns
#                L-sized window partitions of image data.
def GetImageWindows(ImagePathName,L):
    # Initialize image object list
    ImageWindows = [];
    
    # Open image
    MyImage = Image.open(ImagePathName);
    
    # Get image size
    imHeight, imWidth = MyImage.size;
    
    lastPixelRight = imWidth - 1;
    lastPixelLower = imHeight - 1;
    
    # Partition image
    EndOfImage = False;
    EndOfLine  = False;
    
    left  = 0;
    upper = 0;
    right = L;
    lower = L;
    
    while not EndOfImage:             
        thisWindow = [left, upper, right, lower];
        CurImage = MyImage.crop(thisWindow);
        
        ImageWindows.append(CurImage);
        
        if right >= lastPixelRight:
            EndOfLine = True;
            if lower >= lastPixelLower:
                EndOfImage = True;
        else:
            EndOfLine = False;
        
        if not EndOfLine:
            left  += L;
            right += L;
        else:
            # Return
            left  = 0;
            right = L;
            
            upper += L;
            lower += L;
    
    return ImageWindows;

# Function Name: GetImageStats
# Function Desc: Accepts Pillow Image object for satellite image and mask and
#                and returns statistics on the pixels of interest.
def GetImageStats(PilImageObj,PilMaskObj):
    ImageData = PilImageObj.getdata();
    MaskData  = PilMaskObj.getdata();
    
    # Generate logical mask
    LogicalMask = [];
    for ii in range(len(MaskData)):
        pixel_i = MaskData[ii];
        if IsBlack(pixel_i):
            LogicalMask.append(False);
        elif IsWhite(pixel_i):
            LogicalMask.append(True);
        else:
            # Error handling here
            pass;
    
    # Compute statistics
    numPixelsOfInterest = 0;
    r = [];
    g = [];
    b = [];
    c = []; #R
    m = []; #R
    y = []; #R
    k = []; #R
    
    
    for ii in range(len(LogicalMask)):
        if LogicalMask[ii]:
            pixel_i = ImageData[ii];
            
            numPixelsOfInterest += 1;
            r.append(pixel_i[0]);
            g.append(pixel_i[1]);
            b.append(pixel_i[2]);
            kTemp = ( 1-np.max( [(pixel_i[0]/255),(pixel_i[1]/255),(pixel_i[2]/255)] ) ); # k = 1 - max(R',G',B') w R' = R/255
            k.append(kTemp); 
            try:
                c.append( (1 - (pixel_i[0]/255) - kTemp ) / (1 - kTemp) ); #C = (1-R'-K) / (1-K)
                m.append( (1 - (pixel_i[1]/255) - kTemp ) / (1 - kTemp) );
                y.append( (1 - (pixel_i[2]/255) - kTemp ) / (1 - kTemp) );
            except:
                print("kTemp value is 1, divide by zero.  Assing values to -1")
                c.append(10);
                m.append(10);
                y.append(10);
            
    
    mu_r  = np.mean(r);
    mu_g  = np.mean(g);
    mu_b  = np.mean(b);
    mu_c  = np.mean(c); #R
    mu_m  = np.mean(m); #R
    mu_y  = np.mean(y); #R
    mu_k  = np.mean(k); #R
    
    sig_r = np.std(r);
    sig_g = np.std(g);
    sig_b = np.std(b);
    sig_c = np.std(c); #R
    sig_m = np.std(m); #R
    sig_y = np.std(y); #R
    sig_k = np.std(k); #R
    
    ent_r = entropy(np.array(r));
    ent_g = entropy(np.array(g));
    ent_b = entropy(np.array(b));
    ent_c = entropy(np.array(c)); #R
    ent_m = entropy(np.array(m)); #R
    ent_y = entropy(np.array(y)); #R
    ent_k = entropy(np.array(k)); #R
    
    
    stats = [numPixelsOfInterest, mu_r, mu_g, mu_b, sig_r, sig_g, sig_b, ent_r, ent_g, ent_b, mu_c, mu_m, mu_y, mu_k, ent_c, ent_m, ent_y, ent_k];
    
    return stats;

# Function Name: IsWhite
# FUnction Desc: determine if a pixel is white	
def IsWhite(pixel):
    returnVal = False;
    if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
        returnVal = True;
        
    return returnVal;

# Function Name: IsBlack	
# FUnction Desc: determine if a pixel is Black
def IsBlack(pixel):
    returnVal = False;
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
        returnVal = True;
        
    return returnVal;
		
# Function Name: loadImageData
# Function Desc: takes in a list of files and saves the info in the same format as 
# 				 The model was trained       
def loadImagesData(files):
    images    = []
    filepaths = []        

    for file in files:        
        filepaths.append(file)
        image = cv2.imread(file)
        image = image.astype(np.float32)
        images.append(image)

    ## normalize the input because that's how the model was trained
    images = np.array(images) / 255
	
    return (images, filepaths)
	
# Function Name: LoadModel
# Function Desc: Checks if there are checkpoints and loads the weights, if not
# 				 load teh file in the model path passed in
def loadModel(model_filepath):
    if checkpoint_dir is None:
        return tf.keras.models.load_model(model_filepath)

    model   = create_model()
    weights = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(weights)
	
    return model

# Function Name: make_class_folders
# Function Desc: creates a new folder for each type of class defined by the 
#				 Nueral network       
def make_class_folders(predictions):
    if len(output_folders):
        return 

    for pred_class in range(0, predictions.shape[1]):
        name                 = className(pred_class)
        output_folders[name] = os.path.join(predections_output, name)
        os.mkdir(output_folders[name])

# Function Name: predict2
# Function Desc: Takes in two (binary output) models and their respective filepaths and 
#				 passes forward a single array of the predictions        
def predict2(model_1, model_2, files_AllPix, files_RoadPix):
    ## load the images to predict
    (predict_images_AllPix,  filepaths_AllPix)  = loadImagesData(files_AllPix)
    (predict_images_RoadPix, filepaths_RoadPix) = loadImagesData(files_RoadPix)
	## predict with the respective model 
    predictions_2 = model_2.predict(predict_images_AllPix) #results from second model
    predictions_1 = model_1.predict(predict_images_RoadPix) #results from first model

    PR = np.concatenate((predictions_1 ,predictions_2 ) , axis = 1 ) #stack the results of the two
    copy_predicted_images(PR, filepaths_AllPix)	

#Function Name: read_image
#Function Desc: read in image info as floats
def read_image(path):
    image = cv2.imread(path)
    return image.astype(np.float32)	
	
# Function Name: runModels
# Function Desc: gather the images in the directories and run the models on the appropriate files 
def runModels(Mod1, Mod2, file_dir):
    ## Load both individual models 
    model_Pave   = loadModel(Mod1)
    model_UrbRur = loadModel(Mod2)
	
    ## predict in batches so we don't store all the images in RAM
    predict_images_RoadPix = gather_image_filepaths(file_dir + '/RoP')   #These images only have pixels on the roads, rest is black 
    predict_images_AllPix  = gather_image_filepaths(file_dir)            #These images have all the pixels
    num_of_images = len(predict_images_AllPix)

    batch_size = 25
    count = 0
    while len(predict_images_AllPix) > batch_size:        
        predict2(model_Pave, model_UrbRur, predict_images_AllPix[:batch_size], predict_images_RoadPix[:batch_size])
        predict_images_RoadPix = predict_images_RoadPix[batch_size:]
        predict_images_AllPix  = predict_images_AllPix[batch_size:]
        count += batch_size
        print("Completed %s of %s" % (count, num_of_images))

    ## remainder
    predict2( model_Pave, model_UrbRur, predict_images_AllPix[:batch_size], predict_images_RoadPix[:batch_size])
    print("Completed")

# Function Name: save_image
# Function Desc: Saves an image in the respective image path 
def save_image(image_path, separated):
    save_dir  = create_output_folder(image_path)
    file_name = os.path.basename(image_path)
    file_path = os.path.join(save_dir, file_name + ".jpg")
    
    cv2.imwrite(file_path, separated)	
	
# Function Name: ScanDir
# Function Desc: Accepts full path to image directory and window size and 
#                generates image windows and CSV-file containing window
#                statistics.
def ScanDir(DirPathName, OutputFolder, L):
	# Get image lists
    ImageId, ImageSat, ImageMask = GetImageLists(DirPathName);
    
    # Generate image windows, window statistics, and save
    for ii in range(len(ImageId)):
        # Display status message
        msg = "Scanning image " + str(ii + 1) + " of " + str(len(ImageId)) + "...";
        print(msg);
        SaveName = OutputFolder + ImageId[ii]
        
        # Get image windows
        ImageSatWindows  = GetImageWindows(ImageSat[ii],L);
        ImageMaskWindows = GetImageWindows(ImageMask[ii],L);
        
        # Loop through windows
        ImageStats = [];
        for jj in range(len(ImageSatWindows)):
            SatWin_j  = ImageSatWindows[jj];
            MaskWin_j = ImageMaskWindows[jj];
            
            # Get statistics for current window
            CurImageStats = GetImageStats(SatWin_j,MaskWin_j); # Changing things here to only save if >100 pixel of interest
            if CurImageStats[0] > 100: #changed from 0 to 100 so that screens will have at least 100 pixel of interest
                CurImageStats.insert(0,jj);
                ImageStats.append(CurImageStats);
            
                #moved logic to save the file in here so it only saves if there are more than 100 pixels of interest
                # Save window image files
                SaveNameSat  = SaveName + "_sat_w" + str(jj) + ".jpg";
                SaveNameMask = SaveName + "_mask_w" + str(jj) + ".png";
            
                SatWin_j.save(SaveNameSat);
                MaskWin_j.save(SaveNameMask);
            
# Function Name: seperate_image
# Function Desc: Takes a satellite image and its mask and returns an image 
#				 with only the road pixels present, the rest are black  
def separate_image(image_path, mask_path):
    image = read_image(image_path)
    mask  = read_image(mask_path)

    ## make array same shape filled with zeros
    separated = np.zeros_like(mask) 

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            pixels = mask[x][y]
            if pixels[0] != 0.0 or pixels[1] != 0.0 or pixels[2] != 0.0:
                separated[x][y] = image[x][y]

    save_image(image_path, separated)	

#####################################################################################
################################## Run the program ##################################
#####################################################################################

## make output folder
create_dir_or_is_empty(predections_output)

# Convert full size images into 256x256 sized
ScanDir(Full_Image_Predict, predict_data_root_dir, 256)

#Take 256x256 pixel images/masks and create road only images
util.create_dir_or_is_empty(RPoI_Out)

(images, masks) = find_images(predict_data_root_dir)

for i in range(0, len(images)):
    print('Extracting road pixels: Image ' + str(i) + ' out of ' + str(len(images)))
    separate_image(images[i], masks[i])

	
# run the models 
runModels(model_PaveVUnpaved, model_UrbanVRural, predict_data_root_dir)


