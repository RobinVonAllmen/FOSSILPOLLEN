from pylibCZIrw import czi as pyczi
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Rectangle as czrect
import bioformats as bf
from torchvision.ops import nms
from torch import tensor
from cv2 import rectangle
import shortuuid
import xml.etree.ElementTree as ET
from operator import indexOf
from multiprocessing import Pool
import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import cv2
import os
import shutil
import tqdm
import random


def get_focused_from_ROI(img_list, size : int = 100, shift : int = 50):

    """ Creates focused images from a four dimensional image array containing images of the same scene at different focal planes
    img_list:     np.ndarray of dimension 4 or 5 (will be converted to 4)
    size:           size of kernel size X size to search for most focused part across all image tiles
    shift:          Overlay between different ROI's
    """
    try:
        img_shp = np.shape(img_list[0])
        canvas = np.full(img_shp, 255, dtype=int)

        laplace = []

        ### Sliding window across image to find most in focus image.
        for x in range(0, img_shp[0] - size + shift, shift):
            for y in range(0, img_shp[1] - size +shift, shift):

                laplace = []
                for img in np.array(img_list)[:, x : x + size, y:y + size, :]:
                    val = cv2.Laplacian(img, cv2.CV_32F).var()
                    laplace.append(val)
                    
                
                canvas[ x : x + size, y:y + size, :] = np.min([canvas[ x : x + size, y:y + size, :], np.array(img_list)[indexOf(laplace, np.max(laplace)), x : x + size, y:y + size, :]], axis=0)          
        
        canvas = np.where(canvas != 0, np.float32(np.log(canvas)), 0)*175
        canvas = cv2.normalize(canvas , dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    except Exception as e:
        return e
    
    finally:
        return canvas
    

def squared_box(x1, y1, x2, y2, xabs : int, yabs : int):
    """
    Function returns a squared box from input values
    x1,x2,y1,y2:    coordinates from annotations
    xabs:           length of original iamge
    yabs:           width of original image
    """

    x = x2 - x1
    y = y2 - y1

    if y > x :
        x = y

    elif y < x:
        y = x

    xmid = int(np.mean([x1 , x2]))
    ymid = int(np.mean([y1 , y2]))

    x1 = int(xmid - x/2)
    x2 = int(xmid + x/2)
    y1 = int(ymid - y/2)
    y2 = int(ymid + x/2)

    if x1 < 0 :
        x1 = 0
        x2 = x
    
    if x2 > xabs-1:
        x2 = xabs - 1
        x1 = xabs - 1 - x 
    
    if y1 < 0:
        y1 = 0
        y2 = y

    if y2 > yabs -1:
        y2 = yabs - 1
        y1 = yabs - 1 - y

    return [int(x1), int(y1), int(x2), int(y2)]


def extract_images(file_path : os.PathLike, output_path: os.PathLike, tilesize: tuple = (1200,1200), ksize : tuple = 100, shift : int = 50, min_border : int = 100, num_cores = 1):
    """ extracts images from a scan (.czi file format)

    """
    filename = os.path.splitext(file_path)[0]
    width = tilesize[0]
    height = tilesize[1]

    np.seterr(divide="ignore")

    ### Start Extraction.

    try: 
        with pyczi.open_czi(file_path) as czidoc:
            
            
            tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=width,
                                                        total_tile_height=height,
                                                        min_border_width=min_border)
            

            scene_dims = czidoc.scenes_bounding_rectangle
            
            for scene in scene_dims:
                db = os.path.join(output_path, filename, "SCENE{}".format(scene))

                if not os.path.exists(db):
                    os.makedirs(db)

                dim = scene_dims[scene]
                x = dim.x
                y = dim.y
                w = dim.w
                h = dim.h

                
                tiles = tiler.tile_rectangle(czrect(x = 0, y=0, w = w, h = h))
                tile_count = len(tiles)

                if __name__ == "__main__":
                    
                    while len(tiles) != 0:

                        try:
                            pool = Pool(processes = num_cores)

                            starttime = time.time()
                            averagetime = 0
                            items = []
                            save_paths = []



                            for i in range(0, num_cores):
                                if len(tiles) != 0: break

                                tile =  tiles.pop(0)
                                print("\n", tile)

                                ch = [czidoc.read(roi=(x + tile.roi.x, y + tile.roi.y, tile.roi.w, tile.roi.h), plane={'C' : 0, 'Z' : z}) for z in range(0,12)]

                                if np.max(ch[0]) < 10:
                                    continue
                                
                                save_path = os.path.join(db, "S{}_X{}_Y{}.png".format(scene, tile.roi.x, tile.roi.y))
                                items.append(ch)
                                save_paths.append(save_path)

                            for idx, result in enumerate(pool.map(get_focused_from_ROI, items)):
                                print(save_paths[idx])
                                cv2.imwrite(save_paths[idx], result)

                        except IndexError as e:
                            print(e)
                            break

                        finally:
                            pool.close()
                            pool.join()
                            
                            averagetime = np.average([averagetime, time.time() - starttime])
                            timeestimated = averagetime * (tile_count/num_cores)

                            print("tiles finished: {} | {}\t time per batch: {}\t time estimated: {}\t\t\t".format(tile_count - len(tiles), tile_count, averagetime, timeestimated), end="\r", flush=True)
                    


        
    except KeyboardInterrupt:
        print("the extraction was stopped")

    except Exception as e:
        print(e)
    
    finally:
        return


#This function extracts all pollen grains from images using annotation files (.xml format!!). Both, images and annotation files need to be in the same folder and the filenames (not extension) must match.
def extract_cropped_pollen(INPUT_PATH, OUTPUT_PATH, tileDimensions):
    """ Uses .xml annotation files to extract annotated objects from original scanned image tile by using box-coordinates
    INPUT_PATH:     PATH to directory that contains image and annotation files (both need the same filename)
    OUTPUT_PATH:    PATH to store cropped images from objects contained in the annotation file"""
    # Tracks information on annotated images
    class_counts = {}
    grain_count = 0

    # Iterates through annoation files (.xml format) and extracts previously annotated objects by using the boundig box to calculate
    # a squared box around the object and cutting this part from the original image.
    for file in os.listdir(INPUT_PATH):

        if os.path.splitext(file)[1] != ".xml":
            continue
        tree = ET.parse(os.path.join(INPUT_PATH, file))
        root = tree.getroot()
        pollen = root.iter("object")

        # Filename of image that was referenced in annotation file.
        img_file = os.path.splitext(file)[0] + ".png"
        # Reads the image as numpy array.
        img = np.array(cv2.imread(os.path.join(INPUT_PATH, img_file)))

        for index, grain in enumerate(pollen):
            c = grain.find("name").text
            box = [int(child.text) for child in grain.find("bndbox").getchildren()]

            # Calculates a squared area around the annotated pollen grain.
            box = squared_box(box[0], box[1], box[2], box[3], tileDimensions[0], tileDimensions[1])

            if c not in class_counts.keys():
                CLASS_PATH = os.path.join(OUTPUT_PATH, c)
                if not os.path.exists(CLASS_PATH):
                    os.makedirs(CLASS_PATH)
                class_counts[c] = 0

            class_counts[c] = class_counts[c] + 1
            
            # Stores image as .PNG file in predifned SAVEDIR
            cv2.imwrite(os.path.join(OUTPUT_PATH,c, os.path.splitext(file)[0] + "_{}_{}".format(c, index) + ".png"), img[box[1]:box[3],box[0]:box[2],:])
            grain_count += 1

    print(class_counts)
    print(grain_count)



# Creates or clears the DB folder in which all original copy will be stored in. 

def build_database(source_dir : os.PathLike, dst_dir : os.PathLike, min_pollen_counts = 35):
    """ Choses equal amounts of images from different scans and stores them as a database in one folder.
    source_dir:         PATH to where directories are contained with cropped images from all scans used to build this dataset.
    dst_dir:            PATH to where to store the chosen images.
    min_pollen_counts:  Min amount of pollen per class in database.
    """

    image_per_slide = {}

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    for pollen_class in tqdm.tqdm(os.listdir(source_dir)):

        POLLEN_CLASS_PATH = os.path.join(source_dir, pollen_class)

        ### Sorts all images with the corresponding sample origin.
        for pollen_image in os.listdir(POLLEN_CLASS_PATH):
            FILE = pollen_image.split("__")[0]
            IMAGE_PATH = os.path.join(POLLEN_CLASS_PATH, pollen_image)

            if FILE not in image_per_slide.keys():
                image_per_slide[FILE] = [IMAGE_PATH]

            else:
                image_per_slide[FILE].append(IMAGE_PATH)

        POLLEN_DB_PATH = os.path.join(dst_dir, pollen_class)

        if not os.path.exists(POLLEN_DB_PATH):
            os.makedirs(POLLEN_DB_PATH)

        keys = list(image_per_slide.keys())

        while len(os.listdir(POLLEN_DB_PATH)) < min_pollen_counts:

            try:
                # dynamic list------------------
                key = keys[0]
                keys.append(keys.pop(0))
                # ------------------------------
            except:
                print(image_per_slide)
                print(keys)

            if len(image_per_slide[key]) == 0:
                keys.pop()
                continue
            
            
            try:
                index = random.randint(0,len(image_per_slide[key])-1)
            except:
                index = 0

            RANDOM_IMG_PATH = image_per_slide[key].pop(index)

            DESTINATION = os.path.join(POLLEN_DB_PATH, os.path.basename(RANDOM_IMG_PATH))
            shutil.copy(RANDOM_IMG_PATH, DESTINATION)

        image_per_slide = {}


def make_split(a, n:int):
    """Splits the vector a in n equal parts.
    a:      [] vector
    n:      amount of vectors of equal length to create
    """

    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)]   for i in range(n)]

def create_train_val_split(source_dir : os.PathLike, dst_dir : os.PathLike, fold : int = 5):
    """ This function creates the training and validation splits for n-fold cross-validation by partitioning the dataset in n-fold equal pieces. Each of these partitions demonstrate a validation split
    in n-fold many training loops. 

    source_dir:        os.PathLike      input directory with original dataset.
    dst_dir:           os.PathLike      Where tos tore the training/validation splits.
    fold:              int              how many different folds should be created.
    """

    # create Folder structure--------------
    if  os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)
    # -------------------------------------


    for subdir in os.listdir(source_dir):
        SUB_PATH = os.path.join(source_dir, subdir)

        files = os.listdir(SUB_PATH)
        random.shuffle(files)
        splits = make_split(files, fold)

        for index, split in tqdm.tqdm(enumerate(splits)):
            SPLIT_PATH = os.path.join(dst_dir, "split_{}".format(index))
            VAL_SPLIT = os.path.join(SPLIT_PATH, "validation", subdir)
            TRAIN_SPLIT = os.path.join(SPLIT_PATH, "train", subdir)

            if not os.path.exists(SPLIT_PATH):
                os.makedirs(SPLIT_PATH)
            
            os.makedirs(VAL_SPLIT)
            os.makedirs(TRAIN_SPLIT)

            for file in split:
                shutil.copy(os.path.join(SUB_PATH, file), os.path.join(VAL_SPLIT, file))

            for split in [x for i, x in enumerate(splits) if i != index]:
                for file in split:
                    shutil.copy(os.path.join(SUB_PATH, file), os.path.join(TRAIN_SPLIT, file))


def get_img_array(img_path, size):
    """
    """
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = heatmap.numpy()
    heatmap  = (heatmap - np.min(heatmap)) / (np.max(heatmap)-np.min(heatmap))
    #heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)
    return heatmap




def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.95):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap.save(os.path.splitext(cam_path)[0] + "_heatmap.jpg")


    # Display Grad CAM
    display(Image(cam_path))


def euclidean_distance_matrix(box_list):
    centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in box_list])  # Calculate centerpoints of all boxes

    input_centers = centers.reshape(1, -1, 2)  # Reshape input centers to match broadcasting dimensions
    box_centers = centers.reshape(-1, 1, 2)  # Reshape box centers to match broadcasting dimensions

    distances = np.linalg.norm(box_centers - input_centers, axis=2)  # Compute Euclidean distances using broadcasting

    return distances


def is_overlaying(box1, box2, threshold):
    # Extract the coordinates of the bounding boxes
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(xmin1, xmin2)
    y_top = max(ymin1, ymin2)
    x_right = min(xmax1, xmax2)
    y_bottom = min(ymax1, ymax2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)

    # Calculate the area of both bounding boxes
    
    box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    # Return True if IoU is above the threshold, False otherwise
    return iou >= threshold , iou

def calculate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var()
    return blur_score

def count_pollen (dir : os.PathLike, output : os.PathLike, seg_threshold : float = 0.15, threshold = 0.8, cnn_model = None , inspect_boxes : bool = False, categories = categories):
    
    """ This function reads all images in a directory and uses a segmentation model to extract and then a classification model to classify the images correctly.

    dir:            PathLike argument, source directory for input images.
    output:         PathLike argument, destination directory for classified pollen.
    seg_threshold:  float, used to decide minimal confidence to classify pollen.
    cnn_model:      of Type, tf.Model needed for classification.
    inspect_boxes:  bool, if True, input images will be saved with detected boxes in output folder.
    categories:     list, categories used as model classes.
    """
    
    #------------- Setting Up: -----------------
    normalization_layer = keras.layers.Rescaling(scale = 1./127.5, offset = -1)

    categories = [e.split("_")[0] for e in categories] # Original categories (excluding different sides such as polar views)
    categories += ["OTHERS" , "OFP" , "DUPLICATE"] # Adding Additional categories that are not trained by the model.

    pollen_DB = {"coordinates" : [], "scores" : [], "cnn_scores" : [], "multi_scores" : [], "paths" : [], "predicted_class" : [], "predicted_index" : []} # Output on model 
    DB = dict.fromkeys(np.unique(categories), 0)
    
    for category in np.unique(categories):  # Build Folder Structure:
        if os.path.exists(os.path.join(output, category)):
            shutil.rmtree(os.path.join(output, category))
        os.makedirs(os.path.join(output, category))

    if not os.path.exists(os.path.join(output, "VALIDATE_SEGMENTATION")):
        os.makedirs(os.path.join(output, "VALIDATE_SEGMENTATION"))
    #--------------------------------------------

    print("Initialized automated Pollen Analysis on {}".format(os.path.basename(dir)))
    
    frames = os.listdir(dir)
    random.shuffle(frames)
    frame_count = 0
    #------------- Analyze Sample ----------------
    for frame in tqdm.tqdm(frames, position=0, leave=True):

        if os.path.splitext(frame)[1] not in [".jpg", ".png"]:
            continue

        if frame_count >= len(frames)/10:
            break

        frame_count += 1

        #------------- Load and Preprocess input images ---------------------
        filename = os.path.splitext(frame)[0]
        X = int(filename.split("_X")[1].split("_")[0]) # X-Location on Microscope (relative px. Value)
        Y = int(filename.split("_Y")[1].split("_")[0]) # Y-Location on Microscope (relative px. Value)

        image_np = np.asarray(cv2.imread(os.path.join(dir, frame )))
        image_draw_on = image_np.copy()
        tileDimensions = np.shape(image_np)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        #---------------------------------------------------------------------

        #------------- Detection of Pollen -----------------------------------
        try:
            detections = detect_fn(input_tensor)
        except:
            continue
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()  for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        boxes = detections['detection_boxes']*1200
        scores = detections['detection_scores']
        classes = detections["detection_classes"]
        multiclass_scores = detections["detection_multiclass_scores"]
        #---------------------------------------------------------------------
        
        #------------- Postprocessing -----------------------------------------
        indeces = nms(boxes = tensor(boxes), scores = tensor(scores), iou_threshold=0.2)  # Filter out overlying pollen with non-maximal supression
        boxes = boxes[indeces]
        boxes += 1
        scores = scores [indeces]
        classes = classes[indeces]
        multiclass_scores = multiclass_scores[indeces]
        #---------------------------------------------------------------------

        #------------- Classification -----------------------------------------
        for idx, box in enumerate(boxes): # Loop through all reminding boxes
            
            UUID = shortuuid.uuid()  #Generate a unique identifier for the detected box.

            #------------- Preprocessing of segmented pollen objects -----------------------------------------
            box = squared_box(box[0], box[1], box[2], box[3], tileDimensions[0], tileDimensions[1])  # Crop out pollen image and preprocess it for classification.
            pollen = image_np[box[0]:box[2],box[1]:box[3],:]
            pollen_input = cv2.resize(pollen, dsize=(140,140))
            pollen_input = normalization_layer(pollen_input)

            if scores[idx] < seg_threshold:  # Filter out NPM with low prediction values.
                continue
            
            try:
                prediction = cnn_model.predict(np.expand_dims(pollen_input,axis=0))

            except Exception as e: 
                if np.shape(pollen_input)[0] == 0 or np.shape(pollen_input)[1] == 0: continue
                else: print(e)
            #---------------------------------------------------------------------------------------------------


            #------------- PostProcessing of Predcition Results -----------------------------------------
            npp_prediction = prediction[0][10:14] # Predictions for Non Pollen Palynomorph Classes.
            prediction = np.concatenate([prediction[0][:10],prediction[0][14:]])
            
            fused_scores = np.average([prediction, multiclass_scores[idx]], axis = 0)
         # Fuse scores from object detection and classification.
            predicted_index = np.argmax(fused_scores)
            predicted_class = categories[predicted_index].split("_")[0]
            predicted_score = np.max(fused_scores)
            #----------------------------------------------------------------------------------------------

            BOX_PATH = None
            #------------- Select the Correct Class depending on Confidence and Bluriness -----------------
            try:
                if  predicted_score > threshold[predicted_index]:
                    DB[predicted_class] = DB[predicted_class] + 1
                    BOX_PATH = os.path.join(output, predicted_class, "{}_{}_{}.png".format(UUID, predicted_class, int(predicted_score*100)))

                elif calculate_blurriness(pollen) < 50:
                    #BOX_PATH = os.path.join(output, "OFP", "{}_{}_{}.png".format(UUID, predicted_class, int(predicted_score*100))) 
                    BOX_PATH = None
                elif np.max(npp_prediction) > 0.8:
                    #BOX_PATH = os.path.join(output, "NPP", "{}_{}_{}.png".format(UUID, predicted_class, int(predicted_score*100)))
                    BOX_PATH = None
                elif   predicted_score > threshold[predicted_index]*0.75:
                    DB["OTHERS"] = DB["OTHERS"] + 1
                    BOX_PATH = os.path.join(output, "OTHERS", "{}_{}_{}.png".format(UUID, predicted_class, int(predicted_score*100)))  
            #------------------------------------------------------------------------------------------------    
            except:
                continue

            finally:
            #------------- Save Images and Box Detections ---------------------------------------------------
                if BOX_PATH:
                    cv2.imwrite(BOX_PATH, pollen)

                if inspect_boxes: 
                    image_draw_on = rectangle(image_draw_on, (box[1], box[0]), (box[3], box[2]), color = (250,0,0))

                    if box[2] + 10 > tileDimensions[1]:
                        coordinates =  (int(box[1]), int(box[0])-10)
                    else:
                        coordinates =  (int(box[1]), int(box[2])+10)

                    cv2.putText(
                    image_draw_on,
                    "{}:{}".format(categories[classes[idx]], int(np.max(multiclass_scores[idx])*100)),
                    coordinates,
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (0, 255, 0),
                    thickness=2
                    )

                pollen_DB["coordinates"].append([X + box[1], Y + box[0], X + box[3], Y + box[2]])
                pollen_DB["scores"].append(predicted_score)
                pollen_DB["multi_scores"].append(multiclass_scores[idx])
                pollen_DB["cnn_scores"].append(prediction)
                pollen_DB["predicted_index"].append(predicted_index)
                pollen_DB["predicted_class"].append(predicted_class)

                if BOX_PATH == None:
                    pollen_DB["paths"].append(UUID)
                else:
                    pollen_DB["paths"].append(BOX_PATH)
                BOX_PATH = None
            #-----------------------------------------------------------------------------------------------
        if inspect_boxes: cv2.imwrite(os.path.join(output, "VALIDATE_SEGMENTATION", frame) , image_draw_on)
        

    #------------- Find closely situated pollen ------------------------------------------------------------
    box_coordinates = pollen_DB["coordinates"]
    distances = euclidean_distance_matrix(box_coordinates)
    distances[distances == 0] = np.inf
    filtered_indices = np.argwhere(distances < 100)
    duplicates = [] # Store boxes that already were classified as duplicates.
    #--------------------------------------------------------------------------------------------------------


    #------------- Filtering out overlaying boxes and mark them as duplicates. ------------------------------
    for pairs in filtered_indices:

        if any(pair in duplicates for pair in pairs): continue

        overlay , iou = is_overlaying(box_coordinates[pairs[0]], box_coordinates[pairs[1]], threshold= 0.2)

        if overlay:
            duplicate = np.argmin([pollen_DB["scores"][pairs[0]], pollen_DB["scores"][pairs[1]]]) # Mark the pair with the lower confidence score as duplicate.
            DUPLICATE = pollen_DB["paths"][pairs[duplicate]]
            duplicates.append(pairs[duplicate])

            if pollen_DB["scores"][pairs[duplicate]] > threshold[pollen_DB["predicted_index"][pairs[duplicate]]]: # If the duplicate was counted as actual pollen type, reduce pollen counts by 1.
                shutil.move(DUPLICATE, os.path.join(output, "DUPLICATE", os.path.basename(DUPLICATE)))
                DB[pollen_DB["predicted_class"][pairs[duplicate]]] -=  1
    #--------------------------------------------------------------------------------------------------------


    #-------------- Store individual pollen detections as CSV File ------------------------------------------
    with open(os.path.join(output, "output.csv"), "w", newline = "") as outputfile:
        
        for idx, predclass in enumerate(pollen_DB["predicted_class"]):
    
            if idx in duplicates: continue

            outputfile.write("{};".format(predclass))
            outputfile.write("{};".format(pollen_DB["scores"][idx]))
            outputfile.write(";".join(str(e) for e in pollen_DB["multi_scores"][idx]))
            outputfile.write(";")
            outputfile.write(";".join(str(e) for e in pollen_DB["cnn_scores"][idx]))
            outputfile.write("\n")
    #--------------------------------------------------------------------------------------------------------

    return DB