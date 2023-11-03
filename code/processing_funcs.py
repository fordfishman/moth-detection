import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

def show_rgb_image(image, title=None, conversion=cv2.COLOR_BGR2RGB):
    # https://gist.github.com/mstfldmr/45d6e47bb661800b982c39d30215bc88?permalink_comment_id=3380868

    # Converts from one colour space to the other. this is needed as RGB
    # is not the default colour space for OpenCV
    image = cv2.cvtColor(image, conversion)

    # Show the image
    plt.imshow(image)

    # remove the axis / ticks for a clean looking image
    plt.xticks([])
    plt.yticks([])

    # if a title is provided, show it
    if title is not None:
        plt.title(title)

    plt.show()
    return None

def bgr2hsv(b, g, r):
    """
    Convert open cv BGR data to HSV
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def h_proportions(pixels):
    """
    Grabs Hue of all pixels in an image, and
    creates bins of hue to see how common different
    hues are in a given image.
    """

    # grab height and width of image
    npixels, _ = np.shape(pixels)

    # reshape image data into a shape conversion function likes
    brg = pixels.reshape((1, npixels, 3))

    # convert to hsv and only grab h, reshape to just one dimension
    hsvs = cv2.cvtColor(brg, cv2.COLOR_BGR2HSV)[:,:,0].reshape(-1) # multiply by 2 bc of open cv's weird hue scale
    # print(hsvs[hsvs==360])
    # create 18 bands of hue
    nbands = 18
    bands = np.linspace(0, 360, num=nbands)
    # each band starts where the other ones stops
    start = bands[0:nbands-1]
    end = bands[1:nbands]

    proportions = list()
    # for each band
    for i in range(nbands-1):
        # how many pixels have a hue that falls within the band?
        npixels = np.where((hsvs >= start[i]) & (hsvs<end[i]))[0].shape[0]
        # what's the proportion of all of the pixels in the image fall in this bin?
        proportions.append(npixels/hsvs.shape[0])

    return np.array(proportions) # return proportions in an array

def image_process(img, imshow=False):

    """
    https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv

    """
    #convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of background (white) brightness in HSV
    sensitivity = 105
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,255,255])

    # Threshold the HSV image to get only bright background colors as a mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    #invert the mask to get only non-background items
    notmask = cv2.bitwise_not(mask)

    # get the largest contour (cropping sheet)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = cv2.findContours(at, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # get bounding box of the largest contour
    x,y,w,h = cv2.boundingRect(big_contour) #straight rectangle

    # crop the image at the bounds
    cropped_img = img[y:y+h, x:x+w]
    cropped_notmask = notmask[y:y+h, x:x+w]

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= notmask)

    # show outputs and masks
    if imshow:
        
        show_rgb_image(cropped_img)
        show_rgb_image(cropped_notmask)
    
    return cropped_img, cropped_notmask

def image_data(cropped_img, cropped_notmask, image_id):
    #blur to remove noise
    blur = cv2.blur(cropped_notmask, (10,10))

    #Apply thresholding to resharpen
    threshValue = 127
    ret, thresh = cv2.threshold(blur, threshValue, 255, cv2.THRESH_BINARY)
    # cv2_imshow(thresh)

    # Find the big contours/blobs on "thresh" (the filtered image):
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    try: hierarchy = hierarchy[0]
    except: hierarchy = []
    print("num total contours:", len(contours)) #print out how many total contours were found

    height, width = thresh.shape
    print("height, width:", height, width)
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    i=0

    image_data_dict = {
                'id' : [],
                'box_SA' : [],
                'cont_SA' : [],
                'B_avg' : [],
                'G_avg' : [],
                'R_avg' : [],
                'B_dom' : [],
                'G_dom' : [],
                'R_dom' : [],
                'B_cont_mean' : [],
                'G_cont_mean' : [],
                'R_cont_mean' : [],
                'H_cont_mean' : [],
                'S_cont_mean' : [],
                'V_cont_mean' : [],
                'H_diff_180' : []
    }

    hue_proportions = list()


    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)

        area = cv2.contourArea(contour)

        sub_id = f'{image_id}_{i}'

        # if w > 20 and h > 20:
        if area > 1000:
            # test if contour touches sides of image
            if x == 0 or y == 0 or x+w == width or y+h == height:
                print(sub_id,': region touches the sides')
                i=i+1
            else:
                cv2.rectangle(thresh, (x,y), (x+w,y+h), (255, 0, 0), 2) #draw a rectangle on thresh
                moth_thumbnail = cropped_img[y:y+h, x:x+w]
                surface = h*w #surface area of bounding box
                area = cv2.contourArea(contour) #surface area of contour

                average = moth_thumbnail.mean(axis=0).mean(axis=0) #average color of thumbnail (bounding box) [B,G,R]
                #get mean color by contour
                mask = np.zeros(cropped_notmask.shape,np.uint8)
                cv2.drawContours(mask,[contour],0,255,-1)
                cont_mean = cv2.mean(cropped_img,mask = mask) #get mean color within the mask from the image named "crop" [B,G,R]
                H_cont_mean,S_cont_mean,V_cont_mean = bgr2hsv(cont_mean[0],cont_mean[1],cont_mean[2])
                #get dominant color
                pixels = np.float32(moth_thumbnail.reshape(-1, 3))
                n_colors = 5
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv2.KMEANS_RANDOM_CENTERS

                _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
                _, counts = np.unique(labels, return_counts=True)
                dominant = palette[np.argmax(counts)]
                print(sub_id,": box_SA:",surface,"cont_SA:",area,"avg:",average,"dom:",dominant,"cont_mean:",cont_mean)
                print("HSV_cont_mean: ",H_cont_mean,S_cont_mean,V_cont_mean)
                print("H diff from 180:", abs(H_cont_mean-180))

                # save the data
                image_data_dict['id'].append(sub_id)
                image_data_dict['box_SA'].append(surface)
                image_data_dict['cont_SA'].append(area)
                image_data_dict['B_avg'].append(average[0])
                image_data_dict['G_avg'].append(average[1])
                image_data_dict['R_avg'].append(average[2])
                image_data_dict['B_dom'].append(dominant[0])
                image_data_dict['G_dom'].append(dominant[1])
                image_data_dict['R_dom'].append(dominant[2])
                image_data_dict['B_cont_mean'].append(cont_mean[0])
                image_data_dict['G_cont_mean'].append(cont_mean[1])
                image_data_dict['R_cont_mean'].append(cont_mean[2])
                image_data_dict['H_cont_mean'].append(H_cont_mean)
                image_data_dict['S_cont_mean'].append(S_cont_mean)
                image_data_dict['V_cont_mean'].append(V_cont_mean)
                image_data_dict['H_diff_180'].append(abs(H_cont_mean-180))

                hue_proportions.append(h_proportions(pixels))

                # Using cv2.copyMakeBorder() method
                bordertemp = cv2.copyMakeBorder(moth_thumbnail, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value = cont_mean) #make a thumbnail with a 10px border of the avg contour color
                show_rgb_image(bordertemp)
                i=i+1
        else:
            print(sub_id,': region is too small')
            i=i+1

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(thresh, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    # cv2_imshow(thresh)
    df = pd.DataFrame(image_data_dict)
    hues = pd.DataFrame(hue_proportions, columns=["h_{:.1f}".format(n) for n in np.linspace(0, 360, num=18) if n != 360 ] )
    df_hues = df.join(hues)
    return df_hues

def image_process2(img):
    """
    IN PROGRESS, NOT PROPERLY FUNCTIONAL
    """

    gryimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(gryimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 111, 15); #block size must be odd

    notmask = cv2.bitwise_not(th2)
    # get the largest contour (cropping sheet)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # get bounding box of the largest contour
    x,y,w,h = cv2.boundingRect(big_contour) #straight rectangle
    print(x,y,w,h)
    # cv2_imshow(notmask)
    return notmask

def standardize(x):
    """
    Standardizes columns of a Pandas DataFrame
    """

    return (x - x.mean(axis=0))/x.std(axis=0)

def hierarchical_clustering(df, labels, box_SA=True, cont_SA=True, avg_cont_color=True, hue_band=True, dend_title="", color_threshold=0.8):
    
    # columns we don't want to include
    drop_columns = ['id','B_avg', 'G_avg', 'R_avg', 'B_dom', 'G_dom', 
                    'R_dom', 'h_63.5', 'h_84.7', 'h_105.9', 'h_127.1', 
                    'h_148.2', 'h_169.4', 'h_190.6', 'H_diff_180'] 
    
    n_size_features = 0
    
    if not box_SA:
        drop_columns += ['box_SA']
        n_size_features += 1
    
    if not cont_SA:
        drop_columns += ['cont_SA']
        n_size_features += 1
    
    if not avg_cont_color:
        drop_columns += ['B_cont_mean', 'G_cont_mean', 'R_cont_mean', 
                         'H_cont_mean','S_cont_mean', 'V_cont_mean']
    
    if not hue_band:
        drop_columns += ['h_0.0', 'h_21.2', 'h_42.4', 'h_211.8', 'h_232.9', 
                         'h_254.1', 'h_275.3', 'h_296.5', 'h_317.6', 'h_338.8']
        
    data = df.drop(columns=drop_columns)
    
    data_std = standardize(data)
    
    n_color_features = len(data.columns) - n_size_features
    
    size_w = [ 1/n_size_features for _ in range(n_size_features) ]
    color_w = [ 1/n_color_features for _ in range(n_color_features) ]
    
    w = np.array(size_w + color_w)
    data_w = data_std*w
    
    clusters = linkage(data_w, method='ward', metric='euclidean')
    
    plt.figure(figsize=(13, 12))
    dendrogram(
        clusters,
        orientation='right',
        labels=labels,
        distance_sort='descending',
        show_leaf_counts=False,
        leaf_font_size=10,
        color_threshold = color_threshold
    )
    plt.title(dend_title)
    plt.show()
    
    return None