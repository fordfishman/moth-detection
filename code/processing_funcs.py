import cv2 
import numpy as np
import matplotlib.pyplot as plt

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