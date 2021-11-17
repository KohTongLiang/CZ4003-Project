import cv2
import argparse
import numpy as np
import pytesseract
import matplotlib.pyplot as pl

debug_dir = './debug'
im_dir = './images'

def otsu_t (img):
    print('Calculating Otsu threshold')
    ## de-noising
    # add gaussian blur to reduce image noise
    img = cv2.GaussianBlur(img, (5,5),0)

    cv2.imwrite(f'{debug_dir}/gblur_{args.imname}.jpg', img) # checkpoint 1 gaussian blur

    ## threshold calculation

    # get image histogram
    bins = 256
    hist, bin_edges = np.histogram(img, bins=bins)

    # get histogram plot
    pl.hist(hist)
    pl.title(f'{args.imname}')
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(f"{debug_dir}/histoplot_{args.imname}.jpg")

    # perform normalization
    hist = np.divide(hist.ravel(), hist.max())

    # get centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # iterate over all possible thresholds and get probabilities w1 and w2
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    # get mean of class mu0(t)
    mu0 = np.cumsum(hist * bin_mids) / w1

    # get mean of class mu1(t)
    mu1 = (np.cumsum((hist * bin_mids)[::-1]) / w2[::-1])[::-1]

    # get inter class variance
    inter_class_variance = w1[:-1] * w2[1:] * (mu0[:-1] - mu1[1:]) ** 2

    # maximise inter class variance
    i = np.argmax(inter_class_variance)

    th = bin_mids[:-1][i]

    print(f'Threshold calculated: {th}')

    _, output = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    print('Otsu thresholding Applied')

    cv2.imwrite(f'{debug_dir}/processed_{args.imname}.jpg', output)
    return output

# for sanity check, cross-checking output with above function
def otsu_lib(img):
    _, output = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(f'{debug_dir}/debug_otsu_{args.imname}.jpg', output)

#### CV2 preprocessing functions
# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


## Tesseract function
def translate(img):
    print('Translating image...')
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(img, config=custom_config)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teserract OCR program parser.')
    image_path = parser.add_argument('--imname', default='sample01')
    args = parser.parse_args()

    print('Reading image...')

    # read image in greyscale mode
    img = cv2.imread(f'{im_dir}/{args.imname}.png', 0)

    img = otsu_t(img)
    otsu_lib(img) # sanity check with cv2 otsu thresholding
    ocr_result = translate(img)
    with open(f'./result_{args.imname}.txt', 'w') as tf:
        tf.write(ocr_result)