import cv2
import argparse
import numpy as np
import pytesseract
import matplotlib.pyplot as pl
import math

debug_dir = './debug'
im_dir = './images'

def otsu_t (img):
    print('Calculating Otsu threshold')
    ## de-noising
    # add gaussian blur to reduce image noise
    # img = cv2.GaussianBlur(img, (5,5),0)

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

    return output

def improvement1(img):
    output = img
    output1 = cv2.medianBlur(output,3) # blur 1
    output2 = cv2.medianBlur(output,51) # blur 2

    # divide the 2 blur images and normalize the pixel values
    output = np.ma.divide(output1, output2).data
    output = np.uint8(255*output/output.max())

    # use simple 2D filter to sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(output, -1, kernel)
    return output


def improvement2 (img):
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(img)  # we need to know the shape of the input grayscale image
    output = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

    # Now we "sweep" the image in both x and y directions and compute the output
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))  # y direction
            output[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(output, -1, kernel)
    return output

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

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
    parser.add_argument('--imname', default='sample01')
    parser.add_argument('--otsu', default=False)
    parser.add_argument('--otsu_improved1', default=False)
    parser.add_argument('--otsu_improved2', default=False)
    args = parser.parse_args()

    print('Reading image...')

    # read image in greyscale mode
    img = cv2.imread(f'{im_dir}/{args.imname}.png', 0)

    ## Base Otsu thresholding
    if args.otsu:
        output = otsu_t(img)
        cv2.imwrite(f'./processed_{args.imname}.jpg', output)
        otsu_lib(img) # sanity check with cv2 otsu thresholding to ensure that our implementation is correct
        ocr_result = translate(output)

        with open(f'./result_otsu_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)

    ## Improvement1: Equalized illumination
    if args.otsu_improved1:
        output = improvement1(img)
        cv2.imwrite(f'./processed_improved1_{args.imname}.jpg', output)
        ocr_result = translate(output)

        with open(f'./result_otsu_improved1_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)

    ## Improvement2: 
    if args.otsu_improved2:
        output = improvement2(img)
        cv2.imwrite(f'./processed_improved2_{args.imname}.jpg', output)
        output = cv2.imread(f'./processed_improved2_{args.imname}.jpg', 0)
        ocr_result = translate(output)

        with open(f'./result_otsu_improved2_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)