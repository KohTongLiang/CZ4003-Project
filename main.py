import cv2
import argparse
import numpy as np
import pytesseract
import matplotlib.pyplot as pl
import math

debug_dir = './debug'
im_dir = './images'

# otsu global thresholding implementation
def otsu_t (img):
    print('Calculating Otsu threshold')
    ## de-noising
    # add gaussian blur to reduce image noise
    img = cv2.GaussianBlur(img, (5,5),0)

    cv2.imwrite(f'{debug_dir}/gblur_{args.imname}.jpg', img) # checkpoint 1 gaussian blur

    # get image histogram
    bins = 256
    hist, bin_edges = np.histogram(img, bins=bins)

    # save histogram plot for analysis later
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

    # get mean of class u0(t) and u1(t)
    mu0 = np.cumsum(hist * bin_mids) / w1
    mu1 = (np.cumsum((hist * bin_mids)[::-1]) / w2[::-1])[::-1]

    # get inter class variance
    inter_class_variance = w1[:-1] * w2[1:] * (mu0[:-1] - mu1[1:]) ** 2

    # maximise inter class variance
    i = np.argmax(inter_class_variance)

    # get threshold value
    th = bin_mids[:-1][i]

    print(f'Threshold calculated: {th}')

    _, output = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
    print('Otsu thresholding Applied')

    return output

# Using edge detectors
# takes in an additional flag parameter to 
def improvement1 (img, flag=1):
    if flag == 1:
        # create sobel mask
        Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        [rows, columns] = np.shape(img)  # we need to know the shape of the input grayscale image
        output = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

        # Compute output in x and y directions
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))
                gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))
                output[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        output = cv2.filter2D(output, -1, kernel)
    elif flag == 2:
        # create prewitt mask
        Gx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
        [rows, columns] = np.shape(img)
        output = np.zeros(shape=(rows, columns)) 

        # Compute output in x and y directions
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(Gx, img[i:i + 3, j:j + 3]))
                gy = np.sum(np.multiply(Gy, img[i:i + 3, j:j + 3]))
                output[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        output = cv2.filter2D(output, -1, kernel)
        output =otsu_t(output)
    elif flag == 3:
        # use canny edge detector
        output = cv2.Canny(img, 40, 70)
    return output

# Pixel Division
def improvement2(img):
    # image
    output1 = cv2.medianBlur(img,3)
    # background mask
    output2 = cv2.medianBlur(img,51)

    # divide the 2 blur images to remove the background. Afterwards, normalize pixel intensity.
    output = np.ma.divide(output1, output2).data
    output = np.uint8(255 * output / output.max())

    # use simple 2D filter to sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    output = cv2.filter2D(output, -1, kernel)
    return output

# adaptive thresholding
def improvement3(img, threshold = 25):
    # Original image size
    orignrows, origncols = img.shape
    
    # region size
    width = int(np.floor(orignrows/16) + 1)
    height = int(np.floor(origncols/16) + 1)
    
    # border padding
    Mextend = round(width / 2) - 1
    Nextend = round(height / 2) - 1
    
    # image padding
    aux = cv2.copyMakeBorder(img, top=Mextend, bottom=Mextend, left=Nextend,right=Nextend, borderType=cv2.BORDER_REFLECT)
    
    region = np.zeros((width,height),np.int32)
    
    # calculate image integral
    imageIntegral = cv2.integral(aux, region, -1)
    
    # get integral image size
    nrows, ncols = imageIntegral.shape
    
    result = np.zeros((orignrows, origncols))
    
    # calculate cumulative pixels
    for i in range(nrows - width):
        for j in range(ncols - height):
            result[i, j] = imageIntegral[i + width, j + height] - imageIntegral[i, j + height]+ imageIntegral[i, j] - imageIntegral[i + width, j]
     
    binary = np.ones((orignrows, origncols), dtype=np.bool)
    img = (img).astype('float64') * width * height
    
    # image binarization
    binary[img <= result * (100.0 - threshold) / 100.0] = False
    output = (255 * binary).astype(np.uint8)
    return output

## Tesseract function
# takes in an image to be fed into the LSTM-based engine
def translate(img):
    print('Translating image...')
    # Adding custom options
    custom_config = r'--oem 3 --psm 6'
    result = pytesseract.image_to_string(img, config=custom_config)
    return result

# program entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teserract OCR program parser.')
    parser.add_argument('--imname', default='sample01')
    parser.add_argument('--otsu', default=True)
    parser.add_argument('--improvement1', default=True)
    parser.add_argument('--improvement2', default=True)
    parser.add_argument('--improvement3', default=True)
    args = parser.parse_args()

    print('Reading image...')

    # read image in greyscale mode
    img = cv2.imread(f'{im_dir}/{args.imname}.png', 0)

    ## Base Otsu thresholding
    if args.otsu:
        output = otsu_t(img)
        cv2.imwrite(f'./processed_{args.imname}.jpg', output)
        ocr_result = translate(output)

        with open(f'./result_otsu_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)

    ## Improvement1: Edge detectors
    if args.improvement1:
        # sobel mask
        output = improvement1(img, 1)
        cv2.imwrite(f'./processed_improved1_sobel_{args.imname}.jpg', output)
        output = cv2.imread(f'./processed_improved1_sobel_{args.imname}.jpg', 0)
        ocr_result = translate(output)

        with open(f'./result_otsu_improved1_sobel_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)

        # prewitt mask
        output = improvement1(img, 2)
        cv2.imwrite(f'./processed_improved1_prewitt_{args.imname}.jpg', output)
        output = cv2.imread(f'./processed_improved1_prewitt_{args.imname}.jpg', 0)
        ocr_result = translate(output)

        with open(f'./result_otsu_improved1_prewitt_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)

        # canny
        output = improvement1(img, 3)
        cv2.imwrite(f'./processed_improved1_canny_{args.imname}.jpg', output)
        output = cv2.imread(f'./processed_improved1_canny_{args.imname}.jpg', 0)
        ocr_result = translate(output)

        with open(f'./result_otsu_improved1_canny_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)

    ## Improvement2: Pixel Division
    if args.improvement2:
        output = improvement2(img)
        cv2.imwrite(f'./processed_improved2_{args.imname}.jpg', output)
        ocr_result = translate(output)

        with open(f'./result_otsu_improved2_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)
    
    ## Improvement3: Adaptive thresholding
    if args.improvement3:
        output = improvement3(img)
        cv2.imwrite(f'./processed_improved3_{args.imname}.jpg', output)
        ocr_result = translate(output)

        with open(f'./result_improved3_{args.imname}.txt', 'w') as tf:
            tf.write(ocr_result)