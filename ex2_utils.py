import math
import numpy as np
import cv2



def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    signal_len, kernel_len = in_signal.size, k_size.size
    pad_signal = np.pad(in_signal, (kernel_len - 1,))  # padding with zeroes
    flip_kernel = np.flip(k_size)  # flip the kernel
    return np.array([np.dot(pad_signal[i:i + kernel_len], flip_kernel) for i in range(signal_len + kernel_len - 1)])


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    #check dim
    if (kernel.ndim == 1):
        width = kernel.shape[0]
        width_pad = int((width - 1) / 2)
        height = 1
        height_pad = 0
        kernel = kernel[-1::-1]
    else:
        height = kernel.shape[0]
        height_pad = int((height - 1) / 2)  # assuming kernel size is odd
        width = kernel.shape[1]
        width_pad = int((width - 1) / 2)  # assuming kernel size is odd
        kernel = kernel[-1::-1, -1::-1]
    #extara padding useing cv2.copyMakeBorder ’border Type’=cv2.BORDER_REPLICATE
    padded_image = cv2.copyMakeBorder(in_image, height_pad, height_pad, width_pad, width_pad, borderType=cv2.BORDER_REPLICATE)

    new_image = np.zeros(in_image.shape)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            sub_image = padded_image[i: i + height, j: j + width]
            new_image[i, j] = (sub_image * kernel).sum()
    return new_image


def calc_magnitude(image: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    :param image: first derive image
    :param image2: second derive image
    :return: magnitude image between image1 and image2
    """
    return np.sqrt(np.square(image) + np.square(image2))


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """

    kernel = np.array([[1, 0, -1]])
    kernel_transposed = kernel.T

    x_der = conv2D(in_image, kernel)
    y_der = conv2D(in_image, kernel_transposed)

    # Calculating magnitude => sqrt(iX**2 + iY**2)
    mag = calc_magnitude(x_der,y_der)

    # formula: tan^-1(x) == arctan(x)
    direction = np.arctan2(y_der, x_der)

    return direction, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
   # get gaus kernelwith sigma 1
    kernel = get_gaussian_kernel(k_size)
    #conv2D the image
    blur_img = conv2D(in_image, kernel)
    return blur_img


def get_gaussian_kernel(size: int, sigma: float = 1) -> np.ndarray:
    center = (int)(size / 2)
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    #  Gaussian kernel using  OpenCV
    Garr = cv2.getGaussianKernel(k_size, -1)
    Gkernel = Garr @ Garr.transpose()
    # filterd2D with ’border Type’=cv2.BORDER_REPLICATE
    blured = cv2.filter2D(in_image, -1, Gkernel, borderType=cv2.BORDER_REPLICATE)
    return blured


def FindRightTemplate(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges by {a + - or + 0 }
    :param img: image
    :return:Edge matrix
    """
    h, w = img.shape[:2]
    edge_image = np.zeros(img.shape)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == 0:
                if (((img[i - 1, j] > 0 and img[i + 1, j] < 0)
                     or (img[i - 1, j] < 0 and img[i + 1, j] > 0))
                        or ((img[i, j - 1] > 0 and img[i, j + 1] < 0)
                            or (img[i, j - 1] < 0 and img[i, j + 1] > 0))):
                    edge_image[i, j] = 1
            elif img[i, j] > 0:
                if ((img[i - 1, j] < 0 or img[i + 1, j] < 0)
                        or (img[i, j - 1] < 0 or img[i, j + 1] < 0)):
                    edge_image[i, j] = 1
            else:  # img[i, j] < 0
                if img[i - 1, j] > 0:
                    edge_image[i - 1, j] = 1
                elif img[i + 1, j] > 0:
                    edge_image[i + 1, j] = 1
                elif img[i, j - 1] > 0:
                    edge_image[i, j - 1] = 1
                elif img[i, j + 1] > 0:
                    edge_image[i, j + 1] = 1
    return edge_image

def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    #laplacian kernel
    laplacian =  np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    conv_img = conv2D(img, laplacian)
    return FindRightTemplate(conv_img)

def gaussianKernel(kernel_size: int) -> np.ndarray:
    g = np.array([1, 1])
    gaussian = np.array(g)
    for i in range(kernel_size - 2):
        gaussian = conv1D(g, gaussian)
    gaussian = np.array([gaussian])
    gaussian = gaussian.T.dot(gaussian)
    return gaussian / gaussian.sum()


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: Edge matrix
    """
    # guet gaus kernel
    gaussian = gaussianKernel(15)
    # laplacian kernel
    laplacian = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]])
    GLap = conv2D(gaussian, laplacian)
    conv_img = conv2D(img, GLap)
    return FindRightTemplate(conv_img)


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    #init
    Circles=[]
    # imShape
    H, W = img.shape
    #get the max radius // for int
    max_radius = min(min(H, W) // 2, max_radius)
    # Get each pixels gradients direction use cv2.Sobel
    sobel_y = cv2.Sobel(img, -1, 0, 1)
    sobel_x = cv2.Sobel(img, -1, 1, 0)
    #func for direction grad
    direction = np.arctan2(sobel_y, sobel_x)

    # Get Edges using cv2.Canny Edge detector
    Canny_Edge_det = cv2.Canny((img * 255).astype(np.uint8), 550, 100)

    radius_diff = max_radius - min_radius
    circle_hist = np.zeros((H, W, radius_diff))
    #  coordinates from canny
    Ycor, Xcor = np.where(Canny_Edge_det)

    # calculate the sin/cos for each edge pixel
    SinFunc = np.sin(direction[ Ycor, Xcor])
    CosFunc= np.cos(direction[ Ycor, Xcor])

    r_range = np.arange(min_radius, max_radius)

    for y1, x1, y2, x2 in zip( Ycor, Xcor, SinFunc, CosFunc):
        dir_sin = (r_range * y2).astype(np.int)
        dir_cos = (r_range * x2).astype(np.int)
        x_1 = x1 + dir_cos
        y_1 = y1 + dir_sin

        x_2 = x1 - dir_cos
        y_2 = y1 - dir_sin

        # Check centers in the image
        r_idx1 = np.logical_and(y_1 > 0, x_1 > 0)
        r_idx1 = np.logical_and(r_idx1, np.logical_and(y_1 < H, x_1 < W))

        # Check centers in the image
        r_idx2 = np.logical_and(y_2 > 0, x_2 > 0)
        r_idx2 = np.logical_and(r_idx2, np.logical_and(y_2 < H, x_2 < W))

        # Add circles to the histogram
        circle_hist[y_1[r_idx1], x_1[r_idx1], r_idx1] += 1
        circle_hist[y_2[r_idx2], x_2[r_idx2], r_idx2] += 1

    thresh = 11
    # Find all the circles centers
    # Find all the circles centers
    y, x, rad = np.where(circle_hist > thresh)

    Circles = np.array([x, y, rad + min_radius, circle_hist[y, x, rad]]).T

    # remove closes circels
    Circles = DellCloseCircles(Circles, min_radius // 2)
    print(thresh)
    return Circles

def DellCloseCircles(xyr: np.ndarray, radius: int) -> list:
    """
    delete a close circles for "pretty output"
    :param xyr: the Circles
    :param radius: min_radius // 2
    :return:
    """
    let_xyr = []

    while len(xyr) > 0:
        # Choose most ranked circle
        curr_arg = xyr[:, -1].argmax()
        curr = xyr[curr_arg, :]
        let_xyr.append(curr)
        xyr = np.delete(xyr, curr_arg, axis=0)

        # Find close neighbors
        dist_table = np.sqrt(np.square(xyr[:, :2] - curr[:2]).sum(axis=1)) < radius
        what_to_delete = np.where(dist_table)

        # Delete neighbors
        xyr = np.delete(xyr, what_to_delete, axis=0)
    return let_xyr

def gaussianForBIFI(x,sigma):
    return (1.0/(2*np.pi*(sigma**2)))*np.exp(-(x**2)/(2*(sigma**2)))


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    height = in_image.shape[0]
    width = in_image.shape[1]
    image_filtering = np.empty([height, width])
    half_kernel_size  = math.floor(k_size/2)
    extra_img_pad = cv2.copyMakeBorder(in_image, half_kernel_size, half_kernel_size,
                             half_kernel_size, half_kernel_size, borderType=cv2.BORDER_REPLICATE)
    for y in range(half_kernel_size , extra_img_pad .shape[0] - half_kernel_size ):
        for x in range(half_kernel_size , extra_img_pad .shape[1] - half_kernel_size ):
            pivot_v = extra_img_pad [y, x]
            neighbor_hood = extra_img_pad [y - half_kernel_size : y + half_kernel_size  + 1,
                            x - half_kernel_size : x + half_kernel_size  + 1]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-0.5 * np.power(diff / sigma_color, 2))
            Gspace = gaussianForBIFI(2 * half_kernel_size + 1, sigma_space)
            combo = Gspace  * diff_gau
            image_filtering[y - half_kernel_size, x -
                half_kernel_size] = (combo * neighbor_hood).sum() / combo.sum()

    cv2_image = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    return cv2_image, image_filtering
















