import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

IMAGE_PATH = './data/IMG/'


def random_shear(image, angle, shear_factor=100):
    """Returns a randomly affine transformed image and corresponding steering angle.

    Args:
        image: Input image to be affine transformed.
        angle: The steering angle assoicated with image.
        shear_factor: the limit of random shear (default is 100).

    Returns:
        Transformed image and steering angle.
    """
    rows, cols, ch = image.shape
    rand_shear = np.random.randint(-shear_factor, shear_factor + 1)
    rand_point = [cols / 2 + rand_shear, rows / 2]
    # Note the points coordinates are different from points in image.
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], rand_point])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    dangle = rand_shear / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    angle += dangle

    return image, angle

def crop(image):
    """Returns a cropped version of the original image.

    Args:
        image: Input image (default size is (320, 160, 3))

    Returns:
        A cropped version of original image (top 70 and bottom 25 are cropped).
    """
    return image[55:135, :, :]

def resize(image):
    """Returns a resized version of the original image.

    Args:
        image: Input image was cropped from the previous function.

    Returns:
        A resized image with dimension of (64, 64, 3).
    """
    return cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)

def brightness_adjust(image):
    """Randomly adjust the brightness of the image.

    Args:
        image: Original image to be randomly adjusted for brightness.

    Returns:
        A brightness-adjusted image.
    """
    _image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    _image = np.array(_image, dtype = np.float64)
    random_brightness = 0.25 + np.random.random()
    _image[:, :, 2] *= random_brightness
    _image = np.array(_image, dtype = np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_HSV2RGB)
    return _image

def random_rotation(image, angle, rotation_angle=15.0):
    """Returns rotated image and accordingly adjust steering angle.

    Args:
        image: Image to be rotated.
        angle: Steering angle associated with the image.
        rotation_angle: maximum angle to be rotated. Default value is 15°.
    
    Returns:
        Randomly rotated image and steering angle.
    """
    rand_rotate = np.random.uniform(-rotation_angle, rotation_angle + 1)
    rad = np.pi / 180.0 * rand_rotate * -1.0
    angle += rad
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 25, 1.2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    return image, angle

def flip(image, angle, p=0.5):
    """Returns the flipped image and steering angle.

    Args:
        image: Input image.
        angle: associated steering angle.

    Returns:
        Randomly flipped (with 50% chance) image and steering angle.
    """
    if np.random.random() < p:
        image = np.fliplr(image)
        angle *= -1.0
    return image, angle

def augment_image_and_angle(image, angle):
    """Returns augmented image and angle.

    Args:
        image: Original image.
        angle: Original steering angle.

    Returns:
        augmented image and angle. Augmentation includes shear, crop, resize, 
        brightness adjust, rotation and flip based on the order.
    """
    # image, angle = random_shear(image, angle)
    image = crop(image)
    image = resize(image)
    image = brightness_adjust(image)
    # image, angle = random_rotation(image, angle)
    image, angle = flip(image, angle)

    return image, angle

def get_images_and_angles(samples, batch_size):
    """Returns and filename of the image and its related steering angle.

    Args:
        samples_idx: Indices of images at certain time-stamp, with size of  
            batch_size.

    Returns:
        A list of tuples contain the image filename and its associated angle.
    """
    # TODO: randomly choose 1 of 3 images and calculate its related angle. 
    #   Return a list of (filename, angle) with size of batch_size
    num_samples = len(samples)
    samples_idx = np.random.randint(0, num_samples, batch_size)
    images_and_angles = []
    correction = 0.28
    for i in samples_idx:
        camera_idx = np.random.randint(0, 3) # randomly choose one camera.
        line = samples[i]
        filename = line[camera_idx].split('/')[-1]
        angle = float(line[3])
        if camera_idx == 1:
            angle += correction
        elif camera_idx == 2:
            angle -= correction
        images_and_angles.append((filename, angle))
    return images_and_angles

def generate_batch(samples, batch_size=32):
    while 1:
        X_batch = []
        y_batch = []
        
        images_and_angles = get_images_and_angles(samples, batch_size)
        for image_addr, angle in images_and_angles:
            current_path = IMAGE_PATH + image_addr
            image = cv2.imread(current_path)
            aug_image, aug_angle = augment_image_and_angle(image, angle)
            # aug_image = image
            # aug_angle = angle
            X_batch.append(aug_image)
            y_batch.append(aug_angle)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        yield X_batch, y_batch

if __name__ == "__main__":
    # samples = []
    # with open('./data/driving_log.csv') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for line in reader:
    #         samples.append(line)
    #     samples.pop(0) # Remove the first row, which is title of the form
    # num_samples = len(samples)

    # iteration = 0
    # for x, y in generate_batch(samples):
    #     print(x.shape)
    #     cv2.imshow('image', x[0, :])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print(y)
    #     iteration += 1
    #     if iteration == 1:
    #         break

    test_img_addr = './data/IMG/right_2016_12_01_13_46_38_294.jpg'
    test_img = cv2.imread(test_img_addr)
    print(test_img.shape)
    cv2.imshow('image', test_img)
    cv2.imshow('flip', augment_image_and_angle(test_img, 1)[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
