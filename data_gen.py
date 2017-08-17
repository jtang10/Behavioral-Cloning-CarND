import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

IMAGE_PATH = './data/IMG/'


def random_shear(image, angle):
    """Returns a 
    """
    # TODO: finish this function.
    return image, angle

def crop(image):
    """Returns a cropped version of the original image.

    Args:
        image: Input image (default size is (320, 160, 3))

    Returns:
        A cropped version of original image (top 70 and bottom 25 are cropped).
    """
    # TODO: finish this function.
    return image

def resize(image):
    """Returns a resized version of the original image.

    Args:
        image: Input image was cropped from the previous function.

    Returns:
        A resized image with dimension of (64, 64, 3).
    """
    # TODO: finish this function.
    return image

def brightness_adjust(image):
    # TODO: finish this function.
    return image

def random_rotation(image, angle, rotation_angle=15):
    """Returns rotated image and accordingly adjust steering angle.

    Args:
        image: Image to be rotated.
        angle: Steering angle associated with the image.
        rotation_angle: maximum angle to be rotated. Default value is 15°.
    
    Returns:
        Randomly rotated image and steering angle.
    """
    # TODO: finish this function.
    return image, angle

def flip(image, angle):
    """Returns the flipped image and steering angle.

    Args:
        image: Input image.
        angle: associated steering angle.

    Returns:
        Randomly flipped (with 50% chance) image and steering angle.
    """
    # TODO: finish this function.
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
    # image = crop(image)
    # image = resize(image)
    # image = brightness_adjust(image)
    # image, angle = random_rotation(image, angle)
    # image, angle = flip(image, angle)

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
    for i in samples_idx:
        camera_idx = np.random.randint(0, 2) # randomly choose one camera.
        line = samples[i]
        filename = line[camera_idx].split('/')[-1]
        angle = line[3]
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
            X_batch.append(aug_image)
            y_batch.append(aug_angle)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        yield X_batch, y_batch

if __name__ == "__main__":
    samples = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        samples.pop(0) # Remove the first row, which is title of the form
    num_samples = len(samples)

    iteration = 0
    for x, y in generate_batch(samples):
        print(x.shape)
        print(y)
        iteration += 1
        if iteration == 5:
            break