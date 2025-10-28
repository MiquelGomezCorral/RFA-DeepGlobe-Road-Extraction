"""Image transformation.

Functions and classes for image transformations and sample representation.
"""
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


class Transformation(Enum):
    """Enum class that defines a set of image transformation operations.

    Each transformation is represented by a name and a corresponding function
    that applies the transformation to an input image.

    Transformations include:
        - SPLIT: Splits the image into smaller grid pieces.
        - ROTATE: Rotates the image by a random degree.
        - MIRROR: Mirrors the image horizontally.
        - SUB: Extracts a random subimage from the original image.
        - SHUFFLE: Shuffles the pixels of the image.
        - CIRCLES: Adds random circles to the image.
        - BRIGHTNESS: Adjusts the brightness of the image.
        - INVERT: Inverts the colors of the image.
        - SHIFT_COLOR: Shifts the color tones of the image randomly.
        - NOISE: Adds random noise to the image.
    """

    SPLIT = ("_spli", lambda img_in, n_pieces: split_image_into_grid(img_in, n_pieces))

    ROTATE = ("_rot", lambda img_in, seed: rotate_image(img_in, seed))
    MIRROR = ("_mir", lambda img_in, seed: mirror_image(img_in))
    SUB = ("_sub", lambda img_in, seed: random_subimage(img_in, seed))

    CIRCLES = ("_cir", lambda img_in, seed: add_random_circles(img_in, seed))
    BRIGHTNESS = ("_bri", lambda img_in, seed: set_brightness(img_in, seed))

    SHIFT_COLOR = ("_shi", lambda img_in, seed: shift_color_towards_random(img_in, seed))
    NOISE = ("_noi", lambda img_in, seed: add_noise_to_image(img_in, seed))

    def pipe_to_name(pipeline):
        """Convert a list of transformations to a string representation of their combined names.

        Args:
            pipeline (list): A list of transformations to be converted into a name string.

        Returns:
            str: A concatenated string containing the names of the transformations applied.
        """
        name = ""
        if Transformation.ROTATE in pipeline:
            name += Transformation.ROTATE.value[0]
        if Transformation.MIRROR in pipeline:
            name += Transformation.MIRROR.value[0]
        if Transformation.SUB in pipeline:
            name += Transformation.SUB.value[0]
        if Transformation.SHUFFLE in pipeline:
            name += Transformation.SHUFFLE.value[0]
        if Transformation.CIRCLES in pipeline:
            name += Transformation.CIRCLES.value[0]
        if Transformation.BRIGHTNESS in pipeline:
            name += Transformation.BRIGHTNESS.value[0]
        if Transformation.INVERT in pipeline:
            name += Transformation.INVERT.value[0]
        if Transformation.SHIFT_COLOR in pipeline:
            name += Transformation.SHIFT_COLOR.value[0]
        if Transformation.NOISE in pipeline:
            name += Transformation.NOISE.value[0]
        return name

    def name_to_pipe(name: str):
        """Convert string representation of a combined transforms name back into transformas.

        Args:
            name (str):
                A string representing the transformations applied, where each
                transformation is represented by its identifier (e.g. "_rot" -> rotation).

        Returns:
            list: A list of transformations corresponding to the input name string.
        """
        pipe = []
        name_split = name.split("_")
        name_split = ["_" + part for part in name_split if part]
        if Transformation.ROTATE.value[0] in name_split:
            pipe.append(Transformation.ROTATE)
        if Transformation.MIRROR.value[0] in name_split:
            pipe.append(Transformation.MIRROR)
        if Transformation.SUB.value[0] in name_split:
            pipe.append(Transformation.SUB)
        if Transformation.SHUFFLE.value[0] in name_split:
            pipe.append(Transformation.SHUFFL)
        if Transformation.CIRCLES.value[0] in name_split:
            pipe.append(Transformation.CIRCLES)
        if Transformation.BRIGHTNESS.value[0] in name_split:
            pipe.append(Transformation.BRIGHTNESS)
        if Transformation.INVERT.value[0] in name_split:
            pipe.append(Transformation.INVERT)
        if Transformation.SHIFT_COLOR.value[0] in name_split:
            pipe.append(Transformation.SHIFT_COLOR)
        if Transformation.NOISE.value[0] in name_split:
            pipe.append(Transformation.NOISE)
        return pipe

    def get_gt_transformations(self):
        """Get the transformations that can be applied to the ground truth images.

        Returns:
            list: A list of transformations that can be applied to the ground truth images.
        """
        # Return the module-level VALID_GT_TRANSFORMATIONS constant (defined below)
        from .image_transformation import VALID_GT_TRANSFORMATIONS as _VALID

        return _VALID


# DIVISION N x X
def split_image_into_grid(image: Image.Image, n: int) -> list[Image.Image]:
    """Split image into n x n grid.

    Splits an image into an n x n grid, scales each subimage to 400x400 pixels,
    and returns them as a list of PIL Image objects.

    Args:
        image (Image.Image): The input image to be spl This should be a PIL Image object.
        n (int): The number of rows and columns to split the image into: n x n grid of subimages.

    Returns:
        List[Image.Image]
            A list of PIL Image objects representing the subimages. Subimages are scaled to 400x400.

    Notes:
        If the dimensions of the input image are not perfectly divisible by n,
        the subimages will still be of the computed size, but this might lead to some
        cropping. The final size of each subimage is fixed at 400x400 pixels.
    """
    # Open the image
    img = image.copy()
    width, height = img.size

    # Calculate the dimensions of each subimage
    sub_width, sub_height = width // n, height // n

    # Generate the subimages
    subimages = []
    for row in range(n):
        for col in range(n):
            # Define the bounding box for the current subimage
            left = col * sub_width
            upper = row * sub_height
            right = left + sub_width
            lower = upper + sub_height
            box = (left, upper, right, lower)

            # Crop and scale the subimage
            sub_img = img.crop(box)
            sub_img = sub_img.resize((400, 400))
            subimages.append(sub_img)

    return subimages


# ROTATION
def rotate_image(image: Image.Image, seed: int, any_angle: bool = False):
    """Rotate image and crop to original size.

    Args:
        image (Image.Image): The input image to be rotated.
        seed (int): The random seed used for generating the rotation angle.
        any_angle (bool): If True, rotate by any angle 1-360;
                          if False, rotate only by 90, 180, or 270 degrees.
    """
    img = image.copy()
    original_size = img.size

    # Create a larger canvas with a background matching the input image mode
    # so that rotating a single-channel ground-truth (mode 'L') does not
    # force it to become RGB. Choose an appropriate black/transparent fill
    # depending on the mode.
    if img.mode == "RGB":
        bg_color = (0, 0, 0)
    elif img.mode == "RGBA":
        bg_color = (0, 0, 0, 0)
    elif img.mode == "L":
        bg_color = 0
    else:
        # Fallback: create a tuple of zeros matching number of bands, or 0
        try:
            bg_color = tuple([0] * len(img.getbands()))
        except Exception:
            bg_color = 0

    larger_canvas = Image.new(img.mode, img.size, bg_color)
    larger_canvas.paste(img, (0, 0))

    # Set rotation angle
    np.random.seed(seed)
    if any_angle:  # any angle except 0
        degrees = np.random.randint(1, 359)
    else:
        degrees = np.random.choice([90, 180, 270])

    rotated_img = larger_canvas.rotate(degrees, resample=Image.BICUBIC, expand=False)
    rotated_cropped_img = rotated_img.crop((0, 0, original_size[0], original_size[1]))

    return rotated_cropped_img


# MIRROR (HORIZONTAL FLIP)
def mirror_image(image: Image.Image):
    """Mirror the content of an image (flips it horizontally).

    Args:
        image (Image.Image): The input image to be mirrored. This should be a PIL Image object.

    Returns:
        Image.Image: The mirrored image as a PIL Image object, flipped horizontally.
    """
    # Open the image
    img = image.copy()

    # Flip the image horizontally (mirror effect)
    mirrored_img = ImageOps.mirror(img)

    return mirrored_img


# STRETCH
def random_subimage(image: Image.Image, seed: int):
    """Get subimage of random size and position and scale to 400x400.

    Selects a random subimage from the input image using the given seed.
    The subimage size is determined randomly between 10 and 400 using the seed.
    The subimage is then scaled to 400x400 pixels.

    Args:
        image (Image.Image): The input image from which a random subimage will be selected.
        seed (int): The seed value for random number generation, ensuring reproducibility.

    Returns:
        Image.Image: The randomly selected and scaled subimage as a PIL Image object.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Get image dimensions
    width, height = image.size
    # Divide into 2^2
    subimage_size = width // 2

    # Calculate the max allowable top-left corner coordinates
    max_x = width - subimage_size
    max_y = height - subimage_size

    # Randomly select the corner of the subimage
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    # Define the bounding box for the subimage
    box = (x, y, x + subimage_size, y + subimage_size)

    # Crop the subimage from the original image
    subimage = image.crop(box)

    # Resize the subimage original image size
    subimage = subimage.resize((width, height), Image.Resampling.LANCZOS)

    return subimage


# HOLES
def add_random_circles(image: Image.Image, seed: int):
    """Add black circles to an image at random positions with random diameters.

    Args:
        image (Image.Image): The input image to which circles will be added.
        seed (int): The seed for random diameter and position generation. Ensures reproducibility.

    Returns:
        Image.Image: The modified image with added black circles as a PIL Image object.
    """
    # Open the image
    img = image.copy()
    width, height = img.size
    np.random.seed(seed)

    # Maximum circle diameter
    max_diameter = min(width, height) // 8
    num_circles = np.random.randint(1, 8)

    # Create a drawing object
    draw = ImageDraw.Draw(img)
    img_np = np.array(img)
    mean_color = int(np.mean(img_np))
    # Choose fill color according to image mode: for 'L' use single int, for 'RGB' tuple
    if img.mode == "L":
        color = mean_color
    else:
        color = (mean_color,) * len(img.getbands())

    # Generate and draw circles
    for _ in range(num_circles):
        # Random diameter
        diameter = np.random.randint(1, max_diameter)

        # Random position ensuring the circle stays within bounds
        x = np.random.randint(0, width - diameter)
        y = np.random.randint(0, height - diameter)

        # Draw the circle
        draw.ellipse([x, y, x + diameter, y + diameter], fill=color)

    return img


# BRIGHTNESS
def set_brightness(image: Image.Image, seed: int):
    """Adjust the brightness of an image based on the input brightness level.

    Args:
        image (Image.Image): The input image to adjust the brightness of.
        seed (int): The seed for random brightness level generation. Ensures reproducibility.

    Returns:
        Image.Image: The adjusted image with modified brightness as a PIL Image object.
        The adjusted image with modified brightness as a PIL Image object.
    """
    # Brightness
    np.random.seed(seed)
    brightness_level = np.random.randint(25, 75)  # From 0 to 100

    # Open the image
    img = image.copy()

    # Calculate brightness factor
    # 0 -> 0.0 (black), 50 -> 1.0 (no change), 100 -> 2.0 (white)
    brightness_factor = brightness_level / 50

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    adjusted_img = enhancer.enhance(brightness_factor)

    return adjusted_img


# COLORS
def shift_color_towards_random(image: Image.Image, seed: int):
    """
    Shifts the color scale of the image towards a randomly generated color based on the seed.

    Args:
        image (Image.Image): The input image to shift the colors of.
        seed (int): The seed used to generate the random color for the color shift.

    Returns:
        Image.Image: The color-shifted image as a PIL Image object.
    """
    img = image.copy()

    r_scale = np.random.uniform(0.8, 1.2)
    g_scale = np.random.uniform(0.8, 1.2)
    b_scale = np.random.uniform(0.8, 1.2)

    # Split channels
    r, g, b = img.split()

    # Apply scaling and clip values to 0-255
    r = r.point(lambda i: np.clip(i * r_scale, 0, 255))
    g = g.point(lambda i: np.clip(i * g_scale, 0, 255))
    b = b.point(lambda i: np.clip(i * b_scale, 0, 255))

    # Merge channels back
    shifted_img = Image.merge("RGB", (r, g, b))
    return shifted_img


# NOISE
def add_noise_to_image(image: Image.Image, seed):
    """Add random noise to an image.

    Args:
        image (Image.Image): The input image to which noise will be added.
        seed (int): The seed used to generate random noise.

    Returns:
        Image.Image: The noisy image as a PIL Image object.
    """
    # Open the image and convert it to a NumPy array
    img = image.copy()
    img_array = np.array(img)

    np.random.seed(seed)
    noise_level = np.random.randint(5, 40)

    # Generate random noise
    noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape, dtype=np.int16)

    # Add noise to the image and clip the values to keep them valid (0-255)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Convert the noisy image array back to a PIL Image
    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img


# Transformations that are valid to apply on ground truth images
VALID_GT_TRANSFORMATIONS = [
    Transformation.ROTATE,
    Transformation.MIRROR,
    Transformation.SUB,
    Transformation.CIRCLES,
]
