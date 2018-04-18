import cv2
import numpy as np
import pandas as pd
from sklearn import cluster, mixture

# %%
MIN_HEIGHT = 600
MIN_WIDTH = 800
num_colors = 5

# %%
def resize_image(img: np.ndarray,
                 width: int = 800,
                 height: int = 600,
                 ensure_min: bool = True) -> np.ndarray:
    """Resize image.

    If ensure_min is True, then the resized image dimensions will be at least
    the ones specified in teh function call. Otherwise, they are at most
    specified in the call.

    Args:
        img:        image.
        width:      desired width of the image.
        height:     desired height of the image.
        ensure_min: if True, dimensions are at least (width, height).

    Returns:
        Resized image.
    """
    image_height, image_width = img.shape[:2]
    if ensure_min:
        scale_factor = min(width / image_width, height / image_height)
    else:
        scale_factor = max(width / image_width, height / image_height)

    return cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)


def _image2data(img: np.ndarray) -> np.array:
    """Transform image into data suitable for clusteringself.

    Image is read as a 3D tensor, but clusteing requires 2D tensor with each
    row corresponding to a single point.

    Args:
        img: image to tranform.

    Returns:
        Image as a 2D tensor.
    """
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def get_pallete(img: np.ndarray,
                num_colors: int = 3,
                algorithm: str = "k-means") -> dict:
    """Get pallette colours.

    Args:
        img:        image.
        num_colors: number of colors in pallette.
        algorithm:  one of "k-means", "mixture".

    Returns:
        Dictionary with keys "colors" and "proportions".
    """
    img = _image2data(img)

    if algorithm == "k-means":
        kmeans = cluster.KMeans(n_clusters=num_colors,
                                verbose=0,
                                algorithm="elkan")
        tmp = kmeans.fit(img)

        colors = np.round(kmeans.cluster_centers_)
        colors = colors.tolist()

        counts = np.bincount(kmeans.predict(img))
        proportions = counts / np.sum(counts)

    elif algorithm == "mixture":
        gaussian_mix = mixture.GaussianMixture(n_components=num_colors)
        gaussian_mix.fit(img_2d)

        colors = np.round(gaussian_mix.means_)
        colors = colors.tolist()

        counts = np.bincount(gaussian_mix.predict(img))
        proportions = counts / np.sum(counts)
    else:
        raise ValueError("Wrong value of algorithm parameter.")

    # Reorder colors according to their prevalence
    ordered_idx = np.flip(np.argsort(proportions), axis=0)
    colors = [colors[idx] for idx in ordered_idx]
    proportions = [proportions[idx] for idx in ordered_idx]

    # Create a list of dicts
    pallette = []
    for color, proportion in zip(colors, proportions):
        pallette.append({"rgb": tuple([int(c) for c in color]),
                         "proportion": proportion})

    return pallette

def plot_pallette(pallette: dict,
                  width: int = 800,
                  height: int = 200) -> np.ndarray:
    """Create an image of palletteself.

    Width of colored rectangles is proportional to the prevalance of the color.

    Args:
        pallette: output of get_pallete() function.
        width:    width of the desired image in pixels.
        height:   height of the desired image in pixels.

    Returns:
        Image of pallette.
    """
    # Initialise empty image
    img = np.zeros((height, width, 3), np.uint8)

    # Add a rectangle for each cluster with width proportional to the size
    # of the cluster
    start_x = 0
    for color in pallette:
        end_x = start_x + int(round(color["proportion"] * width))
        cv2.rectangle(img=img,
                      pt1=(start_x, 0),
                      pt2=(end_x, height),
                      color=color["rgb"],
                      thickness=cv2.FILLED)
        start_x = end_x
    return img

def _rgb_to_hex_string(rgb: tuple) -> str:
    """Convert RGB tuple to hex string."""
    def clamp(x):
        return max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(rgb[0]),
                                           clamp(rgb[1]),
                                           clamp(rgb[2]))


def plot_pallette_with_text(pallette: dict,
                            color_max_width: int = 100,
                            vertical_padding: int = 5,
                            horizontal_padding: int = 5) -> np.ndarray:
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate the text area size
    str_width = []
    str_height = []
    for color in pallette:
        rgb_tuple = str(color["rgb"])
        rgb_hex = _rgb_to_hex_string(color["rgb"])
        color_str = rgb_tuple  # + "," + rgb_hex
        str_size = cv2.getTextSize(text=color_str,
                                   fontFace=FONT,
                                   fontScale=1,
                                   thickness=1)
        str_width.append(str_size[0][0])
        str_height.append(str_size[0][1])

    str_width = max(str_width)
    str_height = max(str_height)

    # Compute the size of box for each color
    box_height = str_height + 2 * vertical_padding
    box_width = color_max_width + str_width + 2 * horizontal_padding

    # Initialise empty image
    num_colors = len(pallette)
    img = np.zeros((num_colors * box_height, box_width, 3))
    img[:] = 255

    # Add colors
    max_color_proportion = np.max([c["proportion"] for c in pallette])
    for i, color in enumerate(pallette):
        # Add rectangle
        color_width = int(round(color["proportion"] * color_max_width /
                                max_color_proportion))
        top_left = (0, i * box_height)
        bottom_right = (color_width, (i + 1) * box_height - 1)
        cv2.rectangle(img=img,
                      pt1=top_left,
                      pt2=bottom_right,
                      color=color["rgb"],
                      thickness=cv2.FILLED)

        bottom_left_text = (color_max_width + horizontal_padding,
                            (i + 1) * box_height - vertical_padding)
        rgb_tuple = str(color["rgb"])
        rgb_hex = _rgb_to_hex_string(color["rgb"])
        color_str = rgb_tuple  # + "," + rgb_hex
        cv2.putText(img=img,
                    text=color_str,
                    org=bottom_left_text,
                    fontFace=FONT,
                    fontScale=1,
                    color=(0, 0, 0),
                    thickness=1)

    return img

def pallette_to_csv(pallette, filename, *args, **kwargs) -> pd.DataFrame:
    """Output pallette to csv.

    Args:
        pallette:        output of get_pallete() function.
        filename:        file name.
        *args, **kwargs: other arguments to pandas.DataFrame.to_csv().

    Returns:
        Pandas DataFrame and saves a file.
    """
    df = pd.DataFrame(pallette)
    df[["red", "green", "blue"]] = df["rgb"].apply(pd.Series)
    df.drop("rgb", inplace=True, axis=1)
    df.to_csv(filename, *args, **kwargs)
    return df


# %%
# Read in image
source_img = cv2.imread("test-image.jpg")
img = resize_image(source_img, 200, 100, ensure_min=True)

# %%
pallette = get_pallete(img, num_colors=20, algorithm="k-means")
pallete_img = plot_pallette_with_text(pallette, color_max_width=600, vertical_padding=10)
cv2.imwrite("tmp.png", pallete_img)

# %%
pallette = get_pallete(img, num_colors=10, algorithm="mixture")
pallete_img = plot_pallette_with_text(pallette, color_max_width=600, vertical_padding=10)
cv2.imwrite("tmp.png", pallete_img)

# %%
img_2d = _image2data(img)
gaussian_mix = mixture.GaussianMixture(n_components=5)
gaussian_mix.fit(img_2d)

gaussian_mix.means_
np.unique(gaussian_mix.predict(img_2d))

# %%
pallete_img = plot_pallette(pallette, height=200, width=800)
cv2.imwrite("tmp.png", pallete_img)
# %%
pallete_img = plot_pallette_with_text(pallette, color_max_width=600, vertical_padding=10)
cv2.imwrite("tmp.png", pallete_img)

# %%
# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 2
fontColor              = (255,255,255)
lineType               = 6

cv2.putText(img,'Hello World!',
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    1,
    lineType)



#Display the image
cv2.imwrite("tmp.jpeg",img)
