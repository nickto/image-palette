import cv2
import numpy as np
from sklearn import cluster
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
        kmeans.get_params()
        colors = np.round(kmeans.cluster_centers_)
        colors = colors.tolist()

        counts = np.bincount(kmeans.predict(img))
        proportions = counts / np.sum(counts)

    elif algorithm == "mixture":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value of algorithm parameter.")

    # Reorder colors according to their prevalence
    ordered_idx = np.flip(np.argsort(proportions), axis=0)
    counts = [counts[idx] for idx in ordered_idx]
    proportions = [proportions[idx] for idx in ordered_idx]

    return {"colors": colors, "proportions": proportions}

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
    for i, color in enumerate(pallette["colors"]):
        proportion = pallette["proportions"][i]

        end_x = start_x + int(round(proportion * width))
        cv2.rectangle(img=img,
                      pt1=(start_x, 0),
                      pt2=(end_x, height),
                      color=tuple(color),
                      thickness=cv2.FILLED)
        start_x = end_x
    return img


# %%
# Read in image
source_img = cv2.imread("test-image.jpg")
img = resize_image(source_img, 200, 100, ensure_min=True)
pallette = get_pallete(img, num_colors=10, algorithm="k-means")
# %%
pallete_img = plot_pallette(pallette, height=200, width=800)
cv2.imwrite("tmp.png", pallete_img)
