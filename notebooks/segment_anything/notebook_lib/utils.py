from math import ceil
from typing import Any, Dict, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely.geometry as shpg
from numpy.typing import NDArray
from skimage import measure

from vibe_core.data import CategoricalRaster, Raster, Sentinel2Raster
from vibe_core.data.core_types import BaseGeometry
from vibe_notebook.plot import lw_plot, transparent_cmap
from vibe_notebook.raster import read_raster, s2_to_img

MANUAL_PROMPT_TITLE = (
    "Press 'f' to add a new foreground point to the prompt.\n"
    "Press 'b' to add a new background point to the prompt.\n"
    "Press 'n' to save the prompt.\n"
    "Press 'e' to exit the prompt selection process."
)

PROMPT_COLOR_LIST = ["red", "green"]


def extract_countours_from_mask_list(mask_list):
    """Extract contours from a list of masks and return a mask with the boundaries"""
    boundaries_mask = np.zeros(mask_list[0].shape, dtype="bool")

    for mask in mask_list:
        # extract contour, get first element as there is only one contour
        contour = measure.find_contours(mask)[0]
        boundaries_mask[
            np.round(contour[:, 0]).astype("int"), np.round(contour[:, 1]).astype("int")
        ] = True

    return boundaries_mask


def show_anns(anns):
    """Plot all masks overlaid on the image"""
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def show_mask(mask, ax, random_color=False):
    """Plot a single mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def key_press(event, new_prompt, new_labels, prompt_list, bg_img):
    """Handle key press events for manual prompt selection"""
    if event.key in ("f", "b"):
        ix, iy = event.xdata, event.ydata
        new_prompt.append((ix, iy))
        new_labels.append(1 if event.key == "f" else 0)

        plt.clf()
        plt.scatter(
            [p[0] for p in new_prompt],
            [p[1] for p in new_prompt],
            color=[PROMPT_COLOR_LIST[label] for label in new_labels],
        )
        plt.scatter(
            [p[0] for prompt, _ in prompt_list for p in prompt],
            [p[1] for prompt, _ in prompt_list for p in prompt],
            color="gray",
        )
        plt.imshow(bg_img)
        plt.title(MANUAL_PROMPT_TITLE)
        plt.draw()
    elif event.key == "n":
        prompt_list.append((new_prompt[:], new_labels[:]))
        [new_prompt.pop() for _ in prompt_list[-1][0]]  # empty new_prompt
        [new_labels.pop() for _ in prompt_list[-1][0]]  # empty new_labels

        plt.clf()
        plt.scatter(
            [p[0] for prompt, _ in prompt_list for p in prompt],
            [p[1] for prompt, _ in prompt_list for p in prompt],
            color="gray",
        )
        plt.imshow(bg_img)
        plt.title("Prompt saved!\n" + MANUAL_PROMPT_TITLE)
        plt.draw()
    elif event.key == "e":
        plt.clf()
        plt.scatter(
            [p[0] for prompt, _ in prompt_list for p in prompt],
            [p[1] for prompt, _ in prompt_list for p in prompt],
            color="gray",
        )
        plt.imshow(bg_img)
        plt.title("All prompts saved!")
        plt.draw()
        plt.close()


def plot_rasters_prompts_masks(
    raster, segmented_raster, geometry, prompt_gdf, labels, img_plot_size=7
):
    # Reprojecting the raster and points to the same CRS
    with rasterio.open(raster.raster_asset.url) as src:
        proj_geom = gpd.GeoSeries(geometry, crs="epsg:4326").to_crs(src.crs).iloc[0].envelope
        shpg_points = list(prompt_gdf.to_crs(src.crs)["geometry"])

    # Reading the raster
    ar, transform = read_raster(raster, projected_geometry=proj_geom)

    if type(raster) == Sentinel2Raster:
        img = s2_to_img(ar)
    else:
        img = ar.transpose((1, 2, 0))

    # Reading the segmentation mask
    mask_ar, _ = read_raster(segmented_raster, projected_geometry=proj_geom)

    # Transforming the points to pixel coordinates for visualization
    ps = [
        ~transform * (shpg_p.x, shpg_p.y)
        for shpg_p in shpg_points
        if isinstance(shpg_p, shpg.Point)
    ]
    foreground_ps = [p for p, l in zip(ps, labels) if l == 1]
    background_ps = [p for p, l in zip(ps, labels) if l == 0]

    bbox = [
        ~transform * (shpg_p.bounds[0], shpg_p.bounds[1])
        + ~transform * (shpg_p.bounds[2], shpg_p.bounds[3])
        for shpg_p in shpg_points
        if isinstance(shpg_p, shpg.Polygon)
    ]

    # Visualizing the results
    plt.figure(figsize=(img_plot_size * (1 + mask_ar.shape[0]), img_plot_size))
    plt.subplot(1, (1 + mask_ar.shape[0]), 1)
    plt.imshow(img)
    if ps:
        plt.scatter([p[0] for p in foreground_ps], [p[1] for p in foreground_ps], color="cyan")
        plt.scatter([p[0] for p in background_ps], [p[1] for p in background_ps], color="red")
    if bbox:
        plt.plot(
            [bbox[0][0], bbox[0][0], bbox[0][2], bbox[0][2], bbox[0][0]],
            [bbox[0][1], bbox[0][3], bbox[0][3], bbox[0][1], bbox[0][1]],
            color="cyan",
        )
    plt.axis("off")

    for i in range(mask_ar.shape[0]):
        plt.subplot(1, (1 + mask_ar.shape[0]), 2 + i)
        plt.imshow(img)
        plt.imshow(mask_ar[i], cmap=transparent_cmap(plt.cm.viridis), vmin=0, vmax=1)
        plt.title(f"Prompt {i}")
        plt.axis("off")
    lw_plot()


def color_autoseg_masks(seg_raster: CategoricalRaster) -> NDArray[Any]:
    with rasterio.open(seg_raster.assets[0].path_or_url) as src:
        m = src.read()

    mask_arr = np.zeros((m.shape[1], m.shape[2], 3), dtype=np.uint8)

    # For each mask, add a random color
    for i in range(m.shape[0]):
        color = np.random.randint(0, 256, 3)  # Random color
        mask_arr[m[i] == 1] = color.reshape(1, -1)  # Apply color to the mask
    return mask_arr


def plot_autoseg_masks(
    raster: Union[Sentinel2Raster, Raster],
    mask_dict: Dict[str, NDArray[Any]],
    geom: BaseGeometry,
    figsize: Tuple[int, int] = (10, 10),
):
    n_images = len(mask_dict) + 1
    if n_images <= 4:
        n_cols = n_images
        n_rows = 1
    else:
        n_cols = int(ceil(n_images**0.5))
        n_rows = int(ceil(n_images / n_cols))

    if isinstance(raster, Sentinel2Raster):
        r = s2_to_img(read_raster(raster, geom)[0])
    else:  # Basemap Raster
        r = read_raster(raster, geom)[0].transpose((1, 2, 0))

    # Display the image
    plt.figure(figsize=figsize)
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(r)

    plot_idx = 2
    for title, mask_arr in mask_dict.items():
        plt.subplot(n_rows, n_cols, plot_idx)
        plt.imshow(mask_arr)
        plt.title(title)
        plot_idx += 1

    if plot_idx < n_rows * n_cols:
        for i in range(plot_idx, n_rows * n_cols):
            plt.subplot(n_rows, n_cols, i)
            plt.axis("off")

    lw_plot()
