import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure

from vibe_core.data.core_types import gen_guid

MANUAL_PROMPT_TITLE = (
    "Press 'f' to add a new foreground point to the prompt.\n"
    "Press 'b' to add a new background point to the prompt.\n"
    "Press 'n' to save the prompt.\n"
    "Press 'e' to exit the prompt selection process."
)

PROMPT_COLOR_LIST = ["red", "green"]


def create_geojson_file_from_point(list_of_points, labels, prompt_ids, storage_dirpath):
    """
    Create a geojson file from a list of points, labels, and prompt_ids
    """
    file_name_prefix = gen_guid()
    df = pd.DataFrame({"geometry": list_of_points, "label": labels, "prompt_id": prompt_ids})

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    gdf.to_file(
        os.path.join(storage_dirpath, f"{file_name_prefix}_geometry_collection.geojson"),
        driver="GeoJSON",
    )

    op_points_filepath = f"/mnt/{file_name_prefix}_geometry_collection.geojson"
    return op_points_filepath, gdf, file_name_prefix


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
    polygons = []
    color = []
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
