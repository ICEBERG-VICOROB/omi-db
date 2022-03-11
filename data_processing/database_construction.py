import csv
from typing import List
import cv2
import logging
import omidb
import os
from omidb.episode import Episode

import numpy as np

# from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut
from dataclasses import dataclass

logging.basicConfig(format='%(asctime)s - %(message)s')


class stats:
    def __init__(self, N, B, M, IC):
        self.N = N
        self.M = M
        self.B = 0
        self.IC = IC
        self.image_CC = 0
        self.image_MLO = 0
        self.image_R = 0
        self.image_L = 0
        self.subtype = np.zeros(8, dtype=np.int32)

    def __repr__(self):
        return \
            f'Stats [N: {self.N}, M {self.M}, IC {self.IC},'\
            f'CC: {self.image_CC}, MLO: {self.image_MLO}, '\
            f'R: {self.image_R}, L:{self.image_L}, ' \
            f'Subtype: {np.array2string(self.subtype)} ]'


def get_breast_bbox(image):
    """
    Makes a threshold of the image identifying the regions different from
    the background (0). Takes the largest (area) region (the one corresponding
    to the breast), defines the contour of this region and creates a roi 
    that fits this region.

    Args:
        image (np.ndarray): Breast image to be croped.
    Return:
        out_bbox (np.ndarray): Coordinates of the bounding box.
        img (np.ndarray): binary mask image of the breast.
    """

    # Threshold image with th=0 and get connected comp.
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]
    nb_components, output, stats, _ = \
        cv2.connectedComponentsWithStats(img, connectivity=4)

    # Get the areas of each connected component and keep the largest (non zero)
    sizes = stats[:, -1]
    # Keep the largest connected component
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    # Generate a binary mask for the breast
    img = np.zeros(output.shape, dtype=np.uint8)
    img[output == max_label] = 255

    # Obtain the contour of the breast and generate bbox.
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    aux_im = img
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(aux_im, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Determine what this function does
    out_bbox = omidb.mark.BoundingBox(x, y, x+w, y+h)

    return out_bbox, img


def crop_breast_bbox(image):
    """
    Gets the bbox of the breast
    Args:
        image (np.ndarray): Image of the breast to crop.
    Returns:
        (np.ndarray): Cropped image.
    """
    # Crop
    bbox, _ = get_breast_bbox(image)
    return image[bbox.y1:bbox.y2, bbox.x1:bbox.x2]


def compare_bboxes(bbox1, bbox2):
    # TODO: use an area criteria.
    coords1 = [bbox1.x1, bbox1.x2, bbox1.y1, bbox1.y2]
    coords2 = [bbox2.x1, bbox2.x2, bbox2.y1, bbox2.y2]
    return np.equal(coords1, coords2)


def get_random_bbox(
    bbox: dataclass, fbn_rois: List[dataclass], prev_rois: List[dataclass],
    breast_mask: np.ndarray, normal_roi_noise: int = 500, normal_roi_size: int = 150
):
    """
    From the original image it generates a random bbox that doesn't
    overlap with the foribiden roi (fbn_roi). If this is not possible, no bbox
    is returned and a warning is displayed.
    Args:
        bbox (dataclass): Breast bbox.
        fbn_roi 0]: Foribiden roi, bbox of the area that we
            don't want the generated bbox to overlap with.
        breast_mask (np.ndarray): Breast mask image
        normal_roi_noise (int): Distance (pixels) from the breast bbox center
            from where to extract a potential center.
        normal_roi_size (int): size of the ROIs to extract.
    """
    dims = breast_mask.shape

    # Get the center of the bbox.
    center_x = np.round((bbox.x2 - bbox.x1)/2).astype(int)
    center_y = np.round((bbox.y2 - bbox.y1)/2).astype(int)
    idx = 0
    found = False
    while not found and idx < 100:
        # Perturbe bbox center
        x = center_x + np.random.randint(-normal_roi_noise, +normal_roi_noise)
        # TODO: Isn't this noise from the center of the image generating very similar bboxes?
        y = center_y + np.random.randint(-normal_roi_noise, +normal_roi_noise)

        # Discard coords out of boundary
        y1 = np.maximum(y - normal_roi_size, 0)
        y2 = np.minimum(y + normal_roi_size, dims[0])
        x1 = np.maximum(x - normal_roi_size, 0)
        x2 = np.minimum(x + normal_roi_size, dims[1])
        # Instatiate the bbox dataclass
        bbox_random = omidb.mark.BoundingBox(x1, y1, x2, y2)

        # Check if actual bbox is different from previously generated ones
        for prev_roi in prev_rois:
            if compare_bboxes(bbox_random, prev_roi):
                bboxes_are_diff = False
                break
            bboxes_are_diff = False

        if bboxes_are_diff:
            # Determine if the actual bbox is overlapping with the "forbiden_roi"
            # or if the center is in the boundary of the image
            found = (x1 == 0) or (y1 == 0)
            for fbn_roi in fbn_rois:
                found = found and not (is_overlapping2D(bbox_random, fbn_roi))

            if found:
                # Check if the bboxes are completely inside the breast
                # TODO: shouldn't they contain border cases too?
                bb_mask = np.zeros(breast_mask.shape)
                bb_mask[bbox_random.y1:bbox_random.y2, bbox_random.x1:bbox_random.x2] = 255
                for fbn_roi in fbn_rois:
                    bb_mask[fbn_roi.y1:fbn_roi.y2, fbn_roi.x1:fbn_roi.x2] = 128

                and_image = cv2.bitwise_and(bb_mask, (255 - breast_mask))
                if np.all(and_image > 0):
                    found = False
        idx += 1

    if found:
        return bbox_random
    else:
        logging.warning("*** Could not get a Normal random ROI after 100 iterations")
        return None


def is_overlapping1D(bbox1, bbox2):
    return bbox1[1] >= bbox2[0] and bbox2[1] >= bbox1[0]


def is_overlapping2D(bbox1, bbox2):
    return is_overlapping1D([bbox1.x1, bbox1.x2], [bbox2.x1, bbox2.x2]) \
       and is_overlapping1D([bbox1.y1, bbox1.y2], [bbox2.y1, bbox2.y2])


def store_rois_and_ffdm(
    client: str, episode: str, scanner: str, side: str, view: str,
    subtype: np.ndarray, image: omidb.image.Image, csv_path: str,
    extra_size: int = 50, n_normal_bbox: int = None
):

    output_path = os.path.dirname(csv_path)
    base_path = os.path.join(output_path, str(scanner))
    filename = f'{client}_{episode}_{image.id}_{view}.png'

    # Adjust pixels intensities if needed:
    if 'WindowWidth' in image.dcm:
        image_array = apply_voi_lut(image.dcm.pixel_array, image.dcm)
    else:
        image_array = image.dcm.pixel_array

    # Convert images to uint8
    # TODO: Why are they scaling the image like this?
    image_array = (np.maximum(image_array, 0) / image_array.max()) * 255.0
    image_array = np.uint8(image_array)

    # Make all images left sided
    if (side == 'R'):
        image_array = cv2.flip(image_array, 1)
    dims = image_array.shape

    # Get lesions ROIs
    lession_rois = []
    roi_path = os.path.join(base_path, 'roi', f'st{subtype}')
    os.makedirs(roi_path)
    roi_path = os.path.join(roi_path, filename)
    for mark in image.marks:
        bbox_roi = mark.boundingBox
        # Mirror the bbox
        if (side == 'R'):
            bbox_roi.x2 = dims[1] - mark.boundingBox.x1
            bbox_roi.x1 = dims[1] - mark.boundingBox.x2

        # Adding an extra_size around the lesion.
        y1 = np.maximum(bbox_roi.y1 - extra_size, 0)
        y2 = np.minimum(bbox_roi.y2 + extra_size, dims[0])
        x1 = np.maximum(bbox_roi.x1 - extra_size, 0)
        x2 = np.minimum(bbox_roi.x2 + extra_size, dims[1])

        lession_rois.append(bbox_roi)

        # Crop and save patch
        image_crop = image_array[y1:y2, x1:x2]
        cv2.imwrite(roi_path, image_crop)

    # TODO: This code doesn't contemplate the case in which we
    #       have more than one lession in the image, fix it?

    # Get normal ROIs
    normal_roi_path = os.path.join(base_path, 'normal_roi', f'st{subtype}')
    os.makedirs(normal_roi_path)
    normal_roi_path = os.path.join(normal_roi_path, filename)
    normal_bboxes = []

    # Exploit all potential normal patches
    if n_normal_bbox is None:
        n_normal_bbox = len(image.marks)
    for n in range(n_normal_bbox):
        # Sample ONE normal patch
        breast_bbox, breast_mask = get_breast_bbox(image_array)
        bbox_norm = \
            get_random_bbox(breast_bbox, lession_rois, normal_bboxes, breast_mask)

        if bbox_norm is None:
            break
        else:
            # Crop and save
            normal_bboxes.append(bbox_norm)
            image_crop = \
                image_array[bbox_norm.y1:bbox_norm.y2, bbox_norm.x1:bbox_norm.x2]
            cv2.imwrite(normal_roi_path, image_crop)

    image_path = os.path.join(base_path, 'ffdm', f'st{subtype}')
    os.makedirs(image_path)
    image_path = os.path.join(image_path, filename)

    # Write the PNG file of only the breast region
    image_crop = \
        image_array[breast_bbox.y1:breast_bbox.y2, breast_bbox.x1:breast_bbox.x2]
    cv2.imwrite(image_path, image_crop)

    # Write CSV
    for bbox_roi in lession_rois:
        with open(csv_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                client, subtype, episode, image.id, filename, side,
                scanner, breast_bbox, bbox_roi, extra_size, 'M'
            ])
    for bbox_roi in normal_bboxes:
        with open(csv_path, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                client, subtype, episode, image.id, filename, side,
                scanner, breast_bbox, bbox_roi, extra_size, 'N'
            ])
    # TODO: Log an info line with a summary for each processed case.


def add_receptor_label(
    subtype: np.ndarray, not_known: np.ndarray, receptor: str,
    value_st: str, st_code: dict
):
    """
    According to the label of the receptor it modifies de
    enconding of labels.
    Args:
        subtype (np.ndarray): Subtype code
        not_known (np.ndarray):
            Bool array identifiying the receptor without data
        receptor (str): Name of the receptor ['ER', 'PR', 'HER2']
        value_st (str): Value stored in the database
        st_code (dict): Dictionary to get the positions in the coding

    Returns:
        Updated versions of 'subtype' and 'not_known'
    """
    if value_st == 'RP':
        subtype[st_code[receptor]] = True
    elif value_st == 'RN':
        subtype[st_code[receptor]] = False
    else:
        logging.info(f'*** {receptor} Status NA', value_st)
        not_known[st_code[receptor]] = True
    return subtype, not_known


def get_subtype_from_event(episode_data: Episode, st_code: dict = None):
    """
    Gets the receptor's labels from an episode information.
    Args:
        episode_data (Episode): Episode to be analysed
        st_code (dict, optional):
            Dictionary to get the positions in the subtype coding
    Returns:
        subtype (np.ndarray): Subtype enconding.
        side (str): Side in which the lesion is present.
        not_known (np.ndarray):
            Bool array identifiying the receptor without data
    """

    if st_code is None:
        st_code = {'ER': 0, 'PR': 1, 'HER2': 2}

    subtype = np.zeros(3, dtype=np.int8)
    not_known = np.zeros(3, dtype=np.int8)
    # logging.debugg("EPISODE ->")
    for key, value in episode_data.items():                             # Episode level
        # logging.debugg(f'All new keys: {key}, {value}')
        if key == "SURGERY":                                            # Episodes with Surgery
            # logging.debugg(f'New key: {key}, {value}')
            for key1, value1 in value.items():
                # logging.debugg(f'Sugery keys: {key1}, {value1}')
                if (key1 == 'R' or key1 == 'L'):                        # Breast level
                    side = key1
                    for key_finding, value_finding in value1.items():   # Finding level
                        for key_st, value_st in value_finding.items():  # Subtype level
                            # logging.debugg(f'Hormone key: {key_st}, {value_st}')
                            if (key_st == 'HormoneERStatus'):            # ER Hormone Rec
                                subtype, not_known = add_receptor_label(
                                    subtype, not_known, 'ER', value_st, st_code
                                )
                            if (key_st == 'HormonePRStatus'):             # PR Hormone Rec
                                subtype, not_known = add_receptor_label(
                                    subtype, not_known, 'PR', value_st, st_code
                                )
                            if (key_st == 'HER2ReceptorStatus'):          # HER2 Hormone Rec
                                subtype, not_known = add_receptor_label(
                                    subtype, not_known, 'HER2', value_st, st_code
                                )
                else:
                    side = 'U'
                    logging.warning(f'*** Side unknown {key1}')

                # TODO: log some info of hows the process going, use verbose: bool = False

    return subtype, side, not_known


def generate_database(
    db: omidb.DB, csv_path: str,
    manufact_selection: List[str] = ['HOLOGIC', 'SIEMENS', 'GE']
):

    # Keep the count of the cases processed
    overall = stats(0, 0, 0)
    copied_count = stats(0, 0, 0)
    st_inv_code = ['ER', 'PR', 'HER2']

    # Initialize csv file
    # csv_path = os.path.join(output_path, 'omidb-selection.csv')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "client", "subtype", "episode", "image", "filename", "side",
            "scanner", "bbox", "bbox_roi", "extra_size", 'class'
        ])

    for client in db:                                                                       # Client level
        # Save the general pathological status of the patient:
        #   M: malignant, N:Normal, CI: Interval Cancer
        # In the API, the label is given in the following order of importance:
        # CI > M > B > N
        if client.status.value == 'Interval Cancer':
            overall.IC += 1
        elif client.status.value == 'Malignant':
            overall.M += 1
        elif client.status.value == 'Benign':
            overall.B += 1
        elif client.status == 'Normal':
            overall.N += 1

        # TODO: Recurrent high risk? not used.
        # TODO: Revise all the metadata available in the 'raw' NBSS data
        # Access 'raw' NBSS data
        nbss_data = db._nbss(client.id)
        for episode in client.episodes:                                                    # Episode level
            # Keep only malignant cases
            if episode.has_malignant_opinions:
                if (episode.studies is not None):
                    # TODO: check that one episode has only one study
                    episode_data = nbss_data.get(episode.id, {})

                    # Keep only episodes with surgery events
                    for key, value in episode_data.items():
                        if key == "SURGERY":
                            # Recover the malignacy subtype code for the episode
                            subtype, side, not_known = get_subtype_from_event(episode)
                            continue

                    # TODO: Now the unkown data is not fully recovered, but the rois are extracted anyway.
                    if side == 'U':
                        logging.warning(f'*** Side U for client: {client.id} - episode {episode.id} ignored')
                        continue
                    if not_known.any():
                        msg = st_inv_code[np.where(not_known)]
                        logging.warning(f'The subtypes {np.array2string(msg)} are unkown')
                        logging.warning(f'client: {client.id} - episode {episode.id} ignored')
                        continue
                    else:
                        # logging.info(f'Subtype {subtype}, side {side}')  # TODO: modify this log

                        for study in episode.studies:                                      # Study level
                            for serie in study.series:                                     # Series level
                                for image in serie.images:                                 # Image level
                                    if image.dcm_path.is_file():

                                        # If the image has lesions, is for presentation and has
                                        # the desired laterality.
                                        for_pres = image.dcm.PresentationIntentType == 'FOR PRESENTATION'
                                        correct_side = image.dcm[0x0020, 0x0062].value == side
                                        if image.marks and for_pres and correct_side:

                                            manufacturer = image.dcm['Manufacturer'].value
                                            view = image.dcm[0x0018, 0x5101].value

                                            # TODO: check if this works fine:
                                            if manufacturer in manufact_selection:
                                                # Keep track of copied cases stats
                                                if (side == "R"):
                                                    copied_count.image_R += 1
                                                else:
                                                    copied_count.image_L += 1

                                                # TODO: ML is not the same as MLO, be careful.
                                                if (view == "MLO" or view == "ML"):
                                                    copied_count.image_MLO += 1
                                                elif (view == "CC"):
                                                    copied_count.image_CC += 1
                                                else:
                                                    logging.warning("*** View not supported: ", view)
                                                    break

                                                logging.info(
                                                    f'-->> Copying case: {manufacturer}, {view}, {side}, {subtype}, \
                                                    IDs: {client.id}, {episode.id}, {serie.id}, {image.id}'
                                                )
                                                aux_st = ''.join(map(str, subtype))
                                                copied_count.subtype[int(aux_st, 2)] += 1
                                                copied_count.M += 1

                                                store_rois_and_ffdm(
                                                    manufacturer, aux_st, view, client.id,
                                                    episode.id, image, side, csv_path
                                                )
                                            else:
                                                logging.info(f'The manufacturer {manufacturer} is not in the list')

    logging.info(f'SUMMARY copied  {copied_count}')
    logging.info(f'SUMMARY overall {overall}')
