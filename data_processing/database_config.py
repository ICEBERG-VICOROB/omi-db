# GENERAL:
# ------------
reading_path = '/mnt/usb_optimam/img/optimam/image_db/sharing/omi-db'
output_path = '/home/joaquin/maia_pro/database/selection'
csv_name = 'omidb-selection.csv'
allowed_episode_types = ['SURGERY']  # ['SCREENING', 'BIOPSYWIDE', 'ASSESSMENT']
manufact_selection = ['HOLOGIC', 'SIEMENS', 'GE', 'Philips']
clients_subset = 'all'  # ['demd100018', 'demd128247', 'demd843','demd94678']
views_selection = ["MLO", "ML", "CC"]


# Image Preprocessing:
# ----------------------
# Intensity scaling in uint8 conversion:
#   Originaly the intensit scaling was done weirdly ('zerof_and_img_max'), check the code
# intensity_scale = 'bitwise_range'
intensity_scale = 'zero_and_img_max'

# Pixel size normalization:
# Whether to resize the images to a common pixel size
normalize_pixel_size = False
# This pixel size has been obtained from the db-exploration
# pixel_spacing = (1.0, 1.0, 1.0)


# For the ROIs:
# ----------------------------
# normal_roi_noise:
#   Maximum offset from breast bbox center from where to sample the ROI center.
normal_roi_noise = 500

# normal_roi_size:
#   Size of the square ROIs to extract.
normal_roi_size = 300

# maximum number of normal rois to sample:
# If 'lesion' it matches the number of leasions in the breast, if int that number
max_num_norm_rois = 'lesion'

# Overlapping bboxes tolerance:
#   Maximumn area that can overlap between randomly extracted *normal* patches.
ovlp_bboxes_tol = 1.

# Background bboxes tolerance:
#   Maximumn portion of background that can be present in a *normal* patch.
bkg_tol = 0.

extra_size = 50
n_normal_bbox = 'same'


# For the LABELS:
# -----------------
st_code = {'ER': 0, 'PR': 1, 'HER2': 2}
# Subtupe inversion code:
st_inv_code = ['ER', 'PR', 'HER2']
