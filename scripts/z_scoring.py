import os 
os.environ["BONNER_BRAINIO_CACHE"] = "/data/mbonner5/shared/brainio"

from bonner.brainio import Catalog
from pathlib import Path
from bonner.datasets.allen2021_natural_scenes._utils import filter_betas_by_roi, remove_invalid_voxels_from_betas, z_score_betas_within_sessions, average_betas_across_reps

CATALOG = Catalog(
    identifier="bonner-datasets",
    csv_file=Path("/home/swadhwa5/projects/bonner-datasets/catalog.csv"),
    cache_directory=None,
)

from bonner.datasets.allen2021_natural_scenes import open_subject_assembly

engineered_model = False

if engineered_model:
    selectors = ({"source": "prf-visualrois"},)
else:
    print("using nsd general")
    selectors = ({"source": "nsdgeneral", "label": "nsdgeneral"},)

filepath = CATALOG.load_data_assembly(identifier="allen2021.natural_scenes.1pt8mm.fithrf_GLMdenoise_RR", check_integrity=False)
assembly = open_subject_assembly(0, filepath=filepath)

# assembly = open_assembly(
#     subject=subject, preprocessing=preprocessing, resolution=resolution
# )

betas = filter_betas_by_roi(
    betas=assembly["betas"],
    rois=assembly["rois"],
    selectors=selectors,
)
betas = remove_invalid_voxels_from_betas(betas, validity=assembly["validity"])
betas = z_score_betas_within_sessions(betas)
betas = average_betas_across_reps(betas)
betas = betas.transpose("presentation", "neuroid")
betas.to_netcdf("/data/mbonner5/shared/eric-datasets/nsd_one_subj/new_data.nc")


# 40 sessions for each subject
# 8 subjects
# 750 trials per session
# make sure each session has the same mean and std for each session
# z_score_betas within_session
# take one subject, figure out which roi you want, filter betas by roi, and then z-score betas