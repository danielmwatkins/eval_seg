# Evaluating sea ice segmentation algorithms with manually validated imagery
The goal of this paper is to identify metrics suitable to evaluate the performance of segmentation algorithms with the specific focus of identifying sea ice floes. The metrics should enable us to answer questions about quality of the sample (representativeness), ice floe locations, ice floe area, and ice floe shape parameters.

1. Selection of candidate metrics and mathematical descriptions
2. Manually labeled validation dataset
3. Synthetic dataset
4. Numerical experiment
5. Application to algorithm results

## Setup
The calibration and validation code includes both Python and Julia scripts. The file `cal-val.yml` contains a list of the package dependencies. Using `conda`, you can create an environment with the command 

```> conda env create -f eval_seg.yml```

The package `proplot` is used for figures. There are issues with `proplot` for newer versions of `matplotlib`, hence the version is set as 0.9.7 for `proplot` and 3.4.3 for `matplotlib`, and Python is set to version 3.9.13. If a newer version of `scikit-image` is needed, we may need to choose a different tool for figure creation.

# Manually validated data
## Validation dataset
The notebook `sample_selection.ipynb` sets up the random sample dataset. Sample selection is an iterative process that involves selecting more samples than needed initially, then filtering the samples based on measures of environmental properties (sea ice fraction, sea ice concentration). 

The folder `data/metadata` includes
- Calculated daily sea ice fraction climatology, used for finding start and end times.
- Region definitions (names, bounding boxes)
- A case list formatted for running the IFT-Pipeline
- Files from initial versions of the sample dataset used along the way to creating the final dataset
- A merged validation table collecting initial manual annotations and the new samples
- A final validation table with all image metadata and analysts

Satellite imagery is stored in `data/modis`. This imagery is downloaded using the IFT pipeline at 250 m resolution.
- Truecolor images
- Falsecolor images
- Cloud Fraction
- Land masks

Sea ice extent at 4 km resolution from the MASIE-NH data product is stored in `data/masie`. The data is interpolated to the same resolution and projection as the MODIS imagery. Both ice extent and land masks are available. 

Results from the IFT-pipeline runs are placed in the `data/ift_lopez-acosta` and `data/ift_buckley`. The `ift_lopez-acosta` runs use the Julia implementation of the algorithm used in Lopez-Acosta et al. (2019), and the `ift_buckley` runs use the Python segmentation routine developed for Buckley et al. (2024). In each case, we save GeoTiff files of labeled floes for all images where the segmentation algorithm produced at least one potential sea ice object. 

## Validation
Key properties that we need to account for, in no particular order
- Location of floe centroids, uncertainty in centroid
- Location of floe boundary, uncertainty in boundary
- Sensitivity to cloud cover
- Confusion matrix: false positive rate, true positive rate, false negative rate, true negative rate
- Dependence of confusion matrix on other variables: are all parts of the FSD recovered well?
- Time uncertainty
- Sensitivity to scene size, complexity
- Tracking (similar set of tests, but testing whether a floe can be tracked, rather than whether a floe can be identified)
- Is there a limit on shapes and sizes of floes for them to be trackable?

## Validation data
TBD: update this, add images

The validation data consists of a set of 100 km by 100 km images, randomly sampled across spring and summer months, within 9 regions of the Arctic Ocean and its marginal seas. The folder `data/validation_tables` contains two folders. In the folder `data/validation_tables/quantitative_assessment_tables/`, there are CSV files for each region as well as a file `all_100km_cases.csv` that is a simple concatenation of the other files. The CSV files include case metadata, including a case number, the file name (`long_name`), the region name, start and end dates, satellite name, and image size. The quantitative assessment results are "yes/no" data for `visible_sea_ice`, `visible_landfast_ice`, `visible_floes`, and  `artifacts` (errors in the image, missing data, or obvious overlap of different images), manual assessment of cloud fraction (0 to 1, to the nearest 0.1), and cloud category (none, thin, scattered, opaque). These values were first estimated by `qa_analyst`, then checked by `qa_reviewer`. Adjustments to the values in the first assessment are noted in `notes`. The columns `fl_analyst` and `fl_reviewer` indicate the analysts who manually labeled the images and who reviewed and/or corrected the manual labeling. 

The floe labeling task was carried out by first selecting all the images where the quantitative assessment indicated visible ice floes, then randomly dividing the images between the 5 analysts. The images in Baffin Bay were each labeled twice to provide a measure of the subjectivity in floe labeling. Floe labeling assignments


# Next steps
- Calculate fraction of land pixels using the land mask and the dilated landmask: add to metadata
- Calculate cloud fraction using the false color images: add to metadata
- Download additional MASIE images based on the updated sample selection
- Calculate ice fraction from MASIE image, save MASIE subset with the validation images, add to metadata
- Updating the data tables for Google Drive with the new samples
- Make list of new samples that need to have the quantitative assessment done on them
- Make a list of all the remaining images for floe labeling
- Set up CSV files to run the extended cases on Oscar
- Set up script to copy and rename the falsecolor and truecolor images from the Oscar runs into the validation imagery folders


