"""Functions for evaluating segmentation results"""
import skimage
import pandas as pd
import numpy as np


def select_relevant_set(label_i, gt_image, seg_image, gt_image_props, seg_image_props, thresh=0.5):
    """Identifies the set of labels in the segmented image corresponding
    to the "Relevant Set". label_i is the label in the ground truth image (gt_image),
    seg_image is a labeled image with a set of candidate objects to match. gt_image_props and
    seg_image_props are the results of skimage.measure.regionprops placed into a dictionary by 
    label. Each object sj in seg_image with a nonzero overlap with object gi in the gt_image is checked
    for 4 criteria: centroid(gi) in sj; centroid(sj) in gi, relative overlap(gi, sj) > thresh, or
    relative overlap(sj, gi) > thresh. Relative overlap (a, b) is |a intersect b|/|b|.
    Returns a list of possible indices."""

    # First grab any labels that have a nonzero intersection.
    # This is \tilde Y from Clinton et al. 2010
    possible = [x for x in np.unique(seg_image[gt_image == label_i]) if x != 0]
    
    if len(possible) == 0:
        return []

    reference_set = []
    for j in possible:    
        # A: centroid(gi) in sj
        rg, cg = gt_image_props[label_i].centroid
        s_label = seg_image[int(np.round(rg)), int(np.round(cg))]
        
        in_A = s_label == j
        
        # B: centroid(sj) in gi
        rs, cs = seg_image_props[j].centroid
        g_label = gt_image[int(np.round(rs)), int(np.round(cs))]
        in_B = g_label == label_i
        
        # C: relative_overlap(g, s) > thresh
        overlap = np.sum((gt_image == label_i) & (seg_image == j))
        in_C = overlap/np.sum(seg_image == j) > thresh
        
        # D: relative_overlap(s, g) > thresh
        in_D = overlap/np.sum(gt_image == label_i) > thresh
        
        if np.any([in_A, in_B, in_C, in_D]):
            reference_set.append(j)

    return reference_set

def bdry_dice_score(label_i, label_j, gt_image, seg_image, radius=1, directed=True):
    """Computes the directed boundary F-score, which is an adaptation of the 
    directed boundary Dice score used in Yeghiazaryan and Voiculescu 2018.
    TBD: Add variable, directed=True, so if it is false, compute the symmetric version.
    """
    imsize = gt_image.shape
    obj_bdry = skimage.segmentation.find_boundaries(gt_image, mode='inner')
    index_bdry = np.argwhere((gt_image == label_i) & obj_bdry)

    k = skimage.morphology.disk(radius)
    kdim = k.shape
    
    gi = gt_image == label_i
    sj = seg_image == label_j
    y = np.zeros(len(index_bdry))
    for idx, (row,col) in enumerate(index_bdry):
        left = max(col - radius, 0)
        right = min(col + radius + 1, imsize[0])
        bottom = max(row - radius, 0)
        top = min(row + radius + 1, imsize[1])
        
        gi_k = gi[bottom:top, left:right]
        
        # this selects the segmented spot
        sj_k = sj[bottom:top, left:right]
        
        # To adjust the size of the kernel
        # Check size of overlap in each direction
        
        k_left = 0
        k_right = kdim[0]
        k_bottom = 0
        k_top = kdim[1]
        
        # We have to adjust the size if the kernel overlaps the boundary
        if left - radius < 0:
            k_left = np.abs(left - radius)
        elif right + radius > imsize[0]:
            k_right = np.abs(right + radius - imsize[0])
        if bottom - radius < 0:
            k_bottom = np.abs(bottom - radius)
        elif top + radius > imsize[1]:
            k_top = np.abs(top + radius - imsize[1])
        
        k[k_bottom:k_top, k_left:k_right]
    
        denom = 2 * np.sum(sj_k & gi_k & k)
        num = np.sum(sj_k & k) + np.sum(gi_k & k)
        
        y[idx] = denom/num
    return np.mean(y)


# Metrics for objectwise comparison
def apply_segmentation_metrics_ij(label_i, gt_image, label_j, seg_image,
                               gt_image_props, seg_image_props):
    """Produce a pandas Series object with the names of each measure as the index. Calculates measures for
    indivual pairs of objects. Inputs are the (integer) labels, the labeled images, and the dictionaries 
    with object properties returned by regionprops. If label_j is nan, then it returns 0 for each metric.
    
    Metrics calculated:

    relative_error_area
    area_similarity_ratio
    undersegmentation_error
    oversegmentation_error
    euclidean_distance
    hausdorff_distance
    modified_hausdorff_distance
    shape_dissimilarity
    directed_boundary_dice_score
    """

    metric_results = {'label_i': label_i,
                      'label_j': label_j}

    
    area_g = gt_image_props[label_i].area
    rg, cg = gt_image_props[label_i].centroid
    
    if np.isnan(label_j):
        area_s = np.nan
        rs, cs = np.nan, np.nan
        
    else:
        area_s = seg_image_props[label_j].area
        rs, cs = seg_image_props[label_j].centroid
    area_gs = np.sum((gt_image == label_i) & (seg_image == label_j))


    metric_results['area_i'] = area_g
    metric_results['area_j'] = area_s
    metric_results['overlap'] = area_gs
    
    pr = area_gs/area_s
    re = area_gs/area_g
    f1_score = 2*(pr*re)/(pr+re)
    

    # area
    metric_results['relative_error_area'] = (area_g - area_s)/area_g
    metric_results['area_similarity_ratio'] = min(area_s, area_g)/max(area_s, area_g)
    metric_results['undersegmentation_error'] = 1 - pr
    metric_results['oversegmentation_error'] = 1 - re
    metric_results['segmentation_error'] = 1 - f1_score

    # location
    metric_results['euclidean_distance'] = np.sqrt((rg - rs)**2 + (cg - cs)**2)
    metric_results['hausdorff_distance'] = skimage.metrics.hausdorff_distance(gt_image == label_i,
                                                                              seg_image == label_j,
                                                                              method='standard')
    
    metric_results['modified_hausdorff_distance'] = skimage.metrics.hausdorff_distance(gt_image == label_i,
                                                                                       seg_image == label_j,
                                                                                       method='modified')

    # boundary -- identical to the relative area error?? double check this formula
    metric_results['shape_dissimilarity'] = (np.sum((gt_image == label_i) & (seg_image != label_j)) + 
                                             np.sum((gt_image != label_i) & (seg_image == label_j))) / area_g


    metric_results['directed_boundary_dice_score'] = bdry_dice_score(
        label_i, label_j, gt_image, seg_image, radius=1, directed=True)

    # add symmetric score
    # add f-score
    
    return pd.Series(metric_results)

# Here's what would get called for each image
def compute_metrics(gt_image, seg_image, return_type='weighted'):
    """Extract features from the ground truth and segmented image, then compute a series of metrics
    to measure the quality of the segmentation. Returns a pandas Series with measures calculated for 
    the full image. If "weighted=True", then weight by the ground truth object size before returning.
    """


    gt_img_props = {floe.label: floe for floe in skimage.measure.regionprops(gt_image)}
    seg_img_props = {floe.label: floe for floe in skimage.measure.regionprops(seg_image)}
    
    results = []
    for label_i in gt_img_props:
        ref_set = select_relevant_set(label_i=label_i, gt_image=gt_image, seg_image=seg_image,
                        gt_image_props=gt_img_props, seg_image_props=seg_img_props, thresh=0.5)
        if len(ref_set) > 0:
            for label_j in ref_set:
                results.append(apply_segmentation_metrics_ij(label_i=label_i,
                                                             gt_image=gt_image,
                                                             label_j=label_j,
                                                             seg_image=seg_image,
                                                             gt_image_props=gt_img_props,
                                                             seg_image_props=seg_img_props))

    # image-wide metrics - add another function apply_segmentation_metrics() (i.e. no "ij" at end)
    # and get image-wide Pr, Re, F, MCC.

    # will fail if results is empty -- add method to make row of nans with right labels
    results = pd.DataFrame(results)

    if return_type == 'all':
        return results
        
    # First average over the relevant sets
    results_i = results.groupby('label_j').mean()
    
    # The unweighted average
    unweighted_mean = results_i.mean(axis=0)
    unweighted_mean['n'] = len(results_i)
    
    # Weighted by area
    if return_type == 'weighted':
        w = results_i.area_i / results_i.area_i.sum()
        weighted_mean = pd.Series(np.average(results_i, weights=w, axis=0), index=unweighted_mean.index)
        weighted_mean['n'] = len(results_i)
    
        return weighted_mean

    else:
        return unweighted_mean


# Evaluation metrics operating on full image
# precision

# recall

# f1score




# Evaluation metrics operating on object and relevant set
###### Area Based Methods ###########
# Signed Relative Error in Area

# SimSize (Zhan 2005)

# Area Fit Index 

# Shape Dissimilarity

# Quality Rate

# Oversegmentation Error

# Undersegmentation Error 

##### Location Based Metrics ########
# Mean difference in centroids

# 



###### Boundary Based Methods ##########
# TBD: Hausdorff Distance
# This function operates on two boolean images. Wrap to get data for each element of 
# relevant set.
# skimage.metrics.hausdorff_distance(image0, image1, method='standard')

# TBD: Modified Hausdorff Distance
# skimage.metrics.hausdorff_distance(image0, image1, method='modified')

####### To-be-sorted #########
# Matthews Correlation Coefficient
# Cohen Kappa Score

