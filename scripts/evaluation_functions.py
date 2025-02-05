"""Functions for evaluating segmentation results"""


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
        return

    reference_set = []
    for j in possible:    
        # A: centroid(gi) in sj
        rg, cg = gt_image_props[label_i].centroid
        s_label = seg_image[int(np.round(r)), int(np.round(c))]
        
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

        # test
    print('Label', label_i, 'Size', len(reference_set), 'A:', in_A, 'B: ', in_B, 'C: ', in_C, 'D: ', in_D)
    return reference_set

# Metrics for objectwise comparison
def apply_segmentation_metrics_ij(label_i, gt_image, label_j, seg_image,
                               gt_image_props, seg_image_props):
    """Produce a pandas Series object with the names of each measure as the index. Calculates measures for
    indivual pairs of objects. Inputs are the (integer) labels, the labeled images, and the dictionaries 
    with object properties returned by regionprops.
    
    Metrics calculated:

    relative_error_area
    area_similarity_ratio
    undersegmentation_error
    oversegmentation_error
    euclidean_distance
    hausdorff_distance
    modified_hausdorff_distance
    shape_dissimilarity
    directed_boundary_fscore
    """

    metric_results = {'label_i': label_i,
                      'label_j': label_j}

    area_g = gt_image_props[label_i].area
    area_s = seg_image_props[label_j].area
    area_gs = np.sum((gt_image == label_i) & (seg_image == label_j))


    metric_results['area_i'] = area_g
    metric_results['area_j'] = area_s
    metric_results['overlap'] = area_gs
    
    pr = area_gs/area_s
    re = area_gs/area_g
    f1_score = 2*(pr*re)/(pr+re)

    rg, cg = gt_image_props[label_i].centroid
    rs, cs = seg_image_props[label_j].centroid

    # area
    metric_results['relative_error_area'] = (area_g - area_s)/area_g
    metric_results['area_similarity_ratio'] = min(area_s, area_g)/max(area_s, area_g)
    metric_results['undersegmentation_error'] = 1 - pr
    metric_results['oversegmentation_error'] = 1 - re
    metric_results['segmentation_error'] = 1 - f1_score

    # location
    metric_results['euclidean_distance'] = np.sqrt((rg - rs)**2 + (cg - cs)**2)
    metric_results['hausdorff_distance'] = skimage.metrics.hausdorff_distance(gt_image == label_i, seg_image == label_j, method='standard')
    metric_results['modified_hausdorff_distance'] = skimage.metrics.hausdorff_distance(gt_image == label_i, seg_image == label_j, method='modified')

    # boundary
    metric_results['shape_dissimilarity'] = (np.sum((gt_image == label_i) & (seg_image != label_j)) + 
                                             np.sum((gt_image != label_i) & (seg_image == label_j))) / area_g

    # TBD: write function to calculate directed boundary score
    # metric_results['directed_boundary_fscore'] = dbf(gt_image, seg_image, r=5)
    

    return pd.Series(metric_results)

# Here's what would get called for each image
def compute_metrics(gt_image, seg_image, weighted=True):
    """Extract features from the ground truth and segmented image, then compute a series of metrics
    to measure the quality of the segmentation. Returns a pandas Series with measures calculated for 
    the full image. If "weighted=True", then weight by the ground truth object size before returning.
    """

    # TBD
    # Can add image-wide metrics here

    
    results = []
    for label_i in man_img_props:
        ref_set = select_relevant_set(label_i=label_i, gt_image=man_img, seg_image=ift_img,
                        gt_image_props=man_img_props, seg_image_props=ift_img_props, thresh=0.5)
    
        if len(ref_set) > 0:
            for label_j in ref_set:
                results.append(apply_segmentation_metrics_ij(label_i=label_i,
                                                             gt_image=man_img,
                                                             label_j=label_j,
                                                             seg_image=ift_img,
                                                             gt_image_props=man_img_props,
                                                             seg_image_props=ift_img_props))
    results = pd.DataFrame(results)
    
    # First average over the relevant sets
    results_i = results.groupby('label_j').mean()
    
    # The unweighted average
    unweighted_mean = results_i.mean(axis=0)
    
    # Weighted by area
    w = results_i.area_i / results_i.area_i.sum()
    weighted_mean = pd.Series(np.average(results_i, weights=w, axis=0), index=unweighted_mean.index)
    
    # In both cases, add the number of ground truth elements, as well
    weighted_mean['n'] = len(results_i)
    unweighted_mean['n'] = len(results_i)

    if weighted:
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

