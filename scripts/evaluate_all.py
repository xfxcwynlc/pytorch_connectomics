import os
from matplotlib import pyplot as plt
import torch
import tifffile
import argparse
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from connectomics.utils.evaluate import get_binary_jaccard, adapted_rand, voi
from connectomics.utils.process import binarize_and_median



def evaluate_prediction(pred_path, gt_path, use_median=True,
                        thres=[0.3, 0.4, 0.5, 0.6]):
    binarize = False
    gt = tifffile.imread(gt_path).astype(np.uint8)
    pred = tifffile.imread(pred_path)

    if len(pred.shape) == 4: #remove the first channel
        pred = pred[0,:]
    if pred.max() == 1:
        binarize = True
    else:
        pred = pred / 255. # output is casted to uint8 with range [0,255].

    if use_median:
        '''
        if use_median is True, the prediction must not be binarilized
        A threshold of 0.5 with median filter size of (15,15,15) is used 
        to binarize the volume , then produce the jaccard scores
        '''
        sz = (7,7,7)
        thres = [0.45]
        #print(pred.shape,gt.shape)
        pred = binarize_and_median(pred, size=sz, thres=thres)
        scores = get_binary_jaccard(pred, gt) # prediction is already binarized
        print(f"Median filter size {sz}")
        assert pred.dtype == np.uint8 and gt.dtype == np.uint8 
        adapted_ran_err = {thres[-1]:adapted_rand(pred,gt)}
        vois = {thres[-1]: voi(pred,gt)} 
    elif binarize: 
        scores = get_binary_jaccard(pred, gt) # prediction is already binarized
        print(f"Image has already been binarilized")
        assert pred.dtype == np.uint8 and gt.dtype == np.uint8 
        adapted_ran_err = {'n/a':adapted_rand(pred,gt)}
        vois = {'n/a': voi(pred,gt)} 

    else:
        '''
        Otherwise we will provide a list of threshold and binarlize the image
        and compute the jaccard scores
        '''
        scores = get_binary_jaccard(pred, gt, thres)
        adapted_ran_err = {}
        vois = {}
        for t in thres:
            pred_temp = (pred>t).astype(np.uint8)
            assert pred_temp.dtype == np.uint8 and gt.dtype == np.uint8 
            adapted_ran_err[t] = adapted_rand(pred_temp,gt)
            vois[t] = voi(pred_temp, gt)

    for i,t in enumerate(thres):
        print(f"The prediction from {pred_path} compared to {gt_path} with use_median {use_median} w Binarilize threshold: {t}")
        print(f"\tForeground IoU: {scores[i][0]} \n\tIoU: {scores[i][1]}\n\tprecision: {scores[i][2]}\n\trecall: {scores[i][3]}") 
        print(f"\tAdapted Rand Error: {adapted_ran_err[t]}.")
        print(f"\tSplit error: {vois[t][0]}. \n\tMerge error: {vois[t][1]}")
        #foreground IoU, IoU, precision and recall

    return pred_path,scores #,adapted_ran_err,vois #return precision, recall
    


if __name__ == '__main__':
    '''/Users/yananw/Downloads/pred_txyz.tif 
    /Users/yananw/Downloads/V4_seg_gtdownsampled.tif
    /Users/yananw/Desktop/results_training/val_server/1_val_curvature_txyz.tif
    /Users/yananw/Desktop/results_training/val_server/mask_1val_gt.tif
    '''
    parser = argparse.ArgumentParser("Evaluation of astrocytic volume, rand err, variation of information, jac scores, precision")
    parser.add_argument('--gt_path',  type=str, help='path to groundtruth mask tiff, default 0-1')
    parser.add_argument('--pred_folder_path',  type=str, help='path to FOLDER predictions tiff , can be 0-255 or 0-1')
    parser.add_argument('--use_median', action='store_false', default=False, help='Use binarize and median filter')
    parser.add_argument('--threslist', nargs='*', type=float, default=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85], help='List of thresholds; seperate by blank space')
    
    args = parser.parse_args()

    pred_paths = [os.path.join(args.pred_folder_path,fn) for fn in os.listdir(args.pred_folder_path) if fn.endswith('txyz.tif')]

    thres = args.threslist

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_prediction, path, args.gt_path, args.use_median,thres) for path in pred_paths]
        results = []
        for future in concurrent.futures.as_completed(futures):
            # Process the results here
            results.append(future.result())


    #plot line graphs:
    # Sort results by pred_path so that the colors will be consistent across subplots
    results.sort(key=lambda x: x[0])

    # Create subplots
    fig, axs = plt.subplots(2,2, figsize=(10, 5))
    # Flatten axs
    axs = axs.ravel()
    # Set y-axis labels
    y_labels = ['Foreground IoU', 'IoU', 'Precision', 'Recall']
    x = thres
    # Plot results
    for i, ax in enumerate(axs):
        for pred_path, scores in results:
            name = os.path.basename(pred_path)[:-9]
            ax.plot(x, scores[:, i], label=name)
        ax.set_title(y_labels[i])
        ax.set_xlabel('Threshold')
        ax.set_ylabel(y_labels[i])
        ax.legend()

    plt.tight_layout()
    plt.show()