import torch
import tifffile
import argparse
import numpy as np
from connectomics.utils.evaluate import get_binary_jaccard, adapted_rand, voi
from connectomics.utils.process import binarize_and_median

if __name__ == '__main__':
    '''/Users/yananw/Downloads/pred_txyz.tif 
    /Users/yananw/Downloads/V4_seg_gtdownsampled.tif
    
    '''
    parser = argparse.ArgumentParser("Evaluation of astrocytic volume, rand err, variation of information, jac scores, precision")
    parser.add_argument('--gt_path',  type=str, help='path to groundtruth mask tiff, default 0-1')
    parser.add_argument('--pred_path',  type=str, help='path to prediction tiff, can be 0-255 or 0-1')
    parser.add_argument('--use_median',  type=bool, default=False, help='Use binarize and median filter')

    args = parser.parse_args()
    binarize = False
    gt = tifffile.imread(args.gt_path).astype(np.uint8)
    pred = tifffile.imread(args.pred_path)
    if len(pred.shape) == 4:
        pred = pred[0,:]
    if pred.max() == 1:
        binarize = True
    else:
        pred = pred / 255. # output is casted to uint8 with range [0,255].

    print(f"The prediction from {args.pred_path}")
    print(f"compared to {args.gt_path}")
    print(f"with use_median {args.use_median}")

    if args.use_median:
        '''
        if use_median is True, the prediction must not be binarilized
        A threshold of 0.8 with median filter size of (15,15,15) is used 
        to binarize the volume , then produce the jaccard scores
        '''
        sz = (7,7,7)
        thres = [0.8]
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
        thres = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # evaluate at multiple thresholds.
        scores = get_binary_jaccard(pred, gt, thres)
        adapted_ran_err = {}
        vois = {}
        for t in thres:
            pred_temp = (pred>t).astype(np.uint8)
            assert pred_temp.dtype == np.uint8 and gt.dtype == np.uint8 
            adapted_ran_err[t] = adapted_rand(pred_temp,gt)
            vois[t] = voi(pred_temp, gt)

    for i,t in enumerate(thres):
        print(f"Binarilize threshold: {t}")
        print(f"\tForeground IoU: {scores[i][0]} \n\tIoU: {scores[i][1]}\n\tprecision: {scores[i][2]}\n\trecall: {scores[i][3]}") 
        print(f"\tAdapted Rand Error: {adapted_ran_err[t]}.")
        print(f"\tSplit error: {vois[t][0]}. \n\tMerge error: {vois[t][1]}")
        #foreground IoU, IoU, precision and recall
    


    