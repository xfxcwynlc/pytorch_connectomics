from __future__ import print_function, division
from typing import Optional, List

import torch
import torch.nn as nn
import scipy.ndimage as ndi
import torch.nn.functional as F


class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.

    Args:
        pred (torch.Tensor): foreground logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """    
    def forward(self, 
                pred: torch.Tensor,
                mask: Optional[torch.Tensor] = None):

        pred = torch.sigmoid(pred)
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = 1.0 / diff

        if mask is not None:
            loss *= mask
        return loss.mean()


class CurvatureReg(nn.Module):
    '''
    Regularization for the curvature over the surface
    Args:
        pred (torch.Tensor): foreground logits.
        sigma (int): sigma to smooth the surface
    '''
    def __init__(self, thres=0.1, sigma=5,kernel_size=9,mode='tv',variant='min') -> None:
        '''
        Args:
        - thres, threshold to binarize prediction logits, default 0.1
        - sigma, Standard deviation of the Gaussian.
        - kernel_size, int, odd number for kernel size in Gaussian
        - mode, str, 'tv' for total variation, 
                    'mean' for mean
        - variant, str, mode for total variation, 
                    'min' for min(diff1,diff2,diff3).sum(), 
                    'sum' for sum(diff1,diff2,diff3).sum()
                    'percentage' for (diff1 + diff2 + diff3)/ max(diff1,diff2,diff3,1)
        '''
        super().__init__()
        assert mode in ['tv','mean']
        assert variant in ['min', 'sum', 'percentage']
        self.thres = thres
        self.sigma = sigma 
        self.kernal_size = kernel_size
        self.mode = mode
        self.variant = variant
        self.gaussian_kernel = self.gaussian_3d(self.kernal_size).unsqueeze(0).unsqueeze(0)
    
    def gaussian_3d(self, kernel_size=9):
        """
        Generates a 3D Gaussian kernel.
        Args:
            kernel_size (int): Size of the kernel (it should be odd).
        Returns:
            torch.Tensor: 3D Gaussian kernel.
        """
        kernel_range = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy, zz = torch.meshgrid(kernel_range, kernel_range, kernel_range)
        kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * self.sigma**2))
        return kernel / kernel.sum()

    # def signed_distance_transform(self, logits):
    #     dist = ndi.distance_transform_edt(binary_map) - ndi.distance_transform_edt(inverse_map)

    def get_mask(self, pred, act_full=False):
        '''
        Computing the narrow band by taking morphology mask

        Args:
        - pred, probabilistic prediction, of shape (N,C,H,W,L)
        - act_full, bool, includes narrow band on surface (F) or all values > thres (T)
        Returns:
        - mask: returns a binary mask which,  
                if act_full = True, return the dilation mask
                if act_fall = False, return the narrow band
        '''
        thres = self.thres
        N,C,H,W,L = pred.shape
        input = (pred >= thres).float() 
        # max pooling, dilation
        maxpool = nn.MaxPool3d(3,1,padding=1)
        m = torch.zeros(pred.shape) 
        dilation = maxpool(input)
        m[:,:,5:H-5,5:W-5,5:L-5] = dilation[:,:,5:H-5,5:W-5,5:L-5]
        if act_full:
            return m
        # else return narrow band 
        erosion = - maxpool(-input) #erosion
        m[:,:,5:H-5,5:W-5,5:L-5] = m[:,:,5:H-5,5:W-5,5:L-5]-erosion[:,:,5:H-5,5:W-5,5:L-5]
        return m #TODO we can return erosion or dilation seperatly, to handle curvature at different regions


    def curvature(self, phi):
        """ Computes divergence of vector field
        phi: input signed distance function / Gaussian smoothed function
        returns:
            curvature
        """
        grad_x, grad_y, grad_z = torch.gradient(phi,dim=(2,3,4))
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)+1e-6
        curvature = (torch.gradient(grad_x/grad_mag, dim=2)[0] + torch.gradient(grad_y/grad_mag, dim=3)[0] + torch.gradient(grad_z/grad_mag, dim=4)[0]) 
        return curvature # ignore this for now /2

    def total_variation_loss(self, batched_volume):
        '''
        Total variation loss with mean reduction [-1,0,1 ]
        '''
        #[-1,0,1 ] #phi = F.conv2d(pred, kernel, padding=self.kernal_size // 2)
        
        N,C,H,W,L = batched_volume.shape

        kernel = torch.tensor([-1, 0, 1], dtype=batched_volume.dtype, device=batched_volume.device)
        kernel_x = kernel.view(1,1,1,1,3)
        kernel_y = kernel.view(1,1,1,3,1)
        kernel_z = kernel.view(1,1,3,1,1)

        diff1  = torch.abs(F.conv3d(batched_volume, kernel_x ,padding=(0,0,1)))
        diff2  = torch.abs(F.conv3d(batched_volume, kernel_y ,padding=(0,1,0)))
        diff3  = torch.abs(F.conv3d(batched_volume, kernel_z ,padding=(1,0,0)))
        assert diff1.shape == batched_volume.shape and diff2.shape == batched_volume.shape and diff3.shape == batched_volume.shape
        if self.variant == 'min':
            score = torch.minimum(torch.minimum(diff1, diff2), diff3) 
        elif self.variant == 'sum':
            score = (diff1 + diff2 + diff3)
        elif self.variant == 'percentage':
            max_elements = torch.maximum(torch.maximum(diff1,diff2),diff3)
            max_elements = torch.where(max_elements<1e-6, 1e-6, max_elements) # 1e-6 avoid zero division
            score = (diff1 + diff2 + diff3)/max_elements 
        return score
    
    def forward(self, pred , act_full=True):
        '''
        Args:
        - pred: (N,C,H,W,L) probablistic prediction   
        - act_full, bool, includes narrow band on surface (F) or all values > thres (T)
        Returns:
        - Curvature, loss with mean reduction, divided by total count of nonzero elements
        '''
        kernel = self.gaussian_kernel.to(pred.device)
        mask = self.get_mask(pred, act_full=act_full).to(pred.device) #Do not act soft from the first place
        phi = F.conv3d(pred, kernel, padding=self.kernal_size // 2)
        k = self.curvature(phi).to(pred.device) 
        
        #Not reduced yet
        if self.mode == 'tv':
            loss = self.total_variation_loss(k)
        else: #if just 'mean'
            loss = k.abs()

        if mask is not None:
            loss *= mask
        return loss.mean()


class ForegroundDTConsistency(nn.Module):    
    """Consistency regularization between the binary foreground mask and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None
    """
    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):

        log_prob_pos = F.logsigmoid(pred1)
        log_prob_neg = F.logsigmoid(-pred1)
        distance = torch.tanh(pred2)
        dist_pos = torch.clamp(distance, min=0.0)
        dist_neg = - torch.clamp(distance, max=0.0)

        loss_pos = - log_prob_pos * dist_pos
        loss_neg = - log_prob_neg * dist_neg
        loss = loss_pos + loss_neg

        if mask is not None:
            loss *= mask
        return loss.mean()


class ContourDTConsistency(nn.Module):
    """Consistency regularization between the instance contour map and
    signed distance transform.

    Args:
        pred1 (torch.Tensor): contour logits.
        pred2 (torch.Tensor): signed distance transform.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """
    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):

        contour_prob = torch.sigmoid(pred1)
        distance_abs = torch.abs(torch.tanh(pred2))
        assert contour_prob.shape == distance_abs.shape
        loss = contour_prob * distance_abs
        loss = loss**2

        if mask is not None:
            loss *= mask
        return loss.mean()


class FgContourConsistency(nn.Module):
    """Consistency regularization between the binary foreground map and 
    instance contour map.

    Args:
        pred1 (torch.Tensor): foreground logits.
        pred2 (torch.Tensor): contour logits.
        mask (Optional[torch.Tensor], optional): weight mask. Defaults: None.
    """
    sobel = torch.tensor([1, 0, -1], dtype=torch.float32)
    eps = 1e-7

    def __init__(self, tsz_h=1) -> None:
        super().__init__()

        self.sz = 2*tsz_h + 1
        self.sobel_x = self.sobel.view(1,1,1,1,3)
        self.sobel_y = self.sobel.view(1,1,1,3,1)

    def forward(self, 
                pred1: torch.Tensor, 
                pred2: torch.Tensor, 
                mask: Optional[torch.Tensor] = None):

        fg_prob = torch.sigmoid(pred1)
        contour_prob = torch.sigmoid(pred2)

        self.sobel_x = self.sobel_x.to(fg_prob.device)
        self.sobel_y = self.sobel_y.to(fg_prob.device)

        # F.conv3d - padding: implicit paddings on both sides of the input. 
        # Can be a single number or a tuple (padT, padH, padW).
        edge_x = F.conv3d(fg_prob, self.sobel_x, padding=(0,0,1))
        edge_y = F.conv3d(fg_prob, self.sobel_y, padding=(0,1,0))

        edge = torch.sqrt(edge_x**2 + edge_y**2 + self.eps)
        edge = torch.clamp(edge, min=self.eps, max=1.0-self.eps)

        # F.pad: the padding size by which to pad some dimensions of input are 
        # described starting from the last dimension and moving forward.
        edge = F.pad(edge, (1,1,1,1,0,0))
        edge = F.max_pool3d(edge, kernel_size=(1, self.sz, self.sz), stride=1)

        assert edge.shape == contour_prob.shape
        loss = F.mse_loss(edge, contour_prob, reduction='none')

        if mask is not None:
            loss *= mask
        return loss.mean()


class NonoverlapReg(nn.Module):
    """Regularization to prevent overlapping prediction of pre- and post-synaptic
    masks in synaptic polarity prediction ("1" in MODEL.TARGET_OPT).

    Args:
        fg_masked (bool): mask the regularization region with predicted cleft. Defaults: True
    """
    def __init__(self, fg_masked: bool = True) -> None:
        super().__init__()
        self.fg_masked = fg_masked

    def forward(self, pred: torch.Tensor):
        # pred in (B, C, Z, Y, X)
        pos = torch.sigmoid(pred[:, 0]) # pre-synaptic
        neg = torch.sigmoid(pred[:, 1]) # post-synaptic
        loss = pos * neg

        if self.fg_masked:
            # masked by the cleft (union of pre and post)
            # detached to avoid decreasing the cleft probability
            loss = loss * torch.sigmoid(pred[:, 2].detach())

        return loss.mean()

if __name__ == '__main__':
    #test curvature consistency 
    rand_tensor = torch.randn(3,1,25,25,25)
    reg1 = CurvatureReg(mode='tv',variant='min')
    reg2 = CurvatureReg(mode='tv',variant='percentage')
    reg3 = CurvatureReg(mode='tv',variant='sum')

    loss = reg1(rand_tensor)
    print(loss)
    loss = reg2(rand_tensor)
    print(loss)
    loss = reg3(rand_tensor)
    print(loss)

    #verify the run check 
    # dumbell shape
    #wanna ask Arnab how to design an experiment!! 