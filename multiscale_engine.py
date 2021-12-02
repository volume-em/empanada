import math
import torch
import torch.nn.functional as F
from mitonet.inference.postprocess import (
    find_instance_center, group_pixels,
    get_instance_segmentation,
    merge_semantic_and_instance,
    get_panoptic_segmentation
)
from collections import deque

# TODO TESTING MEDIAN FILTERING OF SEMANTIC FEATURES
# POSSIBLE THAT POINTREND WILL BREAK WITHOUT IT!

class MultiScaleInferenceEngine:
    def __init__(
        self,
        base_model,
        render_model,
        thing_list,
        scales=[1, 2, 4],
        input_scale=4,
        median_kernel_size=3,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        **kwargs
    ):
        assert median_kernel_size % 2 == 1, "Kernel size must be odd integer!"

        # set models to eval mode
        self.base_model = base_model.eval()
        self.render_model = render_model.eval()

        self.thing_list = thing_list

        # scales are ascending and exponents of 2
        for scale in scales:
            assert math.log(scale, 2).is_integer()

        self.scales = sorted(scales)
        self.input_scale = input_scale

        self.label_divisor = label_divisor
        self.stuff_area = stuff_area
        self.void_label = void_label
        self.nms_threshold = nms_threshold
        self.nms_kernel = nms_kernel
        self.confidence_thr = confidence_thr

        # median parameters
        self.ks = median_kernel_size
        self.median_queue = deque(maxlen=median_kernel_size)
        self.mid_idx = (median_kernel_size - 1) // 2
        
    def _harden_seg(self, sem):
        if sem.size(1) > 1: # multiclass segmentation
            sem = torch.argmax(sem, dim=1, keepdim=True)
        else:
            sem = (sem >= self.confidence_thr).long() # need integers
            
        return sem

    def get_median_sem_logits(self):
        # each item in deque is shape (1, C, H, W)
        # cat on batch dim and take the median
        median_sem = torch.median(
            torch.cat([output['sem_logits'] for output in self.median_queue], dim=0),
            dim=0, keepdim=True
        ).values
        
        # (1, C, H, W)
        return median_sem
    
    @torch.no_grad()
    def infer(self, image):
        return self.base_model(image)

    def get_instance_cells(self, ctr_hmp, offsets):
        # first find the object centers
        ctr = find_instance_center(ctr_hmp, self.nms_threshold, self.nms_kernel)

        # no objects, return zeros
        if ctr.size(0) == 0:
            return torch.zeros_like(ctr_hmp)
        
        step = 4
        return group_pixels(ctr, offsets, step=step).float()[None] # (1, 1, H, W)
    
    @torch.no_grad()
    def postprocess(self, sem_logits, features, instance_cells):
        # generate masks for all scales
        pan_pyramid = []
        seg_scale = self.input_scale * 4
        for scale in self.scales[::-1]: # loop from smallest to largest scale
            # determine the scale factor for upsampling the instance cells
            scale_factor = seg_scale / scale
            if scale_factor >= 2:
                instance_cells = F.interpolate(instance_cells, scale_factor=scale_factor, mode='nearest')

                # each forward pass upsamples sem_logits by a factor of 2
                for _ in range(int(scale_factor / 2)):
                    sem_logits = self.render_model(sem_logits, features)
                    
                seg_scale /= scale_factor

            # apply classification activation and harden
            if sem_logits.size(1) > 1:
                sem = F.softmax(sem_logits, dim=1)
            else:
                sem = torch.sigmoid(sem_logits)

            sem = self._harden_seg(sem)[0]

            # keep only label for instance classes
            instance_seg = torch.zeros_like(sem)
            for thing_class in self.thing_list:
                instance_seg[sem == thing_class] = 1

            # map object ids
            instance_seg = (instance_seg * instance_cells[0]).long()

            pan_seg = merge_semantic_and_instance(
                sem, instance_seg, self.label_divisor, self.thing_list,
                self.stuff_area, self.void_label
            )

            pan_pyramid.append(pan_seg)

        return pan_pyramid[::-1] # largest to smallest
    
    def end(self):
        # any items past self.mid_idx remaining
        # in the queue are postprocessed and returned
        final_segs = []
        for model_out in list(self.median_queue)[self.mid_idx + 1:]:
            instance_cells = self.get_instance_cells(model_out['ctr_hmp'], model_out['offsets'])
            final_segs.append(
                self.postprocess(model_out['sem_logits'], model_out['semantic_x'], instance_cells)
            )
        
        return final_segs
    
    def __call__(self, image):
        # check that image is 4d (N, C, H, W) and has a 
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1
        
        # move image to same device as the model
        device = next(self.base_model.parameters()).device
        image = image.to(device)
        
        # infer labels
        model_out = self.infer(image)
        
        # append results to median queue
        self.median_queue.append(model_out)
        
        nq = len(self.median_queue)
        if nq <= self.mid_idx:
            # take last item in the queue
            median_out = self.median_queue[-1]
        elif nq > self.mid_idx and nq < self.ks:
            # return nothing while the queue builds
            return None
        elif nq == self.ks:
            # use the middle item in the queue
            # with the median segmentation probs
            median_out = self.median_queue[self.mid_idx]
            median_out['sem_logits'] = self.get_median_sem_logits()
        else:
            raise Exception('Queue length cannot exceed maxlen!')
        
        # calculate the instance cells
        instance_cells = self.get_instance_cells(median_out['ctr_hmp'], median_out['offsets'])
        pan_seg = self.postprocess(median_out['sem_logits'], median_out['semantic_x'], instance_cells)

        return pan_seg


