import math
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from empanada.inference.postprocess import (
    factor_pad, find_instance_center, group_pixels,
    get_instance_segmentation,
    merge_semantic_and_instance,
    get_panoptic_segmentation
)
from collections import deque


__all__ = [
    'InferenceEngine',
    'MedianInferenceEngine',
    'MedianInferenceEngineBC',
    'MultiGPUInferenceEngine',
    'MultiScaleInferenceEngine'
]

class InferenceEngine:
    def __init__(
        self,
        model,
        thing_list,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        **kwargs
    ):
        # set model to eval mode
        self.model = model.eval()
        self.thing_list = thing_list
        self.label_divisor = label_divisor
        self.stuff_area = stuff_area
        self.void_label = void_label
        self.nms_threshold = nms_threshold
        self.nms_kernel = nms_kernel
        self.confidence_thr = confidence_thr

    def _harden_seg(self, sem):
        if sem.size(1) > 1: # multiclass segmentation
            sem = torch.argmax(sem, dim=1, keepdim=True)
        else:
            sem = (sem >= self.confidence_thr).long() # need integers

        return sem

    def infer(self, image):
        # run inference
        with torch.no_grad():
            model_out = self.model(image)
            sem_logits = model_out['sem_logits']

            # multiclass or binary logits to probs
            if sem_logits.size(1) > 1:
                sem = F.softmax(sem_logits, dim=1)
            else:
                sem = torch.sigmoid(sem_logits)

        # notice that sem is NOT sem_logits
        model_out['sem'] = sem
        return model_out

    def postprocess(self, sem, ctr_hmp, offsets):
        pan_seg, _ = get_panoptic_segmentation(
            sem, ctr_hmp, offsets, self.thing_list,
            self.label_divisor, self.stuff_area,
            self.void_label, self.nms_threshold, self.nms_kernel
        )
        return pan_seg

    def end(self):
        # no history to finish
        return []

    def __call__(self, image):
        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        device = next(self.model.parameters()).device
        image = image.to(device, non_blocking=True)

        # infer labels and postprocess
        model_out = self.infer(image)

        # harden the segmentation to (N, 1, H, W)
        model_out['sem'] = self._harden_seg(model_out['sem'])

        pan_seg = self.postprocess(
            model_out['sem'], model_out['ctr_hmp'], model_out['offsets']
        )

        return pan_seg
    
class MedianInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model,
        thing_list,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        median_kernel_size=3,
        **kwargs
    ):
        assert median_kernel_size % 2 == 1, "Kernel size must be odd integer!"
        super().__init__(
            model, thing_list, label_divisor, stuff_area,
            void_label, nms_threshold, nms_kernel, confidence_thr,
            **kwargs
        )

        self.ks = median_kernel_size
        self.median_queue = deque(maxlen=median_kernel_size)
        self.mid_idx = (median_kernel_size - 1) // 2

    def get_median_sem(self):
        # each item in deque is shape (1, C, H, W)
        # cat on batch dim and take the median
        median_sem = torch.median(
            torch.cat([output['sem'] for output in self.median_queue], dim=0),
            dim=0, keepdim=True
        ).values

        # (1, C, H, W)
        return median_sem

    def end(self):
        # any items past self.mid_idx remaining
        # in the queue are postprocessed and returned
        final_segs = []
        for model_out in list(self.median_queue)[self.mid_idx + 1:]:
            model_out['sem'] = self._harden_seg(model_out['sem'])
            pan_seg = self.postprocess(
                model_out['sem'], model_out['ctr_hmp'], model_out['offsets']
            )
            final_segs.append(pan_seg)

        return final_segs

    def __call__(self, image):
        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        device = next(self.model.parameters()).device
        image = image.to(device, non_blocking=True)

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
            median_out['sem'] = self.get_median_sem()
        else:
            raise Exception('Queue length cannot exceed maxlen!')

        # harden the segmentation to (N, 1, H, W)
        median_out['sem'] = self._harden_seg(median_out['sem'])

        pan_seg = self.postprocess(
            median_out['sem'], median_out['ctr_hmp'], median_out['offsets']
        )

        return pan_seg
    
class MedianInferenceEngineBC:
    def __init__(
        self,
        model,
        median_kernel_size=3,
        **kwargs
    ):
        assert median_kernel_size % 2 == 1, "Kernel size must be odd integer!"
        self.model = model.eval()
        self.ks = median_kernel_size
        self.median_queue = deque(maxlen=median_kernel_size)
        self.mid_idx = (median_kernel_size - 1) // 2

    def get_median(self):
        # each item in deque is shape (1, C, H, W)
        # cat on batch dim and take the median
        median_out = torch.median(
            torch.cat([output for output in self.median_queue], dim=0),
            dim=0, keepdim=True
        ).values

        # (1, C, H, W)
        return median_out
    
    @torch.no_grad()
    def infer(self, image):
        model_out = self.model(image)
        sem_logits = model_out['sem_logits']
        cnt_logits = model_out['cnt_logits']

        # only works for binary
        assert sem_logits.size(1) == 1
        sem = torch.sigmoid(sem_logits)
        cnt = torch.sigmoid(cnt_logits)

        return torch.cat([sem, cnt], dim=1) # (1, 2, H, W)

    def end(self):
        # list of remaining segs (1, 2, H, W)
        return list(self.median_queue)[self.mid_idx + 1:]

    def __call__(self, image):
        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        device = next(self.model.parameters()).device
        image = image.to(device, non_blocking=True)

        output = self.infer(image)

        # append results to median queue
        self.median_queue.append(output)

        nq = len(self.median_queue)
        if nq <= self.mid_idx:
            # take last item in the queue
            output = self.median_queue[-1]
        elif nq == self.ks:
            output = self.get_median()
        else: # implies nq > self.mid_idx and nq < self.ks
            # return nothing while the queue builds
            return None

        return output

class MultiGPUInferenceEngine(InferenceEngine):
    def __init__(
        self,
        model,
        thing_list,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        **kwargs
    ):
        super().__init__(
            model, thing_list, label_divisor, stuff_area,
            void_label, nms_threshold, nms_kernel, confidence_thr,
            **kwargs
        )

    def get_instance_cells(self, ctr_hmp, offsets):
        # first find the object centers
        ctr = find_instance_center(ctr_hmp, self.nms_threshold, self.nms_kernel)

        # no objects, return zeros
        if ctr.size(0) == 0:
            return torch.zeros_like(ctr_hmp)

        return group_pixels(ctr, offsets, step=1)

    def get_panoptic_seg(self, sem, instance_cells):
        # keep only label for instance classes
        instance_seg = torch.zeros_like(sem)
        for thing_class in self.thing_list:
            instance_seg[sem == thing_class] = 1

        # map object ids
        instance_seg = (instance_seg * instance_cells[None]).long()

        pan_seg = merge_semantic_and_instance(
            sem, instance_seg, self.label_divisor, self.thing_list,
            self.stuff_area, self.void_label
        )

        return pan_seg

class MultiScaleInferenceEngine:
    def __init__(
        self,
        base_model,
        render_model,
        thing_list,
        median_kernel_size=3,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        padding_factor=16,
        coarse_boundaries=True,
        device='gpu',
        **kwargs
    ):
        assert median_kernel_size % 2 == 1, "Kernel size must be odd integer!"

        # set models to eval mode
        self.base_model = base_model.eval()
        self.render_model = render_model.eval()

        if device == 'gpu' and torch.cuda.is_available():
            # generically load onto any gpu
            self.base_model = self.base_model.cuda()
            self.render_model = self.render_model.cuda()
            self.device = 'cuda'
        elif torch.cuda.is_available():
            self.base_model = self.base_model.to(device)
            self.render_model = self.render_model.to(device)
            self.device = device
        else:
            print(f'Using CPU, this may be slow.')
            self.device = 'cpu'

        self.thing_list = thing_list

        self.padding_factor = padding_factor
        self.coarse_boundaries = coarse_boundaries

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

    def reset(self):
        # reset the median queue
        self.median_queue = deque(maxlen=self.ks)

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

    @torch.no_grad()
    def get_instance_cells(self, ctr_hmp, offsets, upsampling=1):
        # if calculating coarse boundaries then
        # don't upsample from 1/4th resolution
        # coarse boundaries are faster and memory friendly
        # but can look "blocky" in the final segmentation
        scale_factor = 1 if self.coarse_boundaries else 4

        if scale_factor > 1:
            ctr_hmp = F.interpolate(ctr_hmp, scale_factor=scale_factor, mode='bilinear', align_corners=True)
            offsets = F.interpolate(offsets, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        # first find the object centers
        ctr = find_instance_center(ctr_hmp, self.nms_threshold, self.nms_kernel)

        # grid step size for pixel grouping
        step = 4 if self.coarse_boundaries else 1

        # no objects, return zeros
        if ctr.size(0) == 0:
            instance_cells = torch.zeros_like(ctr_hmp)
        else:
            # grouped pixels should be integers,
            # but we need them in float type for upsampling
            instance_cells = group_pixels(ctr, offsets, step=step).float()[None] # (1, 1, H, W)

        # scale again by the upsampling factor times step
        instance_cells = F.interpolate(instance_cells, scale_factor=int(upsampling * step), mode='nearest')
        return instance_cells

    @torch.no_grad()
    def upsample_logits(
        self,
        sem_logits,
        coarse_sem_seg_logits,
        features,
        scale_factor
    ):
        # apply render model
        if scale_factor >= 2:
            # each forward pass upsamples sem_logits by a factor of 2
            for _ in range(int(math.log(scale_factor, 2))):
                sem_logits = self.render_model(sem_logits, coarse_sem_seg_logits, features)

        # apply classification activation and harden
        if sem_logits.size(1) > 1:
            sem = F.softmax(sem_logits, dim=1)
        else:
            sem = torch.sigmoid(sem_logits)

        return sem, sem_logits

    @torch.no_grad()
    def postprocess(self, coarse_sem_seg_logits, features, instance_cells, upsampling=1):
        sem_logits = coarse_sem_seg_logits.clone()

        # seg to be upsampled by factor of 4 and then
        # by factor of upsample
        sem, sem_logits = self.upsample_logits(
            sem_logits, coarse_sem_seg_logits, features, upsampling * 4
        )

        # harden the segmentation
        sem = self._harden_seg(sem)[0]

        # keep only label for instance classes
        instance_seg = torch.zeros_like(sem)
        for thing_class in self.thing_list:
            instance_seg[sem == thing_class] = 1

        # map object ids
        instance_seg = (instance_seg * instance_cells[0]).long()

        # get the completed pan_seg
        pan_seg = merge_semantic_and_instance(
            sem, instance_seg, self.label_divisor, self.thing_list,
            self.stuff_area, self.void_label
        )

        return pan_seg

    def empty_queue(self, upsampling=1):
        # any items past self.mid_idx remaining
        # in the queue are processed and returned
        final_segs = []
        for model_out in list(self.median_queue)[self.mid_idx + 1:]:
            h, w = model_out['size']
            instance_cells = self.get_instance_cells(model_out['ctr_hmp'], model_out['offsets'], upsampling)
            pan_seg = self.postprocess(model_out['sem_logits'], model_out['semantic_x'], instance_cells, upsampling)
            final_segs.append(pan_seg[..., :(h * upsampling), :(w * upsampling)])

        return final_segs

    def __call__(self, image, size, upsampling=1):
        assert math.log(upsampling, 2).is_integer(),\
        "Upsampling factor not log base 2!"

        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        h, w = size
        image = factor_pad(image, self.padding_factor)
        image = image.to(self.device, non_blocking=True)

        # infer labels
        model_out = self.infer(image)
        model_out['size'] = size

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
        instance_cells = self.get_instance_cells(median_out['ctr_hmp'], median_out['offsets'], upsampling)
        pan_seg = self.postprocess(median_out['sem_logits'], median_out['semantic_x'], instance_cells, upsampling)

        # remove padding from the pan_seg
        pan_seg = pan_seg[..., :h, :w]

        return pan_seg
