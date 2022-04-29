import math
import torch
import torch.nn.functional as F

from empanada.inference.postprocess import (
    factor_pad, find_instance_center, group_pixels,
    get_instance_segmentation,
    merge_semantic_and_instance,
    get_panoptic_segmentation
)
from collections import deque


__all__ = [
    'PanopticDeepLabEngine',
    'PanopticDeepLabEngine3d',
    'PanopticDeepLabRenderEngine',
    'PanopticDeepLabRenderEngine3d',
    'BCEngine',
    'BCEngine3d',
]

class _Engine:
    def __init__(self, model):
        self.model = model.eval()

    def infer(self, image):
        raise NotImplementedError

    def to_model_device(self, tensor):
        # move tensor to the model device
        device = next(self.model.parameters()).device
        return tensor.to(device, non_blocking=True)

    def __call__(self, image):
        raise NotImplementedError

class _RenderEngine(_Engine):
    def __init__(self, model, render_models, **kwargs):
        super().__init__(model=model, **kwargs)
        self.render_models = {name: rm.eval() for name, rm in render_models.items()}

class _MedianQueue:
    def __init__(self, median_kernel_size, **kwargs):
        # super to allow multiple inheritance
        super().__init__(**kwargs)
        assert median_kernel_size % 2 == 1, "Kernel size must be odd integer!"
        self.ks = median_kernel_size
        self.mid_idx = (median_kernel_size - 1) // 2
        self.median_queue = deque(maxlen=median_kernel_size)

    def reset(self):
        self.median_queue = deque(maxlen=self.ks)

    @torch.no_grad()
    def get_median(self, key):
        median = torch.median(
            torch.cat([output[key] for output in self.median_queue], dim=0),
            dim=0, keepdim=True
        ).values

        return median

    def get_next(self, keys):
        nq = len(self.median_queue)
        if nq <= self.mid_idx:
            # take last item in the queue
            output = self.median_queue[-1]
        elif nq > self.mid_idx and nq < self.ks:
            # return nothing while the queue builds
            return None
        elif nq == self.ks:
            # use the middle item in the queue
            # with the median segmentation probs
            output = self.median_queue[self.mid_idx]
            # replace output with median output
            for key in keys:
                output[key] = self.get_median(key)

        return output

    def enqueue(self, item):
        self.median_queue.append(item)

    def end(self):
        return list(self.median_queue)[self.mid_idx + 1:]

class PanopticDeepLabEngine(_Engine):
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
        super().__init__(model=model)
        self.thing_list = thing_list
        self.label_divisor = label_divisor
        self.stuff_area = stuff_area
        self.void_label = void_label
        self.nms_threshold = nms_threshold
        self.nms_kernel = nms_kernel
        self.confidence_thr = confidence_thr

    @torch.no_grad()
    def _harden_seg(self, sem):
        if sem.size(1) > 1: # multiclass segmentation
            sem = torch.argmax(sem, dim=1, keepdim=True)
        else:
            sem = (sem >= self.confidence_thr).long() # need integers

        return sem

    @torch.no_grad()
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

    @torch.no_grad()
    def postprocess(self, sem, ctr_hmp, offsets):
        pan_seg, _ = get_panoptic_segmentation(
            sem, ctr_hmp, offsets, self.thing_list,
            self.label_divisor, self.stuff_area,
            self.void_label, self.nms_threshold, self.nms_kernel
        )
        return pan_seg

    def __call__(self, image):
        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        image = self.to_model_device(image)

        # infer labels and postprocess
        model_out = self.infer(image)

        # harden the segmentation to (N, 1, H, W)
        model_out['sem'] = self._harden_seg(model_out['sem'])

        pan_seg = self.postprocess(
            model_out['sem'], model_out['ctr_hmp'], model_out['offsets']
        )

        return pan_seg

class PanopticDeepLabEngine3d(_MedianQueue, PanopticDeepLabEngine):
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
        super().__init__(
            model=model, thing_list=thing_list, label_divisor=label_divisor,
            stuff_area=stuff_area, void_label=void_label,
            nms_threshold=nms_threshold, nms_kernel=nms_kernel,
            confidence_thr=confidence_thr, median_kernel_size=median_kernel_size,
            **kwargs
        )

    def end(self):
        f"""
        Define what happens to results left in the queue at the
        end of inference.
        """
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
        image = self.to_model_device(image)

        # infer labels and postprocess
        model_out = self.infer(image)

        self.enqueue(model_out)
        median_out = self.get_next(keys=['sem'])
        if median_out is None:
            # nothing to return, we're building the queue
            return None

        pan_seg = self.postprocess(
            self._harden_seg(median_out['sem']), median_out['ctr_hmp'], median_out['offsets']
        )

        return pan_seg

class PanopticDeepLabRenderEngine(_RenderEngine, PanopticDeepLabEngine):
    def __init__(
        self,
        model,
        render_models,
        thing_list,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        padding_factor=16,
        coarse_boundaries=True,
        **kwargs
    ):
        super().__init__(
            model=model, render_models=render_models, thing_list=thing_list,
            label_divisor=label_divisor, stuff_area=stuff_area, void_label=void_label,
            nms_threshold=nms_threshold, nms_kernel=nms_kernel,
            confidence_thr=confidence_thr
        )

        self.padding_factor = padding_factor
        self.coarse_boundaries = coarse_boundaries

    @torch.no_grad()
    def infer(self, image):
        return self.model(image)

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
                sem_logits = self.render_models['sem_logits'](sem_logits, coarse_sem_seg_logits, features)

        # apply classification activation
        if sem_logits.size(1) > 1:
            sem = F.softmax(sem_logits, dim=1)
        else:
            sem = torch.sigmoid(sem_logits)

        return sem, sem_logits

    @torch.no_grad()
    def get_panoptic_seg(self, sem, instance_cells):
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

        return pan_seg

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
        return self.get_panoptic_seg(sem, instance_cells)

    def __call__(self, image, size, upsampling=1):
        assert math.log(upsampling, 2).is_integer(),\
        "Upsampling factor not log base 2!"

        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        h, w = size
        image = factor_pad(image, self.padding_factor)
        image = self.to_model_device(image)

        # infer labels
        model_out = self.infer(image)

        # calculate the instance cells
        instance_cells = self.get_instance_cells(
            model_out['ctr_hmp'], model_out['offsets'], upsampling
        )
        pan_seg = self.postprocess(
            model_out['sem_logits'], model_out['semantic_x'],
            instance_cells, upsampling
        )

        # remove padding from the pan_seg
        pan_seg = pan_seg[..., :h, :w]

        return pan_seg

class PanopticDeepLabRenderEngine3d(_MedianQueue, PanopticDeepLabRenderEngine):
    def __init__(
        self,
        model,
        render_models,
        thing_list,
        label_divisor=1000,
        stuff_area=64,
        void_label=0,
        nms_threshold=0.1,
        nms_kernel=7,
        confidence_thr=0.5,
        median_kernel_size=3,
        padding_factor=16,
        coarse_boundaries=True,
        **kwargs
    ):
        super().__init__(
            model=model, render_models=render_models, thing_list=thing_list,
            label_divisor=label_divisor, stuff_area=stuff_area, void_label=void_label,
            nms_threshold=nms_threshold, nms_kernel=nms_kernel,
            confidence_thr=confidence_thr, median_kernel_size=median_kernel_size,
            padding_factor=padding_factor, coarse_boundaries=coarse_boundaries
        )

    def end(self, upsampling=1):
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
        image = self.to_model_device(image)

        # infer labels
        model_out = self.infer(image)
        model_out['size'] = size

        # append results to median queue
        self.enqueue(model_out)
        median_out = self.get_next(keys=['sem_logits'])
        if median_out is None:
            # nothing to return, we're building the queue
            return None

        # calculate the instance cells
        instance_cells = self.get_instance_cells(median_out['ctr_hmp'], median_out['offsets'], upsampling)
        pan_seg = self.postprocess(median_out['sem_logits'], median_out['semantic_x'], instance_cells, upsampling)

        # remove padding from the pan_seg
        pan_seg = pan_seg[..., :h, :w]

        return pan_seg

class BCEngine(_Engine):
    def __init__(self, model, **kwargs):
        super().__init__(model=model)

    @torch.no_grad()
    def infer(self, image):
        model_out = self.model(image)
        sem_logits = model_out['sem_logits']
        cnt_logits = model_out['cnt_logits']

        # only works for binary
        assert sem_logits.size(1) == 1
        sem = torch.sigmoid(sem_logits)
        cnt = torch.sigmoid(cnt_logits)

        return {'bc': torch.cat([sem, cnt], dim=1)} # (N, 2, H, W)

    def __call__(self, image):
        # check that image is 4d (N, C, H, W)
        assert image.ndim == 4 and image.size(0) == 1
        return self.infer(self.to_model_device(image))['bc'] # (1, 2, H, W)

class BCEngine3d(BCEngine, _MedianQueue):
    def __init__(self, model, median_kernel_size=3, **kwargs):
        super().__init__(model=model, median_kernel_size=median_kernel_size)

    def end(self):
        # list of remaining segs (1, 2, H, W)
        return list(self.median_queue)[self.mid_idx + 1:]

    def __call__(self, image):
        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        image = self.to_model_device(image)

        # infer labels and postprocess
        model_out = self.infer(image)

        self.enqueue(model_out)
        median_out = self.get_next(keys=['bc'])
        if median_out is None:
            # nothing to return, we're building the queue
            return None

        return median_out['bc'] # (1, 2, H, W)
