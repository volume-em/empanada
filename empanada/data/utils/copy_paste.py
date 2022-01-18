import random
import numpy as np
import albumentations as A

__all__ = [
    'copy_paste_class'
]

def copy_paste_class(dataset_class):
    def _split_transforms(self):
        split_index = None
        if self.transforms is not None:
            for ix, tf in enumerate(list(self.transforms.transforms)):
                if 'CopyPaste' in tf.get_class_fullname():
                    split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index+1:]

            # replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            # recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def load_pasted_example(self, idx):
        img_data = self.load_example(idx)
        if self.copy_paste is not None:
            paste_idx = np.random.choice(
                range(len(self)), p=self.weights.numpy() / self.weights.sum().item()
            )

            random.randint(0, len(self) - 1)
            paste_img_data = self.load_example(paste_idx)
            for k in list(paste_img_data.keys()):
                paste_img_data['paste_' + k] = paste_img_data[k]
                del paste_img_data[k]

            img_data = self.copy_paste(**img_data, **paste_img_data)
            img_data = self.post_transforms(**img_data)
            img_data['paste_index'] = paste_idx

        return img_data
    
    setattr(dataset_class, '_split_transforms', _split_transforms)
    setattr(dataset_class, 'load_pasted_example', load_pasted_example)

    return dataset_class
