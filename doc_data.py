import numpy as np
import torch
from docreader.evaluation.metrics.bbox_evaluation import calculate_iou
from docschema.semantic import Word, Paragraph, TextLine, Section, Document, Field
from typing import Union


# EOC_TOKEN = np.array([0, 0, 0, 0])
EOC_TOKEN = np.array([0, 0, 0, 0])
EOL_TOKEN = [0, 0]


class Preprocessor:
    def __init__(self, target_container, crop_h=500, crop_w=None, random_shuffle: bool = False, only_midpoints: bool = False):
        self.target_container = target_container
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.random_shuffle = random_shuffle
        self.only_midpoints = only_midpoints

    def __call__(self, doc):
        words_and_containers = self.get_words_and_containers(doc, self.target_container)
        datum = self.preprocess_bboxes_pointers(words_and_containers, EOC_TOKEN)

        if self.random_shuffle:
            datum = self.random_shuffle_sequence(datum)
        
        if self.only_midpoints:
            datum = self.process_midpoints(datum, n_decimals=2)

        return datum

    def _get_relative_bbox(self, bbox, start_row, start_col, h, w):
        return (
            max(bbox[0] - start_col, 0),
            max(bbox[1] - start_row, 0),
            min(bbox[2] - start_col, w),
            min(bbox[3] - start_row, h)
        )

    def _get_random_crop(self, doc, image, crop_w=None, crop_h=500):
        h, w = image.shape[:2]
        if crop_w is None:
            crop_w = w
        if crop_h is None:
            crop_h = h

        start_row = np.random.randint(0, high=(h-crop_h+1))
        start_col = np.random.randint(0, high=(w-crop_w+1))

        # Clear the word annotations from the Document Schema
        words = doc.filter_descendants(Word)

        for word in words:
            if word.bbox is None:
                continue

            # Remove the word if it overlaps with the region to delete
            iou = calculate_iou(
                predicted_bboxes=np.array([(start_col, start_row, start_col+crop_w, start_row+crop_h)]),
                gt_bboxes=np.array([word.bbox]),
                is_xywh=False, denominator='gt')

            if iou[0][0] > 0.2:  # Don't delete the word!
                word.bbox = self._get_relative_bbox(word.bbox, start_row, start_col, crop_h, crop_w)
                continue

            word.parent = None
            del word

        image = image[start_row: start_row+crop_h, start_col: start_col+crop_w]
        return doc, image

    def get_words_and_containers(self, doc: Document, target_container: Union[TextLine, Paragraph, Field, str]):
        """
        Inputs: document and the container to generate sequences for.
        Outputs: a dictionary containing the image, the 

        The image/doc gets cropped to a smaller size.
        """
        if isinstance(target_container, str) and target_container not in ['key', 'value']:
            raise ValueError('target_container doesnot have the correct type / value')
        image = doc.rendered_image
        if len(doc.filter_descendants(target_container)) == 0:
            raise ValueError('The doc has no containers from the target container!')
        
        crop_h = self.crop_h or image.shape[0]
        crop_w = self.crop_w or image.shape[1]

        doc, image = self._get_random_crop(doc, image, crop_w=crop_w, crop_h=crop_h)

        list_containers = []
        list_child_words = []

        for el in doc.filter_descendants(target_container):
            child_words = el.filter_descendants(Word)
            if len(child_words) == 0:
                continue
            list_child_words.append(child_words)
            list_containers.append(el)

        assert len(list_child_words) == len(list_containers)
        assert np.all(image.shape[:2] == np.array([crop_h, crop_w]))

        return {
            'containers': list_containers,
            'words': list_child_words,
            'image': image,
        }

    def _discretize(self, vals: np.ndarray, binv: int) -> np.ndarray:
        maxval = max(vals)
        bins = np.arange(0, maxval+1, binv)
        return (np.digitize(vals, bins) - 1) * binv

#     assert np.all(self._discretize([0, 1, 2, 3, 4], 3) == [0, 0, 0, 3, 3])

    def _get_sorted_bboxes_inds(self, bboxes: np.ndarray, binv=1, sort_r: bool =  True, sort_c: bool = True) -> np.ndarray:
        """
        Sort bounding boxes from top to bottom, left to right after discretizing the co-ordinates by binv 
        """
        if bboxes.shape[1] != 4:
            raise ValueError

        # FIXME: The scale might help here
        lefts = self._discretize(bboxes[:, 0], binv)
        tops = self._discretize(bboxes[:, 1], binv)

        # The key allows us to easily implement the desired sorting
        if sort_r and sort_c:
            keys = tops * 1e10 + lefts
        elif sort_r:
            keys = tops
        elif sort_c:
            keys = lefts
        else:
            raise ValueError('At least one of sort_c and sort_r need to be enabled')
        sort_inds = np.argsort(keys)
        return sort_inds

    def preprocess_bboxes_pointers(self, data, EOC_TOKEN):
        # Inputs
        containers = data['containers']
        child_words = data['words']

        container_bboxes = np.array([el.bbox for el in containers if el.bbox is not None])
        if len(container_bboxes) == 0:
    #         raise ValueError('no elements in the cropped area')
            return {'bboxes': np.array([[]]), 'pointers': np.array([]), 'image': np.array([[]]), 'is_empty': True}

        sort_inds_container = self._get_sorted_bboxes_inds(container_bboxes, sort_r=True, sort_c=True)

        all_word_bboxes = [EOC_TOKEN]  # ADD ALL THE FIXED TOKENS
        pointer_seq = []

        word_idx_start = len(all_word_bboxes)
        for idx in sort_inds_container:  # iterate over the containers in the order top-bottom left-right
            words = child_words[idx]
            word_bboxes = np.vstack([word.bbox for word in words if word.bbox is not None])
            if len(word_bboxes) == 0:
                continue

            word_bboxes = word_bboxes[self._get_sorted_bboxes_inds(word_bboxes, sort_r=False, sort_c=True)]

            all_word_bboxes.append(word_bboxes)
            n_words = len(word_bboxes)

            # Get the indices for the words and store them as the "pointers"
            pointer_seq.extend(np.arange(n_words) + word_idx_start)
            pointer_seq.append(0)  # NOTE: 0 is the index of the fixed EOC token
            word_idx_start += n_words

        all_word_bboxes = np.vstack(all_word_bboxes).astype(np.float32)
        pointer_seq = np.array(pointer_seq).astype(np.long)

        scale = all_word_bboxes.max()
        all_word_bboxes /= scale

        return {
            'bboxes': all_word_bboxes,
            'pointers': pointer_seq,
            'image': data['image'],
            'is_empty': False,
            'scale': scale
        }

    @staticmethod
    def random_shuffle_sequence(datum):
        """
        Randomly shuffles the input sequence and the other arrays correspondingly.
        """
        is_empty = datum['is_empty']
        if is_empty:
            return datum
        
        bboxes, pointers = datum['bboxes'], datum['pointers']
        n = len(bboxes)
        
        # Generate new 
        inds_new_order = np.arange(n)
        np.random.shuffle(inds_new_order)
        bboxes = bboxes[inds_new_order].squeeze()
        
        # map the pointers to the new indices
        inds_reverse = np.zeros(n)
        inds_reverse[inds_new_order] = np.arange(n)
        new_pointers = inds_reverse[pointers].astype(np.long)
        assert np.all(new_pointers.shape == pointers.shape)
        
        return {
            'bboxes': bboxes,
            'pointers': new_pointers,
            'image': datum['image'],
            'is_empty': is_empty,
            'scale': datum['scale'],
        }
    
    @staticmethod
    def process_midpoints(datum, n_decimals: int = 4):
        is_empty = datum['is_empty']
        if is_empty:
            return datum
        
        bboxes, pointers = datum['bboxes'], datum['pointers']
        n = len(bboxes)
        
        midpoints = np.hstack([
            np.round((bboxes[:, 0] + bboxes[:, 2]) / 2, decimals=n_decimals).reshape(-1, 1),
            np.round((bboxes[:, 1] + bboxes[:, 3]) / 2, decimals=n_decimals).reshape(-1, 1),
        ])
        
        assert np.all(midpoints.shape == np.array([len(bboxes), 2]))
        
        # Generate new 
        datum.update({
            'bboxes': midpoints,
        })

        return datum


# Data loader specific
def get_padded_tensor_and_lens(list_seqs, pad_constant_value=0, n_dim=2):
    lens = np.array([len(x) for x in list_seqs])
    # Each sequence is an array of shape seq_len*n_dim
    for ix in range(len(list_seqs)):
        seq = list_seqs[ix]
        if len(seq) == 0 or len(seq[0]) == 0:
            list_seqs[ix] = np.zeros(n_dim, dtype=np.float32)[np.newaxis, :]
        seq = list_seqs[ix]
        assert len(seq.shape) == 2, 'Actual shape is: {}'.format(seq.shape)
        assert seq.shape[1] == n_dim

    max_len = max(lens)
    data = np.array([
        np.pad(seq, pad_width=[(0, max_len - len(seq)), (0, 0)], mode='constant', constant_values=pad_constant_value)
        for seq in list_seqs
    ])

    return data, lens


def collate_fn(batch):
    inds_to_take = np.array([not sample['is_empty'] for sample in batch], dtype=np.bool)
    batch = np.array(batch)[inds_to_take]
    assert len(batch) == sum(inds_to_take)
    
    if len(batch) == 0:
        return None

    sequences, lens1 = get_padded_tensor_and_lens([sample['bboxes'] for sample in batch], pad_constant_value=0, n_dim=2)
    pointers, lens2 = get_padded_tensor_and_lens([sample['pointers'][..., np.newaxis] for sample in batch], pad_constant_value=-100, n_dim=1)
    
    # Sort such that the longest sequence is first. Sort the pointers to match the sequences.
    inds_sorted_desc = np.argsort(lens1)[::-1]
    sequences, lens1 = sequences[inds_sorted_desc, ...], lens1[inds_sorted_desc]
    pointers, lens2 = pointers[inds_sorted_desc, ...], lens2[inds_sorted_desc]
    
    sequences = torch.from_numpy(sequences)
    pointers = torch.from_numpy(pointers)
    
    # Get the images
    images = np.array([sample['image'][np.newaxis, ...] for sample in batch])
    images = images[inds_sorted_desc]
    images = torch.from_numpy(images)

    scales = np.array([sample['scale'] for sample in batch])
    scales = torch.from_numpy(scales[inds_sorted_desc])
    
    return {
        'sequence': sequences,
        'sequence_lens': lens1,
        'pointers': pointers,
        'pointer_lens': lens2,
        'images': images,
        'scales': scales
    }