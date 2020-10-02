#!/usr/bin/env python3
""" Process Outputs """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ YOYO V3"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ init """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """ calculates sigmoid function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """ - outputs is a list of numpy.ndarrays containing the predictions
              from the Darknet model for a single image
            - image_size is a numpy.ndarray containing the image’s original
              size [image_height, image_width]
            - Returns a tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = [out[..., :4] for out in outputs]
        for i, box in enumerate(boxes):
            grid_h, grid_w, n_anchors, _ = box.shape

            m_h = np.arange(grid_h).reshape(1, grid_h)
            m_h = np.repeat(m_h, grid_w, axis=0).T
            m_h = np.repeat(m_h[:, :, np.newaxis], n_anchors, axis=2)
            m_w = np.arange(grid_w).reshape(1, grid_w)
            m_w = np.repeat(m_w, grid_h, axis=0)
            m_w = np.repeat(m_w[:, :, np.newaxis], n_anchors, axis=2)

            box[..., :2] = self.sigmoid(box[..., :2])
            box[..., 0] += m_w
            box[..., 1] += m_h

            box[..., 2:] = np.exp(box[..., 2:])
            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]
            box[..., 2] *= anchor_w
            box[..., 3] *= anchor_h

            box[..., 0] /= grid_w
            box[..., 1] /= grid_h

            box[..., 2] /= self.model.input.shape[1].value
            box[..., 3] /= self.model.input.shape[2].value

            box[..., 0] -= box[..., 2] / 2
            box[..., 1] -= box[..., 3] / 2

            box[..., 2] += box[..., 0]
            box[..., 3] += box[..., 1]

            box[..., 0] *= image_size[1]
            box[..., 2] *= image_size[1]
            box[..., 1] *= image_size[0]
            box[..., 3] *= image_size[0]

        box_conf = [self.sigmoid(out[..., 4, np.newaxis]) for out in outputs]
        box_class_probs = [self.sigmoid(out[..., 5:]) for out in outputs]

        return boxes, box_conf, box_class_probs
