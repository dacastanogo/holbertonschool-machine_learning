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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
                        filtered bounding boxes:
        box_classes: a numpy.ndarray of shape (?,) containing the class number
                        that each box in filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
                        each box in filtered_boxes, respectively

        """
        # box_scores
        box_scores_prev = []
        for b_c, b_c_p in zip(box_confidences, box_class_probs):
            box_scores_prev.append(b_c * b_c_p)

        box_scores_max = [box.max(axis=3) for box in box_scores_prev]
        box_scores_max2 = [box.reshape(-1) for box in box_scores_max]
        box_scores_conca = np.concatenate(box_scores_max2)

        indx_to_delete = np.where(box_scores_conca < self.class_t)

        box_scores = np.delete(box_scores_conca, indx_to_delete)

        # box_classes
        box_classes_prev = [box.argmax(axis=3) for box in box_scores_prev]
        box_classes_prev2 = [box.reshape(-1) for box in box_classes_prev]
        box_classes_conca = np.concatenate(box_classes_prev2)
        box_classes = np.delete(box_classes_conca, indx_to_delete)

        # filtered_boxes
        boxes2 = [box.reshape(-1, 4) for box in boxes]
        boxes_conca = np.concatenate(boxes2, axis=0)
        filtered_boxes = np.delete(boxes_conca, indx_to_delete, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def gaps_inter(self, range_1, range_2):
        """ calculates axis intersections"""
        x1, x2 = range_1
        x3, x4 = range_2

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def iou(self, box1, box2):
        """ calculates intersection over union """
        inter_x = self.gaps_inter([box1[0], box1[2]], [box2[0], box2[2]])
        inter_y = self.gaps_inter([box1[1], box1[3]], [box2[1], box2[3]])

        inter = inter_x * inter_y

        w1 = box1[2]-box1[0]
        w2 = box2[2]-box2[0]
        h1 = box1[3]-box1[1]
        h2 = box2[3]-box2[1]

        union = w1*h1 + w2*h2 - inter

        return inter / union

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Non - max
        Returns a tuple of (box_predictions, predicted_box_classes,
                predicted_box_scores):

                box_predictions: a numpy.ndarray of shape (?, 4) containing
                all of the predicted bounding boxes ordered by class and
                box score

                predicted_box_classes: a numpy.ndarray of shape (?,)
                containing the class number for box_predictions ordered by
                class and box score, respectively

                predicted_box_scores: a numpy.ndarray of shape (?)
                containing the box scores for box_predictions ordered by
                class and box score, respectively
        """
        ind = np.lexsort((-box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in ind])
        predicted_box_classes = np.array([box_classes[i] for i in ind])
        predicted_box_scores = np.array([box_scores[i] for i in ind])

        _, counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        ct_prev = 0
        for ct in counts:
            j = 1
            while i < ct + ct_prev:
                while j < ct + ct_prev - i:
                    tmp = self.iou(box_predictions[i], box_predictions[i+j])
                    if tmp > self.nms_t:
                        box_predictions = np.delete(box_predictions, i+j,
                                                    axis=0)
                        predicted_box_scores = np.delete(predicted_box_scores,
                                                         i+j, axis=0)
                        predicted_box_classes = (np.delete
                                                 (predicted_box_classes,
                                                  i+j, axis=0))
                        ct -= 1
                    else:
                        j += 1
                i += 1
                j = 1
            ct_prev += ct
        return box_predictions, predicted_box_classes, predicted_box_scores
