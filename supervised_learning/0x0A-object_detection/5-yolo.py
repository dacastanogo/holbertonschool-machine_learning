#!/usr/bin/env python3
""" Process Outputs """
import tensorflow.keras as K
import numpy as np
import glob
import cv2


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

    def overlap(self, interval_a, interval_b):
        """ claculates axis diferences """
        x1, x2 = interval_a
        x3, x4 = interval_b

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
        # print(box1)
        # print(box2)
        # print("---"*10)
        intersect_w = self.overlap([box1[0], box1[2]], [box2[0], box2[2]])
        intersect_h = self.overlap([box1[1], box1[3]], [box2[1], box2[3]])

        intersect = intersect_w * intersect_h

        w1, h1 = box1[2]-box1[0], box1[3]-box1[1]
        w2, h2 = box2[2]-box2[0], box2[3]-box2[1]

        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union

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
        ind = np.lexsort((box_scores, box_classes))

        box_predictions = np.array([filtered_boxes[i] for i in ind])
        predicted_box_classes = np.array([box_classes[i] for i in ind])
        predicted_box_scores = np.array([box_scores[i] for i in ind])

        i = 0
        c = 0
        idx_to_delete = []
        classes, counts = np.unique(predicted_box_classes, return_counts=True)

        i = 0
        ct_prev = 0
        for cl, ct in zip(classes, counts):
            j = 1
            while i < ct + ct_prev:
                while j < ct + ct_prev - i:
                    tmp = self.iou(box_predictions[i], box_predictions[i+j])
                    # print(tmp if tmp  > self.nms_t else "")
                    if tmp > self.nms_t:
                        # print(tmp)
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

    @staticmethod
    def load_images(folder_path):
        """ folder_path: a string representing the path to the folder holding
                        all the images to load
            Returns a tuple of (images, image_paths):
                images: a list of images as numpy.ndarrays
                image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(folder_path + "/*")
        images = [cv2.imread(img) for img in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """ images: a list of images as numpy.ndarrays

            Resizes the images with inter-cubic interpolation
            Rescales all images to have pixel values in the range [0, 1]

            Returns a tuple of (pimages, image_shapes):
                pimages: a numpy.ndarray of shape (ni, input_h, input_w, 3)
                    containing all of the preprocessed images
                ni: the number of images that were preprocessed
                input_h: the input height for the Darknet model
                        Note: this can vary by model
                input_w: the input width for the Darknet model
                        Note: this can vary by model
                3: number of color channels
                image_shapes: a numpy.ndarray of shape (ni, 2) containing
                    the original height and width of the images
                    2 => (image_height, image_width)
        """

        model_height = self.model.input.shape[2].value
        model_width = self.model.input.shape[1].value
        images_resized = [cv2.resize(img, (model_width, model_height),
                          interpolation=cv2.INTER_CUBIC)
                          for img in images]
        images_rescaled = [img/255 for img in images_resized]

        pimages = np.stack(images_rescaled, axis=0)

        image_shapes_list = [img.shape[:2] for img in images]
        image_shapes = np.stack(image_shapes_list, axis=0)

        return (pimages, image_shapes)
