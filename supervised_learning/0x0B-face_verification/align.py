#!/usr/bin/env python3
"""Face Align """
import dlib
import cv2
import numpy as np


class FaceAlign:
    """ Initialize Face Align """
    def __init__(self, shape_predictor_path):
        """ init for class
            shape_predictor_path is the path to the dlib shape predictor model
            detector - contains dlib‘s default face detector
            shape_predictor - contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """ detects a face in an image:
            - image is a numpy.ndarray of rank 3 containing an image from
            which to detect a face

            Returns: a dlib.rectangle containing the boundary box for the face
            in the image, or None on failure
            If multiple faces are detected, it returns the dlib.rectangle
            with the largest area
            If no faces are detected, it returns a dlib.rectangle that is the
            same as the image
        """
        try:
            faces = self.detector(image, 1)
            area = 0

            for face in faces:
                if face.area() > area:
                    area = face.area()
                    rect = face

            if area == 0:
                rect = (dlib.rectangle(left=0, top=0, right=image.shape[1],
                        bottom=image.shape[0]))

            return rect
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """ finds facial landmarks:
            - image is a numpy.ndarray of an image from which to find
                facial landmarks
            - detection is a dlib.rectangle containing the boundary box
                of the face in the image
            Returns: a numpy.ndarray of shape (p, 2)containing the
                landmark points, or None on failure
                p is the number of landmark points
                2 is the x and y coordinates of the point
        """
        try:
            shape = self.shape_predictor(image, detection)
            coords = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                coords[i] = [shape.part(i).x, shape.part(i).y]
            return coords
        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """ aligns an image for face verification:
            - image is a numpy.ndarray of rank 3 containing the image
                to be aligned
            - landmark_indices is a numpy.ndarray of shape (3,) containing
                the indices of the three landmark points that should be
                used for the affine transformation
            - anchor_points is a numpy.ndarray of shape (3, 2) containing
                the destination points for the affine transformation,
                scaled to the range [0, 1]
            - size is the desired size of the aligned image
            Returns: a numpy.ndarray of shape (size, size, 3) containing
                the aligned image, or None if no face is detected
        """
        rect = self.detect(image)
        coords = self.find_landmarks(image, rect)
        pts = coords[landmark_indices]
        pts = pts.astype('float32')
        new_anchors = anchor_points * size
        warp_mat = cv2.getAffineTransform(pts, new_anchors)
        warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

        return warp_dst
