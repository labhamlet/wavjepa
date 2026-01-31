"""
Common utils for scoring.
"""

from collections import ChainMap
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sed_eval
import torch

# Can we get away with not using DCase for every event-based evaluation??
from dcase_util.containers import MetaDataContainer
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from scipy.optimize import linear_sum_assignment


eps = np.finfo(np.float64).eps

def convert_output_format_cartesian_to_polar(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]
                azimuth = np.arctan2(y, x) * 180 / np.pi
                elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                r = np.sqrt(x**2 + y**2 + z**2)
                out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
    return out_dict

def convert_output_format_new_to_old(in_dict):
    out_dict = {}
    for frame_cnt in in_dict.keys():
        if frame_cnt not in out_dict:
            out_dict[frame_cnt] = []
            for tmp_val in in_dict[frame_cnt]:
                az, el = tmp_val[2], tmp_val[3]
                el = el
                out_dict[frame_cnt].append([tmp_val[0], az, el])
    return out_dict

def compute_seld_metric(sed_error, doa_error):
    """
    Compute SELD metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: seld metric result
    """
    seld_metric = np.mean([
        sed_error[0],
        1 - sed_error[1],
        doa_error[0]/180,
        1 - doa_error[1]]
        )
    return seld_metric

class SELDMetrics(object):
    def __init__(self, doa_threshold=20, nb_classes=11, average='macro'):
        '''
            This class implements both the class-sensitive localization and location-sensitive detection metrics.
            Additionally, based on the user input, the corresponding averaging is performed within the segment.

        :param nb_classes: Number of sound classes. In the paper, nb_classes = 11
        :param doa_thresh: DOA threshold for location sensitive detection.
        '''
        self._nb_classes = nb_classes

        # Variables for Location-senstive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._spatial_T = doa_threshold

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_DE = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        self._average = average

    def early_stopping_metric(self, _er, _f, _le, _lr):
        """
        Compute early stopping metric from sed and doa errors.

        :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
        :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
        :return: early stopping metric result
        """
        seld_metric = np.mean([
            _er,
            1 - _f,
            _le / 180,
            1 - _lr
        ], 0)
        return seld_metric

    def compute_seld_scores(self):
        '''
        Collect the final SELD scores

        :return: returns both location-sensitive detection scores and class-sensitive localization scores
        '''
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            LE = self._total_DE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else 180
            LR = self._DE_TP.sum() / (eps + self._DE_TP.sum() + self._DE_FN.sum())

            SELD_scr = self.early_stopping_metric(ER, F, LE, LR)

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))

            # Class-sensitive localization performance
            LE = self._total_DE / (self._DE_TP + eps)
            LE[self._DE_TP==0] = 180.0
            LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)

            SELD_scr = self.early_stopping_metric(np.repeat(ER, self._nb_classes), F, LE, LR)
            classwise_results = np.array([np.repeat(ER, self._nb_classes), F, LE, LR, SELD_scr])
            F, LE, LR, SELD_scr = F.mean(), LE.mean(), LR.mean(), SELD_scr.mean()
        return ER, F, LE, LR, SELD_scr, classwise_results

    def update_seld_scores(self, pred, gt):
        '''
        Implements the spatial error averaging according to equation 5 in the paper [1] (see papers in the title of the code).
        Adds the multitrack extensions proposed in paper [2]

        The input pred/gt can either both be Cartesian or Degrees

        :param pred: dictionary containing class-wise prediction results for each N-seconds segment block
        :param gt: dictionary containing class-wise groundtruth for each N-seconds segment block
        '''
        for block_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of referece tracks for each class in the segment
                nb_gt_doas = max([len(val) for val in gt[block_cnt][class_cnt][0][1]]) if class_cnt in gt[block_cnt] else None
                nb_pred_doas = max([len(val) for val in pred[block_cnt][class_cnt][0][1]]) if class_cnt in pred[block_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # True positives or False positive case

                    # NOTE: For multiple tracks per class, associate the predicted DOAs to corresponding reference
                    # DOA-tracks using hungarian algorithm and then compute the average spatial distance between
                    # the associated reference-predicted tracks.

                    # Reference and predicted track matching
                    matched_track_dist = {}
                    matched_track_cnt = {}
                    gt_ind_list = gt[block_cnt][class_cnt][0][0]
                    pred_ind_list = pred[block_cnt][class_cnt][0][0]
                    for gt_ind, gt_val in enumerate(gt_ind_list):
                        if gt_val in pred_ind_list:
                            gt_arr = np.array(gt[block_cnt][class_cnt][0][1][gt_ind])
                            gt_ids = np.arange(len(gt_arr[:, -1])) #TODO if the reference has track IDS use here - gt_arr[:, -1]
                            gt_doas = gt_arr[:, 1:]

                            pred_ind = pred_ind_list.index(gt_val)
                            pred_arr = np.array(pred[block_cnt][class_cnt][0][1][pred_ind])
                            pred_doas = pred_arr[:, 1:]

                            if gt_doas.shape[-1] == 2: # convert DOAs to radians, if the input is in degrees
                                gt_doas = gt_doas * np.pi / 180.
                                pred_doas = pred_doas * np.pi / 180.

                            dist_list, row_inds, col_inds = least_distance_between_gt_pred(gt_doas, pred_doas)

                            # Collect the frame-wise distance between matched ref-pred DOA pairs
                            for dist_cnt, dist_val in enumerate(dist_list):
                                matched_gt_track = gt_ids[row_inds[dist_cnt]]
                                if matched_gt_track not in matched_track_dist:
                                    matched_track_dist[matched_gt_track], matched_track_cnt[matched_gt_track] = [], []
                                matched_track_dist[matched_gt_track].append(dist_val)
                                matched_track_cnt[matched_gt_track].append(pred_ind)

                    # Update evaluation metrics based on the distance between ref-pred tracks
                    if len(matched_track_dist) == 0:
                        # if no tracks are found. This occurs when the predicted DOAs are not aligned frame-wise to the reference DOAs
                        loc_FN += nb_pred_doas
                        self._FN[class_cnt] += nb_pred_doas
                        self._DE_FN[class_cnt] += nb_pred_doas
                    else:
                        # for the associated ref-pred tracks compute the metrics
                        for track_id in matched_track_dist:
                            total_spatial_dist = sum(matched_track_dist[track_id])
                            total_framewise_matching_doa = len(matched_track_cnt[track_id])
                            avg_spatial_dist = total_spatial_dist / total_framewise_matching_doa

                            # Class-sensitive localization performance
                            self._total_DE[class_cnt] += avg_spatial_dist
                            self._DE_TP[class_cnt] += 1

                            # Location-sensitive detection performance
                            if avg_spatial_dist <= self._spatial_T:
                                self._TP[class_cnt] += 1
                            else:
                                loc_FP += 1
                                self._FP_spatial[class_cnt] += 1
                        # in the multi-instance of same class scenario, if the number of predicted tracks are greater
                        # than reference tracks count as FP, if it less than reference count as FN
                        if nb_pred_doas > nb_gt_doas:
                            # False positive
                            loc_FP += (nb_pred_doas-nb_gt_doas)
                            self._FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                            self._DE_FP[class_cnt] += (nb_pred_doas-nb_gt_doas)
                        elif nb_pred_doas < nb_gt_doas:
                            # False negative
                            loc_FN += (nb_gt_doas-nb_pred_doas)
                            self._FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                            self._DE_FN[class_cnt] += (nb_gt_doas-nb_pred_doas)
                elif class_cnt in gt[block_cnt] and class_cnt not in pred[block_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[block_cnt] and class_cnt in pred[block_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas

            self._S += np.minimum(loc_FP, loc_FN)
            self._D += np.maximum(0, loc_FN - loc_FP)
            self._I += np.maximum(0, loc_FP - loc_FN)
        return

class OldSELDMetrics(object):
    def __init__(self, nb_frames_1s=None, data_gen=None):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0
        self._block_size = nb_frames_1s

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0
        self._data_gen = data_gen


    def f1_overall_framewise(self, O, T):
        TP = ((2 * T - O) == 1).sum()
        Nref, Nsys = T.sum(), O.sum()
        self._TP += TP
        self._Nref += Nref
        self._Nsys += Nsys

    def er_overall_framewise(self, O, T):
        FP = np.logical_and(T == 0, O == 1).sum(1)
        FN = np.logical_and(T == 1, O == 0).sum(1)
        S = np.minimum(FP, FN).sum()
        D = np.maximum(0, FN - FP).sum()
        I = np.maximum(0, FP - FN).sum()
        self._S += S
        self._D += D
        self._I += I

    def f1_overall_1sec(self, O, T):        
        new_size = int(np.ceil(float(O.shape[0]) / self._block_size))
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.f1_overall_framewise(O_block, T_block)

    def er_overall_1sec(self, O, T):        
        new_size = int(np.ceil(float(O.shape[0]) / self._block_size))
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.er_overall_framewise(O_block, T_block)

    def update_sed_scores(self, pred, gt):
        """
        Computes SED metrics for one second segments

        :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param nb_frames_1s: integer, number of frames in one second
        :return:
        """
        self.f1_overall_1sec(pred, gt)
        self.er_overall_1sec(pred, gt)

    def compute_sed_scores(self):
        ER = (self._S + self._D + self._I) / (self._Nref + 0.0)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        return ER, F

    def update_doa_scores(self, pred_doa_thresholded, gt_doa):
        '''
        Compute DOA metrics when DOA is estimated using classification approach

        :param pred_doa_thresholded: predicted results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                                    with value 1 when sound event active, else 0
        :param gt_doa: reference results of dimension [nb_frames, nb_classes, nb_azi*nb_ele],
                        with value 1 when sound event active, else 0
        :param data_gen_test: feature or data generator class

        :return: DOA metrics

        '''
        self._doa_loss_pred_cnt += np.sum(pred_doa_thresholded)
        self._nb_frames += pred_doa_thresholded.shape[0]

        for frame in range(pred_doa_thresholded.shape[0]):
            nb_gt_peaks = int(np.sum(gt_doa[frame, :]))
            nb_pred_peaks = int(np.sum(pred_doa_thresholded[frame, :]))

            # good_frame_cnt includes frames where the nb active sources were zero in both groundtruth and prediction
            if nb_gt_peaks == nb_pred_peaks:
                self._nb_good_pks += 1
            elif nb_gt_peaks > nb_pred_peaks:
                self._less_est_frame_cnt += 1
                self._less_est_cnt += (nb_gt_peaks - nb_pred_peaks)
            elif nb_pred_peaks > nb_gt_peaks:
                self._more_est_frame_cnt += 1
                self._more_est_cnt += (nb_pred_peaks - nb_gt_peaks)

            # when nb_ref_doa > nb_estimated_doa, ignores the extra ref doas and scores only the nearest matching doas
            # similarly, when nb_estimated_doa > nb_ref_doa, ignores the extra estimated doa and scores the remaining matching doas
            if nb_gt_peaks and nb_pred_peaks:
                pred_ind = np.where(pred_doa_thresholded[frame] == 1)[1]
                pred_list_rad = np.array(self._data_gen.get_matrix_index(pred_ind)) * np.pi / 180

                gt_ind = np.where(gt_doa[frame] == 1)[1]
                gt_list_rad = np.array(self._data_gen.get_matrix_index(gt_ind)) * np.pi / 180

                frame_dist = OldSELDMetrics.distance_between_gt_pred(gt_list_rad.T, pred_list_rad.T)
                self._doa_loss_pred += frame_dist
    
    @classmethod
    def distance_between_gt_pred(cls, gt_list_rad, pred_list_rad):
        """
        Shortest distance between two sets of spherical coordinates. Given a set of groundtruth spherical coordinates,
        and its respective predicted coordinates, we calculate the spherical distance between each of the spherical
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.

        :param gt_list_rad: list of ground-truth spherical coordinates
        :param pred_list_rad: list of predicted spherical coordinates
        :return: cost -  distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
        """
        def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
            """
            Angular distance between two spherical coordinates
            MORE: https://en.wikipedia.org/wiki/Great-circle_distance

            :return: angular distance in degrees
            """
            dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
            # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
            dist = np.clip(dist, -1, 1)
            dist = np.arccos(dist) * 180 / np.pi
            return dist

        gt_len, pred_len = gt_list_rad.shape[0], pred_list_rad.shape[0]
        ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
        cost_mat = np.zeros((gt_len, pred_len))

        if gt_len and pred_len:
            az1, ele1, az2, ele2 = gt_list_rad[ind_pairs[:, 0], 0], gt_list_rad[ind_pairs[:, 0], 1], \
                                pred_list_rad[ind_pairs[:, 1], 0], pred_list_rad[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        cost = cost_mat[row_ind, col_ind].sum()
        return cost

    def compute_doa_scores(self):
        doa_error = self._doa_loss_pred / self._doa_loss_pred_cnt
        frame_recall = self._nb_good_pks / float(self._nb_frames)
        return doa_error, frame_recall

    def reset(self):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0

def distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2):
    """
    Angular distance between two spherical coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance

    :return: angular distance in degrees
    """
    dist = np.sin(ele1) * np.sin(ele2) + np.cos(ele1) * np.cos(ele2) * np.cos(np.abs(az1 - az2))
    # Making sure the dist values are in -1 to 1 range, else np.arccos kills the job
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates
    MORE: https://en.wikipedia.org/wiki/Great-circle_distance
    Check 'From chord length' section

    :return: angular distance in degrees
    """
    # Normalize the Cartesian vectors
    N1 = np.sqrt(x1**2 + y1**2 + z1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + z2**2 + 1e-10)
    x1, y1, z1, x2, y2, z2 = x1/N1, y1/N1, z1/N1, x2/N2, y2/N2, z2/N2

    #Compute the distance
    dist = x1*x2 + y1*y2 + z1*z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
        Shortest distance between two sets of DOA coordinates. Given a set of groundtruth coordinates,
        and its respective predicted coordinates, we calculate the distance between each of the
        coordinate pairs resulting in a matrix of distances, where one axis represents the number of groundtruth
        coordinates and the other the predicted coordinates. The number of estimated peaks need not be the same as in
        groundtruth, thus the distance matrix is not always a square matrix. We use the hungarian algorithm to find the
        least cost in this distance matrix.
        :param gt_list_xyz: list of ground-truth Cartesian or Polar coordinates in Radians
        :param pred_list_xyz: list of predicted Carteisan or Polar coordinates in Radians
        :return: cost - distance
        :return: less - number of DOA's missed
        :return: extra - number of DOA's over-estimated
    """

    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        if len(gt_list[0]) == 3: #Cartesian
            x1, y1, z1, x2, y2, z2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], gt_list[ind_pairs[:, 0], 2], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1], pred_list[ind_pairs[:, 1], 2]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
        else:
            az1, ele1, az2, ele2 = gt_list[ind_pairs[:, 0], 0], gt_list[ind_pairs[:, 0], 1], pred_list[ind_pairs[:, 1], 0], pred_list[ind_pairs[:, 1], 1]
            cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distance_between_spherical_coordinates_rad(az1, ele1, az2, ele2)

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind

  
def segment_labels(_pred_dict, _max_frames, nb_frames_1s):
    '''
        Collects class-wise sound event location information in segments of length 1s from reference dataset
    :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
    :param _max_frames: Total number of frames in the recording
    :return: Dictionary containing class-wise sound event location information in each segment of audio
            dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
    '''
    nb_blocks = int(np.ceil(_max_frames/float(nb_frames_1s)))
    output_dict = {x: {} for x in range(nb_blocks)}
    for frame_cnt in range(0, _max_frames, nb_frames_1s):

        # Collect class-wise information for each block
        # [class][frame] = <list of doa values>
        # Data structure supports multi-instance occurence of same class
        block_cnt = frame_cnt // nb_frames_1s
        loc_dict = {}
        for audio_frame in range(frame_cnt, frame_cnt+nb_frames_1s):
            if audio_frame not in _pred_dict:
                continue
            for value in _pred_dict[audio_frame]:
                if value[0] not in loc_dict:
                    loc_dict[value[0]] = {}

                block_frame = audio_frame - frame_cnt
                if block_frame not in loc_dict[value[0]]:
                    loc_dict[value[0]][block_frame] = []
                loc_dict[value[0]][block_frame].append(value[1:])

        # Update the block wise details collected above in a global structure
        for class_cnt in loc_dict:
            if class_cnt not in output_dict[block_cnt]:
                output_dict[block_cnt][class_cnt] = []

            keys = [k for k in loc_dict[class_cnt]]
            values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

            output_dict[block_cnt][class_cnt].append([keys, values])

    return output_dict


def match_event_roll_lengths_doa(event_roll_a, event_roll_b, length=None):
    """Fix the length of two event rolls (supports 2D and 3D/DOA arrays).

    Parameters
    ----------
    event_roll_a: np.ndarray, shape=(m1, k) or (m1, k, 3)
        Event roll A
    event_roll_b: np.ndarray, shape=(m2, k) or (m2, k, 3)
        Event roll B
    length: int, optional
        Length of the event roll. If None, shorter event roll is padded to match longer one.

    Returns
    -------
    event_roll_a: np.ndarray
        Padded/Cropped event roll A
    event_roll_b: np.ndarray
        Padded/Cropped event roll B
    """
    
    def pad_event_roll(event_roll, target_length):
        current_length = event_roll.shape[0]
        if target_length > current_length:
            padding_length = target_length - current_length
            
            # construct shape to support both 2D (SED) and 3D (DOA)
            pad_shape = list(event_roll.shape)
            pad_shape[0] = padding_length
            
            padding = np.zeros(tuple(pad_shape))
            event_roll = np.vstack((event_roll, padding))
        return event_roll

    # Fix durations of both event_rolls to be equal
    if length is None:
        length = max(event_roll_b.shape[0], event_roll_a.shape[0])
    else:
        length = int(length)

    if length < event_roll_a.shape[0]:
        event_roll_a = event_roll_a[0:length, ...]
    else:
        event_roll_a = pad_event_roll(event_roll_a, length)

    # Handle Event Roll B
    if length < event_roll_b.shape[0]:
        event_roll_b = event_roll_b[0:length, ...]
    else:
        event_roll_b = pad_event_roll(event_roll_b, length)

    return event_roll_a, event_roll_b


def cart2sph(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.
    Returns: azimuth, elevation, radius
    """
    XsqPlusYsq = x ** 2 + y ** 2
    r = np.sqrt(XsqPlusYsq + z ** 2)               # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))      # theta (elevation)
    az = np.arctan2(y, x)                          # phi (azimuth)
    return az, elev, r


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates using elevation angle.

    Parameters:
    x, y, z: Cartesian coordinates

    Returns:
    tuple(float, float, float): (r, azimuth, elevation)
    where:
    - azimuth: angle in the x-y plane (0° to 360°)
    - elevation: angle from the x-y plane (-90° to 90°)
    """

    # Calculate radial distance
    # Calculate azimuth angle in x-y plane
    azimuth = np.radians(np.rad2deg(np.arctan2(x, y)) + 360) % 360
    elevation = np.atan(z / np.sqrt(x**2 + y**2))
    return azimuth, elevation


def label_vocab_as_dict(df: pd.DataFrame, key: str, value: str) -> Dict:
    """
    Returns a dictionary of the label vocabulary mapping the label column to
    the idx column. key sets whether the label or idx is the key in the dict. The
    other column will be the value.
    """
    if key == "label":
        # Make sure the key is a string
        df["label"] = df["label"].astype(str)
        value = "idx"
    else:
        assert key == "idx", "key argument must be either 'label' or 'idx'"
        value = "label"
    return df.set_index(key).to_dict()[value]


def label_to_binary_vector(label: List, num_labels: int) -> torch.Tensor:
    """
    Converts a list of labels into a binary vector
    Args:
        label: list of integer labels
        num_labels: total number of labels

    Returns:
        A float Tensor that is multi-hot binary vector
    """
    # Lame special case for multilabel with no labels
    if len(label) == 0:
        # BCEWithLogitsLoss wants float not long targets
        binary_labels = torch.zeros((num_labels,), dtype=torch.float)
    else:
        binary_labels = torch.zeros((num_labels,)).scatter(0, torch.tensor(label), 1.0)

    # Validate the binary vector we just created
    assert set(torch.where(binary_labels == 1.0)[0].numpy()) == set(label)
    return binary_labels


def validate_score_return_type(ret: Union[Tuple[Tuple[str, float], ...], float]):
    """
    Valid return types for the metric are
        - tuple(tuple(string: name of the subtype, float: the value)): This is the
            case with sed eval metrics. They can return (("f_measure", value),
            ("precision", value), ...), depending on the scores
            the metric should is supposed to return. This is set as `scores`
            attribute in the metric.
        - float: Standard metric behaviour

    The downstream prediction pipeline is able to handle these two types.
    In case of the tuple return type, the value of the first entry in the
    tuple will be used as an optimisation criterion wherever required.
    For instance, if the return is (("f_measure", value), ("precision", value)),
    the value corresponding to the f_measure will be used ( for instance in
    early stopping if this metric is the primary score for the task )
    """
    if isinstance(ret, tuple):
        assert all(
            type(s) == tuple and type(s[0]) == str and type(s[1]) == float for s in ret
        ), (
            "If the return type of the score is a tuple, all the elements "
            "in the tuple should be tuple of type (string, float)"
        )
    elif isinstance(ret, float):
        pass
    else:
        raise ValueError(
            f"Return type {type(ret)} is unexpected. Return type of "
            "the score function should either be a "
            "tuple(tuple) or float. "
        )


class ScoreFunction:
    """
    A simple abstract base class for score functions
    """

    # TODO: Remove label_to_idx?
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param label_to_idx: Map from label string to integer index.
        :param name: Override the name of this scoring function.
        :param maximize: Maximize this score? (Otherwise, it's a loss or energy
            we want to minimize, and I guess technically isn't a score.)
        """
        self.label_to_idx = label_to_idx
        if name:
            self.name = name
        self.maximize = maximize

    def __call__(self, *args, **kwargs) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Calls the compute function of the metric, and after validating the output,
        returns the metric score
        """
        ret = self._compute(*args, **kwargs)
        validate_score_return_type(ret)
        return ret

    def _compute(
        self, predictions: Any, targets: Any, **kwargs
    ) -> Union[Tuple[Tuple[str, float], ...], float]:
        """
        Compute the score based on the predictions and targets.
        This is a private function and the metric should be used as a functor
        by calling the `__call__` method which calls this and also validates
        the return type
        """
        raise NotImplementedError("Inheriting classes must implement this function")

    def __str__(self):
        return self.name


class Top1Accuracy(ScoreFunction):
    name = "top1_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            if predicted_class == target_class:
                correct += 1

        return correct / len(targets)

class ChromaAccuracy(ScoreFunction):
    """
    Score specifically for pitch detection -- converts all pitches to chroma first.
    This score ignores octave errors in pitch classification.
    """

    name = "chroma_acc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # Compute the number of correct predictions
        correct = 0
        for target, prediction in zip(targets, predictions):
            assert prediction.ndim == 1
            assert target.ndim == 1
            predicted_class = np.argmax(prediction)
            target_class = np.argmax(target)

            # Ignore octave errors by converting the predicted class to chroma before
            # checking for correctness.
            if predicted_class % 12 == target_class % 12:
                correct += 1

        return correct / len(targets)


class SoundEventScore(ScoreFunction):
    """
    Scores for sound event detection tasks using sed_eval
    """

    # Score class must be defined in inheriting classes
    score_class: sed_eval.sound_event.SoundEventMetrics = None

    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        params: Dict = None,
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        """
        :param scores: Scores to use, from the list of overall SED eval scores.
            The first score in the tuple will be the primary score for this metric
        :param params: Parameters to pass to the scoring function,
                       see inheriting children for details.
        """
        if params is None:
            params = {}
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = params
        assert self.score_class is not None

    def _compute(
        self, predictions: Dict, targets: Dict, **kwargs
    ) -> Tuple[Tuple[str, float], ...]:
        # Containers of events for sed_eval
        reference_event_list = self.sed_eval_event_container(targets)
        estimated_event_list = self.sed_eval_event_container(predictions)

        # This will break in Python < 3.6 if the dict order is not
        # the insertion order I think. I'm a little worried about this line
        scores = self.score_class(
            event_label_list=list(self.label_to_idx.keys()), **self.params
        )

        for filename in predictions:
            #We need to understand here.
            #It calculates the scores per filename. This is obvious.
            scores.evaluate(
                reference_event_list=reference_event_list.filter(filename=filename),
                estimated_event_list=estimated_event_list.filter(filename=filename),
            )

        # results_overall_metrics return a pretty large nested selection of scores,
        # with dicts of scores keyed on the type of scores, like f_measure, error_rate,
        # accuracy
        nested_overall_scores: Dict[str, Dict[str, float]] = (
            scores.results_overall_metrics()
        )
        # Open up nested overall scores
        overall_scores: Dict[str, float] = dict(
            ChainMap(*nested_overall_scores.values())
        )
        # Return the required scores as tuples. The scores are returned in the
        # order they are passed in the `scores` argument

        return tuple([(score, overall_scores[score]) for score in self.scores])

    @staticmethod
    def sed_eval_event_container(
        x: Dict[str, List[Dict[str, Any]]],
    ) -> MetaDataContainer:
        # Reformat event list for sed_eval
        reference_events = []
        for filename, event_list in x.items():
            for event in event_list:
                reference_events.append(
                    {
                        # Convert from ms to seconds for sed_eval
                        "event_label": str(event["label"]),
                        "event_onset": event["start"] / 1000.0,
                        "event_offset": event["end"] / 1000.0,
                        "file": filename,
                    }
                )
        return  MetaDataContainer(reference_events)


class OldSELD(ScoreFunction):
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        scores: Tuple[str],
        azimuth_list : Tuple[int, int], 
        elevation_list : Tuple[int, int],
        _doa_resolution: int, 
        name: Optional[str] = None,
        maximize: bool = True
    ):
        
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = {}
        self._azi_list = range(azimuth_list[0], azimuth_list[1], _doa_resolution)
        self._ele_list = range(elevation_list[0], elevation_list[1], _doa_resolution)
        self._height = len(self._ele_list)

    def _compute(self,
        pred_dicts,
        ref_dicts,
        _nb_label_frames_1s,
        _nb_pred_frames_1s,
        _max_frames,
        _max_ref_frames) -> Tuple[Tuple[str, float], ...]:
        
        overall_scores = {}
        eval = OldSELDMetrics(nb_frames_1s=_nb_label_frames_1s, data_gen=self)
        eval.reset()
        for file_name in pred_dicts.keys():
            pred_dict = pred_dicts[file_name]
            ref_dict = ref_dicts[file_name]
            _max_frame = _max_frames[file_name]
            _max_ref_frames = _max_ref_frames[file_name]

            #Our prediction dict is in cartesian format, so convert it to polar!
            pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
            pred_dict = convert_output_format_new_to_old(pred_dict)
            ref_dict = convert_output_format_new_to_old(ref_dict)

            #Convert from regression to classification for DOA, the max_frames is indeed the same as prediction labels.
            gt_labels = self.output_format_dict_to_classification_labels(ref_dict, _max_ref_frames)
            pred_labels = self.output_format_dict_to_classification_labels(pred_dict, _max_frames)
            # Calculated SED and DOA scores

            eval.update_sed_scores(pred_labels.max(2), gt_labels.max(2))
            eval.update_doa_scores(pred_labels, gt_labels)

        # Overall SED and DOA scores
        ER, F = eval.compute_sed_scores()
        LE, LR = eval.compute_doa_scores()
        seld_scr = compute_seld_metric([ER, F], [LE, LR])

        overall_scores["ER"] = ER
        overall_scores["F"] = F 
        overall_scores["LE"] = LE
        overall_scores["LR"] = LR 
        overall_scores["SELD"] = seld_scr

        return tuple([(score, float(overall_scores[score])) for score in self.scores])


    def output_format_dict_to_classification_labels(self, _output_dict, max_frames):

        _nb_classes = len(self.label_to_idx)
        _labels = np.zeros((max_frames, _nb_classes, len(self._azi_list) * len(self._ele_list)))

        for _frame_cnt in _output_dict.keys():
            for _tmp_doa in _output_dict[_frame_cnt]:
                # Making sure the doa's are within the limits
                _tmp_doa[1] = np.clip(_tmp_doa[1], self._azi_list[0], self._azi_list[-1])
                _tmp_doa[2] = np.clip(_tmp_doa[2], self._ele_list[0], self._ele_list[-1])
                # create label
                _labels[_frame_cnt, _tmp_doa[0], int(self.get_list_index(_tmp_doa[1], _tmp_doa[2]))] = 1

        return _labels
     
    def get_list_index(self, azi, ele):
        azi = (azi - self._azi_list[0]) // 10
        ele = (ele - self._ele_list[0]) // 10
        return azi * self._height + ele

    def get_matrix_index(self, ind):
        azi, ele = ind // self._height, ind % self._height
        azi = (azi * 10 + self._azi_list[0])
        ele = (ele * 10 + self._ele_list[0])
        return azi, ele


class SELD(ScoreFunction):
    def __init__(
        self,
        label_to_idx: Dict[str, int],
        doa_threshold:float,
        average:str,
        scores: Tuple[str],
        name: Optional[str] = None,
        maximize: bool = True,
    ):
        
        super().__init__(label_to_idx=label_to_idx, name=name, maximize=maximize)
        self.scores = scores
        self.params = {}
        self.doa_threshold = doa_threshold
        self.average=average
        self.nb_classes = len(self.label_to_idx)

    #MAX FRAMES passed here, but not used. It is for the completeness of the whole class.
    #max_frames refer to the maximum number of frames in the dataset.
    def _compute(self,
        pred_dicts,
        ref_dicts,
        _nb_label_frames_1s,
        _nb_pred_frames_1s,
        _max_frames,
        _max_ref_frames) -> Tuple[Tuple[str, float], ...]:
        
        overall_scores = {}

        eval = SELDMetrics(nb_classes=self.nb_classes, doa_threshold=self.doa_threshold, average=self.average)
        for file_name in pred_dicts.keys():
            pred_dict = pred_dicts[file_name]
            ref_dict = ref_dicts[file_name]
            _max_frame = _max_frames[file_name]
            _max_ref_frame = _max_ref_frames[file_name]
            #Our prediction dict is in cartesian format, so convert it to polar!
            pred_dict = convert_output_format_cartesian_to_polar(pred_dict)
            if _nb_label_frames_1s == _nb_pred_frames_1s:
                max_frames = max(list(pred_dict.keys()))
                pred_labels = segment_labels(pred_dict, max_frames, nb_frames_1s=_nb_label_frames_1s)
                ref_labels = segment_labels(ref_dict, max_frames, nb_frames_1s=_nb_label_frames_1s)
            else:
                pred_labels = segment_labels(pred_dict, _max_frames, nb_frames_1s=_nb_pred_frames_1s)
                ref_labels = segment_labels(ref_dict, _max_ref_frames, nb_frames_1s=_nb_label_frames_1s)
            eval.update_seld_scores(pred_labels, ref_labels)

        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
        overall_scores["ER"] = ER
        overall_scores["F"] = F 
        overall_scores["LE"] = LE
        overall_scores["LR"] = LR 
        overall_scores["SELD"] = seld_scr

        return tuple([(score, float(overall_scores[score])) for score in self.scores])

class SegmentBasedScore(SoundEventScore):
    """
    segment-based scores - the ground truth and system output are compared in a
    fixed time grid; sound events are marked as active or inactive in each segment;

    See https://tut-arg.github.io/sed_eval/sound_event.html#sed_eval.sound_event.SegmentBasedMetrics # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.SegmentBasedMetrics


class EventBasedScore(SoundEventScore):
    """
    event-based scores - the ground truth and system output are compared at
    event instance level;

    See https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html # noqa: E501
    for params.
    """

    score_class = sed_eval.sound_event.EventBasedMetrics


class MeanAveragePrecision(ScoreFunction):
    """
    Average Precision is calculated in macro mode which calculates
    AP at a class level followed by macro-averaging across the classes.
    """

    name = "mAP"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot

        """
        Based on suggestions from Eduardo Fonseca -
        Equal weighting is assigned to each class regardless
        of its prior, which is commonly referred to as macro
        averaging, following Hershey et al. (2017); Gemmeke et al.
        (2017).
        This means that rare classes are as important as common
        classes.

        Issue with average_precision_score, when all ground truths are negative
        https://github.com/scikit-learn/scikit-learn/issues/8245
        This might come up in small tasks, where few samples are available
        """
        return average_precision_score(targets, predictions, average="macro")


class DPrime(ScoreFunction):
    """
    DPrime is calculated per class followed by averaging across the classes

    Code adapted from code provided by Eduoard Fonseca.
    """

    name = "d_prime"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            auc = roc_auc_score(targets, predictions, average=None)

            d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
            # Calculate macro score by averaging over the classes,
            # see `MeanAveragePrecision` for reasons
            d_prime_macro = np.mean(d_prime)
            return d_prime_macro
        except ValueError:
            return np.nan


class AUCROC(ScoreFunction):
    """
    AUCROC (macro mode) is calculated per class followed by averaging across the
    classes
    """

    name = "aucroc"

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            # Macro mode auc-roc. Please check `MeanAveragePrecision`
            # for the reasoning behind using using macro mode
            auc = roc_auc_score(targets, predictions, average="macro")
            return auc
        except ValueError:
            return np.nan


class SourceLocal(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2 and targets.ndim == 2
        assert predictions.shape == targets.shape

        # Compute per-sample Euclidean distance
        mean_error = np.abs(predictions - targets).mean()
        return float(mean_error)


class SourceLocalX(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local_x"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            source_localization_error = np.abs(
                np.subtract(predictions[..., 0], targets[..., 0])
            ).mean()
            return float(source_localization_error)
        except ValueError:
            return np.nan


class SourceLocalY(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local_y"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            source_localization_error = np.abs(
                np.subtract(predictions[..., 1], targets[..., 1])
            ).mean()
            return float(source_localization_error)
        except ValueError:
            return np.nan


class SourceLocalZ(ScoreFunction):
    """
    3D Source Localization Error
    """

    name = "3d_source_local_z"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            source_localization_error = np.abs(
                np.subtract(predictions[..., 2], targets[..., 2])
            ).mean()
            return float(source_localization_error)
        except ValueError:
            return np.nan


class DOE(ScoreFunction):
    """
    3D Source Localization Error W.R.T Direction of Arrivial Estimation.
    """

    name = "DOE"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Mean source localization
            pred_az_rad, pred_el_rad = cartesian_to_spherical(
                predictions[..., 0], predictions[..., 1], predictions[..., 2]
            )
            target_az_rad, target_el_rad = cartesian_to_spherical(
                targets[..., 0], targets[..., 1], targets[..., 2]
            )

            # Calculate the angular distance (great circle distance)
            # cos(angular_distance) = sin(el1)*sin(el2) + cos(el1)*cos(el2)*cos(az1-az2)
            cos_dist = np.sin(target_el_rad) * np.sin(pred_el_rad) + np.cos(
                target_el_rad
            ) * np.cos(pred_el_rad) * np.cos(target_az_rad - pred_az_rad)

            # Clip to handle floating point errors
            cos_dist = np.clip(cos_dist, -1.0, 1.0)

            # Convert back to degrees
            angular_dist = np.rad2deg(np.arccos(cos_dist))
            source_localization_error = np.median(angular_dist)
            return float(source_localization_error)
        except ValueError as e:
            print(e)
            return 0.0


class DOA(ScoreFunction):
    """
    3D Source Localization Error W.R.T Direction of Arrivial Estimation.
    """

    name = "DOA"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            sqrt_pred_error = np.sqrt(np.sum((targets - predictions) ** 2, axis=1))
            source_localization_error = (
                2 * np.arcsin(sqrt_pred_error / 2) * (180 / np.pi)
            )
            source_localization_error = np.nan_to_num(source_localization_error, 180)
            return float(source_localization_error.mean())
        except ValueError as e:
            print(e)
            return 180


class MeanAngularError(ScoreFunction):
    """
    3D Source Localization Error W.R.T Direction of Arrivial Estimation.
    """

    name = "MAE"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        # For each datapoint we predict X,Y,Z coordinate.
        assert predictions.ndim == 2
        assert targets.ndim == 2  # X,Y,Z coordiante of the source
        try:
            # Normalize each row vector separately
            targets_normalized = targets / np.linalg.norm(
                targets, axis=1, keepdims=True
            )
            predictions_normalized = predictions / np.linalg.norm(
                predictions, axis=1, keepdims=True
            )

            # Compute dot product between corresponding vectors
            dot_products = np.sum(targets_normalized * predictions_normalized, axis=1)

            # Clip values to valid arccos domain to avoid numerical issues
            dot_products = np.clip(dot_products, -1.0, 1.0)

            # Calculate angular error in radians
            source_localization_error = np.arccos(dot_products) * (180.0 / np.pi)

            return float(source_localization_error.mean())
        except ValueError as e:
            print(e)
            return 180


class Distance(ScoreFunction):
    """
    Distance Error
    """

    name = "distance"
    maximize = False

    def _compute(self, predictions: np.ndarray, targets: np.ndarray, **kwargs) -> float:
        assert predictions.ndim == 2
        assert targets.ndim == 2  # One hot
        # ROC-AUC Requires more than one example for each class
        # This might fail for data in small instances, so putting this in try except
        try:
            # Macro mode auc-roc. Please check `MeanAveragePrecision`
            # for the reasoning behind using using macro mode
            predictions = np.sqrt(np.power(predictions, 2).sum(axis=1))
            targets = np.sqrt(np.power(targets, 2).sum(axis=1))
            r_error = (np.abs(predictions - targets)).mean()
            return float(r_error)
        except ValueError as e:
            print(e)
            return np.nan


available_scores: Dict[str, Callable] = {
    "top1_acc": Top1Accuracy,
    "pitch_acc": partial(Top1Accuracy, name="pitch_acc"),
    "chroma_acc": ChromaAccuracy,
    # https://tut-arg.github.io/sed_eval/generated/sed_eval.sound_event.EventBasedMetrics.html
    "event_onset_200ms_fms": partial(
        EventBasedScore,
        name="event_onset_200ms_fms",
        scores=("f_measure", "precision", "recall"),
        params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.2},
    ),
    "event_onset_50ms_fms": partial(
        EventBasedScore,
        name="event_onset_50ms_fms",
        scores=("f_measure", "precision", "recall"),
        params={"evaluate_onset": True, "evaluate_offset": False, "t_collar": 0.05},
    ),
    "event_onset_offset_50ms_20perc_fms": partial(
        EventBasedScore,
        name="event_onset_offset_50ms_20perc_fms",
        scores=("f_measure", "precision", "recall"),
        params={
            "evaluate_onset": True,
            "evaluate_offset": True,
            "t_collar": 0.05,
            "percentage_of_length": 0.2,
        },
    ),
    "segment_1s_er": partial(
        SegmentBasedScore,
        name="segment_1s_er",
        scores=["error_rate"],
        params={"time_resolution": 1.0},
        maximize=False,
    ),
    'SELD': partial(
        SELD,
        name="SELD",
        scores=("SELD", "ER", "F", "LE", "LR"),
        maximize=False,
    ),
    "OldSELD": partial(
        OldSELD,
        name="SELD",
        scores=("SELD", "ER", "F", "LE", "LR"),
        maximize=False,
    ),
    "mAP": MeanAveragePrecision,
    "d_prime": DPrime,
    "aucroc": AUCROC,
    "3d_source_local": SourceLocal,
    "3d_source_local_x": SourceLocalX,
    "3d_source_local_y": SourceLocalY,
    "3d_source_local_z": SourceLocalZ,
    "DOE": DOE,
    "DOA": DOA,
    "MAE": MeanAngularError,
    "distance": Distance,
}
