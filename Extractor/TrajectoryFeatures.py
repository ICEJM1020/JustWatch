""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-26
""" 
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from config import *
from Extractor.utils import interplate_and_align, compute_dtw, compute_gradient_dtw


""" 
Description: Trajectory Features Group
""" 
## eps measured as pixel
def compute_traj_overlap(concat_df:pd.DataFrame, eps:float=15.0):
    concat_df["overlap"] = concat_df.apply(
            lambda row : 1 if np.sqrt((row["Screen.x"] - row["Ball.x"])**2 + (row["Screen.y"] - row["Ball.y"])**2) <= eps else 0,
            axis=1
        )
    return concat_df["overlap"].sum() / concat_df.shape[0]


def compute_traj_range_overlap(concat_df:pd.DataFrame, eps:float=15.0, time_range=25):
    concat_df["overlap"] = 0
    for idx, row in concat_df.iterrows():
        _temp_df = concat_df.loc[idx-time_range:idx+time_range, :]
        _temp_df["overlap"] = _temp_df.apply(
            lambda b_row : 1 if np.sqrt((row["Screen.x"] - b_row["Ball.x"])**2 + (row["Screen.y"] - b_row["Ball.y"])**2) <= eps else 0,
            axis=1
        )
        if _temp_df["overlap"].sum() > 0:
            concat_df.loc[idx, "overlap"] = 1
    return concat_df["overlap"].sum() / concat_df.shape[0]


def extract_trajectory(eye_data_df:pd.DataFrame, ball_data_df:pd.DataFrame, video_id, scale_to_percentage=False):
    res = {}

    eye_data_df["frame"] = eye_data_df.index
    
    aligned_df = interplate_and_align(eye_data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    # aligned_df.rename(columns={'x':'Ball.x', "y":"Ball.y"}, inplace=True)
    
    _traj_overlap = compute_traj_overlap(aligned_df)
    _traj_range_overlap = compute_traj_range_overlap(aligned_df)
    _all_dtw = compute_dtw(
            line_1=aligned_df.loc[:, ["Screen.x", "Screen.y"]],
            line_2=aligned_df.loc[:, ["Ball.x", "Ball.y"]],
            scale_to_percentage=scale_to_percentage
        )
    _all_speed_dtw = compute_gradient_dtw(
            line_1=aligned_df.loc[:, ["frame", "Screen.x", "Screen.y"]],
            line_2=aligned_df.loc[:, ["frame", "Ball.x", "Ball.y"]],
            order=1,
            scale_to_percentage=scale_to_percentage
    )
    _all_accelerate_dtw = compute_gradient_dtw(
            line_1=aligned_df.loc[:, ["frame", "Screen.x", "Screen.y"]],
            line_2=aligned_df.loc[:, ["frame", "Ball.x", "Ball.y"]],
            order=2,
            scale_to_percentage=scale_to_percentage
    )

    res["AllTrajectoryOverlap"] = _traj_overlap * 100
    res["AllTrajectoryRangeOverlap"] = _traj_range_overlap * 100
    if scale_to_percentage:
        res["AllTrajectoryDTW"] = _all_dtw
        res["AllTrajectoryDTWPerSec"] = _all_dtw / (eye_data_df.shape[0] * EYE_SAMPLE_TIME / 1000)
        res["AllTrajectorySpeedDTW"] = _all_speed_dtw
        res["AllTrajectorySpeedDTWPerSec"] = _all_speed_dtw / (eye_data_df.shape[0] * EYE_SAMPLE_TIME / 1000)
        res["AllTrajectoryAccelerateDTW"] = _all_accelerate_dtw
        res["AllTrajectoryAcceleratePerSec"] = _all_accelerate_dtw / (eye_data_df.shape[0] * EYE_SAMPLE_TIME / 1000)
    else:
        res["AllTrajectoryDTW"] = _all_dtw * VR_SCALE
        res["AllTrajectoryDTWPerSec"] = _all_dtw * VR_SCALE / (eye_data_df.shape[0] * EYE_SAMPLE_TIME / 1000)
        res["AllTrajectorySpeedDTW"] = _all_speed_dtw * VR_SCALE
        res["AllTrajectorySpeedDTWPerSec"] = _all_speed_dtw * VR_SCALE / (eye_data_df.shape[0] * EYE_SAMPLE_TIME / 1000)
        res["AllTrajectoryAccelerateDTW"] = _all_accelerate_dtw * VR_SCALE
        res["AllTrajectoryAccelerateDTWPerSec"] = _all_accelerate_dtw * VR_SCALE / (eye_data_df.shape[0] * EYE_SAMPLE_TIME / 1000)
    
    return res


