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


def extract_trajectory_whole_game(eye_data_df:pd.DataFrame, ball_data_df:pd.DataFrame, video_id, scale_to_percentage=False):
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


def extract_trajectory_lite(round_index, data:pd.DataFrame, ball_data_df):
    _fea = {}
    
    _ball_df = ball_data_df[ball_data_df["round"]==round_index]
    _dtw = compute_dtw(
            line_1=data.loc[:, ["Screen.x", "Screen.y"]],
            line_2=_ball_df.loc[:, ["Ball.x", "Ball.y"]],
            # scale_to_percentage=True,
            # scale_metrics="start"
        )
    
    _fea["TrajectoryDTW"] = _dtw

    return _fea


def compute_ball_move(ball_line:pd.DataFrame):
    _indeices = ball_line.index.to_list()
    _dist = []
    for i in range(1, len(_indeices)):
        _dist.append(
                np.sqrt(
                        (ball_line.loc[_indeices[i], "Ball.x"] - ball_line.loc[_indeices[i-1], "Ball.x"])**2 + (ball_line.loc[_indeices[i],"Ball.y"] - ball_line.loc[_indeices[i-1],"Ball.y"])**2
                    )
            )
    return np.sum(_dist)

def compute_two_traj_angle(eye_traj:pd.DataFrame, ball_traj:pd.DataFrame):
    # Line 1 points
    x1, y1 = eye_traj.iloc[0, :]["Screen.x"], eye_traj.iloc[0, :]["Screen.y"]
    x2, y2 = eye_traj.iloc[-1, :]["Screen.x"], eye_traj.iloc[-1, :]["Screen.y"]

    # Line 2 points
    x3, y3 = ball_traj.iloc[0, :]["Ball.x"], ball_traj.iloc[0, :]["Ball.y"]
    x4, y4 = ball_traj.iloc[-1, :]["Ball.x"], ball_traj.iloc[-1, :]["Ball.y"]

    # Create vectors for both lines
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Calculate the dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes of both vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

    # Compute the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def add_trajectory_lite(round_index, data:pd.DataFrame, ball_data_df):
    _fea = {}
    
    _ball_df = ball_data_df[ball_data_df["round"]==round_index]
    _dtw = compute_dtw(
            line_1=data.loc[:, ["Screen.x", "Screen.y"]],
            line_2=_ball_df.loc[:, ["Ball.x", "Ball.y"]],
        )
    
    _ball_move = compute_ball_move(_ball_df.loc[:, ["Ball.x", "Ball.y"]].copy())

    _fea["TrajDTW"] = _dtw * VR_SCALE
    _fea["TrajDTWPerBallMove"] = (_dtw * VR_SCALE) / (_ball_move * VR_SCALE)
    _fea["DirecAngle"] = compute_two_traj_angle(
            eye_traj=data.loc[:, ["Screen.x", "Screen.y"]],
            ball_traj=_ball_df.loc[:, ["Ball.x", "Ball.y"]].copy()
        )

    return _fea

def extract_trajectory(round_index, data:pd.DataFrame, ball_data_df):
    _fea = {}
    
    _ball_df = ball_data_df[ball_data_df["round"]==round_index]
    _dtw = compute_dtw(
            line_1=data.loc[:, ["Screen.x", "Screen.y"]],
            line_2=_ball_df.loc[:, ["Ball.x", "Ball.y"]],
        )
    
    _ball_move = compute_ball_move(_ball_df.loc[:, ["Ball.x", "Ball.y"]].copy())

    _fea["TrajDTW"] = _dtw * VR_SCALE
    _fea["TrajDTWPerBallMove"] = (_dtw * VR_SCALE) / (_ball_move * VR_SCALE)
    _fea["DirecAngle"] = compute_two_traj_angle(
            eye_traj=data.loc[:, ["Screen.x", "Screen.y"]],
            ball_traj=_ball_df.loc[:, ["Ball.x", "Ball.y"]].copy()
        )

    return _fea

