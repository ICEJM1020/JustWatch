""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-07
""" 
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from config import *
from Extractor.utils import max_circle_radius, compute_stat



""" 
Description: Saccade Features Group
""" 
def compute_drop_point_bias_pred(eye_data:pd.DataFrame, eye_index, ball_data:pd.DataFrame, ball_round, eye_drop="last"):
    if eye_drop=="last":
        eye_init = eye_data.iloc[0, :]
        eye_final = eye_data.iloc[-1, :]
    elif eye_drop=="fast":
        idx_min = eye_data.index.min()
        eye_init = eye_data.loc[eye_index-1 if eye_index-1>idx_min else idx_min, :]
        eye_final = eye_data.loc[eye_index, :]
    ball_init = ball_data[ball_data["round"]==ball_round].iloc[0, :]
    ball_final = ball_data[ball_data["round"]==ball_round].iloc[-1, :]

    init_bias = np.sqrt((ball_init["Ball.x"] - eye_init["Screen.x"])**2 + (ball_init["Ball.y"] - eye_init["Screen.y"])**2)
    final_bias = np.sqrt((ball_final["Ball.x"] - eye_final["Screen.x"])**2 + (ball_final["Ball.y"] - eye_final["Screen.y"])**2)
    return init_bias, final_bias


def compute_drop_point_bias_delay(eye_data:pd.DataFrame, eye_index, ball_data:pd.DataFrame, ball_round, eye_drop="last"):
    if eye_drop=="last":
        eye_init = eye_data.iloc[0, :]
        eye_final = eye_data.iloc[-1, :]
    elif eye_drop=="fast":
        idx_min = eye_data.index.min()
        eye_init = eye_data.loc[eye_index-1 if eye_index-1>idx_min else idx_min, :]
        eye_final = eye_data.loc[eye_index, :]
    ball_init = ball_data[ball_data["round"]==ball_round-1].iloc[0, :]
    ball_final = ball_data[ball_data["round"]==ball_round-1].iloc[-1, :]

    init_bias = np.sqrt((ball_init["Ball.x"] - eye_init["Screen.x"])**2 + (ball_init["Ball.y"] - eye_init["Screen.y"])**2)
    final_bias = np.sqrt((ball_final["Ball.x"] - eye_final["Screen.x"])**2 + (ball_final["Ball.y"] - eye_final["Screen.y"])**2)
    return init_bias, final_bias


def compute_saccade_path(df:pd.DataFrame):
    _speeds = []
    _angles = []

    for _c in range(0, df["c"].max()):
        _temp_t1 = df[df["c"]==_c]
        _temp_t2 = df[df["c"]==_c+1]
        
        c_x_t1 = _temp_t1["Screen.x"].mean()
        c_y_t1 = _temp_t1["Screen.y"].mean()
        c_x_t2 = _temp_t2["Screen.x"].mean()
        c_y_t2 = _temp_t2["Screen.y"].mean()

        _dura = ((_temp_t2["t"].mean() - _temp_t1["t"].mean()) * EYE_SAMPLE_TIME ) / 1000.0
        _dist = np.sqrt((c_x_t1 - c_x_t2)**2 + (c_y_t1 - c_y_t2)**2)
        _angle = np.arctan(_dist / VR_ZDIST) / np.pi * 180
        _angles.append(_angle)

        if _dura <= EYE_SAMPLE_TIME / 1000.0:
            _speeds.append(0)
        else:
            _speeds.append( np.divide(_angle, _dura))

    return _speeds, _angles


def extract_saccade_features(data, video_id):
    _fea = {}
    temp = np.array([data["Screen.x"], data["Screen.y"], data.index.to_numpy()]).T
    temp = pd.DataFrame(temp)
    temp.columns=["Screen.x", "Screen.y", "t"]

    db = DBSCAN(eps=CLUSTER_MIN_EPS, min_samples=CLUSTER_MIN_SAMPLE)
    cluster_res = db.fit(temp.dropna())
    clusters = pd.value_counts(cluster_res.labels_)

    if clusters.shape[0] <= 2: 
        return _fea

    # _not_na_index = temp.dropna().index
    # temp["c"] = -1
    # temp.loc[_not_na_index, "c"] = cluster_res.labels_
    temp["c"] = cluster_res.labels_
    ## map the pixel value to real distance
    temp["x"] = temp["x"] * VR_SCALE
    temp["y"] = temp["y"] * VR_SCALE

    _fea["NumOfGazePoints"] = len(clusters.index) - 1
    # "_g" for gaze
    _g_duration = []
    _g_radius = []
    _g_density = []
    _g_density_t = []
    for c, v in clusters.items():
        if c == -1: continue
        _duration = v * EYE_SAMPLE_TIME
        _g_duration.append(_duration)
        _circle_r = max_circle_radius(temp[temp["c"]==c]) * 100
        _g_radius.append(_circle_r)
        _density = v / (np.pi * np.power(_circle_r,2))
        _g_density.append(_density)
        _g_density_t.append(_density / _duration)

    _speeds,_angles = compute_saccade_path(temp)

    _fea = {**_fea, **compute_stat("GazeRadius", _g_radius)}
    _fea = {**_fea, **compute_stat("GazeDuratiuon", _g_duration)}
    _fea = {**_fea, **compute_stat("GazeDensity", _g_density)}
    _fea = {**_fea, **compute_stat("GazeDensityFrequency", _g_density_t)}

    _fea = {**_fea, **compute_stat("SaccadeSpeed", _speeds)}
    _fea = {**_fea, **compute_stat("SaccadeAngel", _angles)}

    return _fea


def compute_saccade_path_lite(df:pd.DataFrame):
    _speeds = []
    _angles = []
    indices = list(df.index)
    _dura = (1 * EYE_SAMPLE_TIME ) / 1000.0
    df["Screen.x"] = df["Screen.x"] * VR_SCALE
    df["Screen.y"] = df["Screen.y"] * VR_SCALE

    for _i in range(1, len(indices)):
        row_1 = df.loc[indices[_i - 1], :]
        row_2 = df.loc[indices[_i], :]

        _dist = np.sqrt((row_1["Screen.x"] - row_2["Screen.x"])**2 + (row_1["Screen.y"] - row_2["Screen.y"])**2)
        _angle = np.arctan(_dist / VR_ZDIST) / np.pi * 180

        _angles.append(_angle)
        _speeds.append(np.divide(_angle, _dura))

    return _speeds, _angles


def extract_saccade_features_lite(round_index, data:pd.DataFrame, ball_data_df:pd.DataFrame):
    _fea = {}

    _speeds, _angles = compute_saccade_path_lite(data.copy())
    _fea = {**_fea, **compute_stat("SaccadeSpeed", _speeds)}
    _fea = {**_fea, **compute_stat("SaccadeAngel", _angles)}

    round_start_index = ball_data_df[ball_data_df["round"]==round_index].index[0]
    round_dura = ball_data_df[ball_data_df["round"]==round_index].shape[0]
    eye_start_index = data.index[0]
    _fea["SaccadeDelay"] = (eye_start_index-round_start_index) * EYE_SAMPLE_TIME
    _fea["SaccadeDelayPercent"] = ((eye_start_index-round_start_index) / round_dura) * EYE_SAMPLE_TIME

    return _fea


def compute_saccade_path(df:pd.DataFrame, ball_data:pd.DataFrame):
    _speeds = []
    indices = list(df.index)
    _dura = (1 * EYE_SAMPLE_TIME ) / 1000.0
    df["Screen.x"] = df["Screen.x"] * VR_SCALE
    df["Screen.y"] = df["Screen.y"] * VR_SCALE

    ball_data["Ball.x"] = ball_data["Ball.x"] * VR_SCALE
    ball_data["Ball.y"] = ball_data["Ball.y"] * VR_SCALE

    for _i in range(1, len(indices)):
        row_1 = df.loc[indices[_i - 1], :]
        row_2 = df.loc[indices[_i], :]

        _dist = np.sqrt((row_1["Screen.x"] - row_2["Screen.x"])**2 + (row_1["Screen.y"] - row_2["Screen.y"])**2)
        _angle = np.arctan(_dist / VR_ZDIST) / np.pi * 180
        _speeds.append(np.divide(_angle, _dura))

    _eye_dist = np.sqrt((df.iloc[0, :]["Screen.x"] - df.iloc[-1, :]["Screen.x"])**2 + (df.iloc[0, :]["Screen.y"] - df.iloc[-1, :]["Screen.y"])**2)
    _eye_amp = np.arctan(_eye_dist / VR_ZDIST) / np.pi * 180
    _ball_dist = np.sqrt((ball_data.iloc[0, :]["Ball.x"] - ball_data.iloc[-1, :]["Ball.x"])**2 + (ball_data.iloc[0, :]["Ball.y"] - ball_data.iloc[-1, :]["Ball.y"])**2)
    _ball_amp = np.arctan(_ball_dist / VR_ZDIST) / np.pi * 180

    return _speeds, _eye_amp, _eye_amp/_ball_amp


def modify_saccade_features_lite(round_index, data:pd.DataFrame, ball_data_df:pd.DataFrame):
    _fea = {}

    _speeds, _ampli, _unitampli = compute_saccade_path(data.copy(), ball_data_df[ball_data_df["round"]==round_index].copy())
    _fea = {**_fea, **compute_stat("SaccadeSpeed", _speeds)}
    _fea["Amplitude"] = _ampli
    _fea["UnitAmplitude"] = _unitampli

    round_start_index = ball_data_df[ball_data_df["round"]==round_index].index[0]
    round_dura = ball_data_df[ball_data_df["round"]==round_index].shape[0]
    eye_start_index = data.index[0]
    _fea["SaccadeDelay"] = (eye_start_index-round_start_index) * EYE_SAMPLE_TIME
    _fea["SaccadeDelayPercent"] = ((eye_start_index-round_start_index) / round_dura) * EYE_SAMPLE_TIME

    return _fea


def extract_saccade_features(round_index, data:pd.DataFrame, ball_data_df:pd.DataFrame):
    _fea = {}

    _speeds, _ampli, _unitampli = compute_saccade_path(data.copy(), ball_data_df[ball_data_df["round"]==round_index].copy())
    _fea = {**_fea, **compute_stat("SaccadeSpeed", _speeds)}
    _fea["Amplitude"] = _ampli
    _fea["UnitAmplitude"] = _unitampli

    round_start_index = ball_data_df[ball_data_df["round"]==round_index].index[0]
    round_dura = ball_data_df[ball_data_df["round"]==round_index].shape[0]
    eye_start_index = data.index[0]
    _fea["SaccadeDelay"] = (eye_start_index-round_start_index) * EYE_SAMPLE_TIME
    _fea["SaccadeDelayPercent"] = ((eye_start_index-round_start_index) / round_dura) * EYE_SAMPLE_TIME

    return _fea