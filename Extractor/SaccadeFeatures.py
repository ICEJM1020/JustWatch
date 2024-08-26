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

