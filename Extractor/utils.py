""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-07
""" 
import pandas as pd
import numpy as np
from tslearn.metrics import dtw as ts_dtw
from scipy.spatial.distance import cdist

from config import *

def max_circle_radius(df:pd.DataFrame):
    centroid_x = df['Screen.x'].mean()
    centroid_y = df['Screen.y'].mean()
    
    distances = np.sqrt((df['Screen.x'] - centroid_x)**2 + (df['Screen.y'] - centroid_y)**2)
    
    max_radius = distances.max()

    if max_radius < VR_SCALE:
        return VR_SCALE
    else:
        return max_radius
    
    
def compute_stat(name:str, data_list:list):
    _res = {}
    _res[f"{name}_Mean"] = np.mean(data_list)
    _res[f"{name}_Max"] = np.max(data_list)
    _res[f"{name}_Min"] = np.min(data_list)
    _res[f"{name}_Std"] = np.std(data_list)

    return _res


def given_minmax_scale(data:pd.DataFrame, minmax, feature_range=[0,1], axis=0):
    res = []
    for _i in data.index if axis==1 else data.columns:
        scale = (feature_range[1] - feature_range[0]) / (minmax[1] - minmax[0])
        X_scaled = scale * (data[_i] - minmax[0])
        res.append(X_scaled)

    return pd.DataFrame(res) if axis==1 else pd.DataFrame(res).T


def interplate_and_align(base_df:pd.DataFrame, align_df:pd.DataFrame, base_rate:int, align_rate:int, convert_dist=False):
    share_rate = abs(base_rate * align_rate) // np.gcd(base_rate, align_rate)
    
    align_df["frame"] = align_df["frame"] * (share_rate // align_rate)
    temp_index= np.array(range(1, (int(base_df["frame"].max())+1) * (share_rate // base_rate)))

    temp_time_df = pd.DataFrame(temp_index, index=temp_index, columns=["frame"])
    # _index = temp_time_df.index.to_list()
    temp_time_df = temp_time_df.merge(align_df, how="left", on="frame")
    temp_time_df.index = list(range(1, temp_time_df.shape[0]+1))
    ## interplate
    for col in temp_time_df.columns:
        if col=="round": continue
        temp_time_df[col] = temp_time_df[col].interpolate(method='linear')
    ## ensure first row is not empty
    temp_time_df.ffill(inplace=True)
    temp_time_df.bfill(inplace=True)
    
    base_df["frame"] = base_df["frame"] * (share_rate // base_rate)
    alined_df = base_df.merge(temp_time_df, how="left", on="frame", )
    alined_df.index = list(range(1, alined_df.shape[0]+1))
    alined_df["frame"] = alined_df["frame"] // (share_rate // base_rate)

    if convert_dist:
        alined_df = alined_df * VR_SCALE
        alined_df["frame"] = alined_df["frame"] // VR_SCALE

    alined_df.ffill(inplace=True)
    alined_df.bfill(inplace=True)

    return alined_df


def interplate_enlarge(ori_df:pd.DataFrame, size:int):
    ori_df = ori_df.reset_index(drop=True)
    ori_df.index = [int(np.round(i * (size / ori_df.shape[0]))) for i in ori_df.index ]
    new_index = pd.Index(range(0, min(size, ori_df.index[-1]+1)))
    # print(new_index)
    # print(size / ori_df.shape[0])
    ori_df = ori_df.reindex(new_index).interpolate()
    return ori_df


def compute_dtw(line_1:pd.DataFrame, line_2:pd.DataFrame, scale_to_percentage=False, scale_metrics="mean"):
    assert line_1.shape[-1] == line_2.shape[-1]

    res = ts_dtw(line_1, line_2)

    if scale_to_percentage:
        mat = cdist(line_1, line_2)
        if scale_metrics=="max":
            res = (res / mat.max()) *100
        elif scale_metrics=="start":
            scale = np.sqrt((line_1.iloc[0, :]["Screen.x"] - line_2.iloc[0, :]["Ball.x"])**2 + (line_1.iloc[0, :]["Screen.y"] - line_2.iloc[0, :]["Ball.y"])**2)
            res = (res / scale) * 100
        elif scale_metrics=="end":
            scale = np.sqrt((line_1.iloc[-1, :]["Screen.x"] - line_2.iloc[-1, :]["Ball.x"])**2 + (line_1.iloc[-1, :]["Screen.y"] - line_2.iloc[-1, :]["Ball.y"])**2)
            res = (res / scale) * 100
        else:
            res = (res / mat.mean()) *100 

    return res


def compute_gradient_dtw(line_1:pd.DataFrame, line_2:pd.DataFrame, order:int=1, time_scale=1, scale_to_percentage=False):
    assert line_1.shape == line_2.shape
    assert "time" in line_1.columns or "frame" in line_1.columns, "Please include index, time, or frame (named 'time') in the line_1 DataFrame"
    assert "time" in line_2.columns or "frame" in line_2.columns, "Please include index, time, or frame (named 'time') in the line_2 DataFrame"
    assert order>=1, "Gradient order must larger or equal to 1"
    
    ## convert frame to time
    if "frame" in line_1.columns:
        line_1["time"] = line_1["frame"] * EYE_SAMPLE_TIME * time_scale
    if "frame" in line_2.columns:
        line_2["time"] = line_2["frame"] * EYE_SAMPLE_TIME * time_scale

    ## create gradient column
    line_1_grad_col, line_2_grad_col = [], []
    for col in line_1.columns:
        if col == "time": continue
        line_1[f"{col}.grad"] = line_1[col]
        line_1_grad_col.append(f"{col}.grad")
    for col in line_2.columns:
        if col == "time": continue
        line_2[f"{col}.grad"] = line_2[col]
        line_2_grad_col.append(f"{col}.grad")
    
    ## compute the gradient
    for _ in range(0, order):
        for col in line_1_grad_col:
            line_1[col] = np.gradient(line_1[col], line_1["time"])
        for col in line_2_grad_col:
            line_2[col] = np.gradient(line_2[col], line_2["time"])

    return compute_dtw(line_1[line_1_grad_col], line_2[line_2_grad_col], scale_to_percentage=scale_to_percentage)


def single_video_res(people_fea_res, video_id):
    res = {}
    for _p, person_fea in people_fea_res.items():
        try:
            for _r, _fea in person_fea[video_id].items():
                res[f"{_p}-{_r}"] = _fea
        except:
            continue
    return res


def single_person_res(people_fea_res, people_id):
    res = {}
    for _v, video_fea in people_fea_res[people_id].items():
        try:
            for _r, _fea in video_fea.items():
                res[f"{_v}-{_r}"] = _fea
        except:
            continue
    return res


def single_person_rounds(people_rounds_res, people_id):
    res = {}
    for _v, video_rounds in people_rounds_res[people_id].items():
        try:
            for _r, _fea in video_rounds.items():
                res[f"{_v}-{_r}"] = _fea
        except:
            continue
    return res


def single_person_match_rounds(people_match_rounds_res, people_id):
    res = {}
    for _v, video_rounds in people_match_rounds_res[people_id].items():
        try:
            for _r, _fea in video_rounds.items():
                res[f"{_v}-{_r}"] = _fea
        except:
            continue
    return res

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

def compute_eye_move(ball_line:pd.DataFrame):
    _indeices = ball_line.index.to_list()
    _dist = []
    for i in range(1, len(_indeices)):
        _dist.append(
                np.sqrt(
                        (ball_line.loc[_indeices[i], "Screen.x"] - ball_line.loc[_indeices[i-1], "Screen.x"])**2 + (ball_line.loc[_indeices[i],"Screen.y"] - ball_line.loc[_indeices[i-1],"Screen.y"])**2
                    )
            )
    return np.sum(_dist)