""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-26
""" 

import numpy as np
import pandas as pd

from config import *
from Extractor.RoundMatcher import find_match_round_dtw, find_match_round_dtw_kmp
from Extractor.utils import compute_stat, interplate_and_align, compute_dtw, max_circle_radius 
from Extractor.WholeGameFeatures import extract_features_whole
from Extractor.SaccadeFeatures import extract_saccade_features_lite, modify_saccade_features_lite, extract_saccade_features
from Extractor.TrajectoryFeatures import extract_trajectory_lite, modify_trajectory_lite, extract_trajectory, compute_eye_move, compute_two_traj_angle


def extract_features_round(rounds:dict, data_df:pd.DataFrame, ball_data_df:pd.DataFrame):
    data_df["frame"] = data_df.index
    aligned_df = interplate_and_align(data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    res = {}
    for round_index, round_indices in rounds.items():
        round_res = extract_saccade_features(
                round_index,
                aligned_df.loc[round_indices, ["Screen.x", "Screen.y"]],
                aligned_df.loc[:, ["Ball.x", "Ball.y", "round"]]
            )
        
        traj_res = extract_trajectory(
                round_index,
                aligned_df.loc[round_indices, ["Screen.x", "Screen.y"]],
                aligned_df.loc[:, ["Ball.x", "Ball.y", "round"]]
            )
        res[round_index] = {**round_res, **traj_res}
    return res


def threshold_find_match_round_dtw(eye_data:pd.DataFrame, ball_data_df:pd.DataFrame, order, scale_raw_data, mode="fast", dtw_th=1, dist_th=10):
    if mode=="fast":
        rounds, dtw_res = find_match_round_dtw_kmp(eye_data_df=eye_data, ball_data_df=ball_data_df, order=order, scale_raw_data=scale_raw_data)
    elif mode=="greedy":
        rounds, dtw_res = find_match_round_dtw(eye_data_df=eye_data, ball_data_df=ball_data_df, order=order, scale_raw_data=scale_raw_data)

    res = {}
    for _round, _round_index in rounds.items():
        if _round_index:
            if (max_circle_radius(eye_data.loc[_round_index, :]) * 2 >= dist_th) and (dtw_res[_round] <= dtw_th):
                res[_round] = _round_index

    return res, rounds


def extract_features(data, ball_data, player_box_data, dtw_mode="fast", scale_raw_data=True, dtw_th=1, dist_th=10):
    data_df = pd.DataFrame(data).T
    data_df.ffill(inplace=True)
    data_df.bfill(inplace=True)

    ball_data_df = pd.DataFrame(ball_data)
    
    # match_rounds = label_round_hit(data_df.loc[:, ["Screen.x", "Screen.y"]], video_id)

    match_rounds, rounds = threshold_find_match_round_dtw(data_df.copy(), ball_data_df.copy(), order=0, scale_raw_data=scale_raw_data, mode=dtw_mode, dtw_th=dtw_th, dist_th=dist_th)
    # match_rounds = find_match_round_hit(data_df.loc[:, ["Screen.x", "Screen.y"]], video_id, time_range=7, dist=300)
    saccade_features = extract_features_round(match_rounds, data_df.copy(), ball_data_df.copy())

    attention_features = extract_features_whole(data_df.copy(), ball_data_df.copy(), player_box_data)
    
    return {
        "match_rounds" : match_rounds, 
        "rounds": rounds,
        "saccade_fea": saccade_features,
        "attention_fea": attention_features
        }

    # saccade_features = extract_saccade_features(match_rounds, data_df.loc[:, ["Screen.x", "Screen.y"]], video_id)
    # trajectory_features = extract_trajectory(data_df.loc[:, ["Screen.x", "Screen.y"]], video_id, scale_to_percentage=True)

    # return {**saccade_features, **trajectory_features}


def modify_saccade_features_round(rounds:dict, data_df:pd.DataFrame, ball_data_df:pd.DataFrame):
    data_df["frame"] = data_df.index
    aligned_df = interplate_and_align(data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    res = {}
    for round_index, round_indices in rounds.items():
        round_res = modify_saccade_features_lite(
            float(round_index),
            aligned_df.loc[round_indices, ["Screen.x", "Screen.y"]],
            aligned_df.loc[:, ["Ball.x", "Ball.y", "round"]]
            )
        
        traj_res = modify_trajectory_lite(
            float(round_index),
            aligned_df.loc[round_indices, ["Screen.x", "Screen.y"]],
            aligned_df.loc[:, ["Ball.x", "Ball.y", "round"]]
            )
        res[round_index] = {**round_res, **traj_res}
    return res


def modify_features(data, ball_data, rounds, dist_th=10):
    data_df = pd.DataFrame(data).T
    data_df.ffill(inplace=True)
    data_df.bfill(inplace=True)

    ball_data_df = pd.DataFrame(ball_data)

    match_rounds = {}
    for _round, _round_index in rounds.items():
        if _round_index:
            temp_df = data_df.loc[_round_index, :].copy()
            temp_df["sub_x"] = temp_df["Screen.x"]-temp_df["Screen.x"].shift(-1)
            temp_df["sub_y"] = temp_df["Screen.y"]-temp_df["Screen.y"].shift(-1)
            temp_df["sub_x"] = np.round(temp_df["Screen.x"]-temp_df["Screen.x"].shift(-1))
            temp_df["sub_y"] = np.round(temp_df["Screen.y"]-temp_df["Screen.y"].shift(-1))
            temp_indices = temp_df.index.to_list()

            ## search for where brake the line
            break_indices = []
            for i in range(1, len(temp_indices)):
                if temp_df.loc[temp_indices[i], "sub_x"] * temp_df.loc[temp_indices[i-1], "sub_x"] < 0:
                    if temp_df.loc[temp_indices[i], "sub_y"] * temp_df.loc[temp_indices[i-1], "sub_y"] > 0:
                        break_indices.append(temp_indices[i])
                if temp_df.loc[temp_indices[i], "sub_y"] * temp_df.loc[temp_indices[i-1], "sub_y"] < 0:
                    break_indices.append(temp_indices[i])
            break_indices = list(set(break_indices))
            break_indices.sort()

            ## use max length of single direction line
            max_dist = -1
            _indices = []
            for idx, end_index in enumerate(break_indices):
                if idx==0:
                    start_index=temp_indices[0]
                else:
                    start_index=break_indices[idx-1]
                dist = compute_eye_move(temp_df.loc[start_index:end_index])
                if dist > max_dist:
                    max_dist = dist
                    _indices = temp_df.loc[start_index:end_index].index.to_list()

            if len(_indices)==0: continue
            # test if the line is longer enough
            max_dist = max_circle_radius(temp_df.loc[_indices, :]) * 2
            # test if the angle match
            angle = compute_two_traj_angle(
                eye_traj=temp_df.loc[_indices, ["Screen.x", "Screen.y"]].copy(),
                ball_traj=ball_data_df[ball_data_df["round"]==float(_round)].copy()
            )
            if max_dist >= dist_th and angle <= 60:
                match_rounds[_round] = _indices
            # if max_dist >= dist_th:
            #     match_rounds[_round] = _indices
    
    if match_rounds:
        saccade_features = modify_saccade_features_round(match_rounds, data_df.copy(), ball_data_df.copy())
    else:
        saccade_features = {}

    return {
            "match_rounds" : match_rounds, 
            "rounds": rounds,
            "saccade_fea": saccade_features,
            "attention_fea": {}
        }
