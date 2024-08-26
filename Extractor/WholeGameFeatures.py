""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-26
""" 
import pandas as pd

from config import *
from Extractor.utils import interplate_and_align


def judge_inbox(row):
    TOL = 0
    _in_x = (row["Screen.x"] <= row["RightBottom.x"]+TOL) and (row["Screen.x"] >= row["LeftUp.x"]-TOL)
    _in_y = (row["Screen.y"] <= row["LeftUp.y"]+TOL) and (row["Screen.y"] >= row["RightBottom.y"]-TOL)
    return 1 if (_in_x and _in_y) else 0

def compute_inbox_dist(row):
    circle_x = (row["RightBottom.x"] + row["LeftUp.x"]) / 2
    circle_y = (row["LeftUp.y"] + row["RightBottom.y"]) / 2
    return np.sqrt((row["Screen.x"] - circle_x)**2 + (row["Screen.y"] - circle_y)**2) * VR_SCALE 


def extract_features_whole(eye_data:pd.DataFrame, ball_data_df:pd.DataFrame, player_box_data:dict):
    res = {}
    
    p1_bbox_df = pd.DataFrame(player_box_data["Player-1"])
    p2_bbox_df = pd.DataFrame(player_box_data["Player-2"])

    eye_data["frame"] = eye_data.index
    # aligned_df = interplate_and_align(eye_data, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    aligned_p1_df = interplate_and_align(eye_data, p1_bbox_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    aligned_p2_df = interplate_and_align(eye_data, p2_bbox_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    aligned_p1_df["inbox"] = aligned_p1_df.apply(judge_inbox, axis=1)
    aligned_p2_df["inbox"] = aligned_p2_df.apply(judge_inbox, axis=1)
    aligned_p1_df["box_dist"] = aligned_p1_df.apply(compute_inbox_dist, axis=1)
    aligned_p2_df["box_dist"] = aligned_p2_df.apply(compute_inbox_dist, axis=1)
    
    res["Player1AttentionRatio"] = aligned_p1_df["inbox"].sum() / aligned_p1_df.shape[0]
    res["Player2AttentionRatio"] = aligned_p2_df["inbox"].sum() / aligned_p2_df.shape[0]
    res["Player1MinToCircle"] = aligned_p1_df["box_dist"].min()
    res["Player2MinToCircle"] = aligned_p2_df["box_dist"].min()

    return res