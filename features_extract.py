""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-26
""" 

import pandas as pd
import multiprocessing as mp
import os
import json
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import time

from config import *
from load_data import fetch_data, fetch_trajectory, fetch_player_box
from Extractor.Extractor import extract_features, modify_features


def extract_person(_person_dict:dict):
    start_time = time.time()
    _person=_person_dict[0]
    _person_data=_person_dict[1]["data"]
    _ball_data=_person_dict[1]["ball_data"]
    _player_box_data=_person_dict[1]["player_box_data"]

    _person_fea = {}
    _person_match_rounds = {}
    _person_rounds = {}

    # if not _person == "24071512_AD": 
    #     return None

    for _video in _person_data.keys():
        # if not _video == "p7": continue
        
        res = extract_features(
            data=_person_data[_video], 
            ball_data=_ball_data[_video.split("_")[0]],
            player_box_data=_player_box_data[_video.split("_")[0]],
            scale_raw_data=False,
            dtw_mode="fast",
            dtw_th=9999,
            dist_th=58
        )
        
        _person_fea[_video] = {}
        _person_fea[_video]["All"] = {}
        _person_fea[_video]["All"].update(res["attention_fea"])
        _person_fea[_video]["All"].update({"MatchRoundRatio" : len(res["match_rounds"]) / len(res["rounds"])})
        _person_rounds[_video] = res["rounds"]
        
        if res["saccade_fea"]:
            _person_fea[_video].update(res["saccade_fea"])
            _person_match_rounds[_video] = res["match_rounds"]

    fea_res = {}
    for _v, video_fea in _person_fea.items():
        try:
            for _r, _fea in video_fea.items():
                fea_res[f"{_v}-{_r}"] = _fea
        except:
            continue
    
    out_dir = f"output/{_person}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
 
    pd.DataFrame(fea_res).T.to_csv(f"{out_dir}/features.csv")
    with open(f"{out_dir}/rounds.json", "w") as f:
        json.dump(_person_rounds, f)
    with open(f"{out_dir}/match_rounds.json", "w") as f:
        json.dump(_person_match_rounds, f)

    print(_person, time.time()-start_time)
    return _person


def modify_person(_person_dict:dict):
    start_time = time.time()
    _person=_person_dict[0]
    _person_data=_person_dict[1]["data"]
    _ball_data=_person_dict[1]["ball_data"]

    _person_fea = {}
    _person_match_rounds = {}
    
    try:
        with open(os.path.join(FEA_DIR, f"{_person}/rounds.json"), "r") as f:
            people_rounds = json.load(f)
    except:
        return False

    for _video in _person_data.keys():
        # if not _video == "p10_0_0_1": continue

        if _video in people_rounds.keys():
            res = modify_features(
                    data=_person_data[_video], 
                    ball_data=_ball_data[_video.split("_")[0]],
                    rounds=people_rounds[_video],
                    dist_th=58
                )
        
            _person_fea[_video] = {}
            _person_fea[_video]["All"] = {}
            _person_fea[_video]["All"].update(res["attention_fea"])
            _person_fea[_video]["All"].update({"MatchRoundRatio" : len(res["match_rounds"]) / len(res["rounds"])})
            
            if res["saccade_fea"]:
                _person_fea[_video].update(res["saccade_fea"])
            if res["match_rounds"]:
                _person_match_rounds[_video] = res["match_rounds"]
        else:
            print(f"{_person} no match round in {_video}")

    fea_res = {}
    for _v, video_fea in _person_fea.items():
        try:
            for _r, _fea in video_fea.items():
                fea_res[f"{_v}-{_r}"] = _fea
        except:
            continue
    new_fea_df = pd.DataFrame(fea_res).T

    out_dir = f"{FEA_DIR}/{_person}"
    ori_fea_df = pd.read_csv(f"{out_dir}/features.csv", index_col=0)
    drop_cols = list(filter(lambda x: x in new_fea_df.columns, ori_fea_df.columns))
    ori_fea_df.drop(drop_cols, axis=1, inplace=True)
    # fea_df = pd.merge(left=ori_fea_df, right=new_fea_df, how="left", left_index=True, right_index=True)
    fea_df = pd.merge(left=new_fea_df, right=ori_fea_df, how="left", left_index=True, right_index=True)
    fea_df.to_csv(f"{out_dir}/featuresNew.csv")

    with open(f"{out_dir}/match_rounds.json", "w") as f:
        json.dump(_person_match_rounds, f)

    print(_person, time.time()-start_time)
    return _person


if __name__ == "__main__":
    if not os.path.exists("output"):
        os.mkdir("output")

    file_list = os.listdir(DATA_DIR)
    file_list = list(filter(lambda x: "AD" in x, file_list))
    # drop_list = ['pingpang.csv', 'tennis.csv', '.DS_Store', 'ControlGroupInfo.xlsx',]
    drop_list = [i+".csv" for i in os.listdir("output")]

    all_data = fetch_data(DATA_DIR, file_list, drop_list=drop_list)
    ball_data = fetch_trajectory(DATA_DIR)
    player_box_data = fetch_player_box(os.path.join(DATA_DIR, "PlayerDetectionRes"))

    data = {}
    for _p in all_data.keys():
        data[_p] = {
            "data" : all_data[_p],
            "ball_data" : deepcopy(ball_data),
            "player_box_data" : deepcopy(player_box_data)
        }
        
    print("Extract Features from: {}".format(list(data.keys())))

    all_people_fea = {}
    all_people_rounds = {}
    all_people_match_rounds = {}

    # Using a pool of processes
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.map(extract_person, data.items())
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(modify_person, data.items())
    # for _d in data.items():
    #     # if _d[0] == "24090918_AD": 
    #     modify_person(_d)

