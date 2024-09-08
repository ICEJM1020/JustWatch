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
from Extractor.Extractor import extract_features


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
        break
    print("Extract Features from: {}".format(list(data.keys())))

    all_people_fea = {}
    all_people_rounds = {}
    all_people_match_rounds = {}

    # Using a pool of processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(extract_person, data.items())

