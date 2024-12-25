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
from Extractor.Extractor import extract_features_emo


def extract_person(_person_dict:dict):
    start_time = time.time()
    _person=_person_dict[0]
    _person_data=_person_dict[1]["data"]
    _ball_data=_person_dict[1]["ball_data"]
    _player_box_data=_person_dict[1]["player_box_data"]

    _person_fea = {}

    if not _person == "HC13_杨日": 
        return None

    for _video in _person_data.keys():
        if not _video == "p8_0_0_1": continue
        
        res = extract_features_emo(
                data=_person_data[_video], 
                ball_data=_ball_data[_video.split("_")[0]],
                player_box_data=_player_box_data[_video.split("_")[0]]
            )
        
        _person_fea[_video] = {}
        _person_fea[_video].update(res["attention_fea"])
        _person_fea[_video].update(res["distribution_fea"])
        _person_fea[_video].update(res["saccades_fea"])
 
    pd.DataFrame(_person_fea).T.to_csv(f"output/{_person}_features.csv")
    print(_person, time.time()-start_time)
    return _person


if __name__ == "__main__":
    if not os.path.exists("output"):
        os.mkdir("output")

    file_list = os.listdir(os.path.join(DATA_DIR, "Participant_Video_csv"))
    file_list = [f"Participant_Video_csv/{p}" for p in list(filter(lambda x: ".csv" in x, file_list))]
    # drop_list = ['pingpang.csv', 'tennis.csv', '.DS_Store', 'ControlGroupInfo.xlsx',]

    all_data = fetch_data(DATA_DIR, file_list, [])
    ball_data = fetch_trajectory(DATA_DIR)
    player_box_data = fetch_player_box(os.path.join(DATA_DIR, "PlayerDetectionRes"))

    data = {}
    for _p in all_data.keys():
        data[_p.split("/")[-1]] = {
            "data" : all_data[_p],
            "ball_data" : deepcopy(ball_data),
            "player_box_data" : deepcopy(player_box_data)
        }
        
    print("Extract Features from: {}".format(list(data.keys())))

    all_people_fea = {}
    all_people_rounds = {}
    all_people_match_rounds = {}

    # Using a pool of processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(extract_person, data.items())
    # for _d in data.items():
    #     # if _d[0] == "24090918_AD": 
    #     modify_person(_d)

