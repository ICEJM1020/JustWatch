import os
import pandas as pd
import numpy as np

from config import *



def fetch_data(dir_path:str, file_list:list, drop_list:list=[]):
    
    def fetch_eye_data(_raw_eye_data:str):
        _eye_data_rows = _raw_eye_data.split(";")
        eye_data = {}   
        names = _eye_data_rows[0].split(" ")
        for idx, row in enumerate(_eye_data_rows):
            if idx == 0 : continue

            cur_data = row.split(" ")
            try:
                eye_data[idx] = {
                    names[0] : FILL_NAN if cur_data[0]=="NaN" else float(cur_data[0]),
                    names[1] : FILL_NAN if cur_data[0]=="NaN" else float(cur_data[1]),
                    names[2] : FILL_NAN if cur_data[0]=="NaN" else float(cur_data[2]),
                    names[3] : FILL_NAN if cur_data[0]=="NaN" else float(cur_data[3]),
                }
            except:
                eye_data[idx] = {
                    names[0] : FILL_NAN,
                    names[1] : FILL_NAN,
                    names[2] : FILL_NAN,
                    names[3] : FILL_NAN,
                }

        return eye_data


    _all_data = {}
    for file in file_list:
        if file in drop_list: continue
        _single_person_data_df = pd.read_csv(os.path.join(dir_path, file))
        _single_person_data_dict = {}
        _videos = {}

        for _, row in _single_person_data_df.iterrows():
            _eye_data = fetch_eye_data(row["EyeData"])
            v_id = row["videoName"]
            if v_id in _single_person_data_dict.keys(): 
                _videos[v_id] += 1
                v_id = v_id + f"_{_videos[v_id]}"
            else:
                _videos[v_id] = 1
            _single_person_data_dict[v_id] = _eye_data
        
        _all_data[file.split(".")[0]] = _single_person_data_dict
    
    return _all_data


def fetch_trajectory(dir_path:str):
    res = {}

    p_traj = pd.read_csv(os.path.join(dir_path, "pingpang.csv"))
    w_traj = pd.read_csv(os.path.join(dir_path, "tennis.csv"))
    _temp_all = pd.concat([p_traj, w_traj], axis=0)
    _temp_all["position"] = _temp_all["position"].apply(lambda x : x.replace(" ", ""))

    for v_id, v_df in _temp_all.groupby(by="video_name"):

        _x = v_df["position"].apply(lambda x : x.split(",")[0][1:]).to_numpy(dtype=np.int16)
        _y = v_df["position"].apply(lambda x : x.split(",")[1][:-1]).to_numpy(dtype=np.int16)

        # reverse the y-axis
        _y = (_y - VIDEO_SIZE[1]) * -1

        # scale and shift
        _x = (_x / VIDEO_SIZE[0]) * SCREEN_SIZE[0] - (SCREEN_SIZE[0] // 2)
        _y = (_y / VIDEO_SIZE[1]) * SCREEN_SIZE[1] - (SCREEN_SIZE[1] // 2)

        _f = v_df["frame_number"].to_numpy(dtype=np.int16)
        _r = v_df["round"].to_numpy(dtype=np.int16)
        _temp_df = pd.DataFrame(
            np.array([_f, _x, _y, _r])
        ).T
        _temp_df.index = range(1, _f.size+1)
        _temp_df.columns = ["frame", "Ball.x", "Ball.y", "round"]

        res[v_id] = _temp_df.to_dict()

    return res



