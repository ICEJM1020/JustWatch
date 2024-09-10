""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-07
""" 

import pandas as pd
from scipy.spatial import KDTree
from sklearn.preprocessing import minmax_scale


from config import *
from Extractor.utils import interplate_and_align, interplate_enlarge, compute_dtw, compute_gradient_dtw


def count_points_in_radius(ball_df:pd.DataFrame, eye_df:pd.DataFrame, radius):
    # Build a k-d tree from the points DataFrame
    points_tree = KDTree(eye_df[['Screen.x', 'Screen.y']])
    
    count_list = []
    
    for i, trajectory_point in ball_df.iterrows():
        # Query the k-d tree for points within the radius
        indices = points_tree.query_ball_point([trajectory_point['Ball.x'], trajectory_point['Ball.y']], r=radius)
        count_list += indices
    
    return list(set(count_list))


def find_match_round(eye_data_df:pd.DataFrame, ball_data_df:pd.DataFrame, radius=15, inner_percent=0.9):
    res = []

    eye_data_df["frame"] = eye_data_df.index
    assert "round" in ball_data_df.columns, "Missing 'round' in ball trajectory file"

    aligned_df = interplate_and_align(eye_data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    for _round in aligned_df["round"].value_counts().index:
        temp = aligned_df[aligned_df["round"]==_round]
        inner_counts = count_points_in_radius(temp.loc[:, ["Ball.x", "Ball.y"]], temp.loc[:, ["Screen.x", "Screen.y"]], radius=radius)

        # check_df = temp.iloc[inner_counts, :]
        # max_x, min_x = check_df["Screen.x"].max(), check_df["Screen.x"].min()
        # max_y, min_y = check_df["Screen.y"].max(), check_df["Screen.y"].min()
        # dis = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
        dis = radius

        if len(inner_counts) / temp.shape[0] > inner_percent and dis >= radius:
            res.append(_round)

    return res


def find_match_round_hit(eye_data_df:pd.DataFrame, ball_data_df:pd.DataFrame, time_range=10, dist=250):
    res ={}

    eye_data_df["frame"] = eye_data_df.index
    assert "round" in ball_data_df.columns, "Missing 'round' in ball trajectory file"

    aligned_df = interplate_and_align(eye_data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    aligned_df["hit"] = aligned_df["round"] - aligned_df["round"].shift(1, fill_value=1)
    hits_indices = aligned_df[aligned_df["hit"]==1].index

    for idx, _hit in enumerate(hits_indices):
        index_range = [_hit-time_range, _hit+time_range]
        temp = aligned_df.loc[index_range[0]:index_range[1], :]
        eye_max_dist = np.sqrt((temp["Screen.x"].max() - temp["Screen.x"].min())**2 + (temp["Screen.y"].max() - temp["Screen.y"].min())**2)

        if eye_max_dist >= dist:
            res[aligned_df.loc[_hit, "round"]] = list(temp.index)

    return res


def find_match_round_dtw(eye_data_df:pd.DataFrame, ball_data_df:pd.DataFrame, order=0, scale_raw_data=False):
    res = {}
    dtw_res = {}

    eye_data_df["frame"] = eye_data_df.index
    assert "round" in ball_data_df.columns, "Missing 'round' in ball trajectory file"

    aligned_df = interplate_and_align(eye_data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    if scale_raw_data:
        aligned_df.loc[:, ["Screen.x", "Ball.x", "Screen.y", "Ball.y"]] = minmax_scale(aligned_df.loc[:, ["Screen.x", "Ball.x", "Screen.y", "Ball.y"]], axis=0)

    for _round in aligned_df["round"].value_counts(sort=False).index:
        temp = aligned_df[aligned_df["round"]==_round]

        match_indice = None
        min_dtw = np.inf

        try:
            for idx in temp.index:
                for window_size in range(MIN_DTW_WINDOW, temp.shape[0]):
                    # temp_window = aligned_df.loc[_shift_idx:_shift_idx+window_size, :]
                    temp_window = aligned_df.loc[ idx : idx + window_size, :]
                    if temp_window.shape[0] < MIN_DTW_WINDOW: continue
                    _indices = temp_window.index

                    if temp_window.shape[0] < temp.shape[0]:
                        temp_window = interplate_enlarge(temp_window, size=temp.shape[0])

                    if order==0:
                        _dtw = compute_dtw(
                            line_1 = temp.loc[:, ["Ball.x", "Ball.y"]],
                            line_2 = temp_window.loc[:, ["Screen.x", "Screen.y"]],
                            scale_to_percentage=False
                        )
                    elif order>0:
                        _dtw = compute_gradient_dtw(
                                line_1=temp.loc[:, ["frame", "Ball.x", "Ball.y"]],
                                line_2=temp_window.loc[:, ["frame", "Screen.x", "Screen.y"]],
                                order=order,
                                scale_to_percentage=False
                        )
                    else:
                        raise Exception("Order of gradient must be positive")
                    
                    if _dtw < min_dtw:
                        min_dtw = _dtw
                        match_indice = _indices
        except:
            dtw_res[_round] = np.inf
            res[_round] = None
        else:
            dtw_res[_round] = min_dtw
            res[_round] = match_indice.to_list()

    return res, dtw_res


def find_match_round_dtw_kmp(eye_data_df:pd.DataFrame, ball_data_df:pd.DataFrame, order=0, scale_raw_data=False):
    res = {}
    dtw_res = {}

    eye_data_df["frame"] = eye_data_df.index
    assert "round" in ball_data_df.columns, "Missing 'round' in ball trajectory file"

    aligned_df = interplate_and_align(eye_data_df, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    if scale_raw_data:
        aligned_df.loc[:, ["Screen.x", "Ball.x", "Screen.y", "Ball.y"]] = minmax_scale(aligned_df.loc[:, ["Screen.x", "Ball.x", "Screen.y", "Ball.y"]], axis=0)

    for _round in aligned_df["round"].value_counts(sort=False).index:
        temp = aligned_df[aligned_df["round"]==_round]

        match_indice = None
        min_dtw = np.inf
        _shift_idx = temp.index[0]
        _last_idx = temp.index[0] + MIN_DTW_WINDOW
        # for window_size in range(MIN_DTW_WINDOW, temp.shape[0]):
        max_idx = list(aligned_df[(aligned_df["round"]==_round+1) | (aligned_df["round"]==_round)].index)[-1]

        try:
            while _last_idx < max_idx:
                # temp_window = aligned_df.loc[_shift_idx:_shift_idx+window_size, :]
                temp_window = aligned_df.loc[ _shift_idx : _last_idx, :]
                _indices = temp_window.index

                if temp_window.shape[0] < temp.shape[0]:
                    temp_window = interplate_enlarge(temp_window, size=temp.shape[0])

                if temp_window.shape[0] < MIN_DTW_WINDOW: continue

                if order==0:
                    _dtw = compute_dtw(
                        line_1 = temp.loc[:, ["Ball.x", "Ball.y"]],
                        line_2 = temp_window.loc[:, ["Screen.x", "Screen.y"]],
                        scale_to_percentage=False
                    )
                elif order>0:
                    _dtw = compute_gradient_dtw(
                            line_1=temp.loc[:, ["frame", "Ball.x", "Ball.y"]],
                            line_2=temp_window.loc[:, ["frame", "Screen.x", "Screen.y"]],
                            order=order,
                            scale_to_percentage=False
                    )
                else:
                    raise Exception("Order of gradient must be positive")
                
                _last_idx += 1
                if _dtw < min_dtw:
                    min_dtw = _dtw
                    match_indice = _indices
                else:
                    _shift_idx = _last_idx
                    _last_idx = _last_idx + MIN_DTW_WINDOW
        except:
            dtw_res[_round] = np.inf
            res[_round] = None
        else:
            dtw_res[_round] = min_dtw
            res[_round] = match_indice.to_list()

    return res, dtw_res


""" 
Description: Test the match or the match hyper-parameters
""" 
def test_match_hit(data, video_id):
    data_df = pd.DataFrame(data).T
    data_df.ffill(inplace=True)
    data_df.bfill(inplace=True)
    
    if "_" in video_id:
        video_id = video_id.split("_")[0]

    res = {}
    for r in [5, 7, 10, 15]:
        for d in [100,200,300,350,400]:
            count_res = find_match_round_hit(data_df.loc[:, ["Screen.x", "Screen.y"]], video_id, time_range=r, dist=d)
            res[f"Time Range {r}-Dist {d}"] = len(count_res)
            # res[f"Radius {r} match round"] = count_res
    return res


def test_match_round(data, video_id):
    data_df = pd.DataFrame(data).T
    data_df.ffill(inplace=True)
    data_df.bfill(inplace=True)
    
    if "_" in video_id:
        video_id = video_id.split("_")[0]

    res = {}
    for r in [25, 50, 75, 100]:
        count_res = find_match_round(data_df.loc[:, ["Screen.x", "Screen.y"]], video_id, radius=r)
        res[f"Cover Radius {r}"] = len(count_res)
    return res


def test_match(all_data, method=""):
    all_people_match = {}
    for _person in all_data.keys():
        _person_fea = {}

        for _video in all_data[_person].keys():
            if method=="hit":
                _person_fea[_video] = test_match_hit(all_data[_person][_video], _video)
            elif method=="round":
                _person_fea[_video] = test_match_round(all_data[_person][_video], _video)
            else:
                raise Exception("methods not support")

        all_people_match[_person] = _person_fea
    
    return all_people_match

