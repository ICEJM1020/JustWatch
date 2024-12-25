from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu, wilcoxon, f_oneway, kruskal

def TTest(_input, _col, label_col):
    group1 = _input[_input.loc[:, label_col] == 0].loc[:, _col]
    group2 = _input[_input.loc[:, label_col] == 1].loc[:, _col]
    t_statistic, p_value = ttest_ind(group1, group2)
    return p_value


def UTest(_input, _col, label_col):
    group1 = _input[_input.loc[:, label_col] == 0].loc[:, _col]
    group2 = _input[_input.loc[:, label_col] == 1].loc[:, _col]
    u_statistic, p_value = mannwhitneyu(group1, group2)
    return p_value


def AnovaTest(_input, _col, label_col):
    groups = []
    for i in _input[label_col].unique():
        groups.append(_input[_input.loc[:, label_col] == i].loc[:, _col])
    u_statistic, p_value = f_oneway(*groups)
    return p_value

def KWTest(_input, _col, label_col):
    groups = []
    for i in _input[label_col].unique():
        groups.append(_input[_input.loc[:, label_col] == i].loc[:, _col])

    u_statistic, p_value = kruskal(*groups)
    return p_value


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