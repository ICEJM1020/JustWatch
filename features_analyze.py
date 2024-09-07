""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-09-05
""" 
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import shapiro
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt

from config import *
from utils import *
""" 
video_type: 
    [] => all
    p => pingpang
    w => tennis
    1 => first watch
    2 => mutiple watch
    N => no label
    R_S => random shining
    A_S => all shining
    R_A => random arrow
    A_A => all arrow
""" 
def StatsFeatures(all_people_fea:dict, all_people_stat:pd.DataFrame, fea_list:list, video_types:list[str], num_classes=3):
    if num_classes==3:
        diag_map ={
            "HC" : 0,
            "MCI" : 1,
            "mildAD" : 2,
            "moderateAD" : 2,
        }
    else:
        diag_map ={
            "HC" : 0,
            "MCI" : 1,
            "mildAD" : 1,
            "moderateAD" : 1,
        } 

    res={}
    for _p in people_list:
        video_list = all_people_fea[_p].index.to_list()
        _label_video_list = []
        for _v_t in video_types:
            if _v_t=="p":
                _v_ids = list(filter(lambda x:x.startswith("p"), all_people_fea[_p].index))
            elif _v_t=="w":
                _v_ids = list(filter(lambda x:x.startswith("w"), all_people_fea[_p].index))
            elif _v_t=="1":
                _v_ids = list(filter(lambda x: not ((x.split("-")[0].endswith("_2")) or (x.split("-")[0].endswith("_3")) or (x.split("-")[0].endswith("_4"))), all_people_fea[_p].index))
            elif _v_t=="2":
                _v_ids = list(filter(lambda x: (x.split("-")[0].endswith("_2")) or (x.split("-")[0].endswith("_3")) or (x.split("-")[0].endswith("_4")), all_people_fea[_p].index))
            elif _v_t=="N_L":
                _v_ids = list(filter(lambda x: not (("_S" in x.split("-")[0]) or ("_A" in x.split("-")[0])), all_people_fea[_p].index))
            elif _v_t=="R_S":
                _label_video_list += list(filter(lambda x: "R_S" in x.split("-")[0], all_people_fea[_p].index))
            elif _v_t=="A_S":
                _label_video_list += list(filter(lambda x: "A_S" in x.split("-")[0], all_people_fea[_p].index))
            elif _v_t=="R_A":
                _label_video_list += list(filter(lambda x: "R_A" in x.split("-")[0], all_people_fea[_p].index))
            elif _v_t=="A_A":
                _label_video_list += list(filter(lambda x: "A_A" in x.split("-")[0], all_people_fea[_p].index))
            else:
                _v_ids = list(all_people_fea[_p].index)
            
            video_list = list(set(video_list).intersection(set(_v_ids)))
        if _label_video_list:
            video_list = list(set(video_list).intersection(set(_label_video_list)))

        temp_df:pd.DataFrame = all_people_fea[_p].loc[video_list, fea_list]
        mean_val = temp_df.mean(skipna=True)
        mean_val.index = [i+"_Mean" for i in mean_val.index]
        std_val = temp_df.std(skipna=True)
        std_val.index = [i+"_Std" for i in std_val.index]
        max_val = temp_df.max(skipna=True)
        max_val.index = [i+"_Max" for i in max_val.index]
        min_val = temp_df.min(skipna=True)
        min_val.index = [i+"_Min" for i in min_val.index]
        
        try:
            _diag = all_people_stat[all_people_stat["PicoFile"]==_p]["Label"].values[0]
        except:
            continue
        else:
            res[_p] = {
                **mean_val.to_dict(),
                **std_val.to_dict(),
                **max_val.to_dict(),
                **min_val.to_dict(),
            }
            res[_p]["label"] = diag_map[_diag]

    res = pd.DataFrame(res).T.astype(np.float32)
    if num_classes==3:
        res["diag"] = res["label"].map({
            0:"HC",
            1:"MCI",
            2:"Mild-AD",
        })
    else:
        res["diag"] = res["label"].map({
            0:"HC",
            1:"PS",
        })
        
    return res


def sig_test(features_df:pd.DataFrame, ):
    if_norm = {}
    para_pvalue = {}

    for key in features_df.columns:
        if key=="diag" or key=="label": continue
        SW_test = shapiro(features_df.loc[:,key])[1]
        # print("Group {}'s \tShapiro—Wilk test P-Value: \t{}".format(key,SW_test))
        if_norm[key] = (True if SW_test >0.05 else False)

    if features_df["label"].value_counts().shape[0]==2:
        for key in features_df.columns:
            if key=="diag" or key=="label": continue
            if if_norm[key]:
                p_value = TTest(features_df, key, label_col='label')
            else:
                p_value = UTest(features_df, key, label_col='label')
            para_pvalue[key] = {
                "p-value": p_value,
                "if_para": True if p_value <= 0.05 else False
            }
    else:
        for key in features_df.columns:
            if key=="diag" or key=="label": continue
            if if_norm[key]:
                p_value = AnovaTest(features_df, key, label_col='label')
            else:
                p_value = KWTest(features_df, key, label_col='label')
            para_pvalue[key] = {
                "p-value": p_value,
                "if_para": True if p_value <= 0.05 else False
            }
    return pd.DataFrame(para_pvalue).T.sort_values(by="p-value")


def draw_violin(features_df:pd.DataFrame, pt_res:pd.DataFrame, video_name):
    custom_palette = ["#00712D", "#FF9100", "#D5ED9F", "#FFFBE6"]
    if not os.path.exists("pics"):
        os.mkdir("pics")
    
    pass_feas = pt_res
    _ncols = 4
    _nrows = (pass_feas.shape[0]//_ncols) if (pass_feas.shape[0]%_ncols)==0 else (pass_feas.shape[0]//_ncols)+1

    f, axs = plt.subplots(nrows=_nrows, ncols=_ncols, figsize=(3*_ncols, 4*_nrows), dpi=200)
    for idx, col in enumerate(pass_feas.index):
        # col = "MatchRoundRatio_Mean"

        # Set up the matplotlib figure
        # Draw a violinplot with a narrower bandwidth than the default
        temp_ax = axs[idx//_ncols, idx%_ncols]
        sns.violinplot(
                data=features_df.loc[:, [col, "diag"]], 
                palette=custom_palette,
                alpha=0.7,
                bw_adjust=.5, 
                cut=1, 
                linewidth=1, 
                x="diag", 
                y=col, 
                inner="point", 
                ax=temp_ax
            )
        # sns.boxplot(
        #         data=features_df.loc[:, [col, "diag"]], 
        #         palette=custom_palette,
        #         linewidth=1, 
        #         x="diag", 
        #         y=col, 
        #         ax=temp_ax
        #     )
        
        temp_ax.set_xlabel("")
        temp_ax.set_ylabel("")
        temp_ax.set_title(col)
        # sns.despine(left=True, bottom=True)
    plt.tight_layout()

    plt.savefig("pics/{}.png".format(video_name), dpi=200)
    plt.close()



if __name__ == "__main__":
    if not os.path.exists("ana_output"):
        os.mkdir("ana_output")

    people_list = os.listdir(FEA_DIR)
    people_list = list(filter(lambda x: "." not in x, people_list))

    people_fea = {}
    for _p in people_list:
        people_fea[_p] = pd.read_csv(os.path.join(f"{FEA_DIR}/{_p}", "features.csv"), index_col=0)

    people_stat = pd.read_excel(os.path.join(DATA_DIR, "ParticipantsInfo.xlsx"), )

    for video_name,video_types in VIDEO_TYPE_LIST.items():
        features_df = StatsFeatures(
            all_people_fea=people_fea,
            all_people_stat=people_stat,
            fea_list=WHOLE_FEA_LIST+SACCADE_FEA_LIST,
            video_types=video_types,
            num_classes=NUM_CLASSES
        )
        features_df.to_csv(os.path.join("ana_output", f"{video_name}.csv"))

        pt_res = sig_test(features_df)
        pt_res.to_csv(os.path.join("ana_output", f"sigfea_{video_name}.csv"))

        draw_violin(features_df=features_df, pt_res=pt_res, video_name=video_name)







