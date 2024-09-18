""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-09-05
""" 
import os
import shutil
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy.stats import shapiro
import seaborn as sns
sns.set_theme(style="whitegrid")
import matplotlib.pyplot as plt        
from itertools import cycle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, auc, roc_curve, roc_auc_score, confusion_matrix

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
def StatsFeatures_V1(all_people_fea:dict, all_people_stat:pd.DataFrame, fea_list:list, video_types:list[str], num_classes=3):
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


def StatsFeatures(all_people_fea:dict, all_people_stat:pd.DataFrame, fea_list:list, session:list=[0], ball:str="p", number:str="all", stimul_type:str="all" ,twice=0, num_classes=3):
    assert (number in ["f", "b", "all"]), "input number like \"f\" first 2, \"b\" last 2, \"all\" all number"
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
    for _p in all_people_fea.keys():
        video_list = list(filter(lambda x:x.startswith(ball), all_people_fea[_p].index))

        _session_list = []
        for _v_s in session:
            _v_ids = list(filter(lambda x: x.split("-")[0].split("_")[-3]==str(_v_s), all_people_fea[_p].index))
            _session_list = list(set(_session_list).union(set(_v_ids)))
        video_list = list(set(video_list).intersection(set(_session_list)))

        _number_list = []
        if number=="f":
            _number_list = list(filter(lambda x: (x.split("-")[0].split("_")[-2]=="0") or (x.split("-")[0].split("_")[-2]=="1"), all_people_fea[_p].index))
        elif number=="b":
            _number_list = list(filter(lambda x: (x.split("-")[0].split("_")[-2]=="2") or (x.split("-")[0].split("_")[-2]=="3"), all_people_fea[_p].index))
        else:
            _number_list = video_list
        video_list = list(set(video_list).intersection(set(_number_list)))

        _stimul_list = []
        if stimul_type=="r":
            _stimul_list = list(filter(lambda x: ("R_A" in x.split("-")[0]) or ("R_S" in x.split("-")[0]), all_people_fea[_p].index))
        elif stimul_type=="a":
            _stimul_list = list(filter(lambda x: ("A_A" in x.split("-")[0]) or ("A_S" in x.split("-")[0]), all_people_fea[_p].index))
        else:
            _stimul_list = video_list
        video_list = list(set(video_list).intersection(set(_stimul_list)))

        __multi_list = []
        if twice==2:
            __multi_list = list(filter(lambda x: not x.split("-")[0].endswith("_1"), video_list))
        elif twice==1:
            __multi_list = list(filter(lambda x: x.split("-")[0].endswith("_1"), video_list))
        else:
            __multi_list = video_list
        video_list = list(set(video_list).intersection(set(__multi_list)))
            
        # print(_p)
        # print(video_list)
        # print(len(video_list))
        # print(set(["_".join(i.split("_")[:-3]) for i in video_list]))
        # print(len(set(["_".join(i.split("_")[:-3]) for i in video_list])))

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
            if not _diag=="-":
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
            2:"MMAD",
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


def draw_violin(features_df:pd.DataFrame, pt_res:pd.DataFrame, type_name):
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

    plt.savefig("pics/{}.png".format(type_name), dpi=200)
    plt.close()


def build_model(features_df, pt_res, num_classes, type_name):
    fea_list = pt_res[pt_res["if_para"]].index.to_list()
    final_result = {}
    if len(fea_list)==0: return {}

    rkf = RepeatedKFold(n_splits=5, n_repeats=20)
    X = features_df[fea_list]
    
    final_result = {}
    if num_classes==2:
        final_result['overall'] = {}
        final_result['overall']['acc'] = []
        final_result['overall']['pre'] = []
        final_result['overall']['recall'] = []
        final_result['overall']['f1'] = []
        final_result['overall']['fpr'] = []
        final_result['overall']['tpr'] = []
        final_result['overall']['thresholds'] = []
        final_result['overall']['auc'] = []
        final_result['overall']['test_index'] = []
        final_result['overall']['trues'] = []
        final_result['overall']['preds'] = []
        final_result['overall']['preds_porb'] = []

        for i, (train_index, test_index) in enumerate(rkf.split(X)):
            final_result['overall']['test_index'].append(test_index.tolist())

            # cls = RandomForestClassifier(20, max_depth=5)
            cls = RandomForestClassifier(n_estimators=5, max_depth=3)
            # cls = AdaBoostClassifier(n_estimators=5, learning_rate=0.75)
            # cls = SVC(C=0.75, kernel='linear')
            # cls = SVC(C=0.75, kernel='rbf')
            x_train, y_train = features_df.iloc[train_index][fea_list], features_df.iloc[train_index]['label']
            x_test, y_test = features_df.iloc[test_index][fea_list], features_df.iloc[test_index]['label']
            cls.fit(X=x_train, y=y_train)

            pred = cls.predict(x_test)
            final_result['overall']['acc'].append(accuracy_score(y_test, pred))
            final_result['overall']['pre'].append(precision_score(y_test, pred))
            final_result['overall']['recall'].append(recall_score(y_test, pred))
            final_result['overall']['f1'].append(f1_score(y_test, pred))
            fpr, tpr, thresholds = roc_curve(y_test, pred.tolist())
            final_result['overall']['fpr'].append(fpr.tolist())
            final_result['overall']['tpr'].append(tpr.tolist())
            final_result['overall']['thresholds'].append(thresholds.tolist())
            final_result['overall']['auc'].append(auc(fpr, tpr))

            final_result['overall']['trues'].append(y_test.to_list())
            final_result['overall']['preds'].append(pred.tolist())
            final_result['overall']['preds_porb'].append(cls.predict_proba(x_test).tolist())
    if num_classes==3:
        final_result['overall'] = {}
        final_result['overall']['acc'] = []
        final_result['overall']['micro_pre'] = []
        final_result['overall']['micro_recall'] = []
        final_result['overall']['micro_f1'] = []
        final_result['overall']['macro_pre'] = []
        final_result['overall']['macro_recall'] = []
        final_result['overall']['macro_f1'] = []
        final_result['overall']['test_index'] = []
        final_result['overall']['trues'] = []
        final_result['overall']['preds'] = []
        final_result['overall']['preds_porb'] = []

        for i, (train_index, test_index) in enumerate(rkf.split(X)):
            final_result['overall']['test_index'].append(test_index.tolist())

            cls = RandomForestClassifier(n_estimators=5, max_depth=5)
            # cls = AdaBoostClassifier(n_estimators=5, learning_rate=0.75)
            # cls = SVC(C=0.75, kernel='linear')
            # cls = SVC(C=0.75, kernel='rbf')
            x_train, y_train = features_df.iloc[train_index][fea_list], features_df.iloc[train_index]['label']
            x_test, y_test = features_df.iloc[test_index][fea_list], features_df.iloc[test_index]['label']
            cls.fit(X=x_train, y=y_train)

            pred = cls.predict(x_test)
            final_result['overall']['acc'].append(accuracy_score(y_test, pred))
            final_result['overall']['micro_pre'].append(precision_score(y_test, pred, average='micro'))
            final_result['overall']['micro_recall'].append(recall_score(y_test, pred, average='micro'))
            final_result['overall']['micro_f1'].append(f1_score(y_test, pred, average='micro'))
            final_result['overall']['macro_pre'].append(precision_score(y_test, pred, average='macro'))
            final_result['overall']['macro_recall'].append(recall_score(y_test, pred, average='macro'))
            final_result['overall']['macro_f1'].append(f1_score(y_test, pred, average='macro'))

            final_result['overall']['trues'].append(y_test.to_list())
            final_result['overall']['preds'].append(pred.tolist())
            final_result['overall']['preds_porb'].append(cls.predict_proba(x_test).tolist())

    stat = pd.DataFrame(final_result["overall"])

    best_acc=0
    best_indices = []
    for indices in np.array(stat.index).reshape(-1, 5):
        _acc = stat.iloc[indices.tolist(),:]["acc"].mean()
        if _acc > best_acc:
            best_acc = _acc
            best_indices = indices
    final_result["overall"] = stat.iloc[best_indices, :].to_dict('list')

    with open(os.path.join("ana_output", f"{type_name}_config.json"), "w") as f:
        json.dump(final_result, f)
    with open(os.path.join("ana_output", f"{type_name}.txt"), "w") as f:
        for key in final_result['overall'].keys():
            if key in ['fpr', 'tpr', 'thresholds', 'test_index', 'trues', 'preds', 'preds_porb']:
                continue
            f.write("{} : {}\n".format(key, np.mean(final_result['overall'][key])))

        if num_classes==3:
            trues = [item for sublist in final_result["overall"]["trues"] for item in sublist]
            preds_prob = [item for sublist in final_result["overall"]["preds_porb"] for item in sublist]

            f.write("Macro One-vs-Rest AUC : {}\n".format(roc_auc_score(trues, preds_prob, multi_class="ovr", average="macro")))
            f.write("Weighted One-vs-Rest AUC : {}\n".format(roc_auc_score(trues, preds_prob, multi_class="ovr", average="weighted")))
            f.write("Macro One-vs-One AUC : {}\n".format(roc_auc_score(trues, preds_prob, multi_class="ovo", average="macro")))
            f.write("Weighted One-vs-One AUC : {}\n".format(roc_auc_score(trues, preds_prob, multi_class="ovo", average="weighted")))
        else:
            trues = [item for sublist in final_result["overall"]["trues"] for item in sublist]
            preds_prob = [item[1] for sublist in final_result["overall"]["preds_porb"] for item in sublist]
            f.write("AUC : {}\n".format(roc_auc_score(trues, preds_prob)))

    return final_result


def draw_roc(final_result, num_classes, type_name):
    trues = [item for sublist in final_result["overall"]["trues"] for item in sublist]
    preds = [item for sublist in final_result["overall"]["preds"] for item in sublist]
    preds_prob = [item for sublist in final_result["overall"]["preds_porb"] for item in sublist]

    trues_np = np.array(trues, dtype=np.int16)
    trues_onehot = np.zeros((trues_np.size, trues_np.max() + 1))
    trues_onehot[np.arange(trues_np.size), trues_np] = 1

    preds_np = np.array(preds, dtype=np.int16)
    preds_onehot = np.zeros((preds_np.size, preds_np.max() + 1))
    preds_onehot[np.arange(preds_np.size), preds_np] = 1

    preds_prob_np = np.array(preds_prob)

    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(trues_onehot[:, i], preds_prob_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(trues_onehot.ravel(), preds_prob_np.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(9,9), dpi=200)
    if NUM_CLASSES==3:
        diease_map = {
                0:"HC",
                1:"MCI",
                2:"Mild-AD",
            }
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )
    else:
        diease_map = {
                0:"HC",
                1:"PG"
            }
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="Mean ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )
        
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(num_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(diease_map[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Area Under the Receiver (AUC) Operating Characteristic Curve (ROC)")
    plt.legend(loc="lower right")
    plt.savefig(f"pics/{type_name}_roc.png", dpi=200)


def draw_cm(final_result):
    trues = [item for sublist in final_result["overall"]["trues"] for item in sublist]
    preds = [item for sublist in final_result["overall"]["preds"] for item in sublist]
    f,ax = plt.subplots(figsize=(5,4), dpi=200)
    C2 = confusion_matrix(trues, preds, labels=[0,1,2] if NUM_CLASSES==3 else [0,1])
    # C2 = C2 / len(trues)
    C2 = C2 / C2.sum(axis=0)

    sns.heatmap(C2, annot=True, ax=ax, cmap="YlGn", linewidth=.8) 

    if NUM_CLASSES==3:
        ax.set_yticklabels(['HC', 'MCI', 'MMAD'])
        ax.set_xticklabels(['HC', 'MCI', 'MMAD'])
    else:
        ax.set_yticklabels(['HC', 'PG'])
        ax.set_xticklabels(['HC', 'PG'])

    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Target')
    plt.tight_layout()
    plt.savefig(f"pics/{type_name}_cm.png", dpi=200)


def fetch_overall(aim_ana_dir):
    groups_file = list(filter(lambda x: x.endswith(".txt"), os.listdir(aim_ana_dir)))
    res = {}
    for file in groups_file:
        groups_name = file.split(".")[0]
        _res = {}
        with open(os.path.join(aim_ana_dir, file), "r") as f:
            metrics_lines = f.readlines()
        for metrics in metrics_lines:
            _res[metrics.split(" : ")[0]] = metrics.split(" : ")[1][:-1]
        res[groups_name] = _res
    return pd.DataFrame(res, dtype=np.float32).T


if __name__ == "__main__":

    if not os.path.exists("ana_output"):
        os.mkdir("ana_output")
    if not os.path.exists("pics"):
        os.mkdir("pics")

    people_list = os.listdir(FEA_DIR)
    people_list = list(filter(lambda x: "." not in x, people_list))

    people_fea = {}
    for _p in people_list:
        if _p in DROP_LIST: continue
        try:
            people_fea[_p] = pd.read_csv(os.path.join(f"{FEA_DIR}/{_p}", "featuresNew.csv"), index_col=0)
        except:
            people_fea[_p] = pd.read_csv(os.path.join(f"{FEA_DIR}/{_p}", "features.csv"), index_col=0)

    people_stat = pd.read_excel(os.path.join(DATA_DIR, "ParticipantsInfo.xlsx"), )

    for type_name,video_types in VIDEO_TYPE_LIST.items():
        # features_df = StatsFeatures_V1(
        #     all_people_fea=people_fea,
        #     all_people_stat=people_stat,
        #     fea_list=WHOLE_FEA_LIST+SACCADE_FEA_LIST,
        #     video_types=video_types,
        #     num_classes=NUM_CLASSES
        # )
        features_df = StatsFeatures(
            all_people_fea=people_fea,
            all_people_stat=people_stat,
            fea_list=FEA_LIST,
            ball=video_types[0],
            session=video_types[1],
            number=video_types[2],
            twice=video_types[3],
            stimul_type=video_types[4],
            num_classes=NUM_CLASSES
        )
        features_df.to_csv(os.path.join("ana_output", f"{type_name}.csv"))

        pt_res = sig_test(features_df)
        pt_res.to_csv(os.path.join("ana_output", f"{type_name}_sigfea.csv"))

        draw_violin(
            features_df=features_df, 
            pt_res=pt_res, 
            type_name=type_name
            )

        model_res = build_model(
            features_df=features_df, 
            pt_res=pt_res, 
            num_classes=NUM_CLASSES, 
            type_name=type_name
            )
        if model_res:
            draw_cm(final_result=model_res)

            draw_roc(
                final_result=model_res,
                num_classes=NUM_CLASSES,
                type_name=type_name
                )

    final_out = f"Ana_{NUM_CLASSES}C"
    if os.path.exists(final_out):
        shutil.rmtree(final_out)
    os.mkdir(final_out)
    res = fetch_overall("ana_output")
    res.to_csv(os.path.join(final_out, "result.csv"))
    shutil.move('ana_output', f'Ana_{NUM_CLASSES}C/ana_output')
    shutil.move('pics', f'Ana_{NUM_CLASSES}C/pics')



