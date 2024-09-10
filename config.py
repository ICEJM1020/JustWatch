import numpy as np

""" 
For extracting features: 
""" 
DATA_DIR = "/Users/timberzhang/Documents/Documents/2024-JustWatch/Data"
# DATA_DIR = "E:\\JustWatchData\\Data"

FILL_NAN = np.nan
CLUSTER_MIN_EPS = 25
CLUSTER_MIN_SAMPLE = 5
MIN_DTW_WINDOW = 2

VR_SCALE = 0.001207812
VR_ZDIST = 1
EYE_SAMPLE_RATE = 50
EYE_SAMPLE_TIME = (1 / EYE_SAMPLE_RATE) * 1000
SCREEN_SIZE = [1280, 720]

VIDEO_SIZE = [1920, 1080]
VIDEO_FPS = 30


""" 
For analyzing data 
""" 
FEA_DIR = "/Users/timberzhang/Documents/Documents/2024-JustWatch/OutputRecord/GreedyNonScale"

NUM_CLASSES=3
VIDEO_TYPES=["p", "1", "N_L"]
# VIDEO_TYPE_LIST={
#     "pingpang_first_nolabel": ["p", "1", "N_L"],
#     "wangqiu_first_nolabel": ["w", "1", "N_L"],
#     "pingpang_multi_nolabel": ["p", "2", "N_L"],
#     "wangqiu_multi_nolabel": ["w", "2", "N_L"],
#     "pingpang_first_alllabel": ["p", "1", "R_S", "A_S", "R_A", "A_A"],
#     "wangqiu_first_alllabel": ["w", "1", "R_S", "A_S", "R_A", "A_A"],
#     "pingpang_first_shining": ["p", "1", "R_S", "A_S"],
#     "wangqiu_first_shining": ["w", "1", "R_S", "A_S"],
#     "pingpang_first_arrow": ["p", "1", "R_A", "A_A"],
#     "wangqiu_first_arrow": ["w", "1", "R_A", "A_A"],
#     "pingpang_first_random": ["p", "1", "R_A", "R_S"],
#     "wangqiu_first_random": ["w", "1", "A_S", "A_A"],
#     "pingpang_first_all": ["p", "1", "A_S", "A_R"],
#     "wangqiu_first_all": ["w", "1", "A_S", "A_R"],
#     "pingpang_first_random_arrow": ["p", "1", "R_A"],
#     "pingpang_first_all_arrow": ["p", "1", "A_A"],
#     "pingpang_first_random_shining": ["p", "1", "R_S"],
#     "pingpang_first_all_shining": ["p", "1", "A_S"],
#     "wangqiu_first_random_arrow": ["p", "1", "R_A"],
#     "wangqiu_first_all_arrow": ["p", "1", "A_A"],
#     "wangqiu_first_random_shining": ["p", "1", "R_S"],
#     "wangqiu_first_all_shining": ["p", "1", "A_S"],
# }

# [ball (w, p), session ([0,1,2,3]), number (f, b, all), twice (0,1,2)]
VIDEO_TYPE_LIST={
	
	# YES or NO label
	"pingpang_label_N": ["p", [0], "all", "0"],
	"wangqiu_label_N": ["w", [0], "all", "0"],

	"pingpang_label_Y": ["p", [1], "all", "0"],
	"wangqiu_label_Y": ["w", [1], "all", "0"],


	# different CONTENT in different LABEL
	"pingpang": ["p", [0, 1, 2, 3], "all", "0"],
	"wangqiu": ["w", [0, 1, 2, 3], "all", "0"],

	"pingpang_noarrow": ["p", [0, 1, 2], "all", "0"],
	"wangqiu_noarrow": ["w", [0, 1, 2], "all", "0"],

	"pingpang_nolabel": ["p", [0, 2], "all", "0"],
	"wangqiu_nolabel": ["w", [0, 2], "all", "0"],
	
	"pingpang_shining": ["p", [1], "all", "0"],
	"wangqiu_shining": ["w", [1], "all", "0"],

	"pingpang_arrow": ["p", [3], "all", "0"],
	"wangqiu_arrow": ["w", [3], "all", "0"],
	
	
	# BEFORE and AFTER shining
	"pingpang_1C": ["p", [0], "all", "0"],
	"pingpang_2C": ["p", [2], "all", "0"],
	"wangqiu_1C": ["w", [0], "all", "0"],
	"wangqiu_2C": ["w", [2], "all", "0"],
	

	# different LABEL in different CONTENT
	"pingpang_shinin_content": ["p", [1], "f", "0"],
	"pingpang_arrow_content": ["p", [3], "f", "0"],

	"wangqiu_shining_content": ["w", [1], "f", "0"],
	"wangqiu_arrow_content": ["w", [3], "f", "0"],

    # first seen vs multiple seen
	"pingpang_firstseen": ["p", [0,1,2,3], "all", "1"],
	"pingpang_multiseen": ["p", [0,1,2,3], "all", "2"],
	"wangqiu_firstseen": ["w", [0,1,2,3], "f", "1"],
	"wangqiu_multiseen": ["w", [0,1,2,3], "f", "2"],
}

WHOLE_FEA_LIST = ["MatchRoundRatio",]
# whole_fea_list = ["Player1AttentionRatio","Player2AttentionRatio","Player1MinToCircle","Player2MinToCircle","MatchRoundRatio",]
SACCADE_FEA_LIST = ["SaccadeSpeed_Mean","SaccadeSpeed_Max","SaccadeSpeed_Std","SaccadeAngel_Mean","SaccadeAngel_Max","SaccadeAngel_Std","SaccadeDelay","SaccadeDelayPercent","TrajectoryDTW"]