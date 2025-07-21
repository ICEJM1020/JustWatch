import numpy as np

""" 
For extracting features: 
""" 
DATA_DIR = "/Users/timberzhang/Documents/Project/2024-JustWatch/Data"
# DATA_DIR = "/Users/timberzhang/Documents/Project/2024-JustWatch/EmoData"

# DATA_DIR = "E:\\Code\\JustWacth\\Data"
# DATA_DIR = "E:\\Code\\JustWacth\\EmoData"

FILL_NAN = np.nan
CLUSTER_MIN_EPS = 15
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
# FEA_DIR = "E:\\Code\\JustWacth\\Code\\output"
FEA_DIR = "/Users/timberzhang/Documents/Project/2024-JustWatch/IJHCS/OutputRecord/20250506"


NUM_CLASSES=3
VIDEO_TYPES=["p", "1", "N_L"]

# [ball (w, p), session ([0,1,2,3]), number (f,b,all), twice (0,1,2), stimul_type (r,a,all)]
VIDEO_TYPE_LIST={
    ## Five main
    "pingpang_no" : ["p", [0], "all", "0", "all"],
    # "pingpang_ra" : ["p", [3], "f", "0", "r"],
    "pingpang_rs" : ["p", [1], "all", "0", "r"],
    # "pingpang_aa" : ["p", [3], "f", "0", "a"],
    "pingpang_as" : ["p", [1], "all", "0", "a"],
    
    "wangqiu_no" : ["w", [0], "all", "0", "all"],
    # "wangqiu_ra" : ["w", [3], "f", "0", "r"],
    "wangqiu_rs" : ["w", [1], "all", "0", "r"],
    # "wangqiu_aa" : ["w", [3], "f", "0", "a"],
    "wangqiu_as" : ["w", [1], "all", "0", "a"],

	# # BEFORE and AFTER shining
	# "pingpang_1C": ["p", [0], "f", "0", "all"],
	# "pingpang_2C": ["p", [2], "f", "0", "all"],
	# "wangqiu_1C": ["w", [0], "f", "0", "all"],
	# "wangqiu_2C": ["w", [2], "f", "0", "all"],
    
	# # Watch Again
	# "pingpang_watch": ["p", [0], "f", "0", "all"],
	# "pingpang_again": ["p", [0], "b", "0", "all"],
	# "wangqiu_watch": ["w", [0], "f", "0", "all"],
	# "wangqiu_again": ["w", [0], "b", "0", "all"],

    # # Watch Same Again
	# "pingpang_firstseen": ["p", [0], "all", "1", "all"],
	# "pingpang_multiseen": ["p", [0], "all", "2", "all"],
	# "wangqiu_firstseen": ["w", [0], "f", "1", "all"],
	# "wangqiu_multiseen": ["w", [0], "f", "2", "all"],

	# YES or NO label
	# "pingpang_label_N": ["p", [0], "all", "0"],
	# "wangqiu_label_N": ["w", [0], "all", "0"],

	# "pingpang_label_Y": ["p", [1], "all", "0"],
	# "wangqiu_label_Y": ["w", [1], "all", "0"],


	# different CONTENT in different LABEL
	# "pingpang": ["p", [0, 1, 2, 3], "all", "0"],
	# "wangqiu": ["w", [0, 1, 2, 3], "all", "0"],

	# "pingpang_noarrow": ["p", [0, 1, 2], "all", "0"],
	# "wangqiu_noarrow": ["w", [0, 1, 2], "all", "0"],

	# "pingpang_nolabel": ["p", [0, 2], "all", "0"],
	# "wangqiu_nolabel": ["w", [0, 2], "all", "0"],
	
	# "pingpang_shining": ["p", [1], "all", "0"],
	# "wangqiu_shining": ["w", [1], "all", "0"],

	# "pingpang_arrow": ["p", [3], "all", "0"],
	# "wangqiu_arrow": ["w", [3], "all", "0"],
	
	
	# BEFORE and AFTER shining
	# "pingpang_1C": ["p", [0], "all", "0"],
	# "pingpang_2C": ["p", [2], "all", "0"],
	# "wangqiu_1C": ["w", [0], "all", "0"],
	# "wangqiu_2C": ["w", [2], "all", "0"],
	
	# different LABEL in different CONTENT
	# "pingpang_shinin_content": ["p", [1], "f", "0"],
	# "pingpang_arrow_content": ["p", [3], "f", "0"],

	# "wangqiu_shining_content": ["w", [1], "f", "0"],
	# "wangqiu_arrow_content": ["w", [3], "f", "0"],

    # # first seen vs multiple seen
	# "pingpang_firstseen": ["p", [0,2], "all", "1", "all"],
	# "pingpang_multiseen": ["p", [0,2], "all", "2", "all"],
	# "wangqiu_firstseen": ["w", [0,2], "f", "1", "all"],
	# "wangqiu_multiseen": ["w", [0,2], "f", "2", "all"],
}



DROP_LIST = [
    # "24082904_AD",
    # "24082203_AD",
    "24090917_AD",
	"24092002_AD",
	"24092009_AD",
	"24090308_AD",
	"24091022_AD",
	"24090310_AD",
	"24092209_AD",
	"24092207_AD",
	"24092111_AD",
	"24092102_AD",
	"24092110_AD",
	"24082803_AD",
	"24090412_AD",
	]

WHOLE_FEA_LIST = ["MatchRoundRatio",]
# whole_fea_list = ["Player1AttentionRatio","Player2AttentionRatio","Player1MinToCircle","Player2MinToCircle","MatchRoundRatio",]
SACCADE_FEA_LIST = ["SaccadeSpeed_Mean","SaccadeSpeed_Max","SaccadeSpeed_Std","SaccadeAngel_Mean","SaccadeAngel_Max","SaccadeAngel_Std","SaccadeDelay","SaccadeDelayPercent","TrajectoryDTW"]
FEA_LIST = ["MatchRoundRatio","SaccadeSpeed_Mean","SaccadeSpeed_Max","SaccadeSpeed_Std","Amplitude","UnitAmplitude","SaccadeDelay","SaccadeDelayPercent","TrajDTW","TrajDTWPerBallMove","TrajDTWPerEyeMove","DirecAngle"]

"""
33 24071011_AD
10 24082904_AD
19 24071618_AD
22 24070907_AD

24090917_AD
24092002_AD
24092009_AD
24090308_AD
24091022_AD
24090310_AD
24092209_AD
24092207_AD
24092111_AD
24092102_AD
24092110_AD

"""

"""
'24090310_AD', 
'24092009_AD', 
'24092002_AD', 
'24092111_AD',
'24092110_AD'
'24092102_AD',
'24090310_AD',
'24092209_AD', 
'24092207_AD'

'24092208_AD', '24092209_AD', '24092207_AD', '24092111_AD',
       '24092102_AD', '24092110_AD', '24090412_AD'
"""

"""
HC
24092107_AD +
24072023_AD
24071009_AD + 
24092012_AD
24092008_AD +
24070907_AD + 
24071011_AD + 
24072926_AD +
24071721_AD
24092010_AD
24092105_AD +

MCI
24090308_AD +
24090917_AD ++
24082803_AD
24092002_AD ++
24082302_AD +
24090915_AD
24092005_AD


MMAD
24083007_AD +
24092007_AD +
24090914_AD +
24090919_AD
24092104_AD +
24092009_AD ++
24092103_AD
"""