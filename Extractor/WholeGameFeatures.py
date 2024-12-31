""" 
Description: 
Author: Xucheng(Timber) Zhang
Date: 2024-08-26
""" 
import pandas as pd

from config import *
from Extractor.utils import interplate_and_align


""" 
Player Attention
""" 

def judge_inbox(row):
    TOL = 0
    _in_x = (row["Screen.x"] <= row["RightBottom.x"]+TOL) and (row["Screen.x"] >= row["LeftUp.x"]-TOL)
    _in_y = (row["Screen.y"] <= row["LeftUp.y"]+TOL) and (row["Screen.y"] >= row["RightBottom.y"]-TOL)
    return 1 if (_in_x and _in_y) else 0

def compute_inbox_dist(row):
    circle_x = (row["RightBottom.x"] + row["LeftUp.x"]) / 2
    circle_y = (row["LeftUp.y"] + row["RightBottom.y"]) / 2
    return np.sqrt((row["Screen.x"] - circle_x)**2 + (row["Screen.y"] - circle_y)**2) * VR_SCALE 


def extract_features_players(eye_data:pd.DataFrame, player_box_data:dict):
    res = {}
    
    p1_bbox_df = pd.DataFrame(player_box_data["Player-1"])
    p2_bbox_df = pd.DataFrame(player_box_data["Player-2"])

    eye_data["frame"] = eye_data.index
    # aligned_df = interplate_and_align(eye_data, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    aligned_p1_df = interplate_and_align(eye_data, p1_bbox_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)
    aligned_p2_df = interplate_and_align(eye_data, p2_bbox_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    aligned_p1_df["inbox"] = aligned_p1_df.apply(judge_inbox, axis=1)
    aligned_p2_df["inbox"] = aligned_p2_df.apply(judge_inbox, axis=1)
    aligned_p1_df["box_dist"] = aligned_p1_df.apply(compute_inbox_dist, axis=1)
    aligned_p2_df["box_dist"] = aligned_p2_df.apply(compute_inbox_dist, axis=1)
    
    res["Player1AttentionRatio"] = aligned_p1_df["inbox"].sum() / aligned_p1_df.shape[0]
    res["Player2AttentionRatio"] = aligned_p2_df["inbox"].sum() / aligned_p2_df.shape[0]
    res["Player1MinToCircle"] = aligned_p1_df["box_dist"].min()
    res["Player2MinToCircle"] = aligned_p2_df["box_dist"].min()

    return res


""" 
Gaze Distribution
""" 
def add_circle_to_distribution(grid, center_x, center_y, radius):
    y_indices, x_indices = np.ogrid[:grid.shape[0], :grid.shape[1]]
    distance = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)
    mask = distance <= radius
    grid[mask] += 1


def add_line_to_distribution(grid, point1, point2, radius):
    x1, y1 = point1
    x2, y2 = point2

    num_steps = int(max(abs(x2 - x1), abs(y2 - y1)) * 2)

    x_values = np.linspace(x1, x2, num_steps)
    y_values = np.linspace(y1, y2, num_steps)

    for x, y in zip(x_values, y_values):
        add_circle_to_distribution(grid, int(x), int(y), radius)


def compute_distmap(points, radius=15):
    screen_distribution = np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0]))

    last = None
    for x, y in points:
        if not last:
            last = (x+(SCREEN_SIZE[0]//2), y+(SCREEN_SIZE[1]//2))
        else:
            add_line_to_distribution(
                screen_distribution, 
                last, 
                (x+(SCREEN_SIZE[0]//2), y+(SCREEN_SIZE[1]//2)),
                radius)
            last = (x+(SCREEN_SIZE[0]//2), y+(SCREEN_SIZE[1]//2))

    return screen_distribution


def calculate_entropy(distribution):
    """Calculate the entropy of a distribution."""
    # Normalize to create a probability distribution
    prob_dist = distribution / np.sum(distribution)
    # Avoid log(0) by masking zero probabilities
    prob_dist = prob_dist[prob_dist > 0]
    entropy = -np.sum(prob_dist * np.log(prob_dist))
    return entropy

def calculate_kl_divergence(distribution_p, distribution_q):
    """
    Calculate the KL divergence between two distributions.
    distribution_p => target distribution
    destribution_q => comparison distribution
    """
    # Normalize both distributions
    prob_p = distribution_p / np.sum(distribution_p)
    prob_q = distribution_q / np.sum(distribution_q)
    prob_p = np.clip(prob_p, a_min=1e-10, a_max=1)
    prob_q = np.clip(prob_q, a_min=1e-10, a_max=1)
    # Avoid log(0) and division by zero by masking zero probabilities
    mask = (prob_p > 0) & (prob_q > 0)
    kl_divergence = np.sum(prob_p[mask] * np.log(prob_p[mask] / prob_q[mask]))
    return kl_divergence


def extract_features_dist(eye_data:pd.DataFrame, ball_data_df:pd.DataFrame):
    eye_data["frame"] = eye_data.index
    aligned_df = interplate_and_align(eye_data, ball_data_df, EYE_SAMPLE_RATE, VIDEO_FPS, convert_dist=False)

    res = {}
    eye_dist = compute_distmap(aligned_df.loc[:,["Screen.x", "Screen.y"]].to_numpy())
    ball_dist = compute_distmap(aligned_df.loc[:,["Ball.x", "Ball.y"]].to_numpy())
    res["EyeDist"] = eye_dist
    res["BallDist"] = ball_dist

    res["GazeEntropy"] = calculate_entropy(eye_dist)
    res["KL-Divergence"] = calculate_kl_divergence(ball_dist, eye_dist)
    
    return res
