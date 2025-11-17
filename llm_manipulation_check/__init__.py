import pandas as pd, numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.stats import pearsonr,kendalltau,spearmanr, ttest_rel, zscore
from statsmodels.stats.inter_rater import fleiss_kappa 
from statsmodels.stats import inter_rater as irr
import json
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import random
import re
import krippendorff as kd
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity

def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5