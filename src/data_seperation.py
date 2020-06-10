import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd
import shutil

data = pd.read_csv('train.csv')
for i in range(0, len(data)):
	from_ = '/media/razat_ag/Common/BE_PROJECT/aptos2019-blindness-detection/train_images/' + data.iloc[i][0] + '.png'
	to_ = '/media/razat_ag/Common/BE_PROJECT/aptos2019-blindness-detection/CLASSES/' + str(data.iloc[i][1]) + '/' + data.iloc[i][0] + '.png'
	shutil.copy(from_, to_)