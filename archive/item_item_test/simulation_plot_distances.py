#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:32:49 2019

@author: dharmesh
"""

import matplotlib.pyplot as plt
import numpy as np
#all_ave_dist
plt.figure()
plt.semilogx(min_playtime,all_ave_dist_bpr,'o')
plt.semilogx(min_playtime,all_ave_dist_warp,'o')
plt.legend(['BPR','WARP'])
plt.grid()
plt.xlabel('Min Logarithmic Hours Played')
plt.ylabel('Average Similarity Distance ')
plt.title('Average Similarity Distance of Games for Min Logarithmic Hours Played')