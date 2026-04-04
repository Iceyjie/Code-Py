#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 09:10:02 2025

@author: zhoubingjie
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parameter as para
import matplotlib.image as mpimg
import seaborn as sns

def plotGroup(data, classify_name):
    group_data = data.groupby(classify_name)
    valid_groups = [(name, group) for name, group in group_data]
    n_rows, n_cols = 3, 3
    plots_per_page = n_rows * n_cols
    for i in range(0, len(valid_groups), plots_per_page):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5), dpi=500)  
        axes = axes.flatten() #change 2D array to 1D array
        for j, (grouped_name, grouped_data) in enumerate(valid_groups[i:i + plots_per_page]):
            ax = axes[j]
            ax.hist(grouped_data[para.arrival_delay], bins=50, color='skyblue', edgecolor='black')
            median_value = grouped_data[para.arrival_delay].median() 
            ax.set_title(f"#{grouped_name}={len(grouped_data)}", fontsize=5)
            ax.tick_params(axis='both', which='major', labelsize=5)
            # ax.axvline(median_value, color='r', linestyle='-', linewidth=0.5)
            ax.set_xlim(-24, 24)
        # delete blank subfigure
        for k in range(j + 1, len(axes)):
            fig.delaxes(axes[k])
        plt.tight_layout()
        plt.savefig(os.path.join(para.figure_path, f'fig-{classify_name}_{i}'))
        plt.show()

def plotArrivalDelay(data):
    print('--- here is plotting the arrival delay ---')
    custom_bins = list(range(-24, 25, 1))   # [-24,-18,...,20,22,24]
    plt.figure(figsize=(6, 5), dpi=600)
    # 返回 counts 和 bins，便于标注
    counts, bins, patches = plt.hist(data[para.arrival_delay], bins=custom_bins, edgecolor='black')
    # 添加红色竖直线 x=0
    plt.axvline(x=0, color="red", linestyle="-", linewidth=1.5)
    # 给每个柱子加上数字标签
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            plt.text((left + right)/2, count, str(int(count)),
                     ha='center', va='bottom', fontsize=7, rotation=90, color="blue")
    plt.title("ATA minus ETA (Hour)")
    plt.xlabel("Arrival Delay (Hour)")
    plt.savefig(os.path.join(para.figure_path, 'fig-arrival_delay'))
    plt.show()

def plotPerformance(results):
    all_image_names = []
    num_labels = 10 # number of labels
    step = max(1, len(results) // num_labels)  
    results['date'] = pd.to_datetime(results['date']).dt.date
    for index in ['R2_train', 'R2_test', 'MAE', 'RMSE']:
        plt.figure(figsize=(10, 6), dpi = 500)
        plt.plot(results.index, results[index], label=index, color='blue')
        plt.xticks(results.index[::step], results['date'][::step], rotation=35)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        # plt.ylim(-0.02, 0.002)
        plt.legend()
        image_name = f'{para.figure_path}{index}.jpg'
        all_image_names.append(image_name)
        plt.savefig(image_name, dpi = 500)
        plt.show()
    print('--- plot r2 succesfully ---')
    # combine all figures
    fig, ax = plt.subplots(2, 2, figsize=(7, 5), dpi = 500)  
    for i, path in enumerate(all_image_names):
        img = mpimg.imread(path)  
        row, col = divmod(i, 2)  
        ax[row, col].imshow(img)  
        ax[row, col].axis('off')  
    plt.tight_layout()
    plt.savefig(f'{para.figure_path}combine.jpg', dpi = 500)
    plt.show()
    
def plotContour(zs, ss, train, zhat0, Zhat, Q):
    xx = np.array([0, 2])
    zp = zhat0 + xx[:, None]@Zhat.T
    plt.scatter(train['x'], train['z1'], linewidth=1.5, color='none', edgecolor='b')
    plt.scatter(train['x'], train['z2'], linewidth=1.5, color='none', edgecolor='r')
    plt.plot(xx, zp[:, 0], color='b')
    plt.plot(xx, zp[:, 1], color='r')
    plt.show()
    
    resid = zs - zhat0 - ss@Zhat.T
    plt.scatter(resid[:, 0], resid[:, 1], color='none', edgecolor='k')
    
    p1, p2 = 2, 2
    xlim = [-0.8, 1.3]
    ylim = [-2.5, 4.2]
    n1, n2 = 150, 150
    z1 = np.linspace(xlim[0], xlim[1], n1)
    z2 = np.linspace(ylim[0], ylim[1], n2)
    zz1, zz2 = np.meshgrid(z1, z2)
    zz = np.concatenate((zz1.flatten()[:, None],
                         zz2.flatten()[:, None]), axis=1)
    
    values = np.linalg.norm(zz@Q.T, ord=p1, axis=1).reshape(n2, n1)
    ct = plt.contour(zz1, zz2, (1/2)*(values)**p2, levels=[1], colors=['b'])
    plt.xlabel('z1', fontsize=14)
    plt.ylabel('z2', fontsize=14)
    plt.grid()
    plt.show()
    
def visualizeTrajectory(data):
    plt.figure(figsize=(10, 8))
    plt.plot(data["LON"], data["LAT"], color="blue", linestyle="-", marker="o", markersize=5)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ship Trajectory")
    plt.grid(True)
    plt.gca().set_aspect('equal')  # 保持经纬度比例一致
    plt.tight_layout()
    plt.show()
    
def plotGap(summary_prediction, summary_model, label_shiptype, label_figure):
    # 你的数据
    gap = summary_prediction["Gap(hour)"]
    
    # 自定义区间，比如 -20 到 20，步长 2
    custom_bins = list(range(-24, 25, 1))   # [-20,-18,...,20,22,24]
    
    plt.figure(figsize=(6, 5), dpi=600)
    
    # 返回 counts 和 bins，便于标注
    counts, bins, patches = plt.hist(gap, bins=custom_bins, edgecolor='black')
    
    # 添加红色竖直线 x=0
    plt.axvline(x=0, color="red", linestyle="-", linewidth=1.5)
    
    # 给每个柱子加上数字标签
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            plt.text((left + right)/2, count, str(int(count)),
                     ha='center', va='bottom', fontsize=7, rotation=90, color="blue")
    percent = round((summary_prediction["Gap(hour)"] >= 0).mean() * 100, 2)
    
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    # plt.suptitle(f"{label_shiptype} ({summary_model.iloc[0]["Count"]})", fontsize=14)
    # plt.title(f"Coef: {summary_model.iloc[0]["Coef"]}, Intercept(h): {summary_model.iloc[0]["Intercept(h)"]}, Percentage(>0): {percent}%", fontsize=10)
    plt.savefig(os.path.join(para.figure_path, f'fig-Gap-{label_figure}'))
    plt.show()

def plotRandomGap(summary_prediction, summary_model, label_figure):
    # 你的数据
    gap = summary_prediction["Gap(hour)"]
    
    # 自定义区间，比如 -20 到 20，步长 2
    custom_bins = list(range(-24, 25, 1))   # [-20,-18,...,20,22,24]
    
    plt.figure(figsize=(6, 5), dpi=600)
    
    # 返回 counts 和 bins，便于标注
    counts, bins, patches = plt.hist(gap, bins=custom_bins, edgecolor='black')
    
    # 添加红色竖直线 x=0
    plt.axvline(x=0, color="red", linestyle="-", linewidth=1.5)
    
    # 给每个柱子加上数字标签
    for count, left, right in zip(counts, bins[:-1], bins[1:]):
        if count > 0:
            plt.text((left + right)/2, count, str(int(count)),
                     ha='center', va='bottom', fontsize=7, rotation=90, color="blue")
    percent = round((summary_prediction["Gap(hour)"] >= 0).mean() * 100, 2)
    
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    # plt.title(f"Percentage(>0): {percent}%, R2_train: {summary_model.iloc[0]["R2_train"]}, R2_test: {summary_model.iloc[0]["R2_test"]}, MAE_test:{summary_model.iloc[0]["MAE"]}, RMSE_test:{summary_model.iloc[0]["RMSE"]}", fontsize=10)
    plt.savefig(os.path.join(para.figure_path, f'fig-Gap-{label_figure}'))
    plt.show()

def plotTwoHist(data_1, data_2, column, month):
    plt.figure(figsize=(8, 6), dpi=500)
    plt.hist(data_1[column].dropna(), bins=30, alpha=0.6, label="early", edgecolor="black")
    plt.hist(data_2[column].dropna(), bins=30, alpha=0.6, label="late", edgecolor="black")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"{month}-th month")
    plt.legend()
    plt.show()

def plotHist(data, column_name):
    data[column_name].dropna().hist(bins=30, edgecolor="black", figsize=(8, 6))
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column_name}")
    plt.xlim(0, 100)
    plt.show()
    
def plotlastPort(data, last_port):
    g = data[data[para.due_last_port] == last_port]
    plotHist(g, para.arrival_delay)
    