#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 13:51:51 2025
@author: zhoubingjie
"""

import time
import re
import os
import pandas as pd
import func
import parameter as para
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def clean_data_dict(data_dict, keep_keys=None):
    """
    清洗爬取到的 data_dict：
    1. 遇到 key/value 中带斜杠 '/' 时，自动拆分
    2. 去掉单位，只保留数字
    3. 如果提供 keep_keys，只保留指定 key
    """
    cleaned = {}

    for key, val in data_dict.items():
        # 如果有 keep_keys 且 key 不在其中 → 跳过
        if keep_keys is not None and all(k not in key for k in keep_keys):
            continue

        val = str(val).strip()

        if "/" in key and "/" in val:
            sub_keys = [k.strip() for k in key.split("/")]
            sub_vals = [v.strip() for v in val.split("/")]

            for sk, sv in zip(sub_keys, sub_vals):
                if keep_keys is not None and all(k not in sk for k in keep_keys):
                    continue
                num = re.findall(r"[-+]?\d*\.?\d+", sv)
                if num:
                    cleaned[sk] = float(num[0])
        else:
            num = re.findall(r"[-+]?\d*\.?\d+", val)
            if num:
                cleaned[key] = float(num[0])

    return cleaned


def scrape_one_imo(driver, imo_number, keep_keys=None):
    """
    爬取单个 IMO 页面，返回清洗后的 dict
    """
    url = f"https://www.vesselfinder.com/vessels/details/{imo_number}"
    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//table"))
        )
    except:
        print(f"⚠️ IMO {imo_number} 表格加载超时")
        return {"IMO_input": imo_number}

    data_dict = {"IMO_input": imo_number}

    # ------- 抓第一个表格 -------
    rows = driver.find_elements(By.XPATH, "(//table)[1]//tr")
    for row in rows:
        cells = row.find_elements(By.XPATH, "./td")
        if len(cells) == 2:
            key = cells[0].text.strip()
            val = cells[1].text.strip()
            if key and val:
                data_dict[key] = val

    # ------- 抓第二个表格 -------
    rows = driver.find_elements(By.XPATH, "(//table)[2]//tr")
    for row in rows:
        cells = row.find_elements(By.XPATH, "./td")
        if len(cells) == 2:
            key = cells[0].text.strip()
            val = cells[1].text.strip()
            if key and val:
                data_dict[key] = val

    # ------- 单独抓 Gross Tonnage & Year -------
    try:
        data_dict['GT'] = driver.find_element(
            By.XPATH, "//td[contains(text(), 'Gross Tonnage')]/following-sibling::td"
        ).text
    except:
        pass

    try:
        data_dict['Year'] = driver.find_element(
            By.XPATH, "//td[contains(text(), 'Year of Build')]/following-sibling::td"
        ).text
    except:
        pass

    # ------- 清洗 -------
    return clean_data_dict(data_dict, keep_keys=keep_keys)


if __name__ == "__main__":
    raw_data = pd.read_csv(os.path.join(para.merge_path, para.merge_filename)) 
    data = raw_data[raw_data[para.arrived_ship_type] == "CONTAINER"] 
    data[para.in_port_imo_no] = data[para.in_port_imo_no].astype(str)
    imo_list = func.getGroupCounts(data, para.in_port_imo_no)[para.in_port_imo_no].tolist()
    keep_keys = ["IMO", "MMSI", "Length", "Beam", "Average", "Max Speed", "GT", "Year"]

    driver = webdriver.Chrome()
    results = []

    for i, imo in enumerate(imo_list, 1):
        print(f"--- is processing {i}/{len(imo_list)} IMO {imo} ---")
        data = scrape_one_imo(driver, imo, keep_keys=keep_keys)
        results.append(data)
        df_out = pd.DataFrame(results)
        df_out.to_csv("../Data/Web/ship_info_cleaned.csv", index=False, encoding="utf-8-sig")
    driver.quit()

