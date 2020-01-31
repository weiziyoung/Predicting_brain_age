# -*- coding: utf-8 -*-
# @Time    : 2019/7/23 1:58 PM
# @Author  : weiziyang
# @FileName: download_n4_bias.py
# @Software: PyCharm
import os
from tqdm import tqdm
import requests

if __name__ == "__main__":
    google_category = ['Beijing', 'Atlanta', 'ICBM', 'Oxford', 'PaloAlto', 'Dallas', 'NewHaven', 'Berlin_Margulies',
                       'AnnArbor', 'Leipzig', 'SaintLouis', 'IXI', 'Pittsburgh', 'Leiden', ]
    ali_category = ['Baltimore', 'Bangor', 'Cambridge', 'NewYork', 'Orangeburg', 'Oulu', 'Queensland']
    google_prefix = 'https://storage.cloud.google.com/dissertation_wzy/n4_bias/n4_bias/'
    ali_prefix = 'https://wzy-zone.oss-cn-shanghai.aliyuncs.com/remote_disk/'
    error_list = []
    for category in google_category:
        if not os.path.exists(os.path.join('n4_bias', category)):
            os.mkdir(os.path.join('n4_bias', category))
        if category in google_category:
            prefix = google_prefix
        else:
            prefix = ali_prefix
        file_names = [each for each in os.listdir(os.path.join('compress', category)) if '.nii' in each]
        for name in tqdm(file_names):
            full_address = os.path.join(prefix, category, name)
            output_path = os.path.join('n4_bias', category, name)
            if os.path.exists(output_path):
                os.remove(output_path)
            # response = requests.get(full_address, stream=True)
            # if response.status_code == 200:
            #     with open(output_path, 'wb') as f:
            #         for chunk in response.iter_content(chunk_size=1024):
            #             if chunk:
            #                 f.write(chunk)
            #                 f.flush()
            # else:
            #     error_list.append((category, name))
            #     print('error', category, name)
        os.removedirs(os.path.join('n4_bias', category))
    print(error_list)
