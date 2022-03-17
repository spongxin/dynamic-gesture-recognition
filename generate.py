from matplotlib import pyplot as plt
import multiprocessing
import pandas as pd
import numpy as np
import threading
import logging
import detect
import time
import json
import os


class Task(threading.Thread):
    def __init__(self, filename):
        super(Task, self).__init__(daemon=True)
        self.filename = filename

    def run(self):
        try:
            start = time.perf_counter()
            logging.info(f"【S】 {self.filename} 文件开始处理.")
            self.generate()
            finish = time.perf_counter()
            logging.info(f"【D】 {self.filename} 文件保存完毕, 共计耗时%.1f ms." % ((finish-start)*1000))
        except Exception as err:
            logging.error(f"【E】 {self.filename} 出现异常：{err.__str__()}")

    def generate(self):
        frame_map = detect.FrameMap(os.path.join(os.getcwd(), 'dataset', 'LSA64', self.filename))
        coordinates, _ = frame_map.frames2coordinates(frame_map.video2frames())
        features = [[[dot.x, dot.y, dot.z] if dot else [0] * 3 for dot in dots] for dots in coordinates]
        plt.figure(figsize=(7, 6), dpi=100)
        matrix = np.array(features, dtype='float32') * 255
        plt.imshow(matrix.astype(np.uint8))
        plt.savefig(os.path.join(os.getcwd(), 'dataset', 'PNG', self.filename[:-4] + '.png'))
        pd.DataFrame(
            {'feature': [json.dumps(features[line]) for line in range(len(features))]}
        ).to_csv(os.path.join(os.getcwd(), 'dataset', 'CSV', self.filename[:-4] + '.csv'))


def execute(files: list):
    logging.getLogger().setLevel(logging.WARNING)
    start = time.perf_counter()
    tasks = [Task(f) for f in files]
    for task in tasks:
        task.start()
        task.join()
    finish = time.perf_counter()
    logging.warning("【D】进程执行完毕, 共计耗时%.3f s." % (finish-start))


if __name__ == '__main__':
    processes = 4
    pool = multiprocessing.Pool(processes=processes)
    targets = np.array_split(os.listdir(os.path.join(os.getcwd(), 'dataset', 'LSA64')), processes)
    for i in range(processes):
        pool.apply_async(execute, args=(targets[i],))
    pool.close()
    pool.join()
