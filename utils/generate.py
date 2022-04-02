from resource import Solution
import multiprocessing
import numpy as np
import logging
import time
import os


class Task:
    def __init__(self, filename):
        self.filename = filename

    def run(self):
        try:
            start = time.perf_counter()
            logging.info(f"[S] {self.filename}")
            self.generate()
            finish = time.perf_counter()
            logging.info(f"[D] {self.filename} costs %.1f ms." % ((finish-start)*1000))
        except Exception as err:
            logging.error(f"[E] {self.filename}: {err.__str__()}")

    def generate(self):
        solution = Solution(model='VGGNet12', interval=2)
        solution.process(os.path.join(os.getcwd(), '../dataset', 'LSA64', self.filename))
        coordinates, index = solution.coords, solution.index
        np.savez(file=os.path.join(os.getcwd(), '../dataset', 'NPZ', self.filename[:-4]), x=index, y=coordinates)


def execute(files: list, idx: int):
    logging.getLogger().setLevel(logging.INFO)
    start = time.perf_counter()
    tasks = [Task(f) for f in files]
    for task in tasks:
        task.run()
    finish = time.perf_counter()
    logging.warning("[PD %d]process finish, costs %.3f s." % (idx, finish-start))


if __name__ == '__main__':
    processes = 4
    pool = multiprocessing.Pool(processes=processes)
    targets = np.array_split(os.listdir(os.path.join(os.getcwd(), '../dataset', 'LSA64')), processes)
    for i in range(processes):
        pool.apply_async(execute, args=(targets[i], i))
    pool.close()
    pool.join()
