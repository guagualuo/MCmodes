import time


class Timer(object):
    def __init__(self, process_name=''):
        self.name = process_name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name} elapsed {time.time() - self.start} s.")

