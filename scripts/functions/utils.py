#!/usr/bin/env python3

import sys
import os

# Logger that duplicates output to terminal and to file
# Not portable on Windows though...
class Logger(object):

    def __init__(self,logfile):
        import warnings
        warnings.filterwarnings("default")

        # Works but we loose colors in the terminal
        import subprocess
        self.tee = subprocess.Popen(["tee", logfile], stdin=subprocess.PIPE)
        os.dup2(self.tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(self.tee.stdin.fileno(), sys.stderr.fileno())

    def __del__(self):
        self.tee.stdin.close()
        sys.stdout.close()
        sys.stderr.close()


# Helper function for filenames
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'k', 'M', 'B', 'T'][magnitude])
