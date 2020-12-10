import argparse
import os
import signal
import subprocess
import time
from queue import Empty, Queue
from threading import Thread

import numpy as np
import wandb

parser = argparse.ArgumentParser()

parser.add_argument("--wbname", type=str, required=True)
parser.add_argument("--wbproj", type=str, required=True)
parser.add_argument("--wbgroup", type=str, default=None)

args = parser.parse_args()

if args.wbgroup is None:
    wandb.init(project=args.wbproj, name=args.wbname, settings=wandb.Settings(_disable_stats=True))
else:
    wandb.init(project=args.wbproj, group=args.wbgroup, name=args.wbname, settings=wandb.Settings(_disable_stats=True))


def enqueue_output(out, queue):
    for line in iter(out.readline, b""):
        queue.put(line)
    out.close()


os.setpgrp()

clock_proc = subprocess.Popen("nvidia-smi dmon -s c", shell=True, stdout=subprocess.PIPE, bufsize=1)
clock_proc.daemon = True

time.sleep(0.5)

throttle_reasons = [
    "clocks_throttle_reasons.gpu_idle",
    "clocks_throttle_reasons.applications_clocks_setting",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sync_boost",
]
throttle_proc = subprocess.Popen(
    f"nvidia-smi --query-gpu=index,{','.join(throttle_reasons)} --format=csv,noheader --loop=1",
    shell=True,
    stdout=subprocess.PIPE,
    bufsize=1,
)
throttle_proc.daemon = True

# create queue that gets the output lines from both processes
q = Queue()
clock_thread = Thread(target=enqueue_output, args=(clock_proc.stdout, q))
clock_thread.daemon = True
thottle_thread = Thread(target=enqueue_output, args=(throttle_proc.stdout, q))
thottle_thread.daemon = True

clock_thread.start()
thottle_thread.start()

throttles = [[], []]
clocks = [[], []]
while clock_proc.poll() is None or not q.empty():
    try:
        line = q.get_nowait()
    except Empty:
        pass
    else:
        line = line.decode("utf-8").strip()
        if "#" in line:
            continue
        if "," in line:
            raw = line.split(",")
            gpu = int(raw[0])
            bits = [0 if "Not" in a else 1 for a in raw[1:]]
            throttles[gpu].append(bits)
            # print(gpu, bits)
        else:
            raw = line.split("  ")
            gpu = int(raw[0])
            clock = int(raw[-1])
            clocks[gpu].append(clock)
            # print(gpu, clock)

    if len(clocks[0]) > 30:
        try:
            throttles = np.array(throttles)
            clocks = np.array(clocks)
            log_dict = {}
            for gpu in [0, 1]:
                log_dict[f"gpu.{gpu}.clock.speed"] = np.mean(clocks[gpu])

                for r, reason in enumerate(throttle_reasons):
                    log_dict[f"gpu.{gpu}.{reason}"] = np.mean(throttles[gpu, :, r])

            print("\n".join([k.ljust(80) + str(v) for k, v in log_dict.items()]))
            wandb.log(log_dict)
        except:
            pass

        throttles = [[], []]
        clocks = [[], []]

os.kill(throttle_proc.pid, signal.SIGINT)
os.kill(clock_proc.pid, signal.SIGINT)
