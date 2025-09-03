class Logger:
    pass

def print0(s, master_process: bool):
    if master_process:
        print(s)

def log0(s, master_process: bool, console: bool = True):
    if console:
        print0(s, master_process)
    if master_process and logfile:
        with open(logfile, "a") as f:
            print(s, file=f)