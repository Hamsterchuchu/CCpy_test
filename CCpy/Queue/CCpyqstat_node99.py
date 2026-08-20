"""
qstat implementation for node99 (parses SLURM `squeue` output).

cms2 uses PBS-style `qstat -f`, so its implementation is completely different.
-> see CCpyqstat_cms2.py.
At run time CCpyqstat.py picks one of the two according to $CCpy_SERVER.
"""
import os, sys, datetime
import getpass
import pandas as pd
import yaml
import warnings

warnings.filterwarnings("ignore")

# -- load config
try:
    CCpy_SCHEDULER_CONFIG = os.environ['CCpy_SCHEDULER_CONFIG']
    queue_info = yaml.safe_load(open(CCpy_SCHEDULER_CONFIG, 'r'))
except:
    print("Error: Check CCpy_SCHEDULER_CONFIG or YAML file.")
    quit()

class bcolors:
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'

# Add or reduce the spaces here to tune the terminal column spacing as you like!
C_ID    = '   ID'
C_JOB   = '           JOBNAME'
C_USER  = '     USER'
C_STAT  = '   STATUS'
C_START = '        START-TIME'
C_RUN   = '   RUN-TIME'
C_NODE  = '   QUEUE-NODE'
C_SLOTS = ' SLOTS'

def unify_time_format(time_str):
    """Unify Slurm time into 'D days, HH:MM:SS' or 'HH:MM:SS'"""
    if '-' in time_str:
        days, rest = time_str.split('-')
        return f"{days} days, {rest}"
    parts = time_str.split(':')
    if len(parts) == 2: # MM:SS -> 00:MM:SS
        return f"0:{parts[0]}:{parts[1]}"
    return time_str

def CCpyqstat(in_user="*", in_status="", node_check=False):
    fmt = "%i|%j|%u|%t|%V|%S|%M|%R|%C"
    cmd = f'squeue -a -o "{fmt}" -h'
    qstat_raw = os.popen(cmd).read().strip()

    if not qstat_raw:
        print("No jobs in queue.")
        return

    now = datetime.datetime.now()
    cms3_job_id, cms3_jobname, cms3_job_user = [], [], []
    cms3_job_state, cms3_start_time, run_time = [], [], []
    cms3_exec_host, cms3_ncpus = [], []

    for line in qstat_raw.split('\n'):
        p = line.split('|')
        if len(p) < 9: continue
        
        raw_state = p[3]
        state = ' R' if raw_state == 'R' else ' Q' if raw_state == 'PD' else ' ' + raw_state
        
        # Time and node handling (follows the style of the example image you sent)
        if raw_state == 'PD':
            # -- Q state: compute the submit time and the waiting time
            start_t_str = p[4].replace('T', ' ')
            node_info = ' ' 
            try:
                submit_t = datetime.datetime.strptime(start_t_str, '%Y-%m-%d %H:%M:%S')
                diff = now - submit_t
                calc_run = str(diff).split(".")[0]
                if 'day' in calc_run and 'days' not in calc_run:
                    calc_run = calc_run.replace('day', 'days')
            except:
                calc_run = "00:00:00"
        else:
            # -- R state: show the start time and the actual run time
            start_t_str = p[5].replace('T', ' ')
            node_info = f"{p[7]}/{p[8]}"
            calc_run = unify_time_format(p[6])

        cms3_job_id.append(p[0])
        cms3_jobname.append(p[1])
        cms3_job_user.append(p[2])
        cms3_job_state.append(state)
        cms3_start_time.append(start_t_str)
        run_time.append(calc_run)
        cms3_exec_host.append(node_info)
        cms3_ncpus.append(p[8])

    #  use the variables defined above as column names
    ps = {C_ID: cms3_job_id, C_JOB: cms3_jobname, C_USER: cms3_job_user,
          C_STAT: cms3_job_state, C_START: cms3_start_time, 
          C_RUN: run_time, C_NODE: cms3_exec_host, C_SLOTS: cms3_ncpus}

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('expand_frame_repr', False)
    
    df = pd.DataFrame(ps)
    df = df.sort_values(by=C_STAT, ascending=False).reset_index(drop=True)

    if in_user != '*':
        df = df[df[C_USER].str.strip() == in_user]
    if in_status == '-s r':
        df = df[df[C_STAT] == ' R']
    
    #  include the index in the to_string output, keep right alignment
    print(df.to_string(index=True, justify='right'))

    if node_check:
        get_waiting_nodes(df)
        get_empty_nodes(df)

def get_waiting_nodes(df):
    print("\n" + bcolors.OKBLUE + "# ---- Pending Jobs ---- #" + bcolors.ENDC)
    # filter using the C_STAT variable
    waiting_df = df[df[C_STAT].str.contains('Q')].reset_index(drop=True)
    if not waiting_df.empty:
        summary = waiting_df[C_USER].str.strip().value_counts().reset_index()
        summary.columns = ['QUEUE', 'WAITING JOBS']
        print(summary.to_string(index=False, justify='right'))
    else:
        print(pd.DataFrame(columns=['QUEUE', 'WAITING JOBS']))

def get_empty_nodes(df):
    print("\n" + bcolors.OKBLUE + "# ---- Empty Nodes SLOTS----- #" + bcolors.ENDC)
    runnings = {}
    for _, row in df[df[C_STAT].str.contains('R')].iterrows():
        node = row[C_NODE].split('/')[0].strip()
        try:
            runnings[node] = runnings.get(node, 0) + int(row[C_SLOTS])
        except: continue

    all_node = sorted(list(set([n for k in queue_info.keys() for n in queue_info[k][3]])))
    empty_data = []
    for node in all_node:
        total = next((queue_info[k][0] for k in queue_info if node in queue_info[k][3]), 0)
        used = runnings.get(node, 0)
        free = total - used
        if free > 0:
            empty_data.append({'SLOTS': total, 'RUN-SLOTS': used, 'EMPTY-SLOTS': free, 'NODE': node})
    
    if empty_data:
        edf = pd.DataFrame(empty_data).set_index('NODE')
        edf.index.name = None
        print(edf.to_string(justify='right'))
    
    print("-" * 40)

if __name__=="__main__":
    username = "*"
    status = ""
    for arg in sys.argv:
        if "-m" in arg: username = getpass.getuser()
        elif "-u" in arg: username = arg.replace("-u","")
        if "-r" in arg: status = "-s r"
    node_chk = True if username == "*" and status == "" else False
    CCpyqstat(username, status, node_check=node_chk)
