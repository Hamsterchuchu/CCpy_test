"""
qstat implementation for cms2 (parses PBS-style `qstat -f` output).

node99 uses SLURM `squeue`, so its implementation is completely different.
-> see CCpyqstat_node99.py.
At run time CCpyqstat.py picks one of the two according to $CCpy_SERVER.
"""
import os, sys, re
import datetime
from datetime import timedelta, date
import getpass
import pandas as pd
import yaml
from CCpy.Queue import CCpyJobControl
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

queue_path = ""
# -- Queue and nodes settings
try:
    CCpy_SCHEDULER_CONFIG = os.environ['CCpy_SCHEDULER_CONFIG']

except:
    print('''Error while load $CCpy_SCHEDULER_CONFIG file.
Please check the example of scheduler config file at https://github.com/91bsjun/CCpy/tree/master/CCpy/Queue''')
    quit()

queue_info = yaml.load(open(CCpy_SCHEDULER_CONFIG, 'r'), Loader=yaml.FullLoader)


def CCpyqstat(in_user="*", in_status="", node_check=False):
    """
    Modules to show SGE qstat more effectively
    """

    q = queue_path + "qstat -f"
    qstat = os.popen(q).read()

    # -- parsing
#    firstline_patt = re.compile("\s+[0-9]+ .*", re.M)
#    firstline = firstline_patt.findall(qstat)

    jobname_patt = re.compile("Job_Name =\s+\S+")
    jobname = jobname_patt.findall(qstat)

    queue_patt = re.compile("Master Queue=\s+\S+")
    queue = queue_patt.findall(qstat)

    requeue_patt = re.compile("Hard requested queues:\s+\S+")
    requeue = requeue_patt.findall(qstat)

    renode_patt = re.compile("Hard Resources:\s+.*")
    renode = renode_patt.findall(qstat)

    job_id_patt = re.compile("Job Id:\s\d+")
    job_id = job_id_patt.findall(qstat)

    job_user_patt = re.compile("Job_Owner =\s+\S+")
    job_user = job_user_patt.findall(qstat)

    exec_host_patt = re.compile("exec_host =\s+\S+")
    exec_host = exec_host_patt.findall(qstat)

    ncpus_patt = re.compile("Resource_List.ncpus =\s\d+")
    ncpus = ncpus_patt.findall(qstat)

    job_state_patt = re.compile("job_state =\s+\S+")
    job_state = job_state_patt.findall(qstat)

    qtime_patt = re.compile("qtime =\s+\S+\D+\w+\s+\d+:\d+:\d+\s+\d+")
    qtime = qtime_patt.findall(qstat)

    mtime_patt = re.compile("mtime =\s+\S+\D+\w+\s+\d+:\d+:\d+\s+\d+")
    mtime = mtime_patt.findall(qstat)

    status = []
    start_time = []
    slot = []
    cms3_jobname = []
    cms3_job_id = []
    cms3_job_user = []
    cms3_exec_host = []
    cms3_ncpus = []
    cms3_job_state = []
    cms3_qtime = []
    cms3_mtime = []

    for n in jobname:
        splt = n.split('=')
        cms3_jobname.append(splt[-1])
    
    for i in job_id:
        splt = i.split('\t')
        cms3_job_id.append(splt[-1])
    
    tmp = []
    for u in job_user:
        splt1 = u.split('=')
        tmp.append(splt1[-1])
    for s in tmp:
        splt = s.split('@')
        cms3_job_user.append(splt[0])

    for cpu in ncpus:
        splt = cpu.split('=')
        cms3_ncpus.append(splt[-1])

    for j in job_state:
        splt = j.split('=')
        cms3_job_state.append(splt[-1])

    for q in qtime:
        splt = q.split('=')
        cms3_qtime.append(splt[-1])

    for m in mtime:
        splt = m.split('=')
        cms3_mtime.append(splt[-1])
        
    j = 0
    tmp2 = []
    for e in exec_host:
        splt = e.split('=')
        tmp2.append(splt[-1])

    for state in cms3_job_state:
        if state == ' R':
            cms3_exec_host.append(tmp2[j])
            j+=1
        else:
            cms3_exec_host.append(' ')
        

#    print(queue)
#    print(requeue)
#    print(renode)

#    for l in firstline:
#        splt = l.split()
#        user.append(splt[3])
#        job_id.append(splt[0])
#        status.append(splt[4])
#        start_time.append(splt[5] + " " + splt[6])
#        slot.append(splt[-1])

#    jobnames = []
#    for j in jobname:
#        jobnames.append(j.split()[2])

#    queues = []
#    for i in range(len(requeue)):
#        if i in range(len(queue)):
#            queues.append(queue[i].split()[2])
#        else:
#            if 'hostname' in renode[i]:
#                queues.append(requeue[i].split()[3] + "@" + renode[i].split("=")[1].split(" ")[0])
#            else:
#                queues.append(requeue[i].split()[3])
#    if len(requeue) == 0:
#        for i in range(len(jobnames)):
#            queues.append("")

    i, c = 0, 0
    for states in cms3_job_state:
        if states == ' R':
            try:
                start_time.append(cms3_mtime[i])
            except:
                if c == 0:
                    num = i - len(cms3_mtime)
                    start_time.append(cms3_mtime[-(num)])
                else:
                    start_time.append(cms3_mtime[-(num) + c])
                c += 1
        else:
            start_time.append(cms3_qtime[i])
        i+=1        
    
    cms3_start_time = []
    for t in start_time:
        t = datetime.datetime.strptime(t, ' %a %b %d %H:%M:%S %Y')
        cms3_start_time.append(t)

    run_time = []
    for t in start_time:
        t = datetime.datetime.strptime(t, ' %a %b %d %H:%M:%S %Y')
        now = datetime.datetime.now()
        run = str(now - t).split(".")[0]
        run_time.append(run)

    ps = {'ID': cms3_job_id, 'JOBNAME': cms3_jobname, 'START-TIME': cms3_start_time, 'RUN-TIME': run_time,
          'QUEUE-NODE': cms3_exec_host, 'SLOTS': cms3_ncpus, '   STATUS': cms3_job_state, 'USER': cms3_job_user}

    pd.set_option('expand_frame_repr', False)
    df = pd.DataFrame(ps)
    df = df.sort_values(by='   STATUS', ascending=False)  # Status : R -> Q 
    df = df.reset_index(drop=True)   #index re indexing
    df = df[['ID', 'JOBNAME', 'USER', '   STATUS', 'START-TIME', 'RUN-TIME', 'QUEUE-NODE', 'SLOTS']]
    pd.set_option('display.max_rows', None)
    #print(bcolors.OKBLUE + "# --- Queue status --- #" + bcolors.ENDC)

    # ------------------ Nodes checking ----------------- #
    if node_check:
        print(df)
        get_waiting_nodes(df)
        get_empty_nodes(df)
#        chk_load()
#### CCpyqstat.py -m
    if in_user != '*':
        df = df[df['USER'] == ' '+ in_user]
        print(df)
##### CCpyqstat.py -r    
    if in_status == '-s r':
        df = df[df['   STATUS'] == ' R']
        print(df)

    exit()
def get_empty_nodes(df):
    # ------------------ Nodes checking ----------------- #
    if len(df) != 0:
        for i in range(len(df)):
            #running_df = df[(df['   STATUS'] == ' R') | (df['   STATUS'] == ' C') | (df['   STATUS'] == ' t')]
            running_df = df[(df['   STATUS'] == ' R')  | (df['   STATUS'] == ' t')]
    else:
        running_df = []
    # -- make all nodes and slots info
    keys = queue_info.keys()   # dict_keys(['opa'])
    queue_nodes = {'QUEUE-NODE':[], 'SLOTS':[]}
    # queue_info : {'opa': [48, 360, 'dummy.q', ['node01', 'node02', 'node03', 'node04', 'node05']]}
    for k in keys:
        for n in queue_info[k][3]:
            queue_nodes['QUEUE-NODE'].append(queue_info[k][2] + "@" + n)
            queue_nodes['SLOTS'].append(queue_info[k][0])
    # -- make running nodes and slots info
    runnings = {}
    print()
    for i in range(len(running_df)):
        if running_df['QUEUE-NODE'][i] in runnings.keys():
            runnings[running_df['QUEUE-NODE'][i]]+= int(running_df['SLOTS'][i])
        else:
            runnings[running_df['QUEUE-NODE'][i]] = int(running_df['SLOTS'][i])

    running_nodes = {'QUEUE-NODE':[], 'RUN-SLOTS':[]}
    
    for key in runnings.keys():
        # key : node01/48 ...
        running_nodes['QUEUE-NODE'].append(key)
        running_nodes['RUN-SLOTS'].append(runnings[key])

    all_RUN_SLOTS = 0
    for i in running_nodes['RUN-SLOTS']:
            all_RUN_SLOTS += int(i)
            
    # -- make pd.DataFrame for empty slots
    all_nodes_df = pd.DataFrame(queue_nodes).set_index('QUEUE-NODE')
    running_nodes_df = pd.DataFrame(running_nodes).set_index('QUEUE-NODE')
    concat_df = pd.concat([all_nodes_df, running_nodes_df], axis=1, sort=True)
    concat_df = concat_df.fillna(0)
    concat_df['EMPTY-SLOTS'] = concat_df['SLOTS'] - concat_df['RUN-SLOTS']
    empty_df = concat_df[(concat_df['EMPTY-SLOTS'] != 0)]
    #empty_df[['SLOTS', 'RUN-SLOTS', 'EMPTY-SLOTS']] = empty_df[['SLOTS', 'RUN-SLOTS', 'EMPTY-SLOTS']].astype(int)
    print(bcolors.OKBLUE + "# ---- Empty Nodes SLOTS----- #" + bcolors.ENDC)
#    print("All SLOTS : 1024")
    result = {}  
    for key, value in runnings.items():
        node_name = key.split('/')[0]  # extract the name of the running node
        if node_name in result:
            result[node_name] += value
        else:
            result[node_name] = value
    
    sorted_a = dict(sorted(result.items()))
    
    data = {
        'SLOTS': [8, 8, 72, 72, 72, 72, 72, 128],
        'RUN-SLOTS': [0, 0, 0, 0, 0, 0, 0, 0],
        'EMPTY-SLOTS': [8, 8, 72, 72, 72, 72, 72, 128] 
            }
    all_node = ['node00', 'node01', 'node02', 'node03', 'node04', 'node05', 'node06', 'node07']

    empty_df = pd.DataFrame(data, index=all_node)
    for key, value in sorted_a.items():
        trimmed_key = key.strip()
        empty_df.at[trimmed_key, 'RUN-SLOTS'] = sorted_a[key]
        empty_node = empty_df.at[trimmed_key, 'SLOTS'] - sorted_a[key]
        empty_df.at[trimmed_key, 'EMPTY-SLOTS'] = empty_node
    
    empty_df = empty_df[empty_df['EMPTY-SLOTS'] != 0]
    print(empty_df)
    print("-"*40)
#    print("NOW RUN-SLOTS : %s" %all_RUN_SLOTS)
#    print("-"*30)
#    print("AVAILABLE SLOTS : %d" % (1024 - int(all_RUN_SLOTS)))
#    print("-"*30)
#    print(empty_df[['SLOTS', 'RUN-SLOTS', 'EMPTY-SLOTS']].astype(int))

def get_waiting_nodes(df):
    # -- make wating nodes and counting
    if len(df) != 0:
        waiting_df = df[(df['   STATUS'] == ' Q')].reset_index()
    else:
        quit() 
    waitings = {}

    waiting_name = {}
    for name in waiting_df['USER']:
        if name in waiting_name.keys():
            waiting_name[name] += 1
        else:
            waiting_name[name] = 1 

    for i in range(len(waiting_df)):
        if waiting_df['QUEUE-NODE'][i] in waitings.keys():
            waitings[waiting_df['QUEUE-NODE'][i]] += 1
        else:
            waitings[waiting_df['QUEUE-NODE'][i]] = 1
    waiting_nodes = {'QUEUE':[], 'WAITING JOBS':[]}
    for key in waiting_name:
        waiting_nodes['QUEUE'].append(key)
        waiting_nodes['WAITING JOBS'].append(waiting_name[key])
    waiting_nodes_df = pd.DataFrame(waiting_nodes)
    print(bcolors.OKBLUE + "# ---- Pending Jobs ---- #" + bcolors.ENDC)
    print(waiting_nodes_df)
################### Not Use cms3 #########################
def chk_load():
    qhost = os.popen('qhost').readlines()
    qhost = [l.replace("\n","") for l in qhost]
    info = {'NODE': [], 'NCPU': [], 'LOAD': [], 'CPU USE (%)': []} 
    for l in qhost:
        spl = l.split()
        if spl[0][:4] == 'node':
            node = spl[0]
            info['NODE'].append(node)

            ncpu = spl[2]
            info['NCPU'].append(ncpu)
            if ncpu == '-':
                ncpu = 1 
            ncpu = float(ncpu)

            load = spl[3]
            info['LOAD'].append(load)
            if load == '-':
                load = -1
            load = float(load)

            cpu_use = round(load / ncpu * 100, 1)
            if cpu_use < 0:
                cpu_use = -1
            info['CPU USE (%)'].append(cpu_use)

    df = pd.DataFrame(info)
    down_df = df[(df['CPU USE (%)'] == -1 )]
    ex_df = df[(df['CPU USE (%)'] > 105)]

    if len(down_df) > 0:
        print(bcolors.OKBLUE + "# ------ DOWN node ----- #" + bcolors.ENDC)
        down_nodes = down_df['NODE'].tolist()
        for down_node in down_nodes:
            print(down_node)
    if len(ex_df) > 0:
        print(bcolors.OKBLUE + "# --- Exceeding 100% cpu use node --- #" + bcolors.ENDC)
        print(ex_df.set_index('NODE'))
################### Not Use cms3 #########################
        
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__=="__main__":
    username = "*"
    status = ""
    for arg in sys.argv:
        if "-m" in arg:
            username = getpass.getuser()
        elif "-u" in arg:
            username = arg.replace("-u","")

        if "-r" in arg:
            status = "-s r"

        if "-h" in arg:
            print("\nHow to use : " + sys.argv[0].split("/")[-1] + " [option] [option2]...")
            print('''--------------------------------------
[option]
-m       : My jobs                  (ex : CCpyqstat.py -m)       (ex : CCpyqstat.py -m -r)
-uNAME   : Specific user's jobs     (ex : CCpyqstat.py -ubsjun)  (ex : CCpyqstat.py -ubsjun -r)
-r       : Current running jobs     (ex : CCpyqstat.py -r)       (ex : CCpyqstat.py -m -r)'''
                  )
            quit()

    if username == "*" and status == '':
        try:
            CCpyqstat(username, status, node_check=True)
        except:
            CCpyqstat(username, status)
    else:
        CCpyqstat(username, status)

