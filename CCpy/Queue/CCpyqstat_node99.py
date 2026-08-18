"""
node99 용 qstat 구현 (SLURM `squeue` 출력 파싱).

cms2 는 PBS 형식 `qstat -f` 를 쓰기 때문에 구현이 완전히 다르다.
-> CCpyqstat_cms2.py 참고.
실행 시에는 CCpyqstat.py 가 $CCpy_SERVER 값에 따라 둘 중 하나를 골라 불러온다.
"""
import os, sys, datetime
import getpass
import pandas as pd
import yaml
import warnings

warnings.filterwarnings("ignore")

# -- 설정 로드
try:
    CCpy_SCHEDULER_CONFIG = os.environ['CCpy_SCHEDULER_CONFIG']
    queue_info = yaml.safe_load(open(CCpy_SCHEDULER_CONFIG, 'r'))
except:
    print("Error: Check CCpy_SCHEDULER_CONFIG or YAML file.")
    quit()

class bcolors:
    OKBLUE = '\033[94m'
    ENDC = '\033[0m'

# 여기서 공백을 추가하거나 줄여서 터미널 간격을 마음대로 조절하세요!
C_ID    = '   ID'
C_JOB   = '           JOBNAME'
C_USER  = '     USER'
C_STAT  = '   STATUS'
C_START = '        START-TIME'
C_RUN   = '   RUN-TIME'
C_NODE  = '   QUEUE-NODE'
C_SLOTS = ' SLOTS'

def unify_time_format(time_str):
    """Slurm 시간을 'D days, HH:MM:SS' 또는 'HH:MM:SS'로 통일"""
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
        
        # 시간 및 노드 처리 (보내주신 예시 이미지 스타일 반영)
        if raw_state == 'PD':
            # -- Q 상태: 제출 시간 및 대기 시간 계산
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
            # -- R 상태: 시작 시간 및 실제 런타임 표시
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

    #  정의한 변수들을 컬럼명으로 사용
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
    
    #  to_string 출력 시 인덱스 포함, 우측 정렬 유지
    print(df.to_string(index=True, justify='right'))

    if node_check:
        get_waiting_nodes(df)
        get_empty_nodes(df)

def get_waiting_nodes(df):
    print("\n" + bcolors.OKBLUE + "# ---- Pending Jobs ---- #" + bcolors.ENDC)
    # C_STAT 변수를 사용하여 필터링
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
