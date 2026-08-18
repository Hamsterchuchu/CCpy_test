#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
#warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

import os, sys
import time

# ----- 여기 추가 -----
import matplotlib
# X 없는 서버에서 쓰는 경우에는 Agg 백엔드가 안전
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ----------------------

import pandas as pd
from CCpy.VASP.VASPio import VASPOutput
from CCpy.Tools.CCpyTools import selectVASPInputs, selectVASPOutputs, selectInputs, linux_command

version = sys.version
if version[0] == '3':
    raw_input = input

def get_energy_from_oszicar(path):
    """
    해당 디렉토리의 OSZICAR에서 마지막 E0 값을 읽어서 리턴.
    못 찾으면 None.
    """
    fname = os.path.join(path, "OSZICAR")
    if not os.path.exists(fname):
        return None
    last_E = None
    with open(fname, "r") as f:
        for line in f:
            if "E0=" in line:
                idx = line.find("E0=")
                rest = line[idx+3:].strip()
                try:
                    val = float(rest.split()[0])
                    last_E = val
                except Exception:
                    continue
    return last_E

def get_natoms_from_poscar(path):
    """
    해당 디렉토리의 POSCAR에서 원자 수를 추정.
    숫자만 있는 첫 줄을 찾아서 합을 natoms로 사용.
    """
    fname = os.path.join(path, "POSCAR")
    if not os.path.exists(fname):
        return None
    with open(fname, "r") as f:
        lines = f.readlines()
    # 6번째 줄 이후에서 '숫자만 있는 줄'을 찾는 방식
    for line in lines[5:]:
        tokens = line.split()
        if not tokens:
            continue
        if all(tok.isdigit() for tok in tokens):
            try:
                return sum(int(tok) for tok in tokens)
            except Exception:
                continue
    return None


try:
    chk = sys.argv[1]
except:
    print("\nHow to use : " + sys.argv[0].split("/")[-1] + " [option] [sub_option1] [sub_option2..]")
    print(""""--------------------------------------
[suboptions]
-sub : deep in subdirectories

[options]
-d : Clear VASP output files (except of POSCAR, POTCAR, KPOINTS, INCAR)
    ex) CCpyVASPAnal.py d

 0 : Check vasp job status.
    ex) CCpyVASPAnal.py 0

 1 : Get final structures
    ex) CCpyVASPAnal.py 1
    ex) CCpyVASPAnal.py 1 -poscar    -> make cif files using POSCAR

 2 : Get final total energy list
    ex) CCpyVASPAnal.py 2 n  : sub option n -> do not show plot
    ex) CCpyVASPAnal.py 2 n -st  : sub option '-st' -> sort by total energy
    ex) CCpyVASPAnal.py 2 n -sa  : sub option '-st' -> sort by energy/atom

 3 : Energy & Cell volume convergence plot
    ex) CCpyVASPAnal.py 3 n  : sub option n -> do not show plot

 4 : Generate cif file from POSCAR or CONTCAR
    ex) CCpyVASPAnal.py 4

 -elastic : Analyze mechanical properties

-e : Handling errors listed '01_unconverged_jobs.csv' file 
     based on Materials Project's custodian module.

-zip : zip unnecessary files (zip CHGCAR DOSCAR PROCAR XDATCAR vasprun.xml)
    ex) CCpyVASPAnal.py -zip      -> user choose directories
    ex) CCpyVASPAnal.py -zip -sub -> user choose directories (include subdirectories)
    ex) CCpyVASPAnal.py -zip -auto        -> automatically detect converged jobs
    ex) CCpyVASPAnal.py -zip -auto -sub   ->               (include subdirectories)
    ex) CCpyVASPAnal.py -zip -bg          -> detect and zip converged jobs every 30 minutes
    ex) CCpyVASPAnal.py -zip -bg -sub     ->               (include sub directories)
    ex) CCpyVaspAnal.py -zip -m           -> remove CHG* DOSCAR* PROCAR*
   
    
"""
          )
    quit()

sub = False
additional_dir = None
if "-sub" in sys.argv:
    sub = True
for arg in sys.argv:
    if "-dir" in arg:
        additional_dir = arg.split("=")[1]

if sys.argv[1] == "-d":
    inputfiles = ["INCAR","POSCAR","POTCAR","KPOINTS"]
    inputs = selectVASPOutputs("./", additional_dir=additional_dir)
    for each_input in inputs:
        print(each_input)
    yn = raw_input("Are you sure to remove these output files? (y/n)")
    if yn == "y" or yn == "yes":
        pass
    else:
        quit()
    pwd = os.getcwd()
    for each_input in inputs:
        os.chdir(each_input)
        files = [f for f in os.listdir("./")]
        for f in files:
            if f in inputfiles:
                pass
            else:
                linux_command("rm -rf "+f)
        os.chdir(pwd)

if sys.argv[1] == "0":
    dirs = selectVASPOutputs("./", ask=False, sub=sub, additional_dir=additional_dir)
    VO = VASPOutput()
    VO.check_terminated(dirs=dirs)

elif sys.argv[1] == "1":
    inputs = selectVASPOutputs("./", sub=sub, additional_dir=additional_dir)
    pwd = os.getcwd()
    for each_input in inputs:
        # 경로 전체를 '_' 로 이어붙여 이름을 만든다.
        #   ./batch1/LiCoO2      -> batch1_LiCoO2
        #   ./LiMnO2/Band-DOS    -> LiMnO2_Band-DOS
        # 마지막 디렉토리 이름만 쓰면 -sub 로 하위까지 훑을 때
        # batch1/LiCoO2 와 batch2/LiCoO2 가 같은 파일명이 되어 덮어써진다.
        dirname = each_input.replace('./', '').replace('/', "_")
        os.chdir(each_input)
        VO = VASPOutput()
        if "-poscar" in sys.argv:
            target_name = dirname + "_poscar.cif"
            VO.getFinalStructure(filename="POSCAR", target_name=target_name, path=pwd+"/")
        elif "CONTCAR" in os.listdir("./") and os.path.getsize("CONTCAR") != 0:
            target_name = dirname + "_contcar.cif"
            VO.getFinalStructure(target_name=target_name, path=pwd+"/")
        else:
            print(each_input + ": CONTCAR is empty!")
        os.chdir(pwd)

elif sys.argv[1] == "2":
    """
    2번 옵션:
    - CCpy 내부 plot을 쓰지 않고
    - OSZICAR/POSCAR를 직접 읽어서
    - energy_list.png 이미지 파일만 생성
    """
    # 서브 옵션 처리: 2 n 이면 plot 자체를 만들지 않음
    print ("XXXXXXXXXX")
    show_plot = True
    sort_mode = None  # "tot" or "atom"
    try:
        show_chk = sys.argv[2]
        if show_chk == "n":
            show_plot = False
    except Exception:
        show_plot = True

    if "-st" in sys.argv:
        sort_mode = "tot"
    elif "-sa" in sys.argv:
        sort_mode = "atom"

    # VASP 결과 디렉토리 목록
    dirs = selectVASPOutputs("./", ask=False, sub=sub, additional_dir=additional_dir)

    data = []
    for d in dirs:
        E = get_energy_from_oszicar(d)
        if E is None:
            continue
        nat = get_natoms_from_poscar(d)
        if nat is None or nat == 0:
            E_per_atom = None
        else:
            E_per_atom = E / nat
        data.append({"dir": d, "E": E, "natoms": nat, "E_per_atom": E_per_atom})

    if len(data) == 0:
        print("No OSZICAR/POSCAR data found. Cannot make energy plot.")
        quit()

    df = pd.DataFrame(data)

    # 정렬 옵션
    if sort_mode == "tot":
        df = df.sort_values(by="E").reset_index(drop=True)
    elif sort_mode == "atom":
        # per-atom 에너지가 없는 경우는 제외
        df = df.dropna(subset=["E_per_atom"])
        df = df.sort_values(by="E_per_atom").reset_index(drop=True)

    # 정렬 후에도 데이터가 없는 경우
    if len(df) == 0:
        print("No valid energy data after sorting.")
        quit()

    # 에너지 리스트를 csv로도 저장
    df.to_csv("energy_list.csv", index=False)
    print("* Saved energy list to energy_list.csv")

    # show_plot=False면 여기서 끝
    if not show_plot:
        quit()

    # ---- 여기서부터 그림 그리기 (Agg backend, 창 안 뜸) ----
    fig, ax = plt.subplots(figsize=(6, 4))

    x = list(range(1, len(df) + 1))
    if sort_mode == "atom":
        y = df["E_per_atom"].values
        ax.set_ylabel("Energy per atom (eV)")
    else:
        y = df["E"].values
        ax.set_ylabel("Total energy (eV)")

    ax.plot(x, y, marker="o")
    ax.set_xlabel("Structure index (sorted)")
    ax.grid(True)

    # x축 tick을 간단히: 1,2,3,... (디렉토리 이름은 csv에서 확인)
    ax.set_title("VASP total energy list")

    outname = "energy_list.png"
    fig.tight_layout()
    fig.savefig(outname, dpi=300)
    plt.close(fig)
    print(f"* Saved energy plot to {outname}")

elif sys.argv[1] == "3":
    # -- Check show plot
    show_plot = True
    try:
        show_chk = sys.argv[2]
        if show_chk == "n":
            show_plot = False
    except:
        show_plot = True

    inputs = selectVASPOutputs("./", additional_dir=additional_dir)
    for each_input in inputs:
        os.chdir(each_input)
        print(each_input)
        VO = VASPOutput()
        VO.getConvergence(show_plot=show_plot)
        os.chdir("../")

elif sys.argv[1] == "4":
    inputs = selectInputs(marker=["POSCAR", "CONTCAR"], directory_path="./")
    for each_input in inputs:
        VO = VASPOutput()
        VO.getFinalStructure(filename=each_input, path="./")

elif sys.argv[1] == "-elastic":
    inputs = selectVASPOutputs("./", sub=sub, additional_dir=additional_dir)
    VO = VASPOutput()
    VO.get_mechanical_properties(dirs=inputs)

elif sys.argv[1] == "-zip":
    """
    zipped status add
    """
    minimize = False
    if '-m' in sys.argv:
        minimize = True
    VO = VASPOutput()
    if "-bg" in sys.argv:
        print("Start loop..")
        cnt = 1
        while True:
            print("\nloop " + str(cnt))
            if sub:
                linux_command("CCpyVASPAnal.py -zip -auto -sub")
            else:
                linux_command("CCpyVASPAnal.py -zip -auto")
            print("Rest 30 minutes..")
            time.sleep(1800)
            cnt+=1
    elif "-auto" in sys.argv:
        if sub:
            print("# ----------- Parsing -------------- #")
            linux_command("CCpyVASPAnal.py 0 -sub")
        else:
            print("\n\n# ----------- Parsing -------------- #")
            linux_command("CCpyVASPAnal.py 0")
        df = pd.read_csv(".00_job_status.csv")
        df[['  Converged', '  Zipped']] = df[['  Converged', '  Zipped']].astype(str)
        df = df[(df['  Converged'] == 'True')]
        df = df[(df['  Zipped'] == 'False')]
        dirs = df['Directory'].tolist()
        if len(dirs) == 0:
            print("Cannot find unzipped VASP job.")
        else:
            print("\n\n# ----------- Zipping -------------- #")
            VO.vasp_zip(dirs, minimize=minimize)
    else:
        dirs = selectVASPOutputs("./", sub=sub)
        print("\n\n# ----------- Zipping -------------- #")
        VO.vasp_zip(dirs, minimize=minimize)

        
elif sys.argv[1] == "-e":
    """
    Handling errors based on Materials Project's custodian module.
    listed 01_unconverged_jobs.csv file
    """
    if "01_unconverged_jobs.csv" not in os.listdir("./"):
        print("\n01_unconverged_jobs.csv was not found in this directory.")
        print("Create it using: CCpyVASPAnal.py 0")
        quit()
    df = pd.read_csv("01_unconverged_jobs.csv")
    try:
        df = df.drop('Unnamed: 0', 1)
    except:
        pass
    print("\n* Handle error or unconverged job(s)")
    print("Unconverged job list in 01_unconverged_jobs.csv")

    dirs = df['Directory'].tolist()
    outputs = selectVASPOutputs("./", dir_list=dirs, additional_dir=additional_dir)

    VO = VASPOutput()
    VO.vasp_error_handle(outputs)









