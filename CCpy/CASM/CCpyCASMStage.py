#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CASM 배열 계산을 다음 단계로 넘긴다.

한 번 완화하고 끝내면 Pulay stress 때문에 부피가 덜 수렴한 채로 남고, 완화용
smearing 이 최종 에너지에 인공 엔트로피 항을 남긴다. 이 명령은 끝난 계산을
받아 다음 단계 입력으로 바꿔 놓는다 — 계산 자체는 다시 제출해야 한다.
"""

import os
import sys

version = sys.version
if version[0] == '3':
    raw_input = input


def _help():
    print("\nHow to use : " + os.path.basename(sys.argv[0])
          + " [option] [sub_option1] [sub_option2..]")
    print('''--------------------------------------
[options]
1 : 다음 완화 준비   (CONTCAR -> POSCAR, Pulay stress 를 걷어낸다)
2 : 최종 정적 준비   (NSW=0, ISMEAR=-5 로 에너지를 확정한다)

끝난 계산의 산출물은 접미사를 붙여 남긴다 (OUTCAR_relax1 ...).
수렴 판정은 CCpyVASPAnal.py 0 이 남기는 01_unconverged_jobs.csv 를 읽어
쓰므로, 이 명령 전에 한 번 돌려 두면 미수렴 배열이 자동으로 빠진다.


[sub_options]
ex) CCpyCASMStage.py 1 -ispin=1 -y

    -ispin=#  : ISPIN 을 지정              (DEFAULT : 기존 값 유지)
                자성 원소(Fe·Co·Ni·Cr·Mn)가 없으면 1 로 내려도 된다
    -force    : 미수렴 배열도 함께 진행
    -keep     : CHG/CHGCAR/WAVECAR 등 큰 파일을 지우지 않음
    -y        : 확인 없이 진행
    -dir=A,B  : 특정 배열만                (DEFAULT : con* 전부)
--------------------------------------''')
    quit()


def _ask(prompt, default=None):
    if default is None:
        return raw_input("%s\n: " % prompt).strip()
    got = raw_input("%s [%s]\n: " % (prompt, default)).strip()
    return got if got else default


def _parse_subopts(argv):
    opt = {}
    for arg in argv:
        if not arg.startswith("-"):
            continue
        if arg.startswith("-ispin="):
            opt["ispin"] = int(arg.split("=", 1)[1])
        elif arg == "-force":
            opt["force"] = True
        elif arg == "-keep":
            opt["keep"] = True
        elif arg == "-y":
            opt["yes"] = True
        elif arg.startswith("-dir="):
            opt["dirs"] = [v.strip() for v in arg.split("=", 1)[1].split(",")]
        else:
            print("\n모르는 옵션입니다: %s" % arg)
            _help()
    return opt


def run(stage, opt):
    from CCpy.CASM import CASMstage as st
    from CCpy.CASM.CASMkpoints import config_dirs

    dirs = [d for d in opt["dirs"]] if opt.get("dirs") else config_dirs(".")
    if not dirs:
        print("\n배열 디렉토리(con*)를 찾지 못했습니다.")
        quit()

    ready = [d for d in dirs if st.is_finished(d)]
    print("\n* 배열 %d개 중 %d개가 끝났습니다 (vasp.done 기준)."
          % (len(dirs), len(ready)))
    if len(ready) < len(dirs):
        print("  아직 안 끝난 %d개는 건너뜁니다." % (len(dirs) - len(ready)))
    if not ready:
        quit()

    skip_unconverged = not opt.get("force")
    bad = st.unconverged_dirs(".")
    if bad is None:
        print("\n  01_unconverged_jobs.csv 가 없습니다.")
        print("  CCpyVASPAnal.py 0 을 먼저 돌리면 미수렴 배열이 자동으로 빠집니다.")
        if not opt.get("yes"):
            if _ask("  그래도 진행할까요? (y/n)", "n").lower() not in ("y", "yes"):
                quit()
    elif skip_unconverged:
        print("  미수렴으로 표시된 배열 %d개는 건너뜁니다." % len(bad))
    else:
        print("  -force 이므로 미수렴 배열 %d개도 함께 진행합니다." % len(bad))

    sample = ready[:3]
    print("\n  현재 상태(앞 %d개):" % len(sample))
    for d in sample:
        try:
            ent = st.entropy_per_atom(d)
            print("    %-10s 부피변화 %+6.2f %%   T*S %-7s meV/atom   NKPTS %s"
                  % (os.path.basename(d), st.volume_drift(d),
                     "%.3f" % ent if ent is not None else "-", st.nkpts(d)))
        except st.StageError as err:
            print("    %-10s %s" % (os.path.basename(d), err))

    if not opt.get("yes"):
        if _ask("\n* 진행할까요? 이전 결과는 접미사를 붙여 남깁니다. (y/n)",
                "y").lower() not in ("y", "yes"):
            print("  아무것도 하지 않았습니다.")
            quit()

    done, skipped = st.advance_all(
        ".", stage=stage, dirs=dirs, skip_unconverged=skip_unconverged,
        ispin=opt.get("ispin"), clean=not opt.get("keep"))
    print("\n" + st.describe(done, skipped))

    if not done:
        return
    print("\n  이제 다시 제출하세요:")
    print("    CCpyJobSubmit.py 2 I5 -batch -scratch -n=8")
    if stage == "relax":
        print("\n  끝나면 부피가 더 움직이는지 보고, 멎었으면 옵션 2 로 최종")
        print("  정적 계산을 준비하세요.")
    else:
        ismears = sorted({str(d.get("ismear")) for d in done})
        if "1" in ismears:
            print("\n  k-점이 4개 미만인 배열은 사면체법(-5)을 쓸 수 없어")
            print("  ISMEAR=1, SIGMA=0.05 로 두었습니다.")
        print("\n  이 계산의 OUTCAR 이 최종 에너지입니다. CCpyCASMhull.py 가")
        print("  OUTCAR 을 읽으므로 따로 손댈 것 없이 hull 이 정적 에너지로 그려집니다.")


if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        _help()

    opt = _parse_subopts(sys.argv[2:])
    if option == "1":
        run("relax", opt)
    elif option == "2":
        run("static", opt)
    else:
        print("\n모르는 옵션입니다: %s" % option)
        _help()
