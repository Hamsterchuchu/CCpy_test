#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from CCpy.Tools.CCpyTools import selectInputs

version = sys.version
if version[0] == '3':
    raw_input = input


def _help():
    print("\nHow to use : " + os.path.basename(sys.argv[0])
          + " [option] [sub_option1] [sub_option2..]")
    print('''--------------------------------------
[options]
1 : CASM 입력 생성   (PRIM · CSPECS · KPOINTS 를 한 번에)
9 : prim.json 생성   (신형 CASM 형식)


[sub_options]
ex) CCpyCASMInputGen.py 1 -sc=2,1,1 -occ=Cu,Ir -p1 -kl=35

    < STRUCTURE >
    -str=FILE   : 구조 파일                   (DEFAULT : 물어봄)
    -sc=#,#,#   : supercell 배수              (DEFAULT : 1,1,1)
                  FCC 는 2,1,1 / HCP·BCC 는 2,2,1 이 8 자리 셀
    -occ=A,B    : 각 자리에 올 수 있는 원소    (DEFAULT : 물어봄)
                  빈자리는 Vac. 원소별로 다르면 -occ=Li:Li,Vac -occ=C:C
    -title=NAME : PRIM 1행                    (DEFAULT : 원소 이름)

    < SYMMETRY >
    -p1         : P1 으로 낮춤                 (전수 열거를 하려면 필요)
    -amp=#      : 흔들 폭 (분수좌표)           (DEFAULT : 0.001)

    < CSPECS >
    -nn=#       : 몇 번째 이웃 껍질까지        (DEFAULT : 1)
    -r=#        : 반경을 직접 지정 (Å)
    -sizes=#,#  : 클러스터 크기                (DEFAULT : 2,3,4)
    -sp=A,B     : 이 원소끼리의 거리만 잼      (Li-C 는 -sp=Li)

    < KPOINTS >
    -kl=#       : 목표 k*L (Å)                (DEFAULT : 35, 금속)
                  절연체는 20 안팎
    -kp=#,#,#   : mesh 를 직접 지정
    -dist       : 만든 KPOINTS 를 con* 전부에 복사

    < PARTIAL >
    -only=A,B   : prim / cspecs / kpoints 중 일부만
                  ex) -only=kpoints  (mesh 만 다시 잡을 때)
--------------------------------------''')
    quit()


def _ask(prompt, default=None):
    if default is None:
        return raw_input("%s\n: " % prompt).strip()
    got = raw_input("%s [%s]\n: " % (prompt, default)).strip()
    return got if got else default


def _parse_subopts(argv):
    """CCpyVASPInputGen 과 같은 방식으로 -key=value / -flag 를 읽는다.

    sub_option 을 하나라도 준 경우에는 대화형으로 되묻지 않고 기본값을 쓴다.
    스크립트에서 부를 수 있어야 하기 때문이다.
    """
    opt = {"only": None, "occ": {}, "occ_all": None,
           "interactive": not any(a.startswith("-") for a in argv)}
    for arg in argv:
        if not arg.startswith("-"):
            continue
        if arg.startswith("-str="):
            opt["str"] = arg.split("=", 1)[1]
        elif arg.startswith("-sc="):
            opt["sc"] = arg.split("=", 1)[1]
        elif arg.startswith("-occ="):
            val = arg.split("=", 1)[1]
            if ":" in val:
                elt, occ = val.split(":", 1)
                opt["occ"][elt] = occ.replace(",", " ")
            else:
                opt["occ_all"] = val.replace(",", " ")
        elif arg.startswith("-title="):
            opt["title"] = arg.split("=", 1)[1]
        elif arg == "-p1":
            opt["p1"] = True
        elif arg.startswith("-amp="):
            opt["amp"] = float(arg.split("=", 1)[1])
        elif arg.startswith("-nn="):
            opt["nn"] = int(arg.split("=", 1)[1])
        elif arg.startswith("-r="):
            opt["r"] = float(arg.split("=", 1)[1])
        elif arg.startswith("-sizes="):
            opt["sizes"] = [int(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-sp="):
            opt["sp"] = arg.split("=", 1)[1].split(",")
        elif arg.startswith("-kl="):
            opt["kl"] = float(arg.split("=", 1)[1])
        elif arg.startswith("-kp="):
            opt["kp"] = [int(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg == "-dist":
            opt["dist"] = True
        elif arg.startswith("-only="):
            opt["only"] = [v.strip().lower() for v in arg.split("=", 1)[1].split(",")]
        elif arg not in ("1", "9"):
            print("\n모르는 옵션입니다: %s" % arg)
            _help()
    return opt


def _wanted(opt, step):
    return opt["only"] is None or step in opt["only"]


# ---------------------------------------------------------------------------
# [1] CASM 입력 생성
# ---------------------------------------------------------------------------

def input_gen(opt):
    from CCpy.CASM.CASMprim import (Prim, make_prim, read_structure,
                                    break_symmetry, PrimError)
    from CCpy.CASM.CASMcspecs import (make_cspecs, describe_shells, Cspecs,
                                      CspecsError)
    from CCpy.CASM.CASMkpoints import (make_kpoints, Kpoints, config_dirs,
                                       distribute, verify_uniform,
                                       describe_uniformity, KpointsError)

    if opt["only"]:
        unknown = set(opt["only"]) - {"prim", "cspecs", "kpoints"}
        if unknown:
            print("\n-only 에 모르는 값이 있습니다: %s" % ", ".join(sorted(unknown)))
            print("prim / cspecs / kpoints 중에서 골라 주세요.")
            quit()

    # -- 기준 구조 -----------------------------------------------------------
    src = opt.get("str")
    if src is None:
        if _wanted(opt, "prim") or not os.path.isfile("PRIM"):
            inputs = selectInputs([".xsd", ".cif", "POSCAR", "CONTCAR"], "./")
            if not inputs:
                print("구조 파일을 찾지 못했습니다 (.cif / POSCAR / CONTCAR).")
                quit()
            src = inputs[0]
        else:
            src = "PRIM"
    if not os.path.isfile(src):
        print("%s 가 없습니다." % src)
        quit()

    made = []

    # -- PRIM ---------------------------------------------------------------
    if _wanted(opt, "prim"):
        try:
            _, _, species = read_structure(src)
        except PrimError as err:
            print("\n%s" % err)
            quit()
        ordered = []
        for s in species:
            if s not in ordered:
                ordered.append(s)
        print("\n* %s : 원자 %d개, 원소 %s" % (src, len(species), ", ".join(ordered)))

        sc = opt.get("sc")
        if sc is None:
            sc = _ask("\n* Supercell 배수 a b c"
                      "\n  (FCC 는 2 1 1, HCP/BCC 는 2 2 1 이 8 자리 셀을 줍니다)",
                      "1 1 1") if opt["interactive"] else "1 1 1"
        try:
            nx, ny, nz = [int(v) for v in sc.replace(",", " ").split()]
        except ValueError:
            print("배수는 정수 3개여야 합니다: %r" % sc)
            quit()

        if opt["occ"]:
            occupancy = dict((e, opt["occ"].get(e, e)) for e in ordered)
            missing = [e for e in ordered if e not in opt["occ"]]
            if missing:
                print("  -occ 에 없는 원소는 고정 자리로 둡니다: %s" % ", ".join(missing))
        elif opt["occ_all"]:
            occupancy = opt["occ_all"]
        elif opt["interactive"]:
            print("\n* 각 자리에 올 수 있는 원소를 적습니다. 빈자리는 Vac 로 적습니다.")
            print("  (예: 이원계 합금 'Cu Ir' / Li 자리 'Li Vac' / 고정 자리 'C')")
            occupancy = dict((e, _ask("  %s 자리" % e, e)) for e in ordered)
        else:
            print("\n-occ 을 주지 않아 모든 자리를 고정으로 둡니다.")
            print("이원계 합금이면 -occ=Cu,Ir 처럼 주세요.")
            occupancy = dict((e, e) for e in ordered)

        try:
            prim = make_prim(src, occupancy=occupancy, supercell=(nx, ny, nz),
                             title=opt.get("title"))
        except PrimError as err:
            print("\n%s" % err)
            quit()

        if os.path.isfile("PRIM"):
            os.rename("PRIM", "PRIM_backup")
        prim.write("PRIM")
        a, b, c = prim.lengths
        print("\n[PRIM] 자리 %d개 / 격자 %.6f %.6f %.6f" % (len(prim), a, b, c))
        print("       " + " / ".join("%d자리 %s" % (n, " ".join(o))
                                     for n, o in prim.groups()))
        mixed = len(prim.mixed_sites)
        if mixed:
            print("       혼합 자리 %d개 -> 배열 수 %d" % (mixed, 2 ** mixed))
        made.append("PRIM")

        if opt.get("p1"):
            try:
                out, msgs = break_symmetry(prim, amplitude=opt.get("amp", 0.001))
            except PrimError as err:
                print("\n%s" % err)
                quit()
            if not os.path.isfile("PRIM_orig"):
                os.rename("PRIM", "PRIM_orig")
            out.write("PRIM")
            for m in msgs:
                print("       " + m)
            print("       P1 으로 낮췄습니다. 원본은 PRIM_orig 입니다.")
            print("       열거가 끝나 make_dirs 를 만든 뒤 원본을 되돌리고")
            print("       디렉토리를 생성하세요:  cp PRIM_orig PRIM")
            made.append("PRIM_orig")
        src_for_rest = prim              # 흔들기 전 구조로 거리·mesh 를 잰다
    else:
        src_for_rest = Prim.read(src) if os.path.basename(src).startswith("PRIM") \
            else src

    # -- CSPECS -------------------------------------------------------------
    if _wanted(opt, "cspecs"):
        try:
            cs, shells = make_cspecs(src_for_rest, nshell=opt.get("nn", 1),
                                     sizes=opt.get("sizes", (2, 3, 4)),
                                     radius=opt.get("r"),
                                     species=opt.get("sp"))
        except (CspecsError, PrimError) as err:
            print("\n%s" % err)
            quit()
        if os.path.isfile("CSPECS"):
            os.rename("CSPECS", "CSPECS_backup")
        cs.write("CSPECS")
        radius = cs.radii[sorted(cs.radii)[0]]
        inside = sum(n for d, n in shells if d <= radius)
        print("\n[CSPECS] 반경 %g Å, 크기 %s"
              % (radius, ", ".join(str(s) for s in sorted(cs.radii))))
        print(describe_shells(shells, limit=4))
        print("       반경 안의 이웃 %d개" % inside)
        made.append("CSPECS")

    # -- KPOINTS ------------------------------------------------------------
    if _wanted(opt, "kpoints"):
        try:
            kp, info = make_kpoints(src_for_rest, target=opt.get("kl", 35.0))
            if opt.get("kp"):
                kp = Kpoints(opt["kp"], mode=kp.mode, comment=kp.comment)
                info = ["mesh 를 직접 지정하셨습니다: %s"
                        % " ".join(str(v) for v in kp.mesh)]
        except (KpointsError, PrimError) as err:
            print("\n%s" % err)
            quit()
        if os.path.isfile("KPOINTS"):
            os.rename("KPOINTS", "KPOINTS_backup")
        kp.write("KPOINTS")
        print("\n[KPOINTS] %s %s" % (kp.mode, " ".join(str(v) for v in kp.mesh)))
        for m in info:
            print("       " + m)
        made.append("KPOINTS")

        dirs = config_dirs(".")
        if dirs:
            go = opt.get("dist")
            if go is None:
                go = _ask("\n* 배열 %d개에 복사할까요? (y/n)" % len(dirs), "y") \
                    .lower() in ("y", "yes") if opt["interactive"] else False
                if not opt["interactive"]:
                    print("       배열 %d개가 있습니다. 복사하려면 -dist 를 주세요."
                          % len(dirs))
            if go:
                distribute(kp, dirs)
                ok, groups = verify_uniform(dirs)
                print("       " + describe_uniformity(groups))

    # -- 정리 ---------------------------------------------------------------
    print("\n만든 파일: %s" % ", ".join(made))
    if "PRIM" in made and not os.path.isdir("con0.0"):
        print("\n다음 단계 — 열거:")
        print("  CCpyCASMRun.py            (준비 중)")
        print("  또는 ./mainclust 를 직접 돌리고 02_ModiMake.sh 를 실행")


# ---------------------------------------------------------------------------
# [9] prim.json (신형 CASM)
# ---------------------------------------------------------------------------

def prim_json_gen():
    from CCpy.CASM.CASMio import CASMInput

    inputs = selectInputs([".xsd", ".cif", "POSCAR", "CONTCAR"], "./")
    for each_input in inputs:
        CASMInput(each_input).primGen()


if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        _help()

    if option == "1":
        input_gen(_parse_subopts(sys.argv[2:]))
    elif option == "9":
        prim_json_gen()
    else:
        print("\n모르는 옵션입니다: %s" % option)
        _help()
