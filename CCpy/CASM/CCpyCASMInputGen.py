#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
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
1 : 배열 생성   구조 파일 하나로 con* 디렉토리까지 끝낸다
                (PRIM · CSPECS · KPOINTS -> mainclust 열거 -> 디렉토리 생성
                 -> 셀 파라미터 평균 -> KPOINTS 배포)
2 : 이어서 손질  1 이 중간에 끊겼거나 손으로 고친 뒤 이어 갈 때
9 : prim.json 생성      (신형 CASM 형식)

인자 없이 `CCpyCASMInputGen.py 1` 로 부르면 물어보면서 진행하고,
sub_option 을 하나라도 주면 되묻지 않는다 (스크립트용).

준비물 — 작업 디렉토리에 INCAR, 원소별 POTCAR_<원소> 가 있어야 한다.
mainclust 는 $CCpy_MAINCLUST / ~/.CCpy_test / 작업 디렉토리 / $PATH 순으로 찾는다.


[sub_options]
ex) CCpyCASMInputGen.py 1
    CCpyCASMInputGen.py 1 -str=CONTCAR -sc=2,1,1 -occ=Cu,Ir -y

    < METHOD >   기본은 Method 1 (비대칭 전수 열거)
    -sym        : Method 2 - 대칭을 유지하고 CSPECS 를 만든다
                  기본값은 좌표를 흔들어 P1 으로 낮추고 CSPECS 를 만들지 않는다.
                  대칭 PRIM 에 CSPECS 가 있으면 열거 수가 달라진다 (8자리 셀에서 9 -> 21)

    < STRUCTURE >   [option 1]
    -str=FILE   : 구조 파일                   (DEFAULT : 물어봄)
    -sc=#,#,#   : supercell 배수              (DEFAULT : 8 자리가 되도록 제안)
                  FCC 관례셀(4원자)은 2,1,1 / BCC·HCP(2원자)는 2,2,1
                  자리가 늘면 배열 수는 2^n 이다 (8자리 256 / 16자리 65536)
    -occ=A,B    : 각 자리에 올 수 있는 원소    (DEFAULT : 물어봄)
                  빈자리는 Vac. 원소별로 다르면 -occ=Li:Li,Vac -occ=C:C
    -title=NAME : PRIM 1행                    (DEFAULT : 원소 이름)
    -amp=#      : P1 으로 낮출 때 흔들 폭      (DEFAULT : 0.001)

    < CSPECS >   [option 1, -sym 일 때]
    -nn=#       : 몇 번째 이웃 껍질까지        (DEFAULT : 1)
    -r=#        : 반경을 직접 지정 (Å)
    -sizes=#,#  : 클러스터 크기                (DEFAULT : 2,3,4)
    -sp=A,B     : 이 원소끼리의 거리만 잼      (Li-C 는 -sp=Li)

    < KPOINTS >
    -kl=#       : 목표 k*L (Å)                (DEFAULT : 35, 금속 / 절연체는 20)
    -kp=#,#,#   : mesh 를 직접 지정

    < CELL PARAM >   [option 2]
    -a=#,#      : 두 원소의 격자상수를 직접
    -ref=F,F    : 두 원소의 구조 파일 경로
                  (DEFAULT : BULK/<원소>/CONTCAR 를 찾고, 없으면 내장 표)
    -iso        : 세 축을 같은 배율로          (원본 03_cellparam.sh 동작)

    < RUN >   [option 1]
    -norun      : 입력 파일만 만들고 mainclust 는 돌리지 않는다
    -vol=#      : 최대 supercell 부피              (DEFAULT : 1)
    -y          : 배열 개수 확인 없이 진행

    < PARTIAL >
    -only=A,B   : [1] prim / cspecs / kpoints  (주면 mainclust 를 돌리지 않는다)
                  [2] makedirs / cellparam / kpoints
--------------------------------------''')
    quit()


def _ask(prompt, default=None):
    if default is None:
        return raw_input("%s\n: " % prompt).strip()
    got = raw_input("%s [%s]\n: " % (prompt, default)).strip()
    return got if got else default


#: 이보다 배열이 많아지면 한 번 물어본다. 8자리 256 / 16자리 65536 이다.
CONFIG_WARN = 512


def _suggest_supercell(natoms, target=8):
    """자리 수가 target 이 되도록 배수를 제안한다.

    FCC 관례셀(4원자)이면 2 1 1, BCC·HCP(2원자)면 2 2 1 이 나온다. 원자 수가
    target 을 나누지 못하면 1 1 1 을 준다 — 억지로 맞추지 않는다.
    """
    if natoms <= 0 or target % natoms:
        return (1, 1, 1)
    need = target // natoms
    for sc in ((1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)):
        if sc[0] * sc[1] * sc[2] == need:
            return sc
    return (need, 1, 1)


def _confirm_size(n, opt):
    """배열 수가 감당할 만한지 확인한다. 자리 하나가 늘 때마다 두 배가 된다."""
    if n <= CONFIG_WARN:
        return
    print("\n  ! 배열이 %d개입니다. 혼합 자리 하나가 늘 때마다 두 배가 됩니다." % n)
    print("    셀을 줄이거나(-sc=2,1,1), 대칭으로 줄이는 Method 2(-sym)를 보세요.")
    if opt.get("yes") or not opt["interactive"]:
        print("    그대로 진행합니다.")
        return
    if _ask("    그래도 진행할까요? (y/n)", "n").lower() not in ("y", "yes"):
        print("    아무것도 만들지 않았습니다.")
        quit()


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
        elif arg == "-sym":
            opt["sym"] = True
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
        elif arg == "-norun":
            opt["norun"] = True
        elif arg == "-y":
            opt["yes"] = True
        elif arg.startswith("-vol="):
            opt["vol"] = int(arg.split("=", 1)[1])
        elif arg == "-iso":
            opt["iso"] = True
        elif arg.startswith("-a="):
            opt["a"] = [float(v) for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-ref="):
            opt["ref"] = [v.strip() for v in arg.split("=", 1)[1].split(",")]
        elif arg.startswith("-only="):
            opt["only"] = [v.strip().lower() for v in arg.split("=", 1)[1].split(",")]
        elif arg not in ("1", "2", "9"):
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

        guess = " ".join(str(v) for v in _suggest_supercell(len(species)))
        sc = opt.get("sc")
        if sc is None:
            sc = _ask("\n* Supercell 배수 a b c"
                      "\n  (원자 %d개짜리 셀이므로 %s 이면 8 자리가 됩니다)"
                      % (len(species), guess), guess) \
                if opt["interactive"] else guess
        try:
            nx, ny, nz = [int(v) for v in sc.replace(",", " ").split()]
        except ValueError:
            print("배수는 정수 3개여야 합니다: %r" % sc)
            quit()

        try:
            prim = make_prim(src, occupancy=occupancy, supercell=(nx, ny, nz),
                             title=opt.get("title"))
        except PrimError as err:
            print("\n%s" % err)
            quit()

        a, b, c = prim.lengths
        print("\n[PRIM] 자리 %d개 / 격자 %.6f %.6f %.6f" % (len(prim), a, b, c))
        print("       " + " / ".join("%d자리 %s" % (n, " ".join(o))
                                     for n, o in prim.groups()))
        mixed = len(prim.mixed_sites)
        if mixed:
            n_config = 2 ** mixed
            if opt.get("sym"):
                print("       혼합 자리 %d개 -> 배열 수 %d 이하 (대칭으로 줄어듭니다)"
                      % (mixed, n_config))
            else:
                print("       혼합 자리 %d개 -> 배열 수 %d" % (mixed, n_config))
                _confirm_size(n_config, opt)

        if os.path.isfile("PRIM"):
            os.rename("PRIM", "PRIM_backup")
        prim.write("PRIM")
        made.append("PRIM")

        if not opt.get("sym"):
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
            print("       열거는 P1 으로, 구조 생성은 원본으로 합니다.")
            made.append("PRIM_orig")
        src_for_rest = prim              # 흔들기 전 구조로 거리·mesh 를 잰다
    else:
        src_for_rest = Prim.read(src) if os.path.basename(src).startswith("PRIM") \
            else src

    # -- CSPECS -------------------------------------------------------------
    if _wanted(opt, "cspecs") and opt.get("sym"):
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

    if opt["only"] is not None or opt.get("norun"):
        print("\nmainclust 는 돌리지 않았습니다. 이어서 하려면:")
        print("  CCpyCASMInputGen.py 1     열거부터 다시 (PRIM 을 새로 만든다)")
        print("  CCpyCASMInputGen.py 2     이미 열거했다면 그 뒤부터")
        return

    build(opt, restore="PRIM_orig" in made)


# ---------------------------------------------------------------------------
# [1-2] mainclust 로 열거하고 배열 디렉토리까지 만든다
# ---------------------------------------------------------------------------

def _preflight(opt):
    """돌리기 전에 준비물을 한 번에 확인한다.

    mainclust 는 준비물이 없어도 오류를 내지 않는다 — POTCAR_<원소> 가 없으면
    0바이트 POTCAR 를, INCAR/KPOINTS 가 없으면 디렉토리를 아예 안 만들면서
    "완료"를 찍는다. 열거에만 몇 분이 걸리므로 시작 전에 다 본다.
    """
    from CCpy.CASM import CASMrun as run

    try:
        binary = run.resolve_binary(workdir=".")
        templates = run.check_templates(".")
        potcars = run.check_potcar_sources(".")
    except run.MainclustError as err:
        print("\n%s" % err)
        quit()

    def _size(n):
        return "%.1f KB" % (n / 1024.0) if n >= 1024 else "%d B" % n

    print("\n* 준비물")
    print("    %-12s %s" % ("mainclust", binary))
    for name, path, size in templates:
        print("    %-12s %s" % (name, _size(size)))
    for elt, path, size in potcars:
        print("    %-12s %s" % (os.path.basename(path), _size(size)))
    return binary


def build(opt, restore=False):
    """mainclust 를 두 번 돌려 con* 까지 만든다.

    restore 가 True 면 (Method 1) 열거가 끝난 뒤 PRIM_orig 를 되돌린다.
    P1 으로 열거하고 대칭 구조로 생성하는 것이 Method 1 이고, 이 되돌리기를
    빠뜨리면 오류 없이 배열 전부가 흔들린 좌표로 만들어진다. 원본
    04_Asym_Alloy.sh 에서 이 줄이 주석 처리돼 있어 실제로 그렇게 나왔다.
    """
    from CCpy.CASM import CASMrun as run
    from CCpy.CASM.CASMkpoints import config_dirs

    if config_dirs("."):
        print("\n배열 디렉토리(con*)가 이미 있습니다.")
        print("다시 만들려면 지운 뒤 실행하시고, 손질만 하려면 옵션 2 를 쓰세요.")
        return

    _preflight(opt)

    if opt["interactive"] and not opt.get("yes"):
        if _ask("\n* mainclust 로 열거하고 배열 디렉토리까지 만들까요? (y/n)",
                "y").lower() not in ("y", "yes"):
            print("  입력 파일만 두고 멈췄습니다.")
            return

    try:
        print("\n[열거] mainclust ...")
        res = run.enumerate_configurations(".", max_volume=opt.get("vol", 1),
                                           dimension=3)
        print("       %s" % res.summary().replace("\n", "\n       "))

        if restore and os.path.isfile("PRIM_orig"):
            shutil.copy("PRIM_orig", "PRIM")
            print("       PRIM 을 대칭 원본으로 되돌렸습니다 (구조는 원본으로 만듭니다).")

        changed, total = run.set_make_flags("make_dirs")
        print("\n[make_dirs] %d / %d 를 1 로 바꿨습니다 (원본은 make_dirs_orig)."
              % (changed, total))

        print("\n[생성] mainclust ... 배열 %d개" % total)
        res2 = run.generate_vasp_inputs(".", energy=0, reference=0)
        print("       %s" % res2.summary().replace("\n", "\n       "))
        print(res2.potcar_report)
    except run.MainclustError as err:
        print("\n%s" % err)
        print("\n고친 뒤 옵션 2 로 이어서 하실 수 있습니다.")
        quit()

    dirs = config_dirs(".")
    print("\n배열 디렉토리 %d개" % len(dirs))
    _finish(opt, dirs)


# ---------------------------------------------------------------------------
# [2] 열거 후 손질
# ---------------------------------------------------------------------------

def post_enum(opt):
    from CCpy.CASM.CASMrun import set_make_flags, MainclustError
    from CCpy.CASM import CASMcellparam as cp
    from CCpy.CASM.CASMkpoints import (Kpoints, config_dirs, distribute,
                                       verify_uniform, describe_uniformity,
                                       KpointsError)

    if opt["only"]:
        unknown = set(opt["only"]) - {"makedirs", "cellparam", "kpoints"}
        if unknown:
            print("\n-only 에 모르는 값이 있습니다: %s" % ", ".join(sorted(unknown)))
            print("makedirs / cellparam / kpoints 중에서 골라 주세요.")
            quit()

    dirs = config_dirs(".")

    # -- make_dirs 플래그 ----------------------------------------------------
    total = None
    if _wanted(opt, "makedirs"):
        try:
            changed, total = set_make_flags("make_dirs")
            print("\n[make_dirs] %d / %d 를 1 로 바꿨습니다 (원본은 make_dirs_orig)."
                  % (changed, total))
        except MainclustError as err:
            print("\n[make_dirs] %s" % err)

    # -- 디렉토리가 아직 없으면 여기서 만든다 --------------------------------
    if not dirs and total:
        from CCpy.CASM import CASMrun as run
        if os.path.isfile("PRIM_orig"):
            shutil.copy("PRIM_orig", "PRIM")
            print("            PRIM 을 대칭 원본으로 되돌렸습니다.")
        _preflight(opt)
        go = True
        if opt["interactive"] and not opt.get("yes"):
            go = _ask("\n* 배열 %d개를 만들까요? (y/n)" % total, "y") \
                .lower() in ("y", "yes")
        if go:
            try:
                print("\n[생성] mainclust ...")
                res = run.generate_vasp_inputs(".", energy=0, reference=0)
                print("       %s" % res.summary().replace("\n", "\n       "))
                print(res.potcar_report)
            except run.MainclustError as err:
                print("\n%s" % err)
                quit()
            dirs = config_dirs(".")

    if not dirs:
        if _wanted(opt, "cellparam") or _wanted(opt, "kpoints"):
            print("\n배열 디렉토리(con*)가 아직 없어 나머지는 건너뜁니다.")
        return

    _finish(opt, dirs)


def _finish(opt, dirs):
    """배열 디렉토리가 생긴 뒤의 손질 — 셀 파라미터 평균과 KPOINTS 배포."""
    from CCpy.CASM import CASMcellparam as cp
    from CCpy.CASM.CASMkpoints import (Kpoints, distribute, verify_uniform,
                                       describe_uniformity, KpointsError)

    # -- 셀 파라미터 평균 ----------------------------------------------------
    if _wanted(opt, "cellparam"):
        lattice, refs = None, None
        elts = None
        try:
            elts = cp.elements_from_prim("PRIM")[:2]
        except cp.CellparamError as err:
            print("\n[cellparam] %s" % err)
            elts = None
        if elts:
            if opt.get("a"):
                if len(opt["a"]) != 2:
                    print("\n-a 는 두 원소의 격자상수 2개여야 합니다: -a=3.6271,3.8707")
                    quit()
                lattice = dict(zip(elts, opt["a"]))
            if opt.get("ref"):
                if len(opt["ref"]) != 2:
                    print("\n-ref 는 두 원소의 구조 파일 2개여야 합니다.")
                    quit()
                refs = dict(zip(elts, opt["ref"]))
            try:
                rec, notes, skipped = cp.apply(".", lattice=lattice, refs=refs,
                                               isotropic=opt.get("iso", False),
                                               dirs=dirs)
                print("\n[cellparam]")
                print("  " + cp.describe(rec, notes, skipped).replace("\n", "\n  "))
            except cp.CellparamError as err:
                print("\n[cellparam] %s" % err)

    # -- KPOINTS 배포 --------------------------------------------------------
    if _wanted(opt, "kpoints"):
        if not os.path.isfile("KPOINTS"):
            print("\n[KPOINTS] KPOINTS 가 없습니다. 옵션 1 로 먼저 만드세요.")
        else:
            try:
                kp = Kpoints.read("KPOINTS")
                if opt.get("kp"):
                    kp = Kpoints(opt["kp"], mode=kp.mode, comment=kp.comment)
                    kp.write("KPOINTS")
                distribute(kp, dirs)
                ok, groups = verify_uniform(dirs)
                print("\n[KPOINTS] %s %s 를 배열 %d개에 복사했습니다."
                      % (kp.mode, " ".join(str(v) for v in kp.mesh), len(dirs)))
                print("  " + describe_uniformity(groups).replace("\n", "\n  "))
            except KpointsError as err:
                print("\n[KPOINTS] %s" % err)

    print("\n다음 단계 — 제출:")
    print("  CCpyJobSubmit.py 2 I5 -batch -scratch -n=8")


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
    elif option == "2":
        post_enum(_parse_subopts(sys.argv[2:]))
    elif option == "9":
        prim_json_gen()
    else:
        print("\n모르는 옵션입니다: %s" % option)
        _help()
