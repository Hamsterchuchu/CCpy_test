#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from CCpy.Tools.CCpyTools import selectInputs

version = sys.version
if version[0] == '3':
    raw_input = input


def _help():
    print("\nHow to use : " + os.path.basename(sys.argv[0]) + " [option]")
    print("""--------------------------------------
[1] : PRIM 생성          (CASM 텍스트 형식, scale 스크립트 대체)
[2] : PRIM 대칭 낮추기   (P1 - 전수 열거를 하려면 필요)
[3] : CSPECS 생성        (최근접 거리 측정, neighbors.pl 대체)
[4] : KPOINTS 생성·통일  (셀에 맞는 mesh 를 정해 전 배열에 복사)
[7] : 다음 완화 준비     (CONTCAR -> POSCAR, Pulay stress 걷어내기)
[8] : 최종 정적 준비     (NSW=0, ISMEAR=-5 로 에너지 확정)
[9] : prim.json 생성     (신형 CASM 형식)
--------------------------------------""")
    quit()


def _ask(prompt, default=None):
    if default is None:
        return raw_input("%s\n: " % prompt).strip()
    got = raw_input("%s [%s]\n: " % (prompt, default)).strip()
    return got if got else default


# ---------------------------------------------------------------------------
# [1] PRIM 생성
# ---------------------------------------------------------------------------

def prim_gen():
    from CCpy.CASM.CASMprim import make_prim, read_structure, PrimError

    input_marker = [".xsd", ".cif", "POSCAR", "CONTCAR"]
    inputs = selectInputs(input_marker, "./")
    if not inputs:
        print("구조 파일을 찾지 못했습니다 (.cif / POSCAR / CONTCAR).")
        quit()
    filename = inputs[0]
    if len(inputs) > 1:
        print("\n여러 개를 고르셨습니다. 첫 번째 파일만 씁니다: %s" % filename)

    try:
        _, _, species = read_structure(filename)
    except PrimError as err:
        print("\n%s" % err)
        quit()

    ordered = []
    for s in species:
        if s not in ordered:
            ordered.append(s)
    print("\n* %s : 원자 %d개, 원소 %s" % (filename, len(species), ", ".join(ordered)))

    sc = _ask("\n* Supercell 배수 a b c"
              "\n  (FCC 는 2 1 1, HCP/BCC 는 2 2 1 이 8 자리 셀을 줍니다. "
              "그대로 쓰려면 1 1 1)", "1 1 1")
    try:
        nx, ny, nz = [int(v) for v in sc.replace(",", " ").split()]
    except ValueError:
        print("배수는 정수 3개여야 합니다: %r" % sc)
        quit()

    print("\n* 각 자리에 올 수 있는 원소를 적습니다. 빈자리는 Vac 로 적습니다.")
    print("  (예: 이원계 합금 'Cu Ir' / Li 자리 'Li Vac' / 고정 자리 'C')")
    occupancy = {}
    for s in ordered:
        occupancy[s] = _ask("  %s 자리" % s, s)

    default_title = "-".join(
        [e for occ in occupancy.values() for e in occ.split()
         if e != "Vac"] or ordered)
    seen, title_elems = set(), []
    for e in default_title.split("-"):
        if e not in seen:
            seen.add(e)
            title_elems.append(e)
    title = _ask("\n* PRIM title", "-".join(title_elems))

    try:
        prim = make_prim(filename, occupancy=occupancy,
                         supercell=(nx, ny, nz), title=title)
    except PrimError as err:
        print("\n%s" % err)
        quit()

    if os.path.isfile("PRIM"):
        os.rename("PRIM", "PRIM_backup")
        print("\n기존 PRIM 을 PRIM_backup 으로 옮겼습니다.")
    prim.write("PRIM")

    a, b, c = prim.lengths
    print("\nPRIM 생성 완료")
    print("  자리 수   : %d" % len(prim))
    print("  격자      : %.6f  %.6f  %.6f" % (a, b, c))
    print("  구성      : " + " / ".join(
        "%d자리 %s" % (n, " ".join(occ)) for n, occ in prim.groups()))
    mixed = len(prim.mixed_sites)
    if mixed:
        print("  혼합 자리 : %d개 -> 배열 수 %d" % (mixed, 2 ** mixed))
        print("\n  대칭이 살아 있으면 CASM 이 대칭인 배열을 하나로 묶습니다.")
        print("  전수 열거를 하시려면 옵션 2 로 P1 까지 낮추세요.")


# ---------------------------------------------------------------------------
# [2] PRIM 대칭 낮추기
# ---------------------------------------------------------------------------

def prim_break_symmetry():
    from CCpy.CASM.CASMprim import Prim, break_symmetry, PrimError

    path = _ask("\n* PRIM 파일", "PRIM")
    if not os.path.isfile(path):
        print("%s 가 없습니다." % path)
        quit()
    try:
        prim = Prim.read(path)
    except PrimError as err:
        print("\n%s" % err)
        quit()

    print("\n* %s : 자리 %d개, 혼합 자리 %d개"
          % (path, len(prim), len(prim.mixed_sites)))

    amp = _ask("\n* 흔들 폭 (분수좌표)"
               "\n  CASM 이 대칭을 못 찾고, VASP 도 SYMPREC 으로 되살리지 못할 "
               "만큼 커야 합니다", "0.001")
    try:
        amp = float(amp)
    except ValueError:
        print("숫자여야 합니다: %r" % amp)
        quit()

    try:
        out, msgs = break_symmetry(prim, amplitude=amp)
    except PrimError as err:
        print("\n%s" % err)
        quit()

    backup = path + "_orig"
    if not os.path.isfile(backup):
        os.rename(path, backup)
    out.write(path)

    for m in msgs:
        print("  " + m)
    print("\n대칭을 낮췄습니다.")
    print("  원본       : %s" % backup)
    print("  흔든 PRIM  : %s" % path)
    print("\n  주의 - 열거가 끝나 make_dirs 를 만든 뒤에는 원본 PRIM 을 되돌린 다음")
    print("  디렉토리를 생성해야 합니다. 흔든 좌표를 그대로 두면 만들어지는 구조")
    print("  전부가 섭동을 안은 채로 출발합니다.")
    print("    cp %s %s" % (backup, path))


# ---------------------------------------------------------------------------
# [3] CSPECS 생성
# ---------------------------------------------------------------------------

def cspecs_gen():
    from CCpy.CASM.CASMcspecs import (neighbor_shells, describe_shells,
                                      suggest_radius, Cspecs, CspecsError)
    from CCpy.CASM.CASMprim import Prim, PrimError

    if os.path.isfile("PRIM"):
        src = _ask("\n* 거리를 잴 구조", "PRIM")
    else:
        inputs = selectInputs([".cif", "POSCAR", "CONTCAR", "PRIM"], "./")
        if not inputs:
            print("구조 파일을 찾지 못했습니다.")
            quit()
        src = inputs[0]

    try:
        structure = Prim.read(src) if os.path.basename(src).startswith("PRIM") \
            else src
    except PrimError as err:
        print("\n%s" % err)
        quit()

    species = _ask("\n* 특정 원소끼리의 거리만 볼까요?"
                   "\n  (Li-C 처럼 한 원소의 배열이 문제일 때 씁니다. 전부 보려면 그냥 Enter)",
                   "")
    species = species.replace(",", " ").split() or None

    try:
        shells = neighbor_shells(structure, species=species)
    except (CspecsError, PrimError) as err:
        print("\n%s" % err)
        quit()

    print("\n%s" % describe_shells(shells))

    nshell = _ask("\n* 몇 번째 껍질까지 한 클러스터로 볼까요?"
                  "\n  (1 = 최근접만. 늘리면 비등가 배열 수가 급격히 커집니다)", "1")
    try:
        nshell = int(nshell)
        proposed = suggest_radius(shells, nshell=nshell)
    except (ValueError, CspecsError) as err:
        print("\n%s" % err)
        quit()

    radius = _ask("\n* 반경 (Å)"
                  "\n  %d껍질 %.6f Å 와 %s 사이 값입니다"
                  % (nshell, shells[nshell - 1][0],
                     ("%d껍질 %.6f Å" % (nshell + 1, shells[nshell][0]))
                     if nshell < len(shells) else "그 바깥"),
                  "%g" % proposed)
    try:
        radius = float(radius)
    except ValueError:
        print("숫자여야 합니다: %r" % radius)
        quit()

    sizes = _ask("\n* 클러스터 크기", "2 3 4")
    try:
        sizes = [int(s) for s in sizes.replace(",", " ").split()]
        cs = Cspecs(dict((s, radius) for s in sizes))
    except (ValueError, CspecsError) as err:
        print("\n%s" % err)
        quit()

    if os.path.isfile("CSPECS"):
        os.rename("CSPECS", "CSPECS_backup")
        print("\n기존 CSPECS 를 CSPECS_backup 으로 옮겼습니다.")
    cs.write("CSPECS")

    print("\nCSPECS 생성 완료")
    print(cs.to_string())
    inside = sum(n for d, n in shells if d <= radius)
    print("  반경 %g Å 안의 이웃 : %d개" % (radius, inside))


# ---------------------------------------------------------------------------
# [4] KPOINTS 생성·통일
# ---------------------------------------------------------------------------

def kpoints_gen():
    from CCpy.CASM.CASMkpoints import (make_kpoints, config_dirs, distribute,
                                       verify_uniform, describe_uniformity,
                                       DEFAULT_TARGET, KpointsError)
    from CCpy.CASM.CASMprim import PrimError

    dirs = config_dirs(".")
    if dirs:
        try:
            ok, groups = verify_uniform(dirs)
            print("\n* 현재 상태 : " + describe_uniformity(groups))
            if not ok:
                print("\n  배열마다 mesh 가 다르면 그 계단이 형성 에너지 차이로 "
                      "둔갑합니다.")
                print("  셀 크기가 서로 같은 배열끼리는(Method 1) 하나로 통일해야 "
                      "합니다.")
                print("  셀 부피가 다른 배열이 섞여 있으면(Method 2) mainclust 가 "
                      "SCEL 배율만큼")
                print("  나눠 k-밀도를 맞춘 것이니 그대로 두는 편이 맞습니다.")
        except KpointsError as err:
            print("\n* 현재 상태를 읽지 못했습니다: %s" % err)

    src = _ask("\n* mesh 를 정할 기준 구조", "PRIM")
    if not os.path.isfile(src):
        print("%s 가 없습니다." % src)
        quit()

    target = _ask("\n* 목표 k*L (Å)"
                  "\n  금속은 35~40, 절연체는 20 안팎입니다", "%g" % DEFAULT_TARGET)
    try:
        target = float(target)
    except ValueError:
        print("숫자여야 합니다: %r" % target)
        quit()

    try:
        kp, info = make_kpoints(src, target=target)
    except (KpointsError, PrimError) as err:
        print("\n%s" % err)
        quit()

    print("")
    for m in info:
        print("  " + m)

    mode = _ask("\n* mode", kp.mode)
    if mode[:1].upper() != kp.mode[:1].upper():
        from CCpy.CASM.CASMkpoints import Kpoints
        kp = Kpoints(kp.mesh, mode=mode, comment=kp.comment)

    mesh = _ask("\n* mesh (그대로 쓰려면 Enter)",
                " ".join(str(v) for v in kp.mesh))
    try:
        from CCpy.CASM.CASMkpoints import Kpoints
        kp = Kpoints([int(v) for v in mesh.replace(",", " ").split()],
                     mode=kp.mode, comment=kp.comment)
    except (ValueError, KpointsError) as err:
        print("\n%s" % err)
        quit()

    kp.write("KPOINTS")
    print("\nKPOINTS 생성 완료")
    print(kp.to_string())

    if not dirs:
        print("  배열 디렉토리가 아직 없습니다. mainclust 로 con* 를 만든 뒤")
        print("  이 옵션을 다시 돌려 전 배열에 복사하세요.")
        return

    go = _ask("* 배열 %d개에 이 KPOINTS 를 복사할까요? (y/n)" % len(dirs), "y")
    if go.lower() not in ("y", "yes"):
        print("  복사하지 않았습니다.")
        return

    n = distribute(kp, dirs)
    ok, groups = verify_uniform(dirs)
    print("\n  %d개에 기록했습니다." % n)
    print("  확인 : " + describe_uniformity(groups))
    if not ok:
        print("  복사가 끝났는데도 어긋난 것이 있습니다. 위 목록을 확인해 주세요.")


# ---------------------------------------------------------------------------
# [7] / [8] 다음 단계 준비
# ---------------------------------------------------------------------------

def _stage_common(stage):
    from CCpy.CASM import CASMstage as st
    from CCpy.CASM.CASMkpoints import config_dirs

    dirs = config_dirs(".")
    if not dirs:
        print("\n배열 디렉토리(con*)를 찾지 못했습니다.")
        quit()

    ready = [d for d in dirs if st.is_finished(d)]
    print("\n* 배열 %d개 중 %d개가 끝났습니다(vasp.done 기준)."
          % (len(dirs), len(ready)))
    if len(ready) < len(dirs):
        print("  아직 안 끝난 %d개는 건너뜁니다." % (len(dirs) - len(ready)))

    bad = st.unconverged_dirs(".")
    if bad is None:
        print("\n  01_unconverged_jobs.csv 가 없습니다.")
        print("  CCpyVASPAnal.py 0 을 먼저 돌리면 미수렴 배열을 자동으로 "
              "건너뜁니다.")
        skip = _ask("  그래도 진행할까요? (y/n)", "n")
        if skip.lower() not in ("y", "yes"):
            quit()
    else:
        print("  미수렴으로 표시된 배열 %d개는 건너뜁니다." % len(bad))

    # 현재 상태를 몇 개만 보여 준다
    sample = ready[:3]
    if sample:
        print("\n  현재 상태(앞 %d개):" % len(sample))
        for d in sample:
            try:
                drift = st.volume_drift(d)
                ent = st.entropy_per_atom(d)
                print("    %-10s 부피변화 %+6.2f %%   T*S %.3f meV/atom   NKPTS %s"
                      % (os.path.basename(d), drift,
                         ent if ent is not None else float("nan"), st.nkpts(d)))
            except st.StageError:
                pass

    ispin = _ask("\n* ISPIN (그대로 두려면 Enter)", "")
    ispin = int(ispin) if ispin else None

    go = _ask("\n* 진행할까요? 이전 결과는 접미사를 붙여 남깁니다. (y/n)", "y")
    if go.lower() not in ("y", "yes"):
        print("  아무것도 하지 않았습니다.")
        quit()

    done, skipped = st.advance_all(".", stage=stage, ispin=ispin)
    print("\n" + st.describe(done, skipped))
    return done


def stage_relax():
    done = _stage_common("relax")
    if done:
        print("\n  이제 다시 제출하세요:")
        print("    CCpyJobSubmit.py 2 I5 -batch -scratch -n=8")
        print("\n  끝나면 부피가 더 움직이는지 보고, 멎었으면 옵션 8 로 최종 "
              "정적 계산을 준비하세요.")


def stage_static():
    done = _stage_common("static")
    if done:
        ismears = {str(d.get("ismear")) for d in done}
        print("\n  ISMEAR %s 로 정적 계산 입력을 만들었습니다." % ", ".join(sorted(ismears)))
        if "1" in ismears:
            print("  k-점이 4개 미만인 배열은 사면체법(-5)을 쓸 수 없어 "
                  "ISMEAR=1, SIGMA=0.05 로 두었습니다.")
        print("\n  이제 다시 제출하세요:")
        print("    CCpyJobSubmit.py 2 I5 -batch -scratch -n=8")
        print("\n  이 계산의 OUTCAR 이 최종 에너지입니다. CCpyCASMhull.py 는 "
              "OUTCAR 을 읽으므로")
        print("  따로 손댈 것 없이 hull 이 정적 에너지로 그려집니다.")


# ---------------------------------------------------------------------------
# [9] prim.json (신형 CASM)
# ---------------------------------------------------------------------------

def prim_json_gen():
    from CCpy.CASM.CASMio import CASMInput

    input_marker = [".xsd", ".cif", "POSCAR", "CONTCAR"]
    inputs = selectInputs(input_marker, "./")
    for each_input in inputs:
        CI = CASMInput(each_input)
        CI.primGen()


if __name__ == "__main__":
    try:
        option = sys.argv[1]
    except IndexError:
        _help()

    if option == "1":
        prim_gen()
    elif option == "2":
        prim_break_symmetry()
    elif option == "3":
        cspecs_gen()
    elif option == "4":
        kpoints_gen()
    elif option == "7":
        stage_relax()
    elif option == "8":
        stage_static()
    elif option == "9":
        prim_json_gen()
    else:
        print("\n모르는 옵션입니다: %s" % option)
        _help()
