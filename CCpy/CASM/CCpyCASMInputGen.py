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
    elif option == "9":
        prim_json_gen()
    else:
        print("\n모르는 옵션입니다: %s" % option)
        _help()
