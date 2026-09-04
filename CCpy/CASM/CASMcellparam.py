# -*- coding: utf-8 -*-
"""배열마다 셀을 조성 가중평균(Vegard)으로 다시 쓴다.

두 원소의 격자상수가 크게 다르면(Cu 3.627 Å vs Ir 3.871 Å) 어느 한쪽 값으로
출발한 완화가 잘 붙지 않는다. 각 배열의 조성비로 두 격자상수를 섞어 출발
셀을 만들어 주면 수렴이 빨라진다.

**이 셀은 출발 추정치일 뿐이다.** ISIF=3 으로 각 배열이 자기 부피로 이완해야
형성 에너지가 옳다. 실측을 보면 완화 후 부피가 1.6 % 까지 움직인다.

원본 ``03_cellparam.sh`` 의 세 가지 문제를 여기서 없앤다.

1. **멱등성.** 원본은 ``$list/POSCAR`` 를 읽어 같은 파일에 덮어썼다. 두 번
   돌리면 배율이 두 번 곱해진다. 여기서는 CASM 이 처음 뱉고 그 뒤로 아무도
   건드리지 않는 ``POS`` 를 읽어 ``POSCAR`` 로 쓴다. 몇 번을 돌려도 같다.
2. **인자 순서.** 원본은 ``03_cellparam.sh Ir Cu`` 로 뒤집어 줘도 그대로
   받아 모든 셀을 반대로 스케일했다. 여기서는 원소 순서를 PRIM 의 occupancy
   에서 읽는다. 사용자가 순서를 주면 PRIM 과 다른지 검사만 한다.
3. **등방 스케일.** 원본은 a·b·c 에 같은 배율을 곱했다. 여기서는 축마다
   대응하는 길이의 비를 쓴다. 입방-입방 쌍에서는 결과가 같고, 육방처럼
   c/a 가 다른 구조에서만 달라진다.

격자상수는 이 순서로 찾는다.

1. ``lattice`` 인자로 직접 준 값
2. ``refs`` 로 준 구조 파일
3. ``BULK/<원소>/CONTCAR`` 관례 (위로 올라가며 찾는다)
4. 아래 :data:`LATTICE_TABLE`

표는 문헌값이 아니라 **이 프로젝트에서 실제로 최적화한 BULK 계산 결과**다
(2026-09, PBE, ENCUT 400, EDIFF 1E-06). 출발 추정치로만 쓰이므로 설정이
조금 달라도 문제되지 않지만, 자기 BULK 파일이 있으면 그쪽이 항상 먼저다.
"""

import os

import numpy as np


class CellparamError(ValueError):
    """셀 파라미터 처리 중 발생한 오류."""


#: 원소 -> (a, c, 구조, POTCAR). 단위 Å.
#: 입방 구조는 a == c 로 채워 두었다.
LATTICE_TABLE = {
    "Ag": (4.144918, 4.144918, "fcc", "Ag"),
    "Au": (4.161795, 4.161795, "fcc", "Au"),
    "Cd": (3.073154, 5.444719, "hcp", "Cd"),
    "Co": (2.493935, 4.023521, "hcp", "Co"),
    "Cr": (2.835053, 2.835053, "bcc", "Cr"),
    "Cu": (3.627129, 3.627129, "fcc", "Cu"),
    "Fe": (2.828922, 2.828922, "bcc", "Fe"),
    "Hf": (3.194600, 5.051100, "hcp", "Hf"),
    "Ir": (3.870726, 3.870726, "fcc", "Ir"),
    "Mo": (3.152529, 3.152529, "bcc", "Mo"),
    "Nb": (3.308373, 3.308373, "bcc", "Nb_sv"),
    "Ni": (3.511475, 3.511475, "fcc", "Ni"),
    "Os": (2.750344, 4.354639, "hcp", "Os"),
    "Pd": (3.934689, 3.934689, "fcc", "Pd"),
    "Pt": (3.965701, 3.965701, "fcc", "Pt"),
    "Re": (2.777776, 4.468487, "hcp", "Re"),
    "Rh": (3.822364, 3.822364, "fcc", "Rh"),
    "Ru": (2.711545, 4.288621, "hcp", "Ru"),
    "Sc": (3.302217, 5.136205, "hcp", "Sc"),
    "Ta": (3.313380, 3.313380, "bcc", "Ta"),
    "Tc": (2.746784, 4.376789, "hcp", "Tc"),
    "Ti": (2.923401, 4.629729, "hcp", "Ti"),
    "V":  (2.979188, 2.979188, "bcc", "V"),
    "W":  (3.174238, 3.174238, "bcc", "W"),
    "Y":  (3.664575, 5.656533, "hcp", "Y_sv"),
    "Zn": (2.605339, 5.309382, "hcp", "Zn"),
    "Zr": (3.221385, 5.193937, "hcp", "Zr_sv"),
}

TABLE_NOTE = ("내장 표 (이 프로젝트 BULK 최적화 결과, PBE / ENCUT 400 / "
              "EDIFF 1E-06)")


# ----------------------------------------------------------------------------
# 격자상수 찾기
# ----------------------------------------------------------------------------

def _read_lengths(path):
    """구조 파일에서 격자벡터 길이 3개를 읽는다."""
    if not os.path.isfile(path):
        raise CellparamError("구조 파일이 없습니다: %s" % path)
    lines = open(path).read().split("\n")
    try:
        scale = float(lines[1].split()[0])
        mat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    except (IndexError, ValueError):
        raise CellparamError("%s 에서 격자를 읽지 못했습니다." % path)
    if scale < 0:                       # 음수 scale = 부피 지정
        scale = (abs(scale) / abs(np.linalg.det(mat))) ** (1.0 / 3.0)
    return np.linalg.norm(mat, axis=1) * scale


def _find_bulk(element, roots):
    for root in roots:
        for name in ("CONTCAR", "POSCAR"):
            p = os.path.join(root, element, name)
            if os.path.isfile(p):
                return p
    return None


def element_lengths(element, lattice=None, refs=None, roots=None):
    """한 원소의 격자벡터 길이 3개와 그 출처를 돌려준다.

    Returns
    -------
    (lengths, source)
    """
    if lattice is not None:
        v = np.atleast_1d(np.asarray(lattice, dtype=float))
        if v.size == 1:
            v = np.repeat(v, 3)
        elif v.size == 2:                # (a, c) 로 준 경우
            v = np.array([v[0], v[0], v[1]])
        if v.size != 3:
            raise CellparamError("격자상수는 1개(a), 2개(a c), 3개(a b c) 중 "
                                 "하나로 주세요: %r" % (lattice,))
        return v, "직접 지정"

    if refs and element in refs:
        return _read_lengths(refs[element]), refs[element]

    if roots is None:
        roots = default_bulk_roots()
    found = _find_bulk(element, roots)
    if found:
        return _read_lengths(found), found

    if element in LATTICE_TABLE:
        a, c, _, _ = LATTICE_TABLE[element]
        return np.array([a, a, c]), TABLE_NOTE

    raise CellparamError(
        "%s 의 격자상수를 찾지 못했습니다.\n"
        "BULK/%s/CONTCAR 를 두거나, -ref 로 경로를, 또는 -a 로 값을 주세요.\n"
        "내장 표에 있는 원소: %s"
        % (element, element, ", ".join(sorted(LATTICE_TABLE))))


def default_bulk_roots(start="."):
    """BULK 폴더가 있을 만한 곳을 가까운 순서로."""
    out = []
    cur = os.path.abspath(start)
    for _ in range(4):
        out.append(os.path.join(cur, "BULK"))
        out.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


# ----------------------------------------------------------------------------
# 적용
# ----------------------------------------------------------------------------

def elements_from_prim(path="PRIM"):
    """PRIM 의 occupancy 순서에서 원소 순서를 읽는다."""
    from CCpy.CASM.CASMprim import Prim, PrimError
    try:
        prim = Prim.read(path)
    except PrimError as err:
        raise CellparamError("PRIM 을 읽지 못했습니다: %s" % err)
    elts = prim.elements
    if len(elts) < 2:
        raise CellparamError(
            "PRIM 에 원소가 %d개뿐입니다 (%s). 이원계여야 조성 가중평균이 "
            "의미가 있습니다." % (len(elts), ", ".join(elts) or "없음"))
    return elts


def read_concentrations(path="make_dirs"):
    """make_dirs 에서 {배열이름: 조성} 을 읽는다.

    3열은 첫 원소의 분율이다.
    """
    if not os.path.isfile(path):
        raise CellparamError(
            "%s 가 없습니다. mainclust 로 열거를 먼저 하세요." % path)
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 3 or parts[0].startswith("#"):
            continue
        try:
            out[parts[0]] = float(parts[2])
        except ValueError:
            continue
    if not out:
        raise CellparamError("%s 에서 조성을 읽지 못했습니다." % path)
    return out


def apply(root=".", elements=None, lattice=None, refs=None, roots=None,
          isotropic=False, dirs=None):
    """모든 배열의 POS 를 조성 가중평균 셀로 스케일해 POSCAR 로 쓴다.

    Parameters
    ----------
    root : str
    elements : (str, str) or None
        원소 순서. None 이면 PRIM 에서 읽는다. 주면 PRIM 과 같은지 검사한다.
    lattice : dict or None
        ``{"Cu": 3.6271, "Ir": [3.87, 3.87, 3.87]}`` 처럼 직접 지정.
    refs : dict or None
        ``{"Cu": "../BULK/Cu/CONTCAR"}`` 처럼 파일로 지정.
    roots : list[str] or None
        BULK 를 찾을 곳.
    isotropic : bool
        True 면 a 축 비 하나로 세 축을 모두 스케일한다(원본 동작).
    dirs : list[str] or None

    Returns
    -------
    (records, notes)
    """
    prim_elts = elements_from_prim(os.path.join(root, "PRIM"))
    if elements is None:
        elements = prim_elts[:2]
    else:
        elements = list(elements)
        if len(elements) != 2:
            raise CellparamError("원소는 2개여야 합니다: %r" % (elements,))
        if elements != prim_elts[:2]:
            raise CellparamError(
                "주신 순서 %s 가 PRIM 의 순서 %s 와 다릅니다.\n"
                "순서를 뒤집으면 조성 가중이 반대로 걸려 모든 셀이 틀린 부피에서 "
                "출발합니다." % (" ".join(elements), " ".join(prim_elts[:2])))
    e1, e2 = elements

    lattice = lattice or {}
    l1, src1 = element_lengths(e1, lattice.get(e1), refs, roots)
    l2, src2 = element_lengths(e2, lattice.get(e2), refs, roots)

    ratio = np.asarray(l2, dtype=float) / np.asarray(l1, dtype=float)
    if isotropic:
        ratio = np.repeat(ratio[0], 3)

    notes = ["%s : %s  (%s)" % (e1, " ".join("%.6f" % v for v in l1), src1),
             "%s : %s  (%s)" % (e2, " ".join("%.6f" % v for v in l2), src2),
             "축별 배율 %s%s" % (" ".join("%.6f" % v for v in ratio),
                              "  (등방)" if isotropic else "")]
    if max(ratio) / min(ratio) > 1.02 and not isotropic:
        notes.append("두 구조의 c/a 가 달라 축마다 배율이 다릅니다. 원본 "
                     "03_cellparam.sh 처럼 등방으로 하려면 -iso 를 주세요.")
    spread = abs(ratio[0] - 1.0)
    if spread > 0.10:
        notes.append("경고: 격자상수 차이가 %.1f %% 입니다. 완화가 잘 붙지 "
                     "않을 수 있습니다." % (spread * 100))

    concentrations = read_concentrations(os.path.join(root, "make_dirs"))

    if dirs is None:
        from CCpy.CASM.CASMkpoints import config_dirs
        dirs = config_dirs(root)
    if not dirs:
        raise CellparamError("배열 디렉토리(con*)를 찾지 못했습니다.")

    records, skipped = [], []
    for d in dirs:
        name = os.path.basename(d)
        if name not in concentrations:
            skipped.append((name, "make_dirs 에 없음"))
            continue
        src = os.path.join(d, "POS")
        if not os.path.isfile(src):
            skipped.append((name, "POS 가 없음 (CASM 원본 구조)"))
            continue
        x = concentrations[name]
        factor = x + ratio * (1.0 - x)
        try:
            _scale_poscar(src, os.path.join(d, "POSCAR"), factor)
        except CellparamError as err:
            skipped.append((name, str(err)))
            continue
        records.append({"name": name, "x": x, "factor": tuple(factor)})

    return records, notes, skipped


def _scale_poscar(src, dst, factor):
    """POS 의 격자벡터를 축마다 factor 로 곱해 POSCAR 로 쓴다."""
    lines = open(src).read().split("\n")
    if len(lines) < 7:
        raise CellparamError("구조 파일이 너무 짧습니다: %s" % src)
    out = list(lines)
    for i, f in zip((2, 3, 4), factor):
        parts = lines[i].split()
        if len(parts) < 3:
            raise CellparamError("%d행이 격자벡터가 아닙니다: %r" % (i + 1, lines[i]))
        vec = [float(v) * f for v in parts[:3]]
        out[i] = "  %15.7f %15.7f %15.7f " % tuple(vec)
    with open(dst, "w") as fh:
        fh.write("\n".join(out))


def describe(records, notes, skipped, limit=4):
    """apply() 결과를 사람이 읽을 문장으로."""
    lines = list(notes)
    lines.append("배열 %d개의 셀을 다시 썼습니다." % len(records))
    if records:
        xs = [r["x"] for r in records]
        lines.append("  조성 범위 %.4g ~ %.4g" % (min(xs), max(xs)))
        for r in sorted(records, key=lambda r: r["x"])[:2] + \
                sorted(records, key=lambda r: -r["x"])[:1]:
            lines.append("    %-10s x=%-6.4g 배율 %s"
                         % (r["name"], r["x"],
                            " ".join("%.6f" % v for v in r["factor"])))
    if skipped:
        lines.append("  건너뛴 것 %d개:" % len(skipped))
        for name, why in skipped[:limit]:
            lines.append("    %-10s %s" % (name, why))
        if len(skipped) > limit:
            lines.append("    ... (%d개 더)" % (len(skipped) - limit))
    lines.append("  POS 를 읽어 POSCAR 로 쓰므로 여러 번 돌려도 결과가 같습니다.")
    return "\n".join(lines)
