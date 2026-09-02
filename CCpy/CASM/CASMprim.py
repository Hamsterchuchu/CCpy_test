# -*- coding: utf-8 -*-
"""CASM PRIM 파일 생성·편집 모듈.

PRIM 은 CASM 이 읽는 기본 셀 정의다. POSCAR 와 거의 같은 형식인데 두 가지가
다르다.

1. 원소 목록 줄(POSCAR 6행)이 없다. 원자 개수 줄만 남는다.
2. 좌표 뒤에 그 자리에 올 수 있는 원소를 모두 적는다(occupancy).

   0.00000000  0.00000000  0.00000000  Cu Ir     <- Cu 또는 Ir
   0.00000000  0.00000000  0.00000000  Li Vac    <- Li 또는 빈자리
   0.00000000  0.16681600  0.12500000  C         <- C 고정

원래 이 파일은 VTST 의 ``scale`` (csh + awk) 로 supercell 을 만든 뒤 손으로
6행을 지우고 원소 이름을 붙여 만들었다. ``scale`` 은 POTCAR 를 요구하지만
거기서 읽는 값(TITEL, RWIGS)은 마지막에 삭제되는 임시 파일에만 쓰이므로
실제로는 필요 없다. 이 모듈은 POTCAR 도 csh 도 없이 같은 일을 한다.

자리 복제 순서는 ``scale`` 과 같게 맞췄다(원자마다 z→x→y 순으로 이미지 생성).
이원계처럼 전 자리 occupancy 가 같으면 순서는 결과에 영향이 없지만, Li-C 처럼
자리마다 occupancy 가 다르면 순서가 곧 의미이므로 원본과 어긋나지 않게 둔다.
"""

import os
import re

import numpy as np


VACANCY = "Vac"


class PrimError(ValueError):
    """PRIM 을 만들거나 읽는 중 발생한 오류."""


class Prim(object):
    """PRIM 파일 한 벌.

    Attributes
    ----------
    title : str
    scale : float
    lattice : (3,3) ndarray
        행이 격자벡터 a, b, c.
    coords : (N,3) ndarray
        분수좌표.
    occupancies : list[list[str]]
        자리마다 올 수 있는 원소 목록. ``["Cu","Ir"]``, ``["Li","Vac"]``, ``["C"]``.
    """

    def __init__(self, title, scale, lattice, coords, occupancies):
        self.title = title
        self.scale = float(scale)
        self.lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
        self.coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        self.occupancies = [list(o) for o in occupancies]
        if len(self.coords) != len(self.occupancies):
            raise PrimError("좌표 수(%d)와 occupancy 수(%d)가 다릅니다."
                            % (len(self.coords), len(self.occupancies)))

    # -- 성질 --------------------------------------------------------------

    def __len__(self):
        return len(self.coords)

    @property
    def lengths(self):
        """격자벡터 길이 a, b, c (Å). scale 을 곱한 값이다."""
        return np.linalg.norm(self.lattice, axis=1) * self.scale

    @property
    def elements(self):
        """등장하는 모든 원소(Vac 제외)를 처음 나온 순서대로."""
        seen = []
        for occ in self.occupancies:
            for e in occ:
                if e != VACANCY and e not in seen:
                    seen.append(e)
        return seen

    @property
    def mixed_sites(self):
        """occupancy 가 둘 이상인 자리의 인덱스."""
        return [i for i, o in enumerate(self.occupancies) if len(o) > 1]

    def groups(self):
        """occupancy 가 같은 자리를 연속 구간으로 묶어 [(개수, occupancy)] 로."""
        out = []
        for occ in self.occupancies:
            if out and out[-1][1] == occ:
                out[-1][0] += 1
            else:
                out.append([1, list(occ)])
        return [(n, o) for n, o in out]

    # -- 입출력 ------------------------------------------------------------

    def to_string(self):
        lines = [self.title,
                 "%20.14f" % self.scale]
        for v in self.lattice:
            lines.append("  %14.8f %14.8f %14.8f" % tuple(v))
        counts = [n for n, _ in self.groups()]
        lines.append(" " + " ".join(str(n) for n in counts))
        lines.append("Direct")
        for xyz, occ in zip(self.coords, self.occupancies):
            lines.append("  %14.8f %14.8f %14.8f   %s"
                         % (xyz[0], xyz[1], xyz[2], " ".join(occ)))
        return "\n".join(lines) + "\n"

    def write(self, path="PRIM"):
        with open(path, "w") as f:
            f.write(self.to_string())
        return path

    @classmethod
    def from_string(cls, text):
        raw = text.rstrip("\n").split("\n")
        if len(raw) < 8:
            raise PrimError("PRIM 이 너무 짧습니다(%d줄). 최소 8줄이 필요합니다." % len(raw))

        title = raw[0].rstrip()
        try:
            scale = float(raw[1].split()[0])
        except (IndexError, ValueError):
            raise PrimError("2행에서 scale 값을 읽지 못했습니다: %r" % raw[1])

        lattice = []
        for i in (2, 3, 4):
            parts = raw[i].split()
            if len(parts) < 3:
                raise PrimError("%d행이 격자벡터가 아닙니다: %r" % (i + 1, raw[i]))
            lattice.append([float(x) for x in parts[:3]])

        counts_line = raw[5].split()
        if not counts_line or not all(p.isdigit() for p in counts_line):
            raise PrimError(
                "6행이 원자 개수 줄이 아닙니다: %r\n"
                "PRIM 에는 원소 이름 줄이 없어야 합니다 — POSCAR 를 그대로 쓰셨는지 "
                "확인해 주세요." % raw[5])
        total = sum(int(p) for p in counts_line)

        mode = raw[6].strip()
        if not mode[:1].lower() in ("d", "c", "k"):
            raise PrimError("7행이 좌표계 표시(Direct/Cartesian)가 아닙니다: %r" % raw[6])
        if mode[:1].lower() != "d":
            raise PrimError("Direct(분수) 좌표만 지원합니다. 현재: %r" % mode)

        coords, occs = [], []
        for i in range(total):
            idx = 7 + i
            if idx >= len(raw):
                raise PrimError("좌표가 %d개여야 하는데 %d개뿐입니다." % (total, len(coords)))
            parts = raw[idx].split()
            if len(parts) < 3:
                raise PrimError("%d행에서 좌표를 읽지 못했습니다: %r" % (idx + 1, raw[idx]))
            coords.append([float(x) for x in parts[:3]])
            occ = [p for p in parts[3:] if re.match(r"^[A-Za-z]", p)]
            if not occ:
                raise PrimError(
                    "%d행에 occupancy 가 없습니다: %r\n"
                    "각 좌표 뒤에 그 자리에 올 수 있는 원소를 적어야 합니다 "
                    "(예: 'Cu Ir', 'Li Vac', 'C')." % (idx + 1, raw[idx]))
            occs.append(occ)

        return cls(title, scale, lattice, coords, occs)

    @classmethod
    def read(cls, path="PRIM"):
        with open(path) as f:
            return cls.from_string(f.read())

    def copy(self):
        return Prim(self.title, self.scale, self.lattice.copy(),
                    self.coords.copy(), [list(o) for o in self.occupancies])

    def __repr__(self):                                # pragma: no cover
        a, b, c = self.lengths
        return ("<Prim %r %d sites %.4f x %.4f x %.4f>"
                % (self.title, len(self), a, b, c))


# ----------------------------------------------------------------------------
# 구조 읽기
# ----------------------------------------------------------------------------

def read_structure(path):
    """POSCAR / CONTCAR / cif 등에서 (lattice, coords, species) 를 읽는다.

    CASM 이 뱉는 POS 는 원소 이름 줄이 없는 VASP4 형식이라 pymatgen 이 같은
    폴더의 POTCAR 를 참조한다. POTCAR 마저 없으면 원소를 알 수 없으므로
    무엇이 없어서 실패했는지 알려 준다.
    """
    from pymatgen.core import Structure

    if not os.path.isfile(path):
        raise PrimError("구조 파일이 없습니다: %s" % path)
    try:
        st = Structure.from_file(path)
    except Exception as err:
        base = os.path.basename(path)
        hint = ""
        if base.startswith(("POSCAR", "CONTCAR", "POS")):
            hint = ("\n원소 이름 줄이 없는 VASP4 형식이면 같은 폴더에 POTCAR 가 "
                    "있어야 원소를 알아낼 수 있습니다.")
        raise PrimError("%s 를 읽지 못했습니다: %s%s" % (path, err, hint))

    lattice = np.array(st.lattice.matrix, dtype=float)
    coords = np.array(st.frac_coords, dtype=float)
    species = [str(s.specie.symbol) for s in st.sites]
    return lattice, coords, species


# ----------------------------------------------------------------------------
# supercell + occupancy -> PRIM
# ----------------------------------------------------------------------------

def _replicate(lattice, coords, species, nx, ny, nz):
    """scale 과 같은 순서로 셀을 복제한다.

    원자 하나마다 z -> x -> y 순으로 이미지를 만든다. 격자벡터 a, b, c 는
    각각 nx, ny, nz 배가 된다.
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    if min(nx, ny, nz) < 1:
        raise PrimError("supercell 배수는 1 이상이어야 합니다: (%d, %d, %d)" % (nx, ny, nz))

    new_lat = np.array(lattice, dtype=float).copy()
    new_lat[0] *= nx
    new_lat[1] *= ny
    new_lat[2] *= nz

    new_coords, new_species = [], []
    for (x, y, z), sp in zip(coords, species):
        for iz in range(nz):
            for ix in range(nx):
                for iy in range(ny):
                    new_coords.append([(x + ix) / nx, (y + iy) / ny, (z + iz) / nz])
                    new_species.append(sp)
    return new_lat, np.array(new_coords), new_species


def _normalize_occupancy(occupancy):
    if isinstance(occupancy, str):
        occ = occupancy.replace(",", " ").split()
    else:
        occ = [str(e) for e in occupancy]
    if not occ:
        raise PrimError("occupancy 가 비어 있습니다.")
    return occ


def _resolve_occupancies(occupancy, species):
    """occupancy 지정을 자리별 목록으로 푼다.

    받아들이는 형태
      "Cu Ir" / ["Cu","Ir"]        : 모든 자리에 같은 occupancy
      {"Li": "Li Vac", "C": "C"}   : 원본 원소별로 지정
      {0: "Li Vac", 3: "C"}        : supercell 이후 자리 번호별로 지정
    """
    n = len(species)

    if isinstance(occupancy, (str, list, tuple)):
        occ = _normalize_occupancy(occupancy)
        return [list(occ) for _ in range(n)]

    if not isinstance(occupancy, dict):
        raise PrimError("occupancy 는 문자열, 리스트, 사전 중 하나여야 합니다. "
                        "받은 형: %s" % type(occupancy).__name__)

    by_index = all(isinstance(k, int) for k in occupancy)
    by_symbol = all(isinstance(k, str) for k in occupancy)
    if not (by_index or by_symbol):
        raise PrimError("occupancy 사전의 키는 자리 번호(int) 아니면 "
                        "원소 기호(str) 로 통일해야 합니다.")

    out = []
    if by_index:
        unknown = [k for k in occupancy if not (0 <= k < n)]
        if unknown:
            raise PrimError("자리 번호가 범위를 벗어났습니다: %s (자리 수 %d)"
                            % (sorted(unknown), n))
        for i, sp in enumerate(species):
            out.append(_normalize_occupancy(occupancy[i]) if i in occupancy else [sp])
    else:
        missing = sorted(set(species) - set(occupancy))
        if missing:
            raise PrimError(
                "occupancy 에 없는 원소가 구조에 있습니다: %s\n"
                "고정 자리도 명시해 주세요 (예: {'C': 'C'})." % ", ".join(missing))
        for sp in species:
            out.append(_normalize_occupancy(occupancy[sp]))
    return out


def make_prim(structure, occupancy, supercell=(1, 1, 1), title=None):
    """구조 파일에서 PRIM 을 만든다.

    Parameters
    ----------
    structure : str or tuple
        구조 파일 경로, 또는 ``(lattice, coords, species)`` 튜플.
    occupancy : str or list or dict
        자리에 올 수 있는 원소. :func:`_resolve_occupancies` 참고.
    supercell : (int, int, int)
        a, b, c 방향 배수. FCC 는 ``(2,1,1)``, HCP/BCC 는 ``(2,2,1)`` 이
        8 자리 셀을 준다(교안 5.2 절).
    title : str
        PRIM 1행. 생략하면 원소들을 이어 붙여 만든다.

    Returns
    -------
    Prim
    """
    if isinstance(structure, str):
        lattice, coords, species = read_structure(structure)
    else:
        lattice, coords, species = structure
        lattice = np.asarray(lattice, dtype=float)
        coords = np.asarray(coords, dtype=float)
        species = list(species)

    nx, ny, nz = supercell
    lattice, coords, species = _replicate(lattice, coords, species, nx, ny, nz)
    occupancies = _resolve_occupancies(occupancy, species)

    if title is None:
        elems = []
        for occ in occupancies:
            for e in occ:
                if e != VACANCY and e not in elems:
                    elems.append(e)
        title = "-".join(elems) if elems else "PRIM"

    return Prim(title, 1.0, lattice, coords, occupancies)


# ----------------------------------------------------------------------------
# P1 으로 낮추기
# ----------------------------------------------------------------------------

def break_symmetry(prim, amplitude=0.001, sites=None, min_displacement=0.005,
                   strict=True):
    """좌표를 미세하게 흔들어 셀의 대칭을 없앤다.

    대칭이 살아 있으면 CASM 이 서로 대칭인 배열을 하나로 묶어 버린다.
    8 자리 셀에서 256 개가 9 개로 줄어드는 것이 그 때문이다(교안 5.3 절).
    전수 열거를 하려면 P1 으로 낮춰야 한다.

    흔드는 폭은 두 기준을 동시에 넘어야 한다(교안 5.4 절).
      1) CASM 이 대칭을 못 찾을 만큼 커야 하고
      2) VASP 가 SYMPREC(기본 1e-5)로 대칭을 되살리지 못할 만큼 커야 한다.

    분수좌표 기준으로 흔들기 때문에 셀이 작으면 실제 변위가 너무 작아질 수
    있다. 그래서 Å 단위 변위를 직접 계산해 ``min_displacement`` 보다 작으면
    알려 준다 — 원본 셸 스크립트에는 없던 확인이다.

    Parameters
    ----------
    prim : Prim
    amplitude : float
        기준 폭(분수좌표). i 번째 자리는 x 로 ``(i+1)*amplitude`` 만큼 밀린다.
    sites : list[int] or None
        흔들 자리. 기본은 occupancy 가 둘 이상인 자리 전부.
    min_displacement : float
        허용 최소 변위(Å).
    strict : bool
        True 면 변위가 모자랄 때 예외를 낸다. False 면 경고만 하고 진행한다.

    Returns
    -------
    (Prim, list[str])
        흔든 새 Prim 과 사람이 읽을 확인 메시지.
    """
    out = prim.copy()
    if sites is None:
        sites = out.mixed_sites or list(range(len(out)))
    if not sites:
        raise PrimError("흔들 자리가 없습니다.")

    for n, i in enumerate(sites):
        if not (0 <= i < len(out)):
            raise PrimError("자리 번호가 범위를 벗어났습니다: %d" % i)
        out.coords[i][0] += (n + 1) * amplitude
        if n < 2:                       # 앞의 두 자리는 y 로도 밀어 축 대칭을 깬다
            out.coords[i][1] += (n + 1) * 10 * amplitude

    delta = (out.coords - prim.coords)
    cart = delta.dot(prim.lattice) * prim.scale
    disp = np.linalg.norm(cart, axis=1)
    moved = disp[disp > 0]
    smallest = float(moved.min()) if len(moved) else 0.0
    largest = float(disp.max())

    msgs = ["흔든 자리 %d개, 실제 변위 %.4f ~ %.4f Å" % (len(sites), smallest, largest)]
    if smallest < min_displacement:
        msg = ("변위 %.4f Å 는 %.4f Å 보다 작습니다. VASP 가 SYMPREC 으로 대칭을 "
               "되살릴 수 있으니 amplitude 를 키우거나 INCAR 에 ISYM=0 을 주세요."
               % (smallest, min_displacement))
        if strict:
            raise PrimError(msg)
        msgs.append("경고: " + msg)
    return out, msgs
