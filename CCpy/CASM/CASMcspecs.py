# -*- coding: utf-8 -*-
"""CASM CSPECS 파일 생성 모듈.

CSPECS 는 CASM 이 클러스터를 어디까지 볼지 정하는 파일이다. 크기(2체·3체·4체)
마다 "이 반경 안에 들어오는 자리를 한 클러스터로 본다"를 적는다.

    Specifications for lattice
    cluster size      within sphere radius
      2                 3.0
      3                 3.0
      4                 3.0

반경은 실제 구조의 최근접 거리에서 정한다. 예를 들어 FCC Ag(a=4.1449 Å)는
1st NN 2.9309 Å, 2nd NN 4.1449 Å 이므로 1st 까지만 보려면 3.0, 2nd 까지 보려면
4.5 를 쓴다(교안 6.2 절). 반경을 늘리면 클러스터가 늘어 비등가 배열 수가
급격히 커진다 — 계산량과 정확도의 맞바꿈이다.

기존에는 이 거리를 VTST 의 ``scale`` 로 supercell 을 만든 뒤 ``neighbors.pl``
로 쟀다. 이 모듈은 pymatgen 으로 원래 셀에서 바로 이웃 껍질을 구한다.
supercell 을 만들 필요가 없는 이유는 ``neighbors.pl`` 이 최소이미지 규약을
쓰기 때문이다 — 셀이 충분히 커야 규약이 성립하므로 미리 키워야 했던 것이고,
pymatgen 은 주기 이미지를 제대로 세므로 그 단계가 통째로 사라진다.

그 단계가 사라지면 딸려 오던 문제도 없어진다. ``scale`` 의 4번째 인자는
진공을 넣는 zvac 이 아니라 c 축 배율(zscale)이다. 교안과 Sym_Alloy.py 가 쓰는
``scale 2 2 2 1`` 은 원자는 2x2x2 로 늘리면서 c 축은 1배로 두므로 z 방향
이미지가 원본 위에 겹친다. cms2 의 Ag/2x2cell 을 실제로 세어 보면 원자 32개
중 서로 다른 위치가 16개뿐이다(부피 284.84 A^3 / 71.21 A^3 x 4 = 16). 거리
자체는 영향을 받지 않지만 배위수는 두 배로 부풀고 거리 0 인 쌍이 생긴다.
제대로 2x2x2 를 만들려면 ``scale 2 2 2 2`` 여야 한다.
"""

import math

import numpy as np


class CspecsError(ValueError):
    """CSPECS 를 만들거나 읽는 중 발생한 오류."""


DEFAULT_SIZES = (2, 3, 4)


class Cspecs(object):
    """CSPECS 파일 한 벌.

    Attributes
    ----------
    radii : dict[int, float]
        클러스터 크기 -> 반경(Å).
    """

    HEADER = "Specifications for lattice"
    COLUMNS = "cluster size      within sphere radius"

    def __init__(self, radii):
        if isinstance(radii, (int, float)):
            radii = dict((s, float(radii)) for s in DEFAULT_SIZES)
        self.radii = dict((int(k), float(v)) for k, v in dict(radii).items())
        if not self.radii:
            raise CspecsError("클러스터 크기가 하나도 없습니다.")
        for size, r in self.radii.items():
            if size < 2:
                raise CspecsError("클러스터 크기는 2 이상이어야 합니다: %d" % size)
            if r <= 0:
                raise CspecsError("반경은 0보다 커야 합니다: 크기 %d -> %g" % (size, r))

    def to_string(self):
        lines = [self.HEADER, self.COLUMNS]
        for size in sorted(self.radii):
            lines.append("  %-16d%s" % (size, _fmt(self.radii[size])))
        return "\n".join(lines) + "\n"

    def write(self, path="CSPECS"):
        with open(path, "w") as f:
            f.write(self.to_string())
        return path

    @classmethod
    def from_string(cls, text):
        radii = {}
        for line in text.split("\n"):
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                size, r = int(parts[0]), float(parts[1])
            except ValueError:
                continue                       # 머리글 줄
            radii[size] = r
        if not radii:
            raise CspecsError(
                "CSPECS 에서 '크기 반경' 줄을 찾지 못했습니다.\n"
                "머리글 두 줄 뒤에 '2  3.0' 같은 줄이 있어야 합니다.")
        return cls(radii)

    @classmethod
    def read(cls, path="CSPECS"):
        with open(path) as f:
            return cls.from_string(f.read())

    def __repr__(self):                                # pragma: no cover
        return "<Cspecs %s>" % ", ".join(
            "%d:%.4g" % (s, self.radii[s]) for s in sorted(self.radii))


# ----------------------------------------------------------------------------
# 이웃 껍질
# ----------------------------------------------------------------------------

def neighbor_shells(structure, species=None, index=None, rmax=8.0, tol=1e-3):
    """이웃을 거리별 껍질로 묶어 돌려준다.

    Parameters
    ----------
    structure : str or pymatgen Structure
        구조 파일 경로 또는 Structure. PRIM 도 받는다(:class:`CCpy.CASM.CASMprim.Prim`).
    species : str or list[str] or None
        이 원소끼리의 거리만 본다. Li-C 처럼 특정 원소쌍의 거리로 반경을 정할 때
        쓴다(교안은 Li-Li 거리로 5.0 Å 을 잡았다). None 이면 모든 원소.
    index : int or None
        중심으로 삼을 자리. None 이면 조건에 맞는 첫 자리.
    rmax : float
        여기까지만 찾는다(Å).
    tol : float
        이 안에 들어오면 같은 껍질로 본다(Å).

    Returns
    -------
    list[(distance, count)]
        거리 오름차순. count 는 그 껍질의 이웃 수(배위수).
    """
    st = _as_structure(structure)

    if species is not None:
        if isinstance(species, str):
            species = [species]
        species = set(species)
        candidates = [i for i, s in enumerate(st) if s.specie.symbol in species]
        if not candidates:
            raise CspecsError(
                "구조에 %s 가 없습니다. 있는 원소: %s"
                % (", ".join(sorted(species)),
                   ", ".join(sorted({s.specie.symbol for s in st}))))
    else:
        candidates = list(range(len(st)))

    center = candidates[0] if index is None else index
    if not (0 <= center < len(st)):
        raise CspecsError("자리 번호가 범위를 벗어났습니다: %s (자리 수 %d)"
                          % (center, len(st)))
    if species is not None and st[center].specie.symbol not in species:
        raise CspecsError(
            "%d번 자리는 %s 인데 %s 만 보라고 하셨습니다."
            % (center, st[center].specie.symbol, ", ".join(sorted(species))))

    dists = []
    for nb in st.get_neighbors(st[center], rmax):
        if species is not None and nb.specie.symbol not in species:
            continue
        dists.append(float(nb.nn_distance))
    if not dists:
        raise CspecsError(
            "반경 %.2f Å 안에 이웃이 없습니다. rmax 를 키워 보세요." % rmax)

    dists.sort()
    shells = []
    for d in dists:
        if shells and abs(d - shells[-1][0]) <= tol:
            shells[-1][1] += 1
        else:
            shells.append([d, 1])
    return [(d, n) for d, n in shells]


def describe_shells(shells, limit=6):
    """이웃 껍질을 사람이 읽을 표로."""
    lines = ["  껍질   거리(Å)     이웃 수"]
    for i, (d, n) in enumerate(shells[:limit], start=1):
        lines.append("  %3d   %10.6f   %6d" % (i, d, n))
    if len(shells) > limit:
        lines.append("  ... (%d개 더)" % (len(shells) - limit))
    return "\n".join(lines)


def suggest_radius(shells, nshell=1, step=0.5):
    """n번째 껍질까지 포함하는 반경을 제안한다.

    껍질 거리 바로 위의 ``step`` 배수를 고르되, 다음 껍질을 넘지 않게 한다.
    넘으면 두 껍질의 중간값을 쓴다.

    실제로 쓰인 값과 맞는지 확인해 보면 — FCC Ag(1st 2.9309, 2nd 4.1449)에서
    1번째 껍질은 3.0, 2번째 껍질은 4.5 가 나온다. 교안 6.2 절의 n3/n4 값과 같다.
    """
    if nshell < 1 or nshell > len(shells):
        raise CspecsError("껍질 번호가 범위를 벗어났습니다: %d (있는 껍질 %d개)"
                          % (nshell, len(shells)))
    d = shells[nshell - 1][0]
    nxt = shells[nshell][0] if nshell < len(shells) else None

    r = math.ceil((d + 1e-6) / step) * step
    if nxt is not None and r >= nxt:
        r = 0.5 * (d + nxt)
    return round(r, 4)


def make_cspecs(structure, nshell=1, sizes=DEFAULT_SIZES, radius=None,
                species=None, index=None, rmax=8.0):
    """구조에서 CSPECS 를 만든다.

    Returns
    -------
    (Cspecs, list[(distance, count)])
        만들어진 CSPECS 와, 반경을 정하는 데 쓴 이웃 껍질.
    """
    shells = neighbor_shells(structure, species=species, index=index, rmax=rmax)
    if radius is None:
        radius = suggest_radius(shells, nshell=nshell)
    radius = float(radius)

    sizes = [int(s) for s in sizes]
    if not sizes:
        raise CspecsError("클러스터 크기를 하나 이상 주세요 (보통 2, 3, 4).")
    return Cspecs(dict((s, radius) for s in sizes)), shells


# ----------------------------------------------------------------------------

def _fmt(value):
    """3.0 은 '3.0' 으로, 4.5 는 '4.5' 로. 소수점을 남긴다."""
    s = ("%.4f" % value).rstrip("0")
    return s + "0" if s.endswith(".") else s


def _as_structure(structure):
    """경로 / Prim / Structure 를 pymatgen Structure 로."""
    from pymatgen.core import Structure

    if isinstance(structure, Structure):
        return structure

    if isinstance(structure, str):
        from CCpy.CASM.CASMprim import read_structure
        lattice, coords, species = read_structure(structure)
        return Structure(lattice, species, coords, coords_are_cartesian=False)

    # Prim: 혼합 자리는 대표 원소 하나로 놓고 거리만 잰다.
    if hasattr(structure, "occupancies") and hasattr(structure, "coords"):
        from CCpy.CASM.CASMprim import VACANCY
        species = []
        for occ in structure.occupancies:
            real = [e for e in occ if e != VACANCY]
            if not real:
                raise CspecsError("occupancy 가 Vac 뿐인 자리가 있습니다.")
            species.append(real[0])
        return Structure(np.asarray(structure.lattice) * structure.scale,
                         species, structure.coords, coords_are_cartesian=False)

    raise CspecsError("구조로 쓸 수 없는 값입니다: %s" % type(structure).__name__)
