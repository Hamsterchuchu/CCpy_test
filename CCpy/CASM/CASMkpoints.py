# -*- coding: utf-8 -*-
"""CASM 배열 계산용 KPOINTS 생성·배포 모듈.

한 hull 안의 배열들은 서로 빼서 형성 에너지를 만든다. 따라서 k-sampling 오차가
배열마다 다르면 그 차이가 물리로 둔갑한다. 교안 5.6 절이 "한 hull 안의 모든
배열이 같은 KPOINTS 파일을 쓸 것"과 md5 검증을 함께 적어 둔 이유다.

이 모듈이 하는 일은 세 가지다.

1. **셀에 맞는 mesh 를 정한다.** 세 축의 k*L 이 목표값에 고르게 맞도록 각
   축의 분할 수를 따로 계산한다. 실측을 보면 Cu-Ir 의 con 256개가 전부
   ``6 6 6`` 을 쓰는데, 이 값은 4원자 정육면체(BULK/Cu)용으로 만들어진 것이
   PRIM 을 ``scale 2 1 1 1`` 로 늘린 뒤에도 그대로 내려온 것이다. 그래서
   k*L 이 22.5 / 22.5 / 45.0 으로 장축만 두 배 조밀하다.

2. **육방 셀을 가려낸다.** 육방 격자에서 짝수 분할 Monkhorst-Pack 은 어긋난
   격자가 육방 대칭을 깨서 잘못된 IBZ 를 만든다(교안 5.7 절). 홀수 분할이면
   MP 와 Gamma-centered 가 같으므로, 육방이면 홀수로 맞춘다. Alloy 대상
   27종 중 12종이 HCP 라 남의 일이 아니다.

3. **한 파일을 전 디렉토리에 복사하고 같은지 확인한다.** 교안이 md5sum 으로
   하던 확인을 코드로 옮겼다.

Method 2(대칭 supercell)에는 쓰지 않는다. 그쪽은 배열마다 셀 부피가 1 과 2 로
다르고 mainclust 가 SCEL 배율만큼 mesh 를 나눠 k-밀도를 일정하게 유지한다
(실측 k*L 23.6~24.9). 같은 파일을 강제하면 오히려 밀도가 배열마다 두 배씩
어긋난다.
"""

import hashlib
import os

import numpy as np


class KpointsError(ValueError):
    """KPOINTS 를 만들거나 배포하는 중 발생한 오류."""


GAMMA = "Gamma"
MONKHORST = "Monkhorst"

#: 금속 기본 목표값. 교안 3절 처방 4 는 금속에 k*a 35~40 을 권한다.
#: (절연체는 20 안팎이면 충분하다.)
DEFAULT_TARGET = 35.0


class Kpoints(object):
    """KPOINTS 파일 한 벌 (자동 mesh 형식).

    Attributes
    ----------
    mesh : tuple[int, int, int]
    mode : str
        ``"Gamma"`` 또는 ``"Monkhorst"``. VASP 는 첫 글자만 본다.
    comment : str
    shift : tuple[float, float, float] or None
    """

    def __init__(self, mesh, mode=GAMMA, comment="CCpy CASM", shift=(0.0, 0.0, 0.0)):
        mesh = tuple(int(v) for v in mesh)
        if len(mesh) != 3:
            raise KpointsError("mesh 는 정수 3개여야 합니다: %r" % (mesh,))
        if min(mesh) < 1:
            raise KpointsError("mesh 분할 수는 1 이상이어야 합니다: %r" % (mesh,))
        self.mesh = mesh
        self.mode = GAMMA if str(mode)[:1].upper() == "G" else MONKHORST
        self.comment = comment
        self.shift = tuple(float(v) for v in shift) if shift is not None else None

    @property
    def is_gamma(self):
        return self.mode == GAMMA

    def to_string(self):
        lines = [self.comment, "0", self.mode,
                 " ".join(str(v) for v in self.mesh)]
        if self.shift is not None:
            lines.append(" ".join(("%g" % v) for v in self.shift))
        return "\n".join(lines) + "\n"

    def write(self, path="KPOINTS"):
        with open(path, "w") as f:
            f.write(self.to_string())
        return path

    @classmethod
    def from_string(cls, text):
        raw = [l for l in text.rstrip("\n").split("\n")]
        if len(raw) < 4:
            raise KpointsError("KPOINTS 가 너무 짧습니다(%d줄). 최소 4줄이 필요합니다."
                               % len(raw))
        comment = raw[0].strip()
        try:
            nk = int(raw[1].split()[0])
        except (IndexError, ValueError):
            raise KpointsError("2행이 숫자가 아닙니다: %r" % raw[1])
        if nk != 0:
            raise KpointsError(
                "자동 mesh 형식(2행이 0)만 지원합니다. 2행이 %d 이면 k-점을 하나씩 "
                "적는 형식입니다." % nk)
        mode = raw[2].strip()
        if not mode[:1].upper() in ("G", "M"):
            raise KpointsError(
                "3행이 Gamma / Monkhorst 가 아닙니다: %r" % raw[2])
        try:
            mesh = [int(v) for v in raw[3].split()[:3]]
        except ValueError:
            raise KpointsError("4행에서 mesh 를 읽지 못했습니다: %r" % raw[3])
        if len(mesh) != 3:
            raise KpointsError("4행에 분할 수가 3개 있어야 합니다: %r" % raw[3])

        shift = None
        if len(raw) > 4 and raw[4].split():
            try:
                shift = [float(v) for v in raw[4].split()[:3]]
            except ValueError:
                shift = None
        return cls(mesh, mode=mode, comment=comment, shift=shift)

    @classmethod
    def read(cls, path="KPOINTS"):
        with open(path) as f:
            return cls.from_string(f.read())

    def __repr__(self):                                # pragma: no cover
        return "<Kpoints %s %d %d %d>" % ((self.mode,) + self.mesh)


# ----------------------------------------------------------------------------
# mesh 정하기
# ----------------------------------------------------------------------------

def _lattice_matrix(structure):
    """구조에서 격자행렬(3x3, Å)을 꺼낸다.

    PRIM 은 확장자가 없어 pymatgen 이 형식을 알아보지 못하므로 파일명으로
    가려내 Prim 파서로 읽는다.
    """
    if hasattr(structure, "occupancies") and hasattr(structure, "lattice"):
        return np.asarray(structure.lattice, dtype=float) * structure.scale
    if hasattr(structure, "lattice") and hasattr(structure.lattice, "matrix"):
        return np.asarray(structure.lattice.matrix, dtype=float)
    if isinstance(structure, str):
        if os.path.basename(structure).startswith("PRIM"):
            from CCpy.CASM.CASMprim import Prim
            prim = Prim.read(structure)
            return np.asarray(prim.lattice, dtype=float) * prim.scale
        from CCpy.CASM.CASMprim import read_structure
        lattice, _, _ = read_structure(structure)
        return np.asarray(lattice, dtype=float)
    raise KpointsError("구조로 쓸 수 없는 값입니다: %s" % type(structure).__name__)


def lattice_lengths(structure):
    """구조에서 격자벡터 길이 (a, b, c) 를 Å 로."""
    mat = _lattice_matrix(structure)
    return tuple(float(v) for v in np.linalg.norm(mat, axis=1))


def is_hexagonal(structure, angle_tol=1.0, length_tol=0.01):
    """육방(또는 삼방) 셀인지. gamma 가 120도(또는 60도)이고 a≈b 이면 참."""
    mat = _lattice_matrix(structure)
    a, b, c = np.linalg.norm(mat, axis=1)
    def ang(u, v):
        return np.degrees(np.arccos(
            np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    alpha, beta, gamma = ang(mat[1], mat[2]), ang(mat[0], mat[2]), ang(mat[0], mat[1])

    ab_close = abs(a - b) < length_tol * max(a, b)
    right = abs(alpha - 90) < angle_tol and abs(beta - 90) < angle_tol
    hex_gamma = abs(gamma - 120) < angle_tol or abs(gamma - 60) < angle_tol
    return bool(ab_close and right and hex_gamma)


def _to_odd(n):
    """가장 가까운 홀수로. 짝수면 위쪽(더 조밀한 쪽)을 고른다."""
    n = int(n)
    return n if n % 2 == 1 else n + 1


def suggest_mesh(structure, target=DEFAULT_TARGET, force_odd=None, min_k=1):
    """세 축의 k*L 이 target 에 맞도록 분할 수를 정한다.

    Parameters
    ----------
    structure : str or Prim or Structure
    target : float
        목표 k*L (Å). 금속은 35~40, 절연체는 20 안팎.
    force_odd : bool or None
        홀수 분할을 강제할지. None 이면 육방 셀에서만 강제한다.
    min_k : int
        각 축의 하한.

    Returns
    -------
    (mesh, info)
        info 는 사람이 읽을 설명 문자열 목록.
    """
    if target <= 0:
        raise KpointsError("target 은 0보다 커야 합니다: %g" % target)

    lengths = lattice_lengths(structure)
    hexagonal = is_hexagonal(structure)
    if force_odd is None:
        force_odd = hexagonal

    mesh = []
    for L in lengths:
        k = int(round(target / L))
        k = max(int(min_k), k)
        if force_odd:
            k = _to_odd(k)
        mesh.append(k)
    mesh = tuple(mesh)

    kl = [m * L for m, L in zip(mesh, lengths)]
    info = ["격자 %.4f / %.4f / %.4f Å" % lengths,
            "mesh %d x %d x %d  ->  k*L %.1f / %.1f / %.1f (목표 %.0f)"
            % (mesh + tuple(kl) + (target,))]
    if hexagonal:
        info.append("육방 셀입니다. 짝수 분할 Monkhorst-Pack 은 육방 대칭을 깨므로 "
                    "홀수로 맞췄습니다.")
    spread = max(kl) - min(kl)
    if spread > 0.35 * target:
        info.append("경고: 축 사이 k*L 편차가 %.1f 로 큽니다. 셀 모양이 많이 "
                    "찌그러졌는지 확인해 보세요." % spread)
    return mesh, info


def make_kpoints(structure, target=DEFAULT_TARGET, mode=None, force_odd=None,
                 comment=None, min_k=1):
    """구조에 맞는 KPOINTS 를 만든다.

    mode 를 주지 않으면 육방 셀은 Gamma-centered, 그 밖에는 Monkhorst 를 쓴다.
    """
    mesh, info = suggest_mesh(structure, target=target, force_odd=force_odd,
                              min_k=min_k)
    if mode is None:
        mode = GAMMA if is_hexagonal(structure) else MONKHORST
    if comment is None:
        comment = "CCpy CASM : k*L ~ %g" % target
    return Kpoints(mesh, mode=mode, comment=comment), info


# ----------------------------------------------------------------------------
# 배포와 확인
# ----------------------------------------------------------------------------

def config_dirs(root=".", prefix="con"):
    """CASM 이 만든 배열 디렉토리 목록 (con0.0, con1.12 ...)."""
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.startswith(prefix) and os.path.isdir(path) \
                and not name.startswith("config"):
            out.append(path)
    return out


def distribute(kpoints, dirs, filename="KPOINTS"):
    """같은 KPOINTS 를 모든 배열 디렉토리에 쓴다.

    Returns
    -------
    int : 쓴 개수
    """
    if not dirs:
        raise KpointsError(
            "배열 디렉토리를 찾지 못했습니다. mainclust 로 con* 를 먼저 만들어 주세요.")
    text = kpoints.to_string() if isinstance(kpoints, Kpoints) else str(kpoints)
    for d in dirs:
        if not os.path.isdir(d):
            raise KpointsError("디렉토리가 아닙니다: %s" % d)
        with open(os.path.join(d, filename), "w") as f:
            f.write(text)
    return len(dirs)


def verify_uniform(dirs, filename="KPOINTS", compare="mesh"):
    """모든 배열이 같은 KPOINTS 를 쓰는지 확인한다.

    교안 5.6 절은 ``md5sum con0.*/KPOINTS | sort -u | wc -l`` 이 1 이어야 한다고
    적었는데, **파일 전체 md5 로 보면 안 된다.** mainclust 는 KPOINTS 1행에
    그 배열의 디렉토리 이름을 적으므로, mesh 가 똑같아도 주석이 달라 md5 가
    전부 다르게 나온다. 실제로 Ag-Pd super2_n3 은 md5 가 21종이지만 mesh 는
    ``6 6 6`` / ``6 6 3`` / ``8 5 5`` 3종뿐이다.

    그래서 기본은 **내용(mode + mesh + shift) 비교**다. 파일 전체를 보려면
    ``compare="bytes"`` 를 쓴다.

    Returns
    -------
    (ok, groups)
        groups 는 {설명: [디렉토리...]}.
    """
    if compare not in ("mesh", "bytes"):
        raise KpointsError("compare 는 'mesh' 또는 'bytes' 여야 합니다: %r" % compare)
    if not dirs:
        raise KpointsError(
            "확인할 배열 디렉토리가 없습니다. mainclust 로 con* 를 먼저 만들어 주세요.")

    groups = {}
    missing, unreadable = [], []
    for d in dirs:
        path = os.path.join(d, filename)
        if not os.path.isfile(path):
            missing.append(d)
            continue
        if compare == "bytes":
            with open(path, "rb") as f:
                key = "md5 " + hashlib.md5(f.read()).hexdigest()[:8]
        else:
            try:
                kp = Kpoints.read(path)
            except KpointsError as err:
                unreadable.append((d, err))
                continue
            key = "%s %d %d %d" % ((kp.mode,) + kp.mesh)
            if kp.shift and any(abs(s) > 1e-9 for s in kp.shift):
                key += " shift " + " ".join("%g" % s for s in kp.shift)
        groups.setdefault(key, []).append(d)

    if missing:
        raise KpointsError(
            "%d개 디렉토리에 %s 가 없습니다: %s"
            % (len(missing), filename,
               ", ".join(os.path.basename(m) for m in missing[:5])
               + (" ..." if len(missing) > 5 else "")))
    if unreadable:
        d, err = unreadable[0]
        raise KpointsError("%s 의 %s 를 읽지 못했습니다: %s"
                           % (os.path.basename(d), filename, err))
    return (len(groups) == 1), groups


def describe_uniformity(groups, limit=6):
    """verify_uniform 결과를 사람이 읽을 문장으로."""
    total = sum(len(v) for v in groups.values())
    if len(groups) == 1:
        key, dirs = next(iter(groups.items()))
        return "배열 %d개가 모두 같은 KPOINTS 를 씁니다  [%s]" % (len(dirs), key)

    lines = ["배열 %d개에 서로 다른 KPOINTS 가 %d 종류 있습니다 — 한 hull 안에서는 "
             "하나여야 합니다." % (total, len(groups))]
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for key, dirs in ordered[:limit]:
        sample = ", ".join(os.path.basename(d) for d in dirs[:4])
        lines.append("  %-24s %4d개  (%s%s)"
                     % (key, len(dirs), sample, " ..." if len(dirs) > 4 else ""))
    if len(ordered) > limit:
        lines.append("  ... (%d 종류 더)" % (len(ordered) - limit))
    return "\n".join(lines)
