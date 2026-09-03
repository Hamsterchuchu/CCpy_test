# -*- coding: utf-8 -*-
"""CASM 배열 계산의 단계 진행 — 2회 완화 + 최종 정적.

지금까지는 각 배열을 ``ISIF=3`` 으로 **한 번만** 돌리고 끝냈다. 두 가지가
빠져 있다.

**Pulay stress.** 고정된 평면파 기저에서 부피를 바꾸면 stress 에 계통 오차가
남는다. 한 번만 돌리면 부피가 덜 수렴한 채로 끝난다. CONTCAR 를 POSCAR 로
옮겨 다시 돌리면 새 부피에 맞는 기저로 다시 푸는 셈이라 이 오차가 걷힌다.
실측을 보면 Cu-Ir con0.128 은 출발 셀 대비 부피가 1.6 % 움직였다 — 한 번
더 돌려 얼마나 더 움직이는지 봐야 수렴을 말할 수 있다.

**Smearing.** 완화에는 ``ISMEAR=1`` 이 맞지만(사면체법은 힘과 stress 가
부정확하다) 최종 에너지에는 인공 엔트로피 항이 남는다. 마지막에 ``NSW=0``,
``ISMEAR=-5`` 로 한 번 더 돌려 에너지를 확정한다.

교안 3절 처방 3·5·7 을 그대로 옮긴 것이고, 형성 에너지 정확도와 MLIP 라벨링
두 목적에 함께 쓰인다.

수렴 판정은 여기서 다시 만들지 않는다. ``CCpyVASPAnal.py 0`` 이 남기는
``01_unconverged_jobs.csv`` 를 읽어 그 배열은 건너뛴다. 실제로 "reached
required accuracy" 문구만 보고 판정하면 안 된다 — Cu-Ir con0.50 은 그 문구가
있는데도 zbrent 로 미수렴 처리된 배열이다.
"""

import csv
import os
import shutil

import numpy as np

from CCpy.VASP.VASPio import read_incar, update_incar, incar_dict_to_str


class StageError(RuntimeError):
    """단계 진행 중 발생한 오류."""


#: 교안 3절 처방 2 가 드는 자성 원소. 하나라도 있으면 ISPIN=2 로 간다.
MAGNETIC_ELEMENTS = ("Fe", "Co", "Ni", "Cr", "Mn")

#: 단순 강자성 초기화로는 바닥 상태가 아닌 원소.
AFM_CAUTION = ("Cr", "Mn")

#: VASP 가 다시 만들어 주는 큰 파일들. 단계를 넘길 때 지운다.
BULKY_FILES = ("CHG", "CHGCAR", "WAVECAR", "DOSCAR", "EIGENVAL", "PROCAR",
               "PCDAT", "XDATCAR", "IBZKPT", "REPORT", "vasprun.xml")

#: 단계를 넘길 때 접미사를 붙여 남겨 두는 파일들.
ARCHIVE_FILES = ("INCAR", "POSCAR", "CONTCAR", "OUTCAR", "OSZICAR", "KPOINTS")

#: 완화 단계 INCAR (교안 3.2 절).
RELAX_SETTINGS = {
    "PREC": "Accurate",
    "EDIFF": "1E-06",
    "EDIFFG": "-0.02",
    "IBRION": 2,
    "NSW": 200,
    "ISIF": 3,
    "ISMEAR": 1,
    "SIGMA": 0.1,
    "LREAL": ".FALSE.",
    "LORBIT": 0,
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}

#: 최종 정적 단계 INCAR. ISMEAR 는 k-점 수를 보고 정한다.
STATIC_SETTINGS = {
    "IBRION": -1,
    "NSW": 0,
    "EDIFF": "1E-07",
    "LWAVE": ".FALSE.",
    "LCHARG": ".FALSE.",
}


# ----------------------------------------------------------------------------
# ISPIN 판정
# ----------------------------------------------------------------------------

def suggest_ispin(elements):
    """원소 목록을 보고 ISPIN 을 정한다.

    Returns
    -------
    (ispin, notes)
    """
    elements = [str(e) for e in elements]
    found = [e for e in MAGNETIC_ELEMENTS if e in elements]
    if not found:
        return 1, ["자성 원소가 없어 ISPIN=1 로 둡니다 (%s). 켜 두면 계산량만 "
                   "두 배가 됩니다." % ", ".join(sorted(set(elements)))]
    notes = ["자성 원소 %s 가 있어 ISPIN=2 로 둡니다." % ", ".join(found)]
    afm = [e for e in found if e in AFM_CAUTION]
    if afm:
        notes.append("%s 는 반강자성이라 단순 강자성 초기화로는 바닥 상태가 "
                     "아닙니다. 계산 후 OUTCAR 의 magnetization 을 반드시 "
                     "확인하세요." % ", ".join(afm))
    notes.append("계산이 끝나면 실제로 모멘트가 섰는지 확인하세요. "
                 "0 으로 수렴했다면 ISPIN=1 로 내려도 됩니다.")
    return 2, notes


# ----------------------------------------------------------------------------
# INCAR 만들기
# ----------------------------------------------------------------------------

def relax_incar(incar, ispin=None, extra=None):
    """완화 단계 INCAR dict 을 만든다."""
    incar_dict = _as_incar_dict(incar)
    settings = dict(RELAX_SETTINGS)
    if ispin is not None:
        settings["ISPIN"] = int(ispin)
    if extra:
        settings.update(extra)
    return update_incar(incar_dict, settings)


def static_incar(incar, nkpts=None, ispin=None, extra=None):
    """최종 정적 단계 INCAR dict 을 만든다.

    ``ISMEAR=-5`` (사면체법 + Blochl 보정)는 k-점이 4개 이상이어야 쓸 수 있다.
    모자라면 ``ISMEAR=1`` 로 두고 SIGMA 를 낮춘다(교안 3.2 절).
    """
    incar_dict = _as_incar_dict(incar)
    settings = dict(STATIC_SETTINGS)

    if nkpts is not None and int(nkpts) < 4:
        settings["ISMEAR"] = 1
        settings["SIGMA"] = 0.05
    else:
        settings["ISMEAR"] = -5

    if ispin is not None:
        settings["ISPIN"] = int(ispin)
    if extra:
        settings.update(extra)
    return update_incar(incar_dict, settings)


def write_incar(incar_dict, path="INCAR"):
    with open(path, "w") as f:
        f.write(incar_dict_to_str(incar_dict))
    return path


def _as_incar_dict(incar):
    if isinstance(incar, dict):
        return dict(incar)
    if isinstance(incar, str):
        if not os.path.isfile(incar):
            raise StageError("INCAR 이 없습니다: %s" % incar)
        return read_incar(incar)
    raise StageError("INCAR 으로 쓸 수 없는 값입니다: %s" % type(incar).__name__)


# ----------------------------------------------------------------------------
# 상태 읽기
# ----------------------------------------------------------------------------

def is_finished(directory="."):
    """CCpy 의 관례대로 vasp.done 유무로 종료를 판정한다."""
    return os.path.isfile(os.path.join(directory, "vasp.done"))


def cell_volume(path):
    """POSCAR / CONTCAR 의 셀 부피 (Å^3)."""
    if not os.path.isfile(path):
        raise StageError("구조 파일이 없습니다: %s" % path)
    lines = open(path).read().split("\n")
    if len(lines) < 5:
        raise StageError("구조 파일이 너무 짧습니다: %s" % path)
    try:
        scale = float(lines[1].split()[0])
        mat = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    except (IndexError, ValueError):
        raise StageError("%s 에서 격자를 읽지 못했습니다." % path)
    return abs(float(np.linalg.det(mat))) * (scale ** 3 if scale > 0 else 1.0)


def volume_drift(directory="."):
    """POSCAR(출발) 대비 CONTCAR(완화 후) 부피 변화율 (%).

    한 단계 더 돌렸을 때 이 값이 충분히 작아지면 Pulay 가 걷힌 것이다.
    """
    v0 = cell_volume(os.path.join(directory, "POSCAR"))
    v1 = cell_volume(os.path.join(directory, "CONTCAR"))
    return (v1 - v0) / v0 * 100.0


def _grep_last(path, needle):
    last = None
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("utf-8", "replace")
            if needle in line:
                last = line
    return last


def nkpts(directory="."):
    """OUTCAR 에서 기약 k-점 수를 읽는다. 없으면 None."""
    path = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(path):
        return None
    line = _grep_last(path, "NKPTS =")
    if not line:
        return None
    try:
        return int(line.split("NKPTS =")[1].split()[0])
    except (IndexError, ValueError):
        return None


def natoms(directory="."):
    """POSCAR 의 원자 수."""
    path = os.path.join(directory, "POSCAR")
    if not os.path.isfile(path):
        raise StageError("POSCAR 이 없습니다: %s" % path)
    lines = open(path).read().split("\n")
    for idx in (5, 6):
        parts = lines[idx].split() if idx < len(lines) else []
        if parts and all(p.isdigit() for p in parts):
            return sum(int(p) for p in parts)
    raise StageError("%s 에서 원자 수를 읽지 못했습니다." % path)


def entropy_per_atom(directory="."):
    """OUTCAR 의 마지막 entropy T*S 를 원자당 meV 로.

    교안 3절 처방 3 은 이 값이 원자당 1 meV 미만이어야 SIGMA 가 적절하다고
    본다. 없으면 None.
    """
    path = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(path):
        return None
    line = _grep_last(path, "entropy T*S")
    if not line:
        return None
    try:
        value = float(line.split("=")[-1].split()[0])
    except (IndexError, ValueError):
        return None
    return abs(value) / natoms(directory) * 1000.0


def unconverged_dirs(root=".", filename="01_unconverged_jobs.csv"):
    """CCpyVASPAnal 이 남긴 미수렴 목록을 읽는다.

    파일이 없으면 빈 집합을 돌려주고, 그 사실은 호출한 쪽에서 알린다.
    """
    path = os.path.join(root, filename)
    if not os.path.isfile(path):
        return None
    out = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            name = (row.get("Directory") or "").strip()
            if name:
                out.add(name)
    return out


# ----------------------------------------------------------------------------
# 단계 넘기기
# ----------------------------------------------------------------------------

def advance(directory, stage="relax", suffix=None, ispin=None,
            clean=True, extra=None):
    """한 배열을 다음 단계로 넘긴다.

    이전 결과를 접미사를 붙여 남기고, CONTCAR 를 POSCAR 로 올린 뒤, 단계에
    맞는 INCAR 을 쓴다. ``vasp.done`` 은 지운다(큐 스크립트가 다시 만든다).

    Parameters
    ----------
    directory : str
    stage : str
        ``"relax"`` 또는 ``"static"``.
    suffix : str
        보관 파일에 붙일 접미사. 생략하면 ``_relax1``, ``_relax2`` ... 로 센다.
    ispin : int or None
    clean : bool
        큰 산출물을 지울지.
    extra : dict or None
        INCAR 에 덧씌울 값.

    Returns
    -------
    dict : 무엇을 했는지 (drift, suffix, ismear ...)
    """
    if stage not in ("relax", "static"):
        raise StageError("stage 는 'relax' 또는 'static' 이어야 합니다: %r" % stage)
    if not os.path.isdir(directory):
        raise StageError("디렉토리가 없습니다: %s" % directory)
    if not is_finished(directory):
        raise StageError(
            "%s 에 vasp.done 이 없습니다. 아직 끝나지 않았거나 재제출된 계산입니다."
            % os.path.basename(directory))

    contcar = os.path.join(directory, "CONTCAR")
    if not os.path.isfile(contcar) or os.path.getsize(contcar) == 0:
        raise StageError("%s 의 CONTCAR 이 없거나 비어 있습니다."
                         % os.path.basename(directory))

    info = {"directory": directory, "stage": stage}
    try:
        info["volume_drift"] = volume_drift(directory)
    except StageError:
        info["volume_drift"] = None
    info["entropy_meV_per_atom"] = entropy_per_atom(directory)
    prev_nkpts = nkpts(directory)
    info["nkpts"] = prev_nkpts

    if suffix is None:
        suffix = _next_suffix(directory)
    info["suffix"] = suffix

    for name in ARCHIVE_FILES:
        src = os.path.join(directory, name)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(directory, name + suffix))

    shutil.copy(os.path.join(directory, "CONTCAR" + suffix),
                os.path.join(directory, "POSCAR"))
    shutil.copy(os.path.join(directory, "KPOINTS" + suffix),
                os.path.join(directory, "KPOINTS"))

    old_incar = os.path.join(directory, "INCAR" + suffix)
    if stage == "relax":
        incar_dict = relax_incar(old_incar, ispin=ispin, extra=extra)
        info["ismear"] = incar_dict.get("ISMEAR")
    else:
        incar_dict = static_incar(old_incar, nkpts=prev_nkpts, ispin=ispin,
                                  extra=extra)
        info["ismear"] = incar_dict.get("ISMEAR")
    write_incar(incar_dict, os.path.join(directory, "INCAR"))

    if clean:
        for name in BULKY_FILES:
            p = os.path.join(directory, name)
            if os.path.isfile(p):
                os.remove(p)
    done = os.path.join(directory, "vasp.done")
    if os.path.isfile(done):
        os.remove(done)

    return info


def _next_suffix(directory):
    n = 1
    while os.path.isfile(os.path.join(directory, "OUTCAR_relax%d" % n)) or \
            os.path.isfile(os.path.join(directory, "OUTCAR_static%d" % n)):
        n += 1
    return "_relax%d" % n


def advance_all(root=".", stage="relax", dirs=None, skip_unconverged=True,
                ispin=None, clean=True, extra=None):
    """여러 배열을 한 번에 넘긴다.

    Returns
    -------
    (done, skipped)
        done 은 advance() 의 결과 목록, skipped 는 (이름, 사유) 목록.
    """
    if dirs is None:
        from CCpy.CASM.CASMkpoints import config_dirs
        dirs = config_dirs(root)
    if not dirs:
        raise StageError("배열 디렉토리를 찾지 못했습니다.")

    bad = unconverged_dirs(root) if skip_unconverged else set()
    done, skipped = [], []
    for d in dirs:
        name = os.path.basename(d)
        if bad and name in bad:
            skipped.append((name, "CCpyVASPAnal 이 미수렴으로 표시"))
            continue
        try:
            done.append(advance(d, stage=stage, ispin=ispin, clean=clean,
                                extra=extra))
        except StageError as err:
            skipped.append((name, str(err)))
    return done, skipped


def describe(done, skipped, limit=5):
    """advance_all 결과를 사람이 읽을 문장으로."""
    lines = ["%d개를 다음 단계로 넘겼습니다." % len(done)]
    if done:
        drifts = [d["volume_drift"] for d in done if d["volume_drift"] is not None]
        if drifts:
            lines.append("  부피 변화 : %+.2f ~ %+.2f %%  (평균 %+.2f %%)"
                         % (min(drifts), max(drifts),
                            sum(drifts) / len(drifts)))
            big = [d for d in done
                   if d["volume_drift"] is not None and abs(d["volume_drift"]) > 1.0]
            if big:
                lines.append("  1 %% 넘게 움직인 배열 %d개 — 한 단계 더 돌려 "
                             "멎는지 확인하세요." % len(big))
        ents = [d["entropy_meV_per_atom"] for d in done
                if d["entropy_meV_per_atom"] is not None]
        if ents:
            worst = max(ents)
            flag = "" if worst < 1.0 else "  <- 1 meV 를 넘습니다. SIGMA 를 낮추세요."
            lines.append("  엔트로피 T*S : 최대 %.3f meV/atom%s" % (worst, flag))
        ismears = sorted({str(d.get("ismear")) for d in done})
        lines.append("  ISMEAR : %s" % ", ".join(ismears))
    if skipped:
        lines.append("건너뛴 것 %d개:" % len(skipped))
        for name, why in skipped[:limit]:
            lines.append("  %-12s %s" % (name, why))
        if len(skipped) > limit:
            lines.append("  ... (%d개 더)" % (len(skipped) - limit))
    return "\n".join(lines)
