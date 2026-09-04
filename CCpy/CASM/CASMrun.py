# -*- coding: utf-8 -*-
"""CASM 외부 바이너리(mainclust) 구동 모듈.

mainclust 는 대화식으로만 동작하도록 만들어져 있고, 어떤 질문이 나오는지는
작업 디렉토리에 어떤 파일이 이미 있는지에 따라 달라진다. 예를 들어 첫 실행에는
eci.in 이 없으므로 그 질문 자체가 나오지 않고, supercells 질문에 read(1) 로
답하면 뒤따르는 max volume / dimension 질문이 통째로 생략된다.

그래서 "정해진 순서로 답을 밀어넣는" 방식은 위험하다. 파일 상태가 예상과
조금만 달라도 답이 밀려서 엉뚱한 질문에 엉뚱한 값이 들어가고, mainclust 는
그것을 조용히 받아들인다. 이 모듈은 대신 **화면에 뜬 질문 문구를 정규식으로
알아보고 그 질문에 해당하는 답을 넣는다.**

mainclust 는 stdout 이 파이프면 블록 버퍼링을 하므로 질문이 제때 보이지 않는다.
따라서 의사 터미널(pty)을 붙여 대화식으로 실행할 때와 같은 줄 버퍼링을 얻는다.
"""

import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time

try:
    import pty
    import termios
    _HAS_PTY = True
except ImportError:                                   # pragma: no cover
    _HAS_PTY = False


#: 실행 파일 이름. 관례상 계 폴더마다 복사해 두고 썼지만, resolve_binary() 가
#: 여러 자리를 순서대로 뒤지므로 이제 한 곳에 두면 된다.
DEFAULT_BINARY = "mainclust"

#: 실행 파일 경로를 직접 지정하는 환경변수 (CCpy_SERVER 와 같은 형식).
BINARY_ENV = "CCpy_MAINCLUST"

#: mainclust 가 이어 붙일 원소별 POTCAR 의 파일명 접두사.
POTCAR_PREFIX = "POTCAR_"

#: 질문 키 -> (정규식, 사람이 읽을 설명). mainclust 바이너리에서 추출한 문구다.
#: 순서는 실제로 나타나는 순서지만, 매칭은 순서와 무관하게 이루어진다.
PROMPT_SPECS = (
    ("eci",
     r"generate\s*\(0\)\s*or\s*keep\s*\(1\)\s*eci\.in",
     "eci.in 을 새로 만들지(0) 그대로 둘지(1)"),
    ("supercells",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*supercells",
     "supercell 을 새로 만들지(0) 읽을지(1)"),
    ("max_volume",
     r"enter the maximum supercell volume",
     "최대 supercell 부피"),
    ("dimension",
     r"generate 3-dimensional or 2-dimensional supercells",
     "3차원(3)인지 2차원(2)인지"),
    ("configuration",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*configuration and configuration\.corr",
     "configuration 을 새로 만들지(0) 읽을지(1)"),
    ("make_dirs",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*make_dirs",
     "make_dirs 를 새로 만들지(0) 읽을지(1)"),
    ("energy",
     r"generate\s*\(0\)\s*or\s*read\s*\(1\)\s*energy and corr\.in",
     "energy / corr.in 을 새로 만들지(0) 읽을지(1)"),
    ("reference",
     r"generate\s*\(0\)\s*or\s*keep\s*\(1\)\s*reference",
     "reference 를 새로 만들지(0) 그대로 둘지(1)"),
)

_COMPILED = tuple((key, re.compile(pat), desc) for key, pat, desc in PROMPT_SPECS)

#: 정상 실행이면 마지막에 반드시 나오는 문구.
_DONE_RE = re.compile(r"CASM code has completed successfully")

#: 정상 진행 표시. 몇 번째 checkpoint 까지 갔는지로 어디서 멈췄는지 알 수 있다.
_CHECKPOINT_RE = re.compile(r"Checkpoint\s+(\d+)")

#: 실행이 정상이어도 뜨는 안내문. 오류로 오해하지 않도록 모아둔다.
BENIGN_MESSAGES = (
    "No SPECIES file in the current directory",
    "No CSPECS file in the current directory",
    "cannot open file custom_structures",
    "No configurations available to determine convex hull",
    "quitting assemble_hull()",
    "No cluster expansion is present",
)


class MainclustError(RuntimeError):
    """mainclust 구동 중 발생한 오류."""

    def __init__(self, message, transcript=None):
        super(MainclustError, self).__init__(message)
        self.transcript = transcript or ""


class MainclustResult(object):
    """한 번의 mainclust 실행 결과."""

    def __init__(self, returncode, transcript, answered, unused, checkpoints, completed):
        self.returncode = returncode
        self.transcript = transcript
        #: [(질문키, 답)] — 실제로 나타난 질문과 넣은 답을 나타난 순서대로
        self.answered = answered
        #: 준비했지만 한 번도 나타나지 않은 질문키
        self.unused = unused
        #: 통과한 checkpoint 번호 목록
        self.checkpoints = checkpoints
        #: "CASM code has completed successfully" 를 보았는가
        self.completed = completed

    @property
    def ok(self):
        return self.returncode == 0 and self.completed

    def summary(self):
        seq = " ".join("%s=%s" % (k, v) for k, v in self.answered)
        return ("exit=%s completed=%s checkpoint=%s\n  answered: %s"
                % (self.returncode, self.completed,
                   max(self.checkpoints) if self.checkpoints else "-",
                   seq or "(없음)"))

    def __repr__(self):                                # pragma: no cover
        return "<MainclustResult %s>" % self.summary().replace("\n", " ")


# ----------------------------------------------------------------------------
# 표준 답안 세트
# ----------------------------------------------------------------------------

def answers_enumerate(max_volume=1, dimension=3):
    """새로 열거한다. PRIM 을 바꾼 뒤 돌리는 경우가 여기에 해당한다.

    기존 결과가 있으면 전부 새로 만들고(0), 없으면 그 질문은 나타나지 않는다.
    """
    return {
        "eci": 0,
        "supercells": 0,
        "max_volume": int(max_volume),
        "dimension": int(dimension),
        "configuration": 0,
        "make_dirs": 0,
        "energy": 0,
        "reference": 0,
    }


def answers_reuse(energy=0, reference=0):
    """이미 만들어 둔 열거 결과를 그대로 읽어 VASP 입력 디렉토리를 만든다.

    make_dirs 의 make 플래그를 1 로 바꾼 뒤 이 답안으로 다시 돌리면
    con* 디렉토리가 생성된다. 교안의 "1 1 1 1 0 0" 이 이것이다.

    energy / reference 는 계에 따라 0(새로 생성) 또는 1(읽기)을 쓴다.
    """
    return {
        "eci": 1,
        "supercells": 1,
        "configuration": 1,
        "make_dirs": 1,
        "energy": int(energy),
        "reference": int(reference),
    }


# ----------------------------------------------------------------------------
# 실행 파일 찾기
# ----------------------------------------------------------------------------

def _config_dir():
    """설정 폴더 경로 문자열. CCpy 설정 모듈을 못 읽으면 관례 경로로 대신한다."""
    try:
        from CCpy.Tools.CCpyConfig import config_home
        return str(config_home())
    except Exception:                                  # pragma: no cover
        return os.path.join(os.path.expanduser("~"), ".CCpy_test")


def _executable(path):
    """실행 가능한 파일이면 절대경로, 없으면 None. 있는데 권한이 없으면 오류."""
    if not path:
        return None
    full = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    if not os.path.isfile(full):
        return None
    if not os.access(full, os.X_OK):
        raise MainclustError(
            "%s 에 실행 권한이 없습니다. chmod +x 로 붙여 주세요." % full)
    return full


def resolve_binary(binary=None, workdir="."):
    """mainclust 실행 파일의 절대 경로를 찾는다.

    mainclust 는 출처와 라이선스가 분명치 않은 컴파일된 외부 바이너리라
    저장소에 넣지 않는다. 그렇다고 계 폴더마다 674 KB 를 복사해 두게 하면
    폴더가 늘어날수록 어느 것이 최신인지 알 수 없어진다. 그래서 아래 순서로
    찾는다 — 앞에서 찾으면 뒤는 보지 않는다.

    1. ``binary`` 에 경로를 준 경우 그 경로 (상대경로면 ``workdir`` 기준)
    2. 환경변수 ``$CCpy_MAINCLUST``
    3. 설정 폴더 ``~/.CCpy_test/mainclust`` (``$CCpy_HOME`` 로 옮길 수 있다)
    4. 작업 디렉토리 안 (기존 관례. 그대로 쓰던 폴더가 깨지지 않도록 남긴다)
    5. ``$PATH``

    1번만 예외로, 경로를 직접 준 것이 없으면 즉시 오류를 낸다. 지정한 경로가
    틀렸는데 조용히 다른 곳의 바이너리로 돌아버리면 안 되기 때문이다.
    """
    name = str(binary) if binary else DEFAULT_BINARY
    tried = []

    def _try(desc, path):
        if not path:
            return None
        tried.append((desc, str(path)))
        return _executable(path)

    if binary and (os.sep in name or name.startswith("~") or name.startswith(".")):
        expanded = os.path.expanduser(os.path.expandvars(name))
        base = expanded if os.path.isabs(expanded) else os.path.join(workdir, expanded)
        found = _try("지정한 경로", base)
        if found:
            return found
        raise MainclustError("mainclust 를 찾지 못했습니다: %s" % tried[-1][1])

    found = _try("환경변수 $%s" % BINARY_ENV, os.environ.get(BINARY_ENV))
    if found:
        return found

    found = _try("설정 폴더", os.path.join(_config_dir(), name))
    if found:
        return found

    found = _try("작업 디렉토리", os.path.join(workdir, name))
    if found:
        return found

    hit = shutil.which(name)
    tried.append(("$PATH", "which %s" % name))
    if hit:
        return os.path.abspath(hit)

    raise MainclustError(
        "mainclust 를 찾지 못했습니다. 아래를 순서대로 찾아보았습니다.\n"
        + "\n".join("  %-14s %s" % (d, p) for d, p in tried)
        + "\n\nmainclust 는 저장소에 들어 있지 않은 외부 바이너리입니다. 한 번 받아\n"
          "  %s\n"
          "에 두거나 $%s 로 경로를 지정해 두면 계 폴더마다 복사하지 않아도 됩니다."
          % (os.path.join(_config_dir(), DEFAULT_BINARY), BINARY_ENV))


# ----------------------------------------------------------------------------
# POTCAR 확인
# ----------------------------------------------------------------------------

def prim_elements(workdir=".", prim="PRIM"):
    """PRIM 에 등장하는 원소를 순서대로. 읽을 수 없으면 MainclustError."""
    path = prim if os.path.isabs(prim) else os.path.join(workdir, prim)
    if not os.path.isfile(path):
        raise MainclustError("%s 가 없습니다." % path)
    try:
        from CCpy.CASM.CASMprim import Prim
    except ImportError as err:                         # pragma: no cover
        raise MainclustError("CASMprim 을 불러오지 못했습니다: %s" % err)
    return Prim.read(path).elements


def check_potcar_sources(workdir=".", elements=None, prim="PRIM"):
    """con* 를 만들기 전에 원소별 POTCAR 가 제대로 있는지 본다.

    mainclust 는 ``POTCAR_<원소>`` 를 이어 붙여 각 배열의 POTCAR 를 만드는데,
    그 파일이 없어도 **오류를 내지 않고 0바이트 POTCAR 를 만든 뒤 그냥 넘어간다.**
    실습 폴더에서 실제로 그런 배열이 나왔다. 계산을 256개 제출하고 나서야
    알게 되는 종류의 실패라, 만들기 전에 여기서 막는다.

    Returns
    -------
    [(원소, 경로, 바이트)]
    """
    if elements is None:
        elements = prim_elements(workdir, prim)
    if not elements:
        raise MainclustError("PRIM 에서 원소를 찾지 못했습니다.")

    ok, missing, empty = [], [], []
    for elt in elements:
        path = os.path.join(workdir, POTCAR_PREFIX + elt)
        if not os.path.isfile(path):
            missing.append(path)
        elif os.path.getsize(path) == 0:
            empty.append(path)
        else:
            ok.append((elt, path, os.path.getsize(path)))

    if missing or empty:
        lines = ["원소별 POTCAR 가 준비되지 않았습니다."]
        if missing:
            lines.append("  없음   : " + ", ".join(os.path.basename(p) for p in missing))
        if empty:
            lines.append("  0바이트: " + ", ".join(os.path.basename(p) for p in empty))
        lines.append("")
        lines.append("  mainclust 는 이 파일이 없어도 오류 없이 0바이트 POTCAR 를 만듭니다.")
        lines.append("  %s 에 %s 를 두고 다시 실행해 주세요."
                     % (os.path.abspath(workdir),
                        " ".join(POTCAR_PREFIX + e for e in elements)))
        raise MainclustError("\n".join(lines))
    return ok


def _titel_element(line):
    """POTCAR 의 TITEL 줄에서 원소 기호만. 못 읽으면 None.

    ``TITEL  = PAW_PBE Cu_pv 22Jun2005`` -> ``Cu``
    """
    try:
        parts = line.split("=", 1)[1].split()
    except IndexError:                                 # pragma: no cover
        return None
    if not parts:
        return None
    name = parts[1] if len(parts) > 1 else parts[0]
    return name.split("_", 1)[0]


def _pos_counts(path):
    """POS/POSCAR 의 원소별 개수 줄을 [정수] 로. 확실치 않으면 None.

    CASM 이 쓰는 POS 는 VASP4 형식이라 원소 이름 줄이 없다. 좌표계 줄
    (Direct/Cartesian) 바로 앞의 정수 줄이 개수 줄이다.
    """
    try:
        with open(path) as f:
            lines = f.read().split("\n")
    except OSError:                                    # pragma: no cover
        return None
    for i, line in enumerate(lines):
        if i < 5:                     # 축척 1 + 격자 3 + 개수 1 보다 앞일 수 없다
            continue
        if line.strip()[:1].upper() not in ("D", "C", "S"):
            continue
        for j in range(i - 1, -1, -1):
            parts = lines[j].split()
            if not parts:
                continue
            try:
                return [int(v) for v in parts]
            except ValueError:
                return None
        return None
    return None


def check_generated_potcars(workdir=".", dirs=None, elements=None, prim="PRIM"):
    """만들어진 con*/POTCAR 가 온전한지 본다.

    POTCAR 의 TITEL 개수는 **그 배열에 실제로 들어 있는 원소 수**와 같아야 한다.
    순수 끝점(전부 Cu, 전부 Ir)은 원소가 하나뿐이라 TITEL 도 하나다. 그래서
    PRIM 의 원소 수와 비교하면 안 되고, POS 의 개수 줄과 맞춰 본다.

    TITEL 에 적힌 원소가 PRIM 에 없는 것이면 엉뚱한 POTCAR_ 를 둔 것이므로
    함께 걸러낸다.

    Returns
    -------
    (검사한 개수, [(디렉토리, 문제 설명)])
    """
    from CCpy.CASM.CASMkpoints import config_dirs
    if dirs is None:
        dirs = config_dirs(workdir)
    if elements is None:
        try:
            elements = prim_elements(workdir, prim)
        except MainclustError:
            elements = None

    bad = []
    for d in dirs:
        path = os.path.join(d, "POTCAR")
        if not os.path.isfile(path):
            bad.append((d, "POTCAR 없음"))
            continue
        if os.path.getsize(path) == 0:
            bad.append((d, "0바이트"))
            continue
        try:
            with open(path) as f:
                titels = [_titel_element(ln) for ln in f
                          if "TITEL" in ln and "=" in ln]
        except OSError as err:                         # pragma: no cover
            bad.append((d, "읽지 못함: %s" % err))
            continue
        titels = [t for t in titels if t]
        if not titels:
            bad.append((d, "TITEL 이 없음 (POTCAR 형식이 아님)"))
            continue

        counts = _pos_counts(os.path.join(d, "POS"))
        if counts is not None and len(titels) != len(counts):
            bad.append((d, "TITEL %d개 (POS 원소 %d종)" % (len(titels), len(counts))))
            continue
        if elements:
            odd = [t for t in titels if t not in elements]
            if odd:
                bad.append((d, "PRIM 에 없는 원소: %s" % ", ".join(odd)))

    return len(dirs), bad


def describe_potcars(total, bad, limit=6):
    """check_generated_potcars 결과를 사람이 읽을 한 덩어리로."""
    if not bad:
        return "  POTCAR %d개 확인 — 모두 정상." % total
    out = ["  POTCAR %d개 중 %d개가 이상합니다:" % (total, len(bad))]
    for d, why in bad[:limit]:
        out.append("    %-12s %s" % (os.path.basename(d), why))
    if len(bad) > limit:
        out.append("    ... 그 밖 %d개" % (len(bad) - limit))
    return "\n".join(out)


# ----------------------------------------------------------------------------
# 드라이버
# ----------------------------------------------------------------------------

class MainclustDriver(object):
    """mainclust 를 사람 없이 구동한다.

    Parameters
    ----------
    workdir : str
        mainclust 를 돌릴 디렉토리. PRIM 등 입력이 여기 있어야 한다.
    binary : str or None
        실행 파일 경로. None 이면 :func:`resolve_binary` 가 환경변수 →
        설정 폴더 → 작업 디렉토리 → PATH 순서로 찾는다.
    idle_timeout : float
        아무 출력도 없이 이만큼(초) 지나면 멈춘 것으로 본다.
        Checkpoint 1(대칭 분석)은 큰 셀에서 수 분이 걸릴 수 있으므로 넉넉히 둔다.
    total_timeout : float or None
        전체 실행 시간 상한. None 이면 제한하지 않는다.
    echo : bool
        True 면 mainclust 출력을 그대로 화면에 흘린다.
    """

    def __init__(self, workdir=".", binary=None,
                 idle_timeout=600.0, total_timeout=None, echo=False):
        self.workdir = os.path.abspath(workdir)
        self.binary = binary
        self.idle_timeout = float(idle_timeout)
        self.total_timeout = total_timeout
        self.echo = echo

    # -- 경로 ---------------------------------------------------------------

    def resolve_binary(self):
        """실행 파일의 절대 경로를 돌려준다. 없으면 MainclustError."""
        return resolve_binary(self.binary, self.workdir)

    # -- 실행 ---------------------------------------------------------------

    def run(self, answers, strict=True):
        """mainclust 를 한 번 돌린다.

        Parameters
        ----------
        answers : dict
            질문키 -> 답. :func:`answers_enumerate` / :func:`answers_reuse` 참고.
        strict : bool
            True 면 답을 준비하지 않은 질문이 나타났을 때 즉시 중단한다.
            False 면 그 질문에 빈 줄을 넣고 계속한다(권장하지 않는다).

        Returns
        -------
        MainclustResult
        """
        unknown = set(answers) - set(k for k, _, _ in _COMPILED)
        if unknown:
            raise MainclustError(
                "모르는 질문키입니다: %s\n사용 가능: %s"
                % (", ".join(sorted(unknown)),
                   ", ".join(k for k, _, _ in _COMPILED)))

        if not _HAS_PTY:                               # pragma: no cover
            raise MainclustError(
                "이 플랫폼에서는 pty 를 쓸 수 없어 mainclust 를 구동할 수 없습니다. "
                "리눅스에서 실행해 주세요.")

        binary = self.resolve_binary()
        return self._run_pty(binary, answers, strict)

    def _run_pty(self, binary, answers, strict):
        master, slave = pty.openpty()
        self._disable_echo(slave)

        proc = subprocess.Popen(
            [binary],
            cwd=self.workdir,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
            preexec_fn=os.setsid,
        )
        os.close(slave)

        chunks = []
        pending = ""          # 아직 질문을 찾지 못한 꼬리
        answered = []
        checkpoints = []
        completed = False
        started = time.time()
        last_output = started

        try:
            while True:
                if self.total_timeout is not None and \
                        time.time() - started > self.total_timeout:
                    self._kill(proc)
                    raise MainclustError(
                        "전체 실행 시간 %.0f 초를 넘겨 중단했습니다."
                        % self.total_timeout, "".join(chunks))

                rlist, _, _ = select.select([master], [], [], 1.0)

                if rlist:
                    try:
                        data = os.read(master, 8192)
                    except OSError:
                        data = b""
                    if not data:
                        break
                    text = data.decode("utf-8", "replace")
                    chunks.append(text)
                    if self.echo:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    pending += text
                    last_output = time.time()

                    for num in _CHECKPOINT_RE.findall(text):
                        n = int(num)
                        if n not in checkpoints:
                            checkpoints.append(n)
                    if _DONE_RE.search(text):
                        completed = True

                    pending = self._consume(pending, answers, answered,
                                            master, proc, strict, chunks)
                else:
                    if proc.poll() is not None:
                        # 종료 후 남은 출력을 마저 읽는다.
                        tail = self._drain(master)
                        if tail:
                            chunks.append(tail)
                            if self.echo:
                                sys.stdout.write(tail)
                            for num in _CHECKPOINT_RE.findall(tail):
                                n = int(num)
                                if n not in checkpoints:
                                    checkpoints.append(n)
                            if _DONE_RE.search(tail):
                                completed = True
                        break
                    if time.time() - last_output > self.idle_timeout:
                        self._kill(proc)
                        raise MainclustError(
                            "%.0f 초 동안 출력이 없어 중단했습니다. mainclust 가 "
                            "답을 준비하지 못한 질문에서 멈춰 있을 수 있습니다.\n"
                            "마지막 출력:\n%s"
                            % (self.idle_timeout, _tail("".join(chunks))),
                            "".join(chunks))
        finally:
            try:
                os.close(master)
            except OSError:
                pass
            if proc.poll() is None:
                self._kill(proc)

        returncode = proc.wait()
        transcript = "".join(chunks)
        used = set(k for k, _ in answered)
        unused = [k for k in answers if k not in used]

        return MainclustResult(returncode, transcript, answered, unused,
                               checkpoints, completed)

    # -- 내부 ---------------------------------------------------------------

    def _consume(self, pending, answers, answered, master, proc, strict, chunks):
        """pending 에서 찾을 수 있는 질문을 모두 처리하고 남은 꼬리를 돌려준다."""
        while True:
            best = None
            for key, rx, desc in _COMPILED:
                m = rx.search(pending)
                if m and (best is None or m.start() < best[1].start()):
                    best = (key, m, desc)
            if best is None:
                # 질문 문구가 두 청크에 걸쳐 잘렸을 수 있으므로 꼬리를 남긴다.
                return pending[-400:] if len(pending) > 400 else pending

            key, m, desc = best
            if key in answers:
                value = str(answers[key])
            elif strict:
                self._kill(proc)
                raise MainclustError(
                    "답을 준비하지 않은 질문이 나왔습니다: '%s' (%s)\n"
                    "answers 에 '%s' 를 넣어 주세요."
                    % (m.group(0).strip(), desc, key),
                    "".join(chunks))
            else:
                value = ""

            os.write(master, (value + "\n").encode("utf-8"))
            answered.append((key, value))
            pending = pending[m.end():]

    @staticmethod
    def _disable_echo(fd):
        """pty 에코를 끈다. 우리가 쓴 답이 출력에 섞여 다시 매칭되지 않도록."""
        try:
            attrs = termios.tcgetattr(fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception:                              # pragma: no cover
            pass

    @staticmethod
    def _drain(fd, limit=1 << 20):
        out = []
        total = 0
        while total < limit:
            r, _, _ = select.select([fd], [], [], 0.2)
            if not r:
                break
            try:
                data = os.read(fd, 8192)
            except OSError:
                break
            if not data:
                break
            out.append(data.decode("utf-8", "replace"))
            total += len(data)
        return "".join(out)

    @staticmethod
    def _kill(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:                                # pragma: no cover
            try:
                proc.terminate()
            except OSError:
                pass
        for _ in range(30):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        try:                                           # pragma: no cover
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass


def _tail(text, lines=15):
    parts = text.rstrip().split("\n")
    return "\n".join(parts[-lines:])


# ----------------------------------------------------------------------------
# 편의 함수
# ----------------------------------------------------------------------------

def enumerate_configurations(workdir=".", max_volume=1, dimension=3,
                             binary=None, echo=False, **kwargs):
    """열거만 수행한다(디렉토리는 만들지 않는다). make_dirs 가 생성된다."""
    drv = MainclustDriver(workdir=workdir, binary=binary, echo=echo, **kwargs)
    return drv.run(answers_enumerate(max_volume=max_volume, dimension=dimension))


def generate_vasp_inputs(workdir=".", energy=0, reference=0,
                         binary=None, echo=False, check_potcar=True, **kwargs):
    """make_dirs 를 읽어 con* 디렉토리를 만든다.

    ``check_potcar`` 가 True 면 만들기 전에 ``POTCAR_<원소>`` 가 있는지 보고,
    만든 뒤에는 각 배열의 POTCAR 가 비지 않았는지 확인한다. mainclust 는 둘 다
    조용히 넘어가므로 여기서 막지 않으면 0바이트 POTCAR 로 계산이 제출된다.

    Returns
    -------
    MainclustResult
        ``potcar_report`` 속성에 확인 결과 문자열이 붙는다(확인했을 때).
    """
    if check_potcar:
        check_potcar_sources(workdir)

    drv = MainclustDriver(workdir=workdir, binary=binary, echo=echo, **kwargs)
    result = drv.run(answers_reuse(energy=energy, reference=reference))

    if check_potcar and result.ok:
        total, bad = check_generated_potcars(workdir)
        result.potcar_report = describe_potcars(total, bad)
        if bad:
            raise MainclustError(
                "con* 는 만들어졌지만 POTCAR 가 온전하지 않습니다.\n"
                + result.potcar_report
                + "\n\n  이대로 제출하면 VASP 가 엉뚱한 계산을 하거나 바로 죽습니다.",
                result.transcript)
    return result


# ----------------------------------------------------------------------------
# make_dirs 플래그
# ----------------------------------------------------------------------------

def set_make_flags(path="make_dirs", value=1, backup=True):
    """make_dirs 의 make 컬럼을 일괄로 바꾼다 (02_ModiMake.sh 대체).

    mainclust 는 이 컬럼이 1 인 배열만 VASP 입력 디렉토리로 만든다. 열거
    직후에는 전부 0 이므로 1 로 바꿔 줘야 한다.

    원본 셸 스크립트는 ``sed -e "s/0     0/1     0/g"`` 처럼 공백 개수에
    의존해, mainclust 출력 폭이 조금만 달라져도 조용히 아무것도 하지 않았다.
    여기서는 열로 나눠 다시 쓴다.

    Returns
    -------
    (changed, total)
    """
    if not os.path.isfile(path):
        raise MainclustError("%s 가 없습니다. mainclust 로 열거를 먼저 하세요." % path)

    lines = open(path).read().split("\n")
    header, rows, changed = None, [], 0
    for line in lines:
        if not line.strip():
            continue
        if header is None and line.lstrip().startswith("#"):
            header = line
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[1] != str(value):
            changed += 1
        rows.append((parts[0], str(value), parts[2]))

    if not rows:
        raise MainclustError("%s 에서 배열 줄을 찾지 못했습니다." % path)

    if backup:
        orig = path + "_orig"
        if not os.path.isfile(orig):
            shutil.copy(path, orig)

    out = [header] if header else ["#    name      make      concentrations  "]
    out += ["%s  %s     %s   " % r for r in rows]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return changed, len(rows)
