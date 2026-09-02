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


DEFAULT_BINARY = "mainclust"

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
# 드라이버
# ----------------------------------------------------------------------------

class MainclustDriver(object):
    """mainclust 를 사람 없이 구동한다.

    Parameters
    ----------
    workdir : str
        mainclust 를 돌릴 디렉토리. PRIM 등 입력이 여기 있어야 한다.
    binary : str
        실행 파일 경로. 상대 경로면 workdir 기준으로 먼저 찾고,
        없으면 PATH 에서 찾는다. 기본값은 관례대로 workdir 안의 ``mainclust``.
    idle_timeout : float
        아무 출력도 없이 이만큼(초) 지나면 멈춘 것으로 본다.
        Checkpoint 1(대칭 분석)은 큰 셀에서 수 분이 걸릴 수 있으므로 넉넉히 둔다.
    total_timeout : float or None
        전체 실행 시간 상한. None 이면 제한하지 않는다.
    echo : bool
        True 면 mainclust 출력을 그대로 화면에 흘린다.
    """

    def __init__(self, workdir=".", binary=DEFAULT_BINARY,
                 idle_timeout=600.0, total_timeout=None, echo=False):
        self.workdir = os.path.abspath(workdir)
        self.binary = binary
        self.idle_timeout = float(idle_timeout)
        self.total_timeout = total_timeout
        self.echo = echo

    # -- 경로 ---------------------------------------------------------------

    def resolve_binary(self):
        """실행 파일의 절대 경로를 돌려준다. 없으면 MainclustError."""
        cand = self.binary
        if os.path.sep in cand or cand.startswith("."):
            path = os.path.abspath(os.path.join(self.workdir, cand)) \
                if not os.path.isabs(cand) else cand
            if os.path.isfile(path):
                if not os.access(path, os.X_OK):
                    raise MainclustError(
                        "%s 에 실행 권한이 없습니다. chmod +x 로 붙여 주세요." % path)
                return path
            raise MainclustError("mainclust 를 찾지 못했습니다: %s" % path)

        local = os.path.join(self.workdir, cand)
        if os.path.isfile(local):
            if not os.access(local, os.X_OK):
                raise MainclustError(
                    "%s 에 실행 권한이 없습니다. chmod +x 로 붙여 주세요." % local)
            return local

        from shutil import which
        found = which(cand)
        if found:
            return found
        raise MainclustError(
            "mainclust 를 찾지 못했습니다. '%s' 가 작업 디렉토리에도 PATH 에도 "
            "없습니다. --binary 로 경로를 지정하거나 CASM_ref 에서 복사해 주세요."
            % cand)

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
                             binary=DEFAULT_BINARY, echo=False, **kwargs):
    """열거만 수행한다(디렉토리는 만들지 않는다). make_dirs 가 생성된다."""
    drv = MainclustDriver(workdir=workdir, binary=binary, echo=echo, **kwargs)
    return drv.run(answers_enumerate(max_volume=max_volume, dimension=dimension))


def generate_vasp_inputs(workdir=".", energy=0, reference=0,
                         binary=DEFAULT_BINARY, echo=False, **kwargs):
    """make_dirs 를 읽어 con* 디렉토리를 만든다."""
    drv = MainclustDriver(workdir=workdir, binary=binary, echo=echo, **kwargs)
    return drv.run(answers_reuse(energy=energy, reference=reference))
