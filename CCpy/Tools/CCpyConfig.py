"""CCpy 개인 설정 폴더(`~/.CCpy_test`)의 위치를 정하는 단일 지점.

CCpy 는 패키지에 동봉된 yaml 을 직접 읽지 않고, 처음 실행할 때 사용자 홈의
설정 폴더로 한 번 복사한 뒤 그 사본을 읽는다. 이 repo(CCpy_test)는 업그레이드
작업용 사본이므로, 프로덕션 CCpy 가 쓰는 `~/.CCpy` 를 그대로 공유하면
서로의 설정을 오염시킨다:

  - 테스트 버전이 만든 default.yaml / queue_config.yaml 을 프로덕션이 읽는다.
  - 반대로 오래된 홈 사본 때문에 repo 에 추가한 yaml 키가 무시된다
    (사본은 최초 1회만 생성되므로).
  - queue_config.yaml 의 `python_path` 가 프로덕션 인터프리터를 가리켜서,
    테스트 버전으로 제출한 잡이 계산 노드에서 프로덕션 파이썬으로 돈다.
    (이건 폴더를 분리해도 남는 문제라 resolve_python_path() 로 따로 다룬다 --
     패키지 템플릿 자체에 Python 3.8 환경 경로가 하드코딩돼 있었기 때문에,
     새로 만든 파일이든 기존 홈에서 승계한 파일이든 같은 값이 들어갔다.)

그래서 이 repo 는 `~/.CCpy_test` 를 쓴다. 경로 문자열을 여러 파일에 흩어놓지
않기 위해 여기 한 곳에서만 정의하고, 나머지 코드는 아래 함수들을 호출한다.

표준 라이브러리만 import 한다 (CCpyTools 등을 import 하면 순환 import 가 된다).
"""

import os
import re
import shutil
import sys
from pathlib import Path

# -- 이 repo 를 production 으로 승격할 때는 이 한 줄만 ".CCpy" 로 되돌린다.
DEFAULT_CONFIG_DIRNAME = ".CCpy_test"

# -- queue_config.yaml 승계 원본 (프로덕션 CCpy 의 설정 폴더).
#    읽기만 한다. 이 폴더의 파일을 수정하거나 지우지 않는다.
LEGACY_CONFIG_DIRNAME = ".CCpy"

# -- 설정 폴더를 다른 곳으로 옮기고 싶을 때 쓰는 환경변수 (CCpy_SERVER 와 같은 형식).
CONFIG_HOME_ENV = "CCpy_HOME"


def config_home():
    """설정 폴더 경로 (Path). $CCpy_HOME 이 설정돼 있으면 그쪽을 쓴다."""
    env = os.environ.get(CONFIG_HOME_ENV)
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env)))
    return Path(os.path.expanduser("~")) / DEFAULT_CONFIG_DIRNAME


def legacy_config_home():
    """프로덕션 CCpy 의 설정 폴더 (`~/.CCpy`). 승계 원본으로만 쓴다."""
    return Path(os.path.expanduser("~")) / LEGACY_CONFIG_DIRNAME


def ensure_config_home():
    """설정 폴더가 없으면 만든다. 있으면 아무 일도 하지 않는다."""
    d = config_home()
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path(*parts):
    """설정 폴더 아래 경로를 만든다. ex) config_path("g09_input.json")"""
    return config_home().joinpath(*parts)


def vasp_config_dir():
    """VASP preset yaml 폴더 (`~/.CCpy_test/vasp`)."""
    return config_home() / "vasp"


def queue_config_path():
    """`~/.CCpy_test/queue_config.yaml`"""
    return config_home() / "queue_config.yaml"


def package_queue_config_template():
    """패키지에 동봉된 queue_config.yaml 템플릿 (`CCpy/Queue/queue_config.yaml`)."""
    return Path(__file__).resolve().parent.parent / "Queue" / "queue_config.yaml"


PYTHON_PATH_KEY = "python_path"
_PYTHON_PATH_NOTICE_SHOWN = False
_PYTHON_PATH_MISMATCH_SHOWN = False


def strip_python_path(text):
    """queue_config.yaml 내용에서 python_path 지정을 주석 처리한다.

    (text, 주석 처리된 원래 줄 or None) 을 돌려준다. yaml 로 다시 쓰지 않고
    줄 단위로 바꾸는 이유는, 원본의 주석과 키 순서를 그대로 남기기 위해서다.
    """
    out = []
    stripped = None
    for line in text.splitlines(True):
        if re.match(r"\s*%s\s*:" % PYTHON_PATH_KEY, line):
            stripped = line.strip()
            out.append("# (승계 시 제외) %s\n" % stripped)
            out.append("# python_path 를 비워두면 CCpy 를 실행 중인 python 이 그대로 쓰입니다.\n")
            continue
        out.append(line)
    return "".join(out), stripped


def resolve_python_path(queue_config=None, verbose=True):
    """잡 스크립트에서 쓸 python 실행 경로를 정한다.

    우선순위
      1) queue_config.yaml 의 python_path -- 단, 그 경로가 실제로 존재할 때만
         (절대경로면 파일 존재 확인, 명령 이름이면 $PATH 에서 탐색)
      2) 지금 CCpy 를 실행 중인 인터프리터 (sys.executable)

    패키지 템플릿에 프로덕션 python 경로가 하드코딩돼 있어서, python_path 를
    고치지 않으면(또는 없는 경로로 고치면) 잡이 조용히 Python 3.8 환경에서
    돌았다. 키를 비워두면 CCpy 를 실행한 환경의 python 이 그대로 쓰인다.
    """
    global _PYTHON_PATH_NOTICE_SHOWN

    configured = None
    if queue_config:
        configured = queue_config.get(PYTHON_PATH_KEY)
    if isinstance(configured, str):
        configured = configured.strip()

    current = sys.executable

    if configured:
        resolved = None
        if os.path.isabs(configured):
            if os.path.isfile(configured):
                resolved = configured
        else:
            resolved = shutil.which(configured)
        if resolved:
            # 명시된 값을 그대로 존중하되, 지금 CCpy 를 돌리는 python 과 다르면
            # 알려준다. 잡이 다른 환경(예: 프로덕션 Python 3.8)에서 돌고 있는데도
            # 모르고 있는 상황을 막기 위한 것이다.
            if verbose and os.path.realpath(resolved) != os.path.realpath(current):
                global _PYTHON_PATH_MISMATCH_SHOWN
                if not _PYTHON_PATH_MISMATCH_SHOWN:
                    print("* 주의: 잡은 queue_config.yaml 의 python_path 로 돕니다: %s" % resolved)
                    print("        지금 CCpy 를 실행 중인 python 은 %s 입니다." % current)
                    print("        같은 환경으로 돌리려면 queue_config.yaml 의 python_path 줄을 지우세요.")
                    _PYTHON_PATH_MISMATCH_SHOWN = True
            return resolved
        if verbose:
            print("* queue_config.yaml 의 python_path 를 찾을 수 없습니다: %s" % configured)
            print("  -> 현재 CCpy 를 실행 중인 python 을 씁니다: %s" % current)
        return current

    if verbose and not _PYTHON_PATH_NOTICE_SHOWN:
        print("* python_path 미지정 -> 현재 CCpy 를 실행 중인 python 을 씁니다: %s" % current)
        _PYTHON_PATH_NOTICE_SHOWN = True
    return current


def ensure_queue_config(template=None):
    """queue_config.yaml 이 없으면 만들고, 그 경로를 돌려준다.

    기존 프로덕션 설정(`~/.CCpy/queue_config.yaml`)이 있으면 그것을 복사해서
    승계한다. 이 파일은 서버·개인마다 값이 다르고(lammps/siesta 경로,
    lammps_mpi_run, lammps_env 등) 템플릿으로 다시 만들면 그 값들을 전부
    잃기 때문이다. 원본은 읽기만 하고 건드리지 않는다.

    단 `python_path` 는 승계하지 않는다 (주석 처리해서 남긴다) -- 기존 홈의
    값은 프로덕션 Python 3.8 환경을 가리키고 있어서, 승계하면 테스트 환경에서
    제출한 잡이 그 인터프리터로 돌아간다. 비워두면 resolve_python_path() 가
    CCpy 를 실행 중인 python 을 쓴다.

    template : 패키지에 동봉된 queue_config.yaml 경로 (승계 원본이 없을 때 사용).
               생략하면 package_queue_config_template() 을 쓴다.
    """
    if template is None:
        template = package_queue_config_template()
    target = queue_config_path()
    if target.is_file():
        return target

    ensure_config_home()
    legacy = legacy_config_home() / "queue_config.yaml"
    if legacy.is_file():
        text, stripped = strip_python_path(legacy.read_text(encoding="utf-8"))
        target.write_text(text, encoding="utf-8")
        print("* queue_config.yaml 을 기존 설정에서 승계했습니다.")
        print("    %s  ->  %s" % (legacy, target))
        if stripped:
            print("  - python_path 는 승계하지 않았습니다 (%s)." % stripped)
            print("    CCpy 를 실행 중인 python 이 자동으로 쓰입니다.")
    else:
        shutil.copy(str(template), str(target))
        print("* queue_config.yaml 을 새로 만들었습니다: %s" % target)
    return target
