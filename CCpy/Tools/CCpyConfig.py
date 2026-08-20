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
    (패키지 템플릿에 그 경로가 하드코딩돼 있었기 때문이다. 그래서 파일을
     처음 만들 때 `python_path` 를 그 시점의 python 절대경로로 기록한다 --
     아래 ensure_queue_config() 참고.)

파일을 만든 뒤에는 CCpy 가 설정 yaml 을 다시 고치지 않는다. 서버·개인별 값은
사용자가 직접 수정하고, 코드는 읽기만 한다.

그래서 이 repo 는 `~/.CCpy_test` 를 쓴다. 경로 문자열을 여러 파일에 흩어놓지
않기 위해 여기 한 곳에서만 정의하고, 나머지 코드는 아래 함수들을 호출한다.

표준 라이브러리만 import 한다 (CCpyTools 등을 import 하면 순환 import 가 된다).
"""

import os
import re
import sys
from pathlib import Path

# -- 이 repo 를 production 으로 승격할 때는 이 한 줄만 ".CCpy" 로 되돌린다.
DEFAULT_CONFIG_DIRNAME = ".CCpy_test"

# -- 설정 폴더를 다른 곳으로 옮기고 싶을 때 쓰는 환경변수 (CCpy_SERVER 와 같은 형식).
CONFIG_HOME_ENV = "CCpy_HOME"


def config_home():
    """설정 폴더 경로 (Path). $CCpy_HOME 이 설정돼 있으면 그쪽을 쓴다."""
    env = os.environ.get(CONFIG_HOME_ENV)
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env)))
    return Path(os.path.expanduser("~")) / DEFAULT_CONFIG_DIRNAME


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


def set_python_path(text, python_path):
    """queue_config.yaml 텍스트의 python_path 값을 주어진 경로로 바꾼다.

    (바뀐 text, 기록한 값) 을 돌려준다. 주석과 키 순서를 그대로 남기려고 yaml
    로 다시 쓰지 않고 줄 단위로 바꾼다. 키가 없으면 맨 앞에 추가한다.
    """
    out = []
    done = False
    for line in text.splitlines(True):
        if not done and re.match(r"\s*%s\s*:" % PYTHON_PATH_KEY, line):
            out.append("%s: %s\n" % (PYTHON_PATH_KEY, python_path))
            done = True
            continue
        out.append(line)
    if not done:
        out.insert(0, "%s: %s\n" % (PYTHON_PATH_KEY, python_path))
    return "".join(out), python_path


def ensure_queue_config(template=None):
    """queue_config.yaml 이 없으면 패키지 템플릿에서 만들고, 그 경로를 돌려준다.

    만들 때 `python_path` 만 지금 CCpy 를 실행 중인 python 의 절대경로
    (sys.executable)로 채운다. 각자 자기 가상환경에서 CCpy 명령을 한 번 돌리면
    그 환경의 python 이 기록되므로, 템플릿에 특정 환경 경로를 하드코딩하거나
    사용자가 처음부터 손으로 고칠 필요가 없다.

    파일이 이미 있으면 아무것도 하지 않는다. **만든 뒤에는 CCpy 가 이 파일을
    다시 고치지 않는다** -- 서버·개인별 경로(vasp/lammps/siesta 등)는 사용자가
    직접 수정하고 코드는 읽기만 한다.

    기존 `~/.CCpy/queue_config.yaml` 을 승계하는 동작은 없다. 프로덕션 값
    (특히 Python 3.8 을 가리키는 python_path)이 딸려 들어와서 폐기했다.

    template : 패키지 동봉 queue_config.yaml 경로. 생략하면
               package_queue_config_template() 을 쓴다.
    """
    if template is None:
        template = package_queue_config_template()
    target = queue_config_path()
    if target.is_file():
        return target

    ensure_config_home()
    text, recorded = set_python_path(Path(template).read_text(encoding="utf-8"), sys.executable)
    target.write_text(text, encoding="utf-8")
    print("* queue_config.yaml 을 새로 만들었습니다: %s" % target)
    print("    python_path: %s   (지금 CCpy 를 실행 중인 python)" % recorded)
    print("  - vasp/lammps/siesta 경로 등 서버·개인별 설정은 이 파일을 직접 수정하세요.")
    print("    CCpy 는 만든 뒤에는 이 파일을 고치지 않습니다.")
    return target
