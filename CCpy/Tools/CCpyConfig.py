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

그래서 이 repo 는 `~/.CCpy_test` 를 쓴다. 경로 문자열을 여러 파일에 흩어놓지
않기 위해 여기 한 곳에서만 정의하고, 나머지 코드는 아래 함수들을 호출한다.

표준 라이브러리만 import 한다 (CCpyTools 등을 import 하면 순환 import 가 된다).
"""

import os
import shutil
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


def ensure_queue_config(template=None):
    """queue_config.yaml 이 없으면 만들고, 그 경로를 돌려준다.

    기존 프로덕션 설정(`~/.CCpy/queue_config.yaml`)이 있으면 그것을 복사해서
    승계한다. 이 파일은 서버·개인마다 값이 다르고(lammps/siesta 경로,
    lammps_mpi_run, lammps_env 등) 템플릿으로 다시 만들면 그 값들을 전부
    잃기 때문이다. 원본은 읽기만 하고 건드리지 않는다.

    승계된 파일의 `python_path` 는 프로덕션 인터프리터를 가리키고 있을 수
    있으므로 경고를 출력한다.

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
        shutil.copy(str(legacy), str(target))
        print("* queue_config.yaml 을 기존 설정에서 승계했습니다.")
        print("    %s  ->  %s" % (legacy, target))
        print("  ! 승계된 python_path 가 프로덕션 환경을 가리킬 수 있습니다.")
        print("    이 repo 를 쓰는 환경의 python 경로로 맞는지 확인하세요.")
    else:
        shutil.copy(str(template), str(target))
        print("* queue_config.yaml 을 새로 만들었습니다: %s" % target)
    return target
