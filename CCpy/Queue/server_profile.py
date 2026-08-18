"""
서버(클러스터)별 하드웨어 프로파일 정의.

cms2 와 node99 는 같은 CCpy 코드를 쓰지만 노드 이름 / 파티션 구성이
서로 다르다. 이전에는 두 서버가 CCpyJobControl.py 와 CCpyqstat.py 를
각자 직접 고쳐서 쓰고 있었고, 그 때문에 두 서버 코드가 갈라져 있었다.

여기서는 두 서버의 설정을 모두 담아두고, 실행 시 어느 서버인지에 따라
골라 쓰도록 한다. 서버 선택 순서는 다음과 같다.

  1) 환경변수 $CCpy_SERVER  ("cms2" 또는 "node99")   <- 권장.
     .bashrc 에    export CCpy_SERVER=cms2    처럼 한 줄 추가하면 된다.
  2) hostname 으로 추정
  3) 위 둘 다 실패하면 DEFAULT_SERVER

새 서버를 추가할 때는 NODE_PROFILES 에 항목을 하나 더 넣으면 된다.
"""

import os
import socket

# 어느 서버인지 판별하지 못했을 때 사용할 기본값
DEFAULT_SERVER = "cms2"

NODE_PROFILES = {
    # ------------------------------------------------------------------ #
    "cms2": {
        "default_partition": "72core",
        # 파티션 이름 -> 그 파티션에 속한 노드 목록
        "node_partitions": {
            "8core":   ["node00", "node01"],
            "72core":  ["node02", "node03", "node04", "node05", "node06"],
            "128core": ["node07"],
        },
        # 노드 이름 대신 코어수만 지정했을 때 (-n 8 처럼) 매핑할 파티션.
        # 이 경우 특정 노드를 지정하지 않고 파티션 전체에 던진다.
        "partition_alias": {
            "8":   "8core",
            "128": "128core",
        },
    },
    # ------------------------------------------------------------------ #
    "node99": {
        "default_partition": "48core",
        "node_partitions": {
            "48core":  ["node01", "node02", "node03", "node04",
                        "node05", "node06", "node07", "node08"],
            "96core":  ["node09", "node10", "node11", "node12"],
            "64core":  ["node13", "node14", "node15", "node16",
                        "node17", "node18"],
            "256core": ["node19"],
        },
        "partition_alias": {
            "96":  "96core",
            "64":  "64core",
            "256": "256core",
        },
    },
}


def get_server_name():
    """현재 실행중인 서버 이름을 돌려준다."""
    name = os.environ.get("CCpy_SERVER", "").strip()
    if name in NODE_PROFILES:
        return name

    try:
        host = socket.gethostname().lower()
    except Exception:
        host = ""
    for key in NODE_PROFILES:
        if key in host:
            return key

    return DEFAULT_SERVER


def get_node_profile(server=None):
    """서버 프로파일 dict 를 돌려준다."""
    if server is None:
        server = get_server_name()
    return NODE_PROFILES[server]
