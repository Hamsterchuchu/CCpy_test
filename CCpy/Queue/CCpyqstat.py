#!/usr/bin/env python
"""
서버별 qstat 구현 선택기 (dispatcher).

cms2 와 node99 는 큐 상태를 읽는 방식이 완전히 다르다.

    cms2   : PBS 형식 'qstat -f' 출력 파싱   -> CCpyqstat_cms2.py
    node99 : SLURM 'squeue' 출력 파싱        -> CCpyqstat_node99.py

두 구현 모두 CCpyqstat / get_empty_nodes / get_waiting_nodes 라는
같은 인터페이스를 갖고 있으므로, 여기서 서버에 맞는 쪽을 골라 그대로 노출한다.
(CCpy/bin/CCpyqstat.py 는 이 파일을 가리키는 심볼릭 링크다.)

서버 판별 규칙은 CCpy/Queue/server_profile.py 참고.
  1) 환경변수 $CCpy_SERVER  ("cms2" / "node99")
  2) hostname 추정
  3) DEFAULT_SERVER
"""

from CCpy.Queue.server_profile import get_server_name

_server = get_server_name()

if _server == "node99":
    from CCpy.Queue import CCpyqstat_node99 as _impl
else:
    from CCpy.Queue import CCpyqstat_cms2 as _impl

# -- 공통 인터페이스
CCpyqstat = _impl.CCpyqstat
get_empty_nodes = _impl.get_empty_nodes
get_waiting_nodes = _impl.get_waiting_nodes

# -- cms2 판에만 있는 기능
if hasattr(_impl, "chk_load"):
    chk_load = _impl.chk_load


if __name__ == "__main__":
    import runpy

    runpy.run_module(_impl.__name__, run_name="__main__")
