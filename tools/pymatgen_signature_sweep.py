#!/usr/bin/env python
"""
CCpy 안의 pymatgen API 호출을 현재 설치된 pymatgen 시그니처와 대조한다.

왜 필요한가
-----------
pymatgen 을 올릴 때 흔히 깨지는 것은 두 가지다.

  (1) import 경로 변경        -> 일반적인 import 검사로 잡힌다
  (2) 호출 시그니처 변경      -> import 는 멀쩡히 되고 실행할 때 터진다

(2)가 문제다. 특히 CCpy 는 계산 노드에서 돌 스크립트를 템플릿 "문자열"로
들고 있어서(NVTLoopQueScript 등), 그 안의 코드는 정적 검사에 아예 걸리지
않는다. 실제로 이런 것들이 이 방식으로 발견됐다.

  MITMDSet(structure, 100.0, temp, nsw, user_incar_settings=...)
      -> pymatgen 2026.x 에서 파라미터 순서가 바뀌어
         TypeError: got multiple values for argument 'user_incar_settings'

  VanHoveAnalysis(..., cellrange=1, ...)
      -> 파라미터명이 cell_range 로 바뀜

동작 방식
---------
1. CCpy 의 모든 .py 를 AST 로 파싱
2. `from pymatgen... import X` 로 들어온 이름을 추적
3. `X(...)` 호출과 `X.method(...)` 속성 호출을 찾음
4. 실제 객체의 시그니처에 inspect.Signature.bind 로 대조 (값은 더미)
5. 템플릿 문자열로 생성되는 스크립트도 꺼내어 같이 검사

한계
----
인자의 개수와 이름만 본다. 값의 의미가 바뀐 경우(단위 변경, 기본값 변경,
반환값 개수 변경 등)는 잡지 못한다. 그런 것은 실제 실행으로 확인해야 한다.

사용법
------
    CCpy_SCHEDULER_CONFIG=<스케줄러설정.yaml> CCpy_SERVER=cms2 MPLBACKEND=Agg \
        python tools/pymatgen_signature_sweep.py

설치된 CCpy 를 검사한다. 소스 트리를 검사하려면 그 디렉토리에서 실행할 것.
불일치가 있으면 종료 코드 1 을 돌려준다.
"""

import ast
import importlib
import inspect
import os
import sys
import warnings

warnings.filterwarnings("ignore")


class _Dummy:
    """bind() 에 넘길 자리 채우개. 값은 보지 않는다."""


DUMMY = _Dummy()


def _load(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _check_call(obj, node, where, label, found):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return                                  # 시그니처를 못 얻는 내장/C 확장 등

    if any(isinstance(a, ast.Starred) for a in node.args):
        return                                  # *args 전개는 판정 불가
    keywords = [k.arg for k in node.keywords]
    if None in keywords:
        return                                  # **kwargs 전개는 판정 불가

    try:
        sig.bind(*([DUMMY] * len(node.args)), **{k: DUMMY for k in keywords})
    except TypeError as exc:
        found.append({
            "where": where,
            "line": node.lineno,
            "label": label,
            "n_positional": len(node.args),
            "keywords": keywords,
            "error": str(exc),
        })


def scan_source(src, where, found):
    """소스 문자열 하나를 검사한다."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return

    names = {}                                  # 지역 이름 -> 실제 객체
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("pymatgen"):
            module = _load(node.module)
            if module is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                obj = getattr(module, alias.name, None)
                if obj is not None:
                    names[alias.asname or alias.name] = obj

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in names:
            _check_call(names[func.id], node, where, func.id, found)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id in names:
            obj = getattr(names[func.value.id], func.attr, None)
            if obj is not None:
                _check_call(obj, node, where,
                            "%s.%s" % (func.value.id, func.attr), found)


# 템플릿 문자열로 계산 노드용 스크립트를 만들어내는 것들.
# 새로 생기면 여기에 추가할 것.
TEMPLATE_SOURCES = [
    ("CCpy.Package.Diffusion.NVTLoopQueScript",
     "NVTLoopQueScriptString", "<생성 스크립트: VASP AIMD (.AIMDLoop.py)>"),
    ("CCpy.Package.Diffusion.SIESTA_NVTLoopQueScript",
     "NVTLoopQueScriptString", "<생성 스크립트: SIESTA NVT>"),
]

SKIP_DIRS = {"__pycache__", "bin", "myGausssum"}


def main():
    found = []
    n_scanned = 0

    import CCpy
    base = os.path.dirname(CCpy.__file__)
    print("검사 대상: %s" % base)

    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in sorted(files):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            with open(path, encoding="utf-8", errors="replace") as fh:
                scan_source(fh.read(), os.path.relpath(path, os.path.dirname(base)), found)
            n_scanned += 1

    for module_name, func_name, label in TEMPLATE_SOURCES:
        module = _load(module_name)
        if module is None or not hasattr(module, func_name):
            print("  (건너뜀: %s.%s 를 불러오지 못함)" % (module_name, func_name))
            continue
        scan_source(getattr(module, func_name)(), label, found)
        n_scanned += 1

    print("검사한 파일/템플릿: %d개" % n_scanned)
    print("시그니처 불일치: %d건" % len(found))
    print()
    for item in found:
        print("  %s:%s" % (item["where"], item["line"]))
        print("    %s(위치 %d개, 키워드 %s)"
              % (item["label"], item["n_positional"], item["keywords"]))
        print("    -> %s" % item["error"])
        print()

    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
