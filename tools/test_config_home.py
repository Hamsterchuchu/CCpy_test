#!/usr/bin/env python
"""개인 설정 폴더 분리(`~/.CCpy` -> `~/.CCpy_test`) 검증 테스트.

repo 루트에서 실행한다:

    python3 tools/test_config_home.py

임시 HOME 안에서만 동작하므로 실제 홈의 `~/.CCpy` / `~/.CCpy_test` 는 건드리지
않는다. pymatgen 이 없는 환경에서는 VASPio 관련 항목만 SKIP 되고 나머지는 그대로
돈다.

확인 항목
  1. CCpyConfig 경로 함수 (기본값, $CCpy_HOME, 하위 경로)
  2. queue_config.yaml 생성 규칙 (템플릿에서 생성 + python_path 기록 / 이미 있으면 유지)
  3. 기존 `~/.CCpy` 를 승계하지도, 수정하지도 않는지
  4. AIMD 루프 생성 스크립트에 절대경로가 박히는지
  5. JobSubmit(init_only=True) 실제 경로
  6. VASPio 가 `~/.CCpy_test/vasp/` 를 만들고 그 사본을 읽는지
  7. 코드에 `.CCpy` 리터럴이 남아있지 않은지
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

RESULTS = []


def check(name, cond, extra=""):
    RESULTS.append((bool(cond), name, extra))
    mark = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (mark, name, ("  <- " + str(extra)) if (extra and not cond) else ""))


def skip(name, why):
    RESULTS.append((None, name, why))
    print("  [SKIP] %s  (%s)" % (name, why))


def fresh_home():
    """새 임시 HOME 을 만들어 os.environ 에 걸고 경로를 돌려준다."""
    home = tempfile.mkdtemp(prefix="ccpy_test_home_")
    os.environ["HOME"] = home
    os.environ.pop("CCpy_HOME", None)
    return Path(home)


def snapshot(d):
    """폴더 안 파일들의 (상대경로, 크기, mtime_ns, 내용) 스냅샷."""
    out = {}
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(d))] = (st.st_size, st.st_mtime_ns, p.read_bytes())
    return out


# ---------------------------------------------------------------- 1. 경로 함수
print("\n1. CCpyConfig 경로 함수")
from CCpy.Tools import CCpyConfig as cfg

home = fresh_home()
check("기본 설정 폴더 = ~/.CCpy_test", cfg.config_home() == home / ".CCpy_test", cfg.config_home())
check("vasp 폴더", cfg.vasp_config_dir() == home / ".CCpy_test" / "vasp")
check("queue_config 경로", cfg.queue_config_path() == home / ".CCpy_test" / "queue_config.yaml")
check("config_path()", cfg.config_path("g09_input.json") == home / ".CCpy_test" / "g09_input.json")
check("패키지 템플릿 존재", cfg.package_queue_config_template().is_file(),
      cfg.package_queue_config_template())

check("ensure_config_home 이 폴더를 만든다", cfg.ensure_config_home().is_dir())
check("ensure_config_home 재호출 안전", cfg.ensure_config_home().is_dir())

os.environ["CCpy_HOME"] = "~/somewhere_else"
check("$CCpy_HOME 오버라이드 (~ 확장)", cfg.config_home() == home / "somewhere_else", cfg.config_home())
os.environ["CCpy_HOME"] = str(home / "env_dir")
check("$CCpy_HOME 오버라이드 (절대경로)", cfg.config_home() == home / "env_dir")
os.environ.pop("CCpy_HOME")

# ------------------------------------------------- 2~3. queue_config 생성 규칙
print("\n2. queue_config.yaml 생성 규칙")
template = cfg.package_queue_config_template()
template_text = template.read_text(encoding="utf-8")
PROD_PYTHON = "/home/shared/anaconda3/envs/CCpy/bin/python"

# (a) 없으면 패키지 템플릿에서 만들고, python_path 만 지금 python 으로 기록
home = fresh_home()
made = cfg.ensure_queue_config()
made_text = made.read_text(encoding="utf-8")
made_cfg = yaml.safe_load(made_text) or {}
check("(a) 템플릿에서 새로 생성", made.is_file(), made)
check("(a) python_path 에 지금 실행 중인 python 이 기록된다",
      made_cfg.get("python_path") == sys.executable, made_cfg.get("python_path"))
check("(a) python_path 외의 값은 템플릿과 같다",
      {k: v for k, v in made_cfg.items() if k != "python_path"}
      == {k: v for k, v in (yaml.safe_load(template_text) or {}).items() if k != "python_path"})
check("(a) 주석 줄도 그대로 남는다",
      [l for l in made_text.splitlines() if l.startswith("#")]
      == [l for l in template_text.splitlines() if l.startswith("#")])
check("(a) 프로덕션 폴더를 만들지 않는다", not (home / ".CCpy").exists())

# (b) 이미 있으면 손대지 않는다 (사용자가 고친 값을 코드가 되돌리지 않는다)
made.write_text("qsub: sbatch\npython_path: /my/env/bin/python\n# edited by user\n",
                encoding="utf-8")
before = made.stat().st_mtime_ns
again = cfg.ensure_queue_config()
after_text = made.read_text(encoding="utf-8")
check("(b) 이미 있으면 덮어쓰지 않는다",
      again == made and made.stat().st_mtime_ns == before
      and "python_path: /my/env/bin/python" in after_text
      and after_text.endswith("# edited by user\n"))

# (c) 기존 ~/.CCpy 가 있어도 승계하지 않고, 원본도 건드리지 않는다
home = fresh_home()
legacy_dir = home / ".CCpy"
legacy_dir.mkdir()
legacy = legacy_dir / "queue_config.yaml"
legacy_text = (template_text
               + "\npython_path: %s\n" % PROD_PYTHON
               + "lammps_path: /home/pomepaw/lammps/build/lmp\n"
               + "lammps_mpi_run: srun --mpi=pmi2\n")
legacy.write_text(legacy_text, encoding="utf-8")
(legacy_dir / "vasp").mkdir()
(legacy_dir / "vasp" / "default.yaml").write_text("INCAR:\n  ENCUT: 500\n", encoding="utf-8")
before_legacy = snapshot(legacy_dir)

made = cfg.ensure_queue_config()
made_text = made.read_text(encoding="utf-8")
made_cfg = yaml.safe_load(made_text) or {}
check("(c) 기존 ~/.CCpy 를 승계하지 않는다", "lammps_mpi_run" not in made_cfg, made_cfg)
check("(c) 프로덕션 python_path 가 따라오지 않는다",
      made_cfg.get("python_path") == sys.executable, made_cfg.get("python_path"))
check("(c) 기존 ~/.CCpy 를 수정하지 않는다", snapshot(legacy_dir) == before_legacy)

# ------------------------------------------- 4. AIMD 루프 생성 스크립트 경로 주입
print("\n4. AIMD 루프 생성 스크립트 (NVTLoopQueScript)")
home = fresh_home()
from CCpy.Package.Diffusion.NVTLoopQueScript import NVTLoopQueScriptString

script = NVTLoopQueScriptString()
expected = str(cfg.queue_config_path())
check("placeholder 가 남지 않는다", "@@CCPY_QUEUE_CONFIG_PATH@@" not in script)
check("절대경로가 박힌다", ('_queue_config_path = "%s"' % expected) in script, expected)
check("`.CCpy/` 리터럴이 없다", "/.CCpy/" not in script)
try:
    compile(script, "<AIMDLoop>", "exec")
    check("생성 스크립트 문법 정상", True)
except SyntaxError as exc:
    check("생성 스크립트 문법 정상", False, exc)

# $CCpy_HOME 을 준 상태로 다시 생성하면 그 경로가 박혀야 한다
os.environ["CCpy_HOME"] = str(home / "env_dir")
script2 = NVTLoopQueScriptString()
check("$CCpy_HOME 이 생성 스크립트에 반영된다",
      str(home / "env_dir" / "queue_config.yaml") in script2)
os.environ.pop("CCpy_HOME")

# ------------------------------------------------- 5. JobSubmit(init_only=True)
print("\n5. JobSubmit(init_only=True)")
home = fresh_home()
(home / ".CCpy").mkdir()
(home / ".CCpy" / "queue_config.yaml").write_text(legacy_text, encoding="utf-8")
sched = home / "scheduler_config.yaml"
sched.write_text('test_queue: [16, 64, "all.q"]\n', encoding="utf-8")
os.environ["CCpy_SCHEDULER_CONFIG"] = str(sched)
os.environ.setdefault("CCpy_SERVER", "cms2")
code = (
    "import sys; sys.path.insert(0, %r)\n"
    "from CCpy.Queue.CCpyJobControl import JobSubmit as JS\n"
    "JS(None, None, None, init_only=True)\n" % str(REPO)
)
proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ))
check("JobSubmit 이 ~/.CCpy_test/queue_config.yaml 을 만든다",
      (home / ".CCpy_test" / "queue_config.yaml").is_file(), proc.stderr[-400:])
js_cfg = yaml.safe_load((home / ".CCpy_test" / "queue_config.yaml").read_text(encoding="utf-8")) or {}
check("JobSubmit 이 기록한 python_path = 그 프로세스의 python",
      js_cfg.get("python_path") == sys.executable, js_cfg.get("python_path"))
check("JobSubmit 도 기존 ~/.CCpy 를 승계하지 않는다", "lammps_mpi_run" not in js_cfg, js_cfg)
check("생성 안내가 출력된다", "새로 만들었습니다" in proc.stdout, proc.stdout[-300:])

# ------------------------------------------------------------------ 6. VASPio
print("\n6. VASPio preset 사본")
home = fresh_home()
try:
    import pymatgen  # noqa: F401
    have_pmg = True
except ImportError:
    have_pmg = False

if not have_pmg:
    skip("VASPio 관련 4항목", "pymatgen 미설치")
else:
    # 이 절의 목적은 preset yaml 을 어느 폴더에서 읽는지 하나뿐이다. 예전에는
    # 여기서 vi.encut_table 길이도 함께 찍었는데, ENCUT 자동 지정이 POTCAR
    # ENMAX 기준으로 일원화되면서(cb13db9) 그런 속성은 존재하지 않게 됐다.
    # 자식 프로세스가 AttributeError 로 죽으면서 아래 5개 검사가 통째로
    # 건너뛰어지고 있었으므로, 낡은 참조를 지우고 ENCUT 은 "계산 지점이
    # 남아 있는지"만 확인한다 (실제 값 검증은 POTCAR 가 필요해 이 절의 범위 밖).
    code = (
        "import sys, os, json; sys.path.insert(0, %r)\n"
        "from CCpy.VASP.VASPio import VASPInput\n"
        "vi = VASPInput()\n"
        "print(json.dumps({'dir': vi.vasp_config_dir, 'yaml': vi.yaml_file,\n"
        "                  'has_set_encut': callable(getattr(vi, 'set_encut', None))}))\n" % str(REPO)
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=dict(os.environ), cwd=str(home))
    info = {}
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            import json
            info = json.loads(line)
    check("VASPInput() 정상 실행", bool(info), proc.stderr[-600:])
    if info:
        check("preset 폴더가 ~/.CCpy_test/vasp/", info["dir"] == str(home / ".CCpy_test" / "vasp") + "/",
              info["dir"])
        check("default.yaml / band_sample.yaml 이 복사된다",
              (home / ".CCpy_test" / "vasp" / "default.yaml").is_file()
              and (home / ".CCpy_test" / "vasp" / "band_sample.yaml").is_file())
        check("읽는 yaml 이 그 사본이다", info["yaml"] == str(home / ".CCpy_test" / "vasp" / "default.yaml"))
        check("프로덕션 ~/.CCpy 를 만들지 않는다", not (home / ".CCpy").exists())
        check("ENCUT 자동 지정 지점(set_encut)이 남아 있다", info.get("has_set_encut") is True)

        # preset_yaml 조회도 새 폴더에서 되는지
        shutil.copy(str(home / ".CCpy_test" / "vasp" / "default.yaml"),
                    str(home / ".CCpy_test" / "vasp" / "my_preset.yaml"))
        code2 = code.replace("VASPInput()", "VASPInput(preset_yaml='my_preset.yaml')")
        proc2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True,
                               env=dict(os.environ), cwd=str(home))
        check("-preset= 조회가 새 폴더에서 된다",
              str(home / ".CCpy_test" / "vasp" / "my_preset.yaml") in proc2.stdout,
              proc2.stderr[-400:])

# ------------------------------------------------- 7. 남은 `.CCpy` 리터럴 검사
print("\n7. 코드에 남은 `.CCpy` 리터럴")
pattern = re.compile(r"""\.CCpy(?=["'/])""")
allowed = {"CCpy/Tools/CCpyConfig.py"}          # 상수 정의 + 설명
leftovers = []
for py in sorted((REPO / "CCpy").rglob("*.py")):
    rel = str(py.relative_to(REPO))
    if rel in allowed:
        continue
    for i, line in enumerate(py.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):            # 주석은 설명 목적이라 통과
            continue
        if pattern.search(line):
            leftovers.append("%s:%d: %s" % (rel, i, stripped[:80]))
check("코드에 하드코딩된 `.CCpy` 경로가 없다", not leftovers, "\n      ".join(leftovers))

# -------------------------------------------------- 8. python_path 기록 규칙
print("\n8. python_path 기록 규칙 (set_python_path)")
text, recorded = cfg.set_python_path("qsub: qsub\npython_path: python\nmpi_run: srun\n",
                                     "/my/env/bin/python")
check("기존 python_path 줄의 값만 바꾼다",
      text == "qsub: qsub\npython_path: /my/env/bin/python\nmpi_run: srun\n"
      and recorded == "/my/env/bin/python", text)
text2, _ = cfg.set_python_path("qsub: qsub\nmpi_run: srun\n", "/my/env/bin/python")
check("키가 없으면 맨 앞에 추가한다",
      text2 == "python_path: /my/env/bin/python\nqsub: qsub\nmpi_run: srun\n", text2)
text3, _ = cfg.set_python_path("python_path: /a/b\npython_path: /c/d\n", "/x/y")
check("첫 python_path 줄만 바꾼다", text3 == "python_path: /x/y\npython_path: /c/d\n", text3)
check("패키지 템플릿에 python_path 키가 있다 (표시용 기본값)",
      "python_path" in (yaml.safe_load(template_text) or {}))
check("코드에 python_path 자동 결정(resolve) 로직이 없다",
      not hasattr(cfg, "resolve_python_path"))
jc = (REPO / "CCpy" / "Queue" / "CCpyJobControl.py").read_text(encoding="utf-8")
check("JobControl 은 queue_config 값을 그대로 읽는다",
      "self.python_path = queue_config['python_path']" in jc)

# ------------------------------------------------------------------- 결과 요약
print("\n" + "=" * 60)
fails = [r for r in RESULTS if r[0] is False]
skips = [r for r in RESULTS if r[0] is None]
print("PASS %d / FAIL %d / SKIP %d" % (len([r for r in RESULTS if r[0] is True]), len(fails), len(skips)))
if fails:
    print("\n실패 항목:")
    for _, name, extra in fails:
        print("  - %s\n      %s" % (name, extra))
sys.exit(1 if fails else 0)
