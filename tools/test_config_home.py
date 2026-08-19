#!/usr/bin/env python
"""개인 설정 폴더 분리(`~/.CCpy` -> `~/.CCpy_test`) 검증 테스트.

repo 루트에서 실행한다:

    python3 tools/test_config_home.py

임시 HOME 안에서만 동작하므로 실제 홈의 `~/.CCpy` / `~/.CCpy_test` 는 건드리지
않는다. pymatgen 이 없는 환경에서는 VASPio 관련 항목만 SKIP 되고 나머지는 그대로
돈다.

확인 항목
  1. CCpyConfig 경로 함수 (기본값, $CCpy_HOME, 하위 경로)
  2. queue_config.yaml 생성 규칙 (템플릿 / 기존 ~/.CCpy 승계 / 이미 있으면 유지)
  3. 프로덕션 `~/.CCpy` 를 수정하지 않는지 (내용·mtime·파일 목록)
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
check("승계 원본 = ~/.CCpy", cfg.legacy_config_home() == home / ".CCpy")
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

# --------------------------------------------------- 2~3. queue_config 생성 규칙
print("\n2. queue_config.yaml 생성 규칙")
template = cfg.package_queue_config_template()

# (a) 승계 원본이 없을 때 -> 패키지 템플릿
home = fresh_home()
made = cfg.ensure_queue_config()
check("(a) 템플릿에서 새로 생성", made.is_file() and made.read_bytes() == template.read_bytes(), made)
check("(a) 프로덕션 폴더를 만들지 않는다", not (home / ".CCpy").exists())

# (b) 이미 있으면 손대지 않는다
made.write_text("qsub: sbatch\n# edited by user\n", encoding="utf-8")
before = made.stat().st_mtime_ns
again = cfg.ensure_queue_config()
check("(b) 이미 있으면 덮어쓰지 않는다",
      again == made and made.read_text(encoding="utf-8").endswith("# edited by user\n")
      and made.stat().st_mtime_ns == before)

# (c) 기존 ~/.CCpy 가 있으면 승계
home = fresh_home()
legacy_dir = home / ".CCpy"
legacy_dir.mkdir()
legacy = legacy_dir / "queue_config.yaml"
legacy_text = (template.read_text(encoding="utf-8")
               + "\nlammps_path: /home/pomepaw/lammps/build/lmp\n"
               + "lammps_mpi_run: srun --mpi=pmi2\n")
legacy.write_text(legacy_text, encoding="utf-8")
(legacy_dir / "vasp").mkdir()
(legacy_dir / "vasp" / "default.yaml").write_text("INCAR:\n  ENCUT: 500\n", encoding="utf-8")
before_legacy = snapshot(legacy_dir)

made = cfg.ensure_queue_config()
check("(c) 기존 ~/.CCpy/queue_config.yaml 을 승계",
      made.read_text(encoding="utf-8") == legacy_text, made)
check("(c) 개인 설정(lammps_path)이 살아있다",
      "lammps_path: /home/pomepaw/lammps/build/lmp" in made.read_text(encoding="utf-8"))
check("(c) 승계 원본을 수정하지 않는다", snapshot(legacy_dir) == before_legacy)
check("(c) 승계 후에도 ~/.CCpy 에 새 파일이 생기지 않는다",
      sorted(p.name for p in legacy_dir.iterdir()) == ["queue_config.yaml", "vasp"])

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
check("JobSubmit 이 승계한다",
      (home / ".CCpy_test" / "queue_config.yaml").read_text(encoding="utf-8") == legacy_text)
check("승계 안내가 출력된다", "승계" in proc.stdout, proc.stdout[-300:])

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
    code = (
        "import sys, os, json; sys.path.insert(0, %r)\n"
        "from CCpy.VASP.VASPio import VASPInput\n"
        "vi = VASPInput()\n"
        "print(json.dumps({'dir': vi.vasp_config_dir, 'yaml': vi.yaml_file,\n"
        "                  'encut_keys': len(vi.encut_table)}))\n" % str(REPO)
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
