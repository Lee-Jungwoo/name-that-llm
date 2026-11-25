# VM 안에서 nohup으로 도는 보조 프로세스: 상태 기록(gpu.log)과, 큰 결과 파일을 40MB 조각 스냅샷으로 /content/dl에 만들어 둡니다.
import glob, os, shutil, subprocess, time
OUT, DL = "/content/out", "/content/dl"
os.makedirs(DL, exist_ok=True)
def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
def snap(name):
    p = f"{OUT}/{name}"
    if not os.path.exists(p):
        return
    m = int(os.path.getmtime(p))
    d = f"{DL}/{name}.{m}"
    if time.time() - m < 10 or os.path.exists(d + "/DONE"):
        return
    for old in glob.glob(f"{DL}/{name}.*"):
        if old != d:
            shutil.rmtree(old, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    sh(f"cp {p} {d}/_copy && split -b 40M -d {d}/_copy {d}/part. && md5sum {d}/_copy | cut -d' ' -f1 > {d}/md5 && rm {d}/_copy && ls {d} | grep -c '^part' > {d}/nparts && touch {d}/DONE")
t0 = time.time()
while time.time() - t0 < 10 * 3600 and not os.path.exists("/content/run/STOP"):
    line = time.strftime("%H:%M:%S ") + sh("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader") + " | ram " + sh("free -m | awk 'NR==2{print $3\"/\"$2\"MB avail \"$7}'") + " | papermill " + (sh("pgrep -f '[p]apermill' | head -1") or "none")
    with open(f"{OUT}/gpu.log", "a") as f:
        f.write(line + "\n")
    for name in ("deb-stage1_ckpt.pt", "ntl_deb_stage1.pt", "ntl_deb_full.pt"):
        try:
            snap(name)
        except Exception as e:
            print("snap error", name, e, flush=True)
    time.sleep(20)
