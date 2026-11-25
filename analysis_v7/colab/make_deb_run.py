"""저장소 노트북 -> Colab용 'DeBERTa만' 실행 사본 (이어 하기 가능).
n-gram 학습과 LoRA는 건너뛰고(3분기 블렌드는 서버에서 저장된 OOF로 계산), Stage 1은 주기적으로 체크포인트를,
Stage 2는 fold마다 OOF를 저장합니다. 사용: make_deb_run.py <repo nb> <out nb> '<NTL_CFG 추가 JSON>'"""
import json, sys
src, dst, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
nb = json.load(open(src))

def rep(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:60])
    return s.replace(old, new)

cells = []
for c in nb["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] != "code":
        cells.append(c); continue
    if "drive.mount" in s:
        continue
    if s.startswith("# Colab 전용 준비"):
        s = rep(s, '    snapshot_download("Qwen/Qwen3-4B-Base", local_dir="/content/models/Qwen3-4B-Base")\n', "")
        s = rep(s, 'snapshot_download("microsoft/deberta-v3-large", local_dir="/content/models/deberta-v3-large")',
                'snapshot_download("microsoft/deberta-v3-large", local_dir="/content/models/deberta-v3-large",\n'
                '                      allow_patterns=["config.json", "spm.model", "tokenizer_config.json", "pytorch_model.bin"])')
        s = rep(s, '        NTL_BASE_MODEL="/content/models/Qwen3-4B-Base",\n', "")
        cfg = {"use_lora": False, **extra}
        s = rep(s, """NTL_CFG='{"batch_size": 4, "grad_accum": 2, "ngram_train_weights": [12.0]}'""", "NTL_CFG=" + repr(json.dumps(cfg)))
    elif s.startswith("cv_folds = list("):
        s = s.split("def quick_cv")[0] + '''
# 실행 사본 전용: fold는 Kaggle 커널의 분할(oof_v4.npz의 fold)을 그대로 씁니다. scikit-learn 버전이 달라도 같은 분할이 됩니다.
_z = np.load(os.environ.get("NTL_FOLD_REF", "/content/run/oof_v4.npz"), allow_pickle=True)
assert (_z["y"] == y).all() and (_z["prompt_id"] == groups).all()
_f = np.zeros(len(train), dtype=int)
for _k, (_, _b) in enumerate(cv_folds):
    _f[_b] = _k
print("StratifiedGroupKFold가 Kaggle 커널 분할과 같은가:", bool((_f == _z["fold"]).all()), "| scikit-learn", __import__("sklearn").__version__)
cv_folds = [(np.where(_z["fold"] != k)[0], np.where(_z["fold"] == k)[0]) for k in range(CFG["n_folds"])]
'''
    elif s.startswith("FUNCTION_WORDS = set("):
        s = s.split("class MultiViewVectorizer")[0]          # sample_pool까지만 (n-gram 학습은 서버에서 따로)
        assert "def sample_pool" in s
    elif s.startswith("plt.rcParams.update"):
        continue
    elif s.startswith("DEB_STAGE1_FILE"):
        # --- Stage 1 체크포인트/이어 하기 ---
        s = rep(s, "def deb_train_stage(model, tok, device, texts, labels, epochs, lr, budget_h, name):",
                   "def deb_train_stage(model, tok, device, texts, labels, epochs, lr, budget_h, name, ckpt_every_s=None):")
        s = rep(s, '''    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / n_steps))))
    labels = torch.tensor(np.asarray(labels), dtype=torch.long)
    start, step, skipped, running = time.time(), 0, 0, 0.0
''', '''    labels = torch.tensor(np.asarray(labels), dtype=torch.long)
    start, step, skipped, running = time.time(), 0, 0, 0.0
    ck_path, resume_bi, resume_step, last_ck = WORK_DIR / f"{name}_ckpt.pt", 0, None, time.time()
    if ckpt_every_s and ck_path.exists():
        assert epochs == 1
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state"])
        resume_bi, step, running, skipped = ck["bi"], ck["step"], ck["running"], ck["skipped"]
        resume_step = step
        print(f"[{name}] 체크포인트에서 이어 하기: {step}/{n_steps} 스텝 (마이크로 배치 {resume_bi}부터), loss {running:.3f}")

    def lr_at(s):
        f = min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / n_steps)))
        if resume_step is not None:  # 옵티마이저 상태는 저장하지 않으므로 이어 할 때 30스텝 동안 lr을 다시 올립니다.
            f *= min(1, (s - resume_step + 1) / 30)
        return f

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_at)
    if resume_step is not None:
        sched.last_epoch = step
        for g, base in zip(optim.param_groups, sched.base_lrs):
            g["lr"] = base * lr_at(step)
''')
        s = rep(s, '''        for bi, batch in enumerate(batches):
            if time.time() - start > budget_h * 3600:
                print(f"[{name}] 시간 예산 도달, {step}/{n_steps} 스텝에서 중단")
                model.eval()
                return
            ids, mask = deb_collate(''', '''        for bi, batch in enumerate(batches):
            if bi < resume_bi:
                continue
            if time.time() - start > budget_h * 3600:
                print(f"[{name}] 시간 예산 도달, {step}/{n_steps} 스텝에서 중단")
                model.eval()
                return
            ids, mask = deb_collate(''')
        s = rep(s, '''            if step % max(1, n_steps // 10) == 0:
                print(f"[{name}] {step}/{n_steps} loss {running:.3f} | {(time.time() - start) / 60:.1f}분")
    print(f"[{name}] 완료: {step}스텝, 비정상 손실로 건너뜀 {skipped}회, {(time.time() - start) / 60:.1f}분")
    model.eval()


@torch.no_grad()
def deb_predict(''', '''            if step % max(1, n_steps // 20) == 0:
                print(f"[{name}] {step}/{n_steps} loss {running:.3f} | {(time.time() - start) / 60:.1f}분 | skip {skipped}", flush=True)
            if ckpt_every_s and time.time() - last_ck > ckpt_every_s and bi + 1 < len(batches):
                torch.save({"state": deb_state(model), "bi": bi + 1, "step": step, "running": float(running),
                            "skipped": skipped}, str(ck_path) + ".tmp")
                os.replace(str(ck_path) + ".tmp", ck_path)
                last_ck = time.time()
                print(f"[{name}] 체크포인트 저장: {step}/{n_steps} 스텝, loss {running:.3f} | {(time.time() - start) / 60:.1f}분", flush=True)
    print(f"[{name}] 완료: {step}스텝, 비정상 손실로 건너뜀 {skipped}회, {(time.time() - start) / 60:.1f}분")
    model.eval()


@torch.no_grad()
def deb_predict(''')
        # --- Stage 2: fold마다 OOF 저장, 끝난 fold는 건너뜀 ---
        s = rep(s, '''    oof, tests = np.zeros((len(train), K), dtype=np.float32), []
    for k, (a, b) in enumerate(cv_folds):
        model.load_state_dict(stage1)
        deb_train_stage(model, tok, device, train.text.values[a].tolist(), y[a], CFG["deb_stage2_epochs"],
                        CFG["deb_stage2_lr"], CFG["deb_stage2_budget_h"], f"deb-stage2-fold{k}")
        if CFG["deb_save_folds"]:
            save_deb(model, DEB_FOLD_FILES[k])
        oof[b] = deb_predict(model, tok, device, train.text.values[b])
        if CFG["deb_test"] == "folds":
            tests.append(deb_predict(model, tok, device, test.text.values))
        print(f"fold {k}: DeBERTa BA {balanced_accuracy_score(y[b], oof[b].argmax(1)):.4f} | {elapsed()}")
''', '''    assert CFG["deb_test"] == "full" and not CFG["deb_save_folds"]
    oof, tests, done = np.zeros((len(train), K), dtype=np.float32), [], []
    part_path = WORK_DIR / "ntl_deb_oof_partial.npz"
    if part_path.exists():
        _p = np.load(part_path)
        oof, done = _p["oof"].astype(np.float32), [int(v) for v in _p["done"]]
        print("이어 하기: 이미 끝난 fold", done)
    for k, (a, b) in enumerate(cv_folds):
        if k in done:
            print(f"fold {k}: (저장된 값) DeBERTa BA {balanced_accuracy_score(y[b], oof[b].argmax(1)):.4f}")
            continue
        model.load_state_dict(stage1)
        deb_train_stage(model, tok, device, train.text.values[a].tolist(), y[a], CFG["deb_stage2_epochs"],
                        CFG["deb_stage2_lr"], CFG["deb_stage2_budget_h"], f"deb-stage2-fold{k}")
        oof[b] = deb_predict(model, tok, device, train.text.values[b])
        done.append(k)
        np.savez(WORK_DIR / "ntl_deb_oof_partial.tmp.npz", oof=oof, done=np.array(done))
        os.replace(WORK_DIR / "ntl_deb_oof_partial.tmp.npz", part_path)
        print(f"fold {k}: DeBERTa BA {balanced_accuracy_score(y[b], oof[b].argmax(1)):.4f} | {elapsed()}", flush=True)
''')
    elif s.startswith("deb_oof, deb_test = None, None"):
        s = rep(s, '''CFG["deb_stage1_epochs"], CFG["deb_stage1_lr"], CFG["deb_stage1_budget_h"], "deb-stage1")
                save_deb(model, DEB_STAGE1_FILE)
''', '''CFG["deb_stage1_epochs"], CFG["deb_stage1_lr"], CFG["deb_stage1_budget_h"], "deb-stage1",
                                ckpt_every_s=float(os.environ.get("NTL_CKPT_S", "600")))
                save_deb(model, DEB_STAGE1_FILE)
                load_deb(model, WORK_DIR / DEB_STAGE1_FILE)  # 이어 하기와 같게: 모든 fold가 저장된(fp16) Stage 1 가중치에서 시작
''')
        s += '\nprint("== DEB RUN DONE ==", elapsed())\n'
        c["source"] = s.splitlines(keepends=True); cells.append(c)
        break
    c["source"] = s.splitlines(keepends=True)
    cells.append(c)
nb["cells"] = cells
import ast
for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
json.dump(nb, open(dst, "w"), ensure_ascii=False, indent=1)
print(dst, len(cells), "cells", extra)
