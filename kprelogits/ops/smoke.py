"""Rent one box, extract one backbone, validate it, destroy the box.

The GPU path in this package has never run. Every part of it is tested offline
-- the bundle contract, the shard state machine, preflight, teardown -- and
none of that says a timm forward pass produces a loadable bundle on rented
hardware. This rents the cheapest adequate box, extracts the smallest backbone
of the pilot 50, and tears down, for well under a dollar.

It is also the first driver that ties ``ops.vast`` together. Those functions
were written and tested individually and then never composed, which left
renting a manual step -- and a manual rental is how this project paid for a
4.3-hour orphan. Treat this as the seed of the real multi-shard driver, not as
the finished thing.

Three things here are not arbitrary:

**A per-attempt S3 prefix.** Bundle keys are ``{dataset}_{model}_features.npz``
-- no ``max_train``, no precision. The production prefix already holds this
model's bundle, so writing there would overwrite real data; and the worker's
resume oracle is S3, so a *reused* smoke prefix makes the second attempt skip
the model and report success having done nothing. Every attempt gets virgin
ground.

**SSH mode, not the image entrypoint.** The worker image starts extracting at
boot, which would race credential staging: the job reaches S3 before its keys
do. Renting with ``ssh=True`` leaves sshd holding the box open so credentials
land first and the job starts deliberately.

**S3 state files are the only liveness signal.** Never ``pgrep`` over SSH: the
remote shell matches its own pattern, so the poll reports the job alive
forever, or reports a kill that never happened. That mistake has cost this
project three separate incidents.

Usage::

    python -m kprelogits.ops.smoke            # plan: prints, rents nothing
    python -m kprelogits.ops.smoke --go       # rent, extract, validate, destroy
    python -m kprelogits.ops.smoke --abort    # destroy anything labelled smoke-*
    python -m kprelogits.ops.smoke --compare BUNDLE  # offline validation only
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import s3, state, vast

# The smallest of the frozen 50 (legacy/artifacts/pilot_backbones.json), and
# not in the local feature_cache -- so this is genuinely new work, not a
# re-derivation of something already on disk.
MODEL = "mobilenetv2_050.lamb_in1k"
MODEL_PARAMS = 687_680
DATA_FLAG = "octmnist"

BUCKET = "pandora-linear-probe-inputs-939723541836-us-east-1-an"
PROD_PREFIX = "medmnist_prelogits"
SMOKE_PREFIX = "medmnist_prelogits_smoke"
DATA_KEY = f"s3://{BUCKET}/medmnist/{DATA_FLAG}_224.npz"

# Stock image: there is no published worker image, and no Docker locally to
# build one. The legacy pipeline shipped code to a stock image the same way.
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"

# cpu_cores_effective, never cpu_cores: the advertised count is routinely 4-8x
# the slice actually granted, and nproc inside the container reports the host's
# count, so nothing on the box can detect the difference.
#
# cuda_max_good is the host driver's CUDA ceiling. The image is cu124, so a
# host below that cannot start the container at all -- a failure that costs a
# whole rental cycle to discover and looks like a mystery from here.
# inet_down matters more than it appears: this job pulls 3.9 GB before it
# computes anything, and that transfer is on the clock.
#
# compute_cap>=750 (Turing and later) is about representativeness, not cost:
# the cheapest offers are Pascal cards, which have neither tensor cores nor
# TF32. A smoke test that passes on Pascal leaves untested exactly the
# numerical path a real run takes -- fp16 autocast on tensor cores, and cudnn
# picking TF32 kernels for convolutions. Ampere-class boxes are within a
# rounding error of the Pascal price anyway.
#
# compute_cap<=900 is the other end of the same question: the image's torch
# 2.5.1/cu124 carries kernels up to sm_90 and nothing beyond. A Blackwell card
# (RTX 5060 Ti, compute capability 12.0) boots, stages and launches normally,
# then fails every model with "no kernel image is available for execution on
# the device" -- and those cards are the cheapest 16 GB offers, so cheapest-
# first ordering walks straight into them (2026-09-14). The remote preflight
# refuses such a box too (config._check_cuda_arch); this keeps them from being
# rented in the first place. Raise the cap only together with the image.
OFFER_QUERY = ("reliability>0.98 num_gpus=1 gpu_ram>=8 dph<0.40 "
               "inet_down>=200 disk_space>=50 rentable=true "
               "cpu_cores_effective>=8 cuda_max_good>=12.4 "
               "compute_cap>=750 compute_cap<=900")
DISK_GB = 40          # legacy probe default; ops' 100 is 2.5x what this needs

# MAX_TRAIN matches the production bundles' identity, so the result is
# comparable 1:1 against the bundle already in S3 for this model. A smaller
# value would be faster and prove less.
MAX_TRAIN = 10_000

REMOTE_PATH = "export PATH=/opt/conda/bin:$PATH"
# Each ssh call gets a fresh shell, so anything touching S3 has to load the
# staged credentials itself. `set -a` exports what the file defines so awscli
# and huggingface_hub see it without kprelogits reading the file.
SOURCE_SECRETS = "set -a; . /workspace/.env-secrets; set +a;"
# timm is pinned because no bundle records the timm version, and timm decides
# what an architecture's pre-logits ARE: the InceptionNeXt zero-width head is a
# 1.0.x constructor bug, and build_encoder's workaround was verified against
# 1.0.29 specifically. The DermaMNIST sweep of 2026-09-13 installed 1.0.29 (the
# latest release then and since); an unpinned install would let two sweeps of
# the same backbone differ with nothing in either manifest to say so. Keep the
# Dockerfile's pin identical -- test_run checks.
PIP_PACKAGES = "timm==1.0.29 medmnist huggingface_hub awscli"

# Deadlines. Every one of these is a window where money burns with nothing to
# show, so each has an abort rather than an open-ended wait.
# provisioning -> actual_status running. 20 min, not 10: "loading" is mostly the
# host pulling the multi-GB image, and on an uncached host that routinely runs
# past 10 min. At 10, 10 of 13 boxes lost on 2026-09-13 died here, and the one
# whose status was logged was mid-pull ("Verifying Checksum ... Download
# complete") -- killed moments from ready, only to start the same pull on
# another uncached host. A genuinely wedged box now costs 10 more minutes; a
# slow one no longer costs a whole retry.
RUNNING_DEADLINE = 20 * 60
SSH_DEADLINE = 5 * 60          # running -> sshd answering
SETUP_TIMEOUT = 8 * 60         # pip install
DATA_TIMEOUT = 15 * 60         # 3.9 GB pull from S3
FIRST_STATE_DEADLINE = 3 * 60  # launch -> state_shard0.json appears
STALE_DEADLINE = 10 * 60       # no state update while running
JOB_DEADLINE = 45 * 60         # total, from launch
POLL_SECONDS = 30


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def smoke_prefix(rid: str) -> str:
    """Virgin ground per attempt -- see the module docstring."""
    return f"s3://{BUCKET}/{SMOKE_PREFIX}/run-{rid}/{DATA_FLAG}"


def prod_bundle_uri() -> str:
    return (f"s3://{BUCKET}/{PROD_PREFIX}/{DATA_FLAG}/"
            f"{DATA_FLAG}_{MODEL}_features.npz")


def worker_env(rid: str) -> Dict[str, str]:
    """Non-secret config for the worker. Credentials are staged, never here."""
    return {
        "S3_BUCKET": BUCKET,
        "S3_FEATURES_PREFIX": f"{SMOKE_PREFIX}/run-{rid}",
        "DATA_FLAG": DATA_FLAG,
        "MAX_TRAIN": str(MAX_TRAIN),
        "DATA_DIR": "/workspace/data",
        "RESULTS_DIR": "/workspace/results",
        "PARALLEL_RANK": "0",
        "PARALLEL_SHARDS": "1",
    }


def selection_doc() -> dict:
    return {"schema": 1, "models": [{"name": MODEL, "params": MODEL_PARAMS}]}


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def _sh(cmd: List[str], *, timeout: int, label: str) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)[:160]}", flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _ssh(target: List[str], remote_cmd: str, *, timeout: int,
         label: str, check: bool = True, stdin_null: bool = False) -> str:
    # -n detaches ssh's own stdin. Without it ssh holds the channel open for a
    # backgrounded remote process, which looks exactly like a hung job.
    flags = ["-n"] if stdin_null else []
    r = _sh(["ssh", *flags, *target, remote_cmd], timeout=timeout, label=label)
    if check and r.returncode != 0:
        raise RuntimeError(f"{label} failed rc={r.returncode}: "
                           f"{(r.stderr or r.stdout).strip()[:2000]}")
    return r.stdout


def code_tarball(dest: Path) -> Path:
    """Ship the package as source. No image to build, nothing to keep in sync."""
    repo = Path(__file__).resolve().parents[2]
    with tarfile.open(dest, "w:gz") as tf:
        tf.add(repo / "kprelogits", arcname="kprelogits",
               filter=lambda ti: None if "__pycache__" in ti.name
               or ti.name.endswith(".pyc") else ti)
    return dest


def aws_credentials() -> Dict[str, str]:
    """Read local AWS creds for staging. Never logged, never in argv."""
    out = {}
    for env_key, cfg_key in (("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
                             ("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
                             ("AWS_DEFAULT_REGION", "region")):
        val = os.environ.get(env_key)
        if not val:
            r = subprocess.run(["aws", "configure", "get", cfg_key],
                               capture_output=True, text=True, timeout=30)
            val = r.stdout.strip()
        if val:
            out[env_key] = val
    missing = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"} - set(out)
    if missing:
        raise RuntimeError(f"no AWS credentials found for {sorted(missing)}")
    return out


# ---------------------------------------------------------------------------
# Validation (local; the box is already gone by the time this runs)
# ---------------------------------------------------------------------------

def compare_bundles(smoke: Path, reference: Optional[Path]) -> bool:
    """Validate the new bundle, and check it against the production one.

    Tiered on purpose. Labels and shapes must be *exact*: they prove the
    seed-42 train subsample and the whole data path are byte-equivalent, and
    that check is free. Features are only checked loosely, because both
    extractions run fp16 autocast on CUDA and cudnn convolutions default to
    TF32 on Ampere and later -- differences around 1e-2 relative are what
    correct code produces on two different GPUs. A cosine similarity below
    ~0.99, though, is not GPU noise: it means preprocessing, weights, or
    normalization diverged, which is exactly what this is here to catch.
    """
    import numpy as np
    from ..compare_subset import feature_agreement
    from ..contracts import validate_prelogit_bundle

    ok = True
    with np.load(smoke, allow_pickle=False) as z:
        man = validate_prelogit_bundle(z)
        print(f"  contract OK: {man.model_name} feat_dim={man.feat_dim} "
              f"weights={man.weights_revision}")
        print(f"    producer={man.producer_version} max_train={man.max_train}")
        new = {k: z[k] for k in z.files if k.startswith(("x_", "y_"))}

    for split in ("train", "val", "test"):
        x, y = new[f"x_{split}"], new[f"y_{split}"]
        print(f"    x_{split}={x.shape}/{x.dtype} y_{split}={y.shape}/{y.dtype}")
        # The contract validates names, versions and shapes -- not values. A
        # non-finite feature is the failure mode that costs the most to find
        # late: it survives upload, passes validation, and only shows up as a
        # fit that will not converge, long after the GPU is gone.
        if not np.isfinite(x).all():
            bad = int((~np.isfinite(x)).sum())
            print(f"  NON-FINITE: x_{split} has {bad} NaN/inf value(s)")
            ok = False
        if y.min() < 0 or y.max() >= man.num_classes:
            print(f"  LABELS OUT OF RANGE: x_{split} labels span "
                  f"[{y.min()}, {y.max()}] with num_classes={man.num_classes}")
            ok = False

    if reference is None:
        print("  (no reference bundle; skipping comparison)")
        return ok

    with np.load(reference, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files if k.startswith(("x_", "y_"))}

    for split in ("train", "val", "test"):
        xk, yk = f"x_{split}", f"y_{split}"
        if new[xk].shape != old[xk].shape:
            print(f"  MISMATCH {xk}: {new[xk].shape} vs {old[xk].shape}")
            ok = False
            continue
        if not np.array_equal(new[yk], old[yk]):
            print(f"  MISMATCH {yk}: labels differ -- the subsample or data "
                  f"path changed")
            ok = False
        cmin, cmean, rel, mx, good = feature_agreement(new[xk], old[xk])
        ok &= good
        print(f"  {xk}: cos min={cmin:.6f} mean={cmean:.6f}  "
              f"rel_fro={rel:.2e}  max|d|={mx:.2e}  {'OK' if good else 'FAIL'}")

    return ok


# ---------------------------------------------------------------------------
# Plan mode
# ---------------------------------------------------------------------------

def plan(rid: str) -> int:
    prefix = smoke_prefix(rid)
    print(f"SMOKE PLAN  run-{rid}")
    print(f"  model      {MODEL} ({MODEL_PARAMS:,} params)")
    print(f"  image      {IMAGE}")
    print(f"  disk       {DISK_GB} GB")
    print(f"  s3 prefix  {prefix}")
    print(f"  reference  {prod_bundle_uri()}")
    print(f"\n  worker env:")
    for k, v in sorted(worker_env(rid).items()):
        print(f"    {k}={v}")

    print(f"\n  offer query: {OFFER_QUERY}")
    try:
        offers = vast.search_offers(OFFER_QUERY, limit=5)
        print(f"  {len(offers)} offer(s):")
        for o in offers:
            print(f"    id={o.get('id')} ${float(o.get('dph_total') or 0):.3f}/hr "
                  f"{o.get('gpu_name')} cc={o.get('compute_cap')} "
                  f"cpu_eff={float(o.get('cpu_cores_effective') or 0):.1f} "
                  f"down={o.get('inet_down')}Mbps {o.get('geolocation')}")
    except Exception as e:
        print(f"  could not search offers: {type(e).__name__}: {e}")
        offers = []

    print("\n  would rent:")
    vast.create(int(offers[0]["id"]) if offers else 0, image=IMAGE, disk=DISK_GB,
                label=f"smoke-{rid}", ssh=True, env=worker_env(rid), dry_run=True)

    print("\n  then, over ssh:")
    for line in remote_script(rid).splitlines():
        print(f"    {line}")
    print("\n  rents nothing. Re-run with --go to execute.")
    return 0


def remote_script(rid: str) -> str:
    """The remote commands, as one printable block (also the runbook).

    Every command that reaches S3 sources the staged secrets first. Each ssh
    call is its own shell, so nothing carries over between them -- and an
    unsourced `aws` fails with "Unable to locate credentials", which is
    trivially fixable and expensive to discover on a rented box.
    """
    env = " ".join(f"{k}={v}" for k, v in sorted(worker_env(rid).items()))
    return "\n".join([
        f"{REMOTE_PATH}; pip install -q {PIP_PACKAGES}",
        f"{REMOTE_PATH}; {SOURCE_SECRETS} "
        f"aws s3 cp {DATA_KEY} /workspace/data/{DATA_FLAG}_224.npz",
        f"{REMOTE_PATH}; {SOURCE_SECRETS} "
        f"cd /workspace && {env} python -m kprelogits.extract "
        f"--selection /workspace/selection.json --dry-run",
        # setsid + nohup + all three descriptors redirected: the job must
        # outlive this ssh session, and ssh must not sit holding the channel
        # open waiting on an inherited descriptor. The trailing echo is how we
        # know the shell got that far -- it is not a claim the job succeeded.
        f"{REMOTE_PATH}; {SOURCE_SECRETS} "
        f"cd /workspace && {env} setsid nohup python -m kprelogits.extract "
        f"--selection /workspace/selection.json "
        f"> /workspace/extract.log 2>&1 < /dev/null & "
        f"sleep 2; echo launched",
    ])


# ---------------------------------------------------------------------------
# Abort / recovery
# ---------------------------------------------------------------------------

def abort() -> int:
    """Destroy anything this driver rented. The first post-mortem command."""
    rec = vast.reconcile()
    print(rec.render())
    live = vast.show_instances()
    mine = [i for i in live if str(i.get("label") or "").startswith("smoke-")]
    if not mine:
        print("nothing labelled smoke-* is running")
        return 0
    print(f"destroying {len(mine)}: {[i.get('id') for i in mine]}")
    for i in mine:
        vast.destroy(int(i["id"]))
    remaining = vast.show_instances()
    print(f"after: ${vast.burn_rate(remaining):.3f}/hr")
    print(vast.describe(remaining))
    return 0 if not remaining else 1


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class Stopped(KeyboardInterrupt):
    """Raised in a renting thread when the driver has decided to stop.

    Only the main thread receives Ctrl-C and SIGTERM, so a thread renting in
    parallel learns of it through an Event, checked wherever it would
    otherwise sit for minutes. It subclasses KeyboardInterrupt so every
    handler already written for an interrupt treats it as one: ``acquire``
    destroys the half-provisioned box and re-raises, and ``bring_up`` does
    not mistake it for a staging failure worth a replacement box.
    """


def check_stop(stop) -> None:
    if stop is not None and stop.is_set():
        raise Stopped()


def _pause(seconds: float, stop) -> None:
    """Sleep, but wake at once -- and raise -- if ``stop`` is set."""
    if stop is None:
        time.sleep(seconds)
    elif stop.wait(seconds):
        raise Stopped()


def wait_for_running(instance_id: int, *, stop=None, tag: str = "") -> dict:
    """Poll until the box is running; on timeout, say what vast last reported.

    ``actual_status`` alone ("loading") cannot tell a slow image pull from a
    wedged host. vast's ``status_msg`` usually can, and 10 of 13 boxes lost on
    2026-09-13 died here with nothing recorded but the deadline -- so the
    message is printed whenever it changes and carried into the error.
    """
    deadline = time.time() + RUNNING_DEADLINE
    last_msg = ""
    while time.time() < deadline:
        check_stop(stop)
        inst = next((i for i in vast.show_instances()
                     if int(i.get("id", -1)) == instance_id), None)
        status = (inst or {}).get("actual_status")
        msg = " ".join(str((inst or {}).get("status_msg") or "").split())[:200]
        print(f"  {tag}status={status}" + (f"  [{msg}]" if msg and msg != last_msg else ""),
              flush=True)
        last_msg = msg or last_msg
        if status == "running":
            return inst
        _pause(15, stop)
    raise RuntimeError(f"instance {instance_id} never reached running in "
                       f"{RUNNING_DEADLINE // 60} min; vast last said: "
                       f"{last_msg or '(nothing)'}")


def wait_for_ssh(routes: List[tuple], *, user: str = "root", on_retry=None,
                 known_hosts: Optional[Path] = None, stop=None, tag: str = ""):
    """Return ssh args for the first route that answers, trying each in turn.

    Both routes are tried every round rather than committing to one, because
    they fail in ways that are indistinguishable from here until tried: the
    direct route hangs until timeout when the host is firewalled, and the
    proxy refuses the login while a freshly attached key propagates. Waiting
    out one route's deadline before discovering the other works is the
    expensive mistake.

    ``on_retry`` is re-invoked partway through to re-attach the key, since a
    lost attach otherwise looks exactly like a box that will never answer.
    """
    deadline = time.time() + SSH_DEADLINE
    last = ""
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        for label, host, port in routes:
            check_stop(stop)
            target = vast.ssh_args(host, port, user, known_hosts=known_hosts)
            r = subprocess.run(["ssh", *target, "true"], capture_output=True,
                               text=True, timeout=40)
            if r.returncode == 0:
                print(f"  {tag}ssh up via {label} ({host}:{port})")
                return target
            last = f"{label}: " + (r.stderr or "").strip().replace("\n", " ")[:100]
            print(f"  {tag}ssh not ready ({attempt}) {last}", flush=True)
        if on_retry is not None and attempt == 2:
            print(f"  {tag}re-attaching ssh key", flush=True)
            try:
                on_retry()
            except Exception as e:
                print(f"  {tag}re-attach failed: {type(e).__name__}: {e}")
        _pause(10, stop)
    raise RuntimeError(f"ssh never came up on any route; last: {last}")


def poll_to_completion(prefix: str) -> Tuple[bool, str]:
    """Watch S3 state files. Returns (ok, reason).

    Reads the remote log only when something has already gone wrong -- reading
    a file is safe, but polling it routinely invites inferring liveness from
    output, which is the habit that leads back to pgrep.
    """
    launched = time.time()
    last_seen = None
    last_change = launched

    while True:
        elapsed = time.time() - launched
        if elapsed > JOB_DEADLINE:
            return False, f"job deadline ({JOB_DEADLINE // 60} min) exceeded"

        states = state.read_states(prefix)
        st = states.get(0)

        if st is None:
            if elapsed > FIRST_STATE_DEADLINE:
                return False, "no state file published; the worker never started"
            print(f"  [{elapsed:5.0f}s] waiting for first state file", flush=True)
        else:
            fingerprint = (st.phase, st.extracted, st.uploaded, st.failed,
                           st.updated_at, st.current)
            if fingerprint != last_seen:
                last_seen, last_change = fingerprint, time.time()
                print(f"  [{elapsed:5.0f}s] {state.summarize(states)}", flush=True)
            if st.phase == "done":
                return True, "done"
            if st.phase == "failed":
                err = st.errors[-1] if st.errors else "(no error recorded)"
                return False, f"worker reported failed: {err}"
            if time.time() - last_change > STALE_DEADLINE:
                return False, (f"no state change for "
                               f"{STALE_DEADLINE // 60} min while {st.phase}")

        time.sleep(POLL_SECONDS)


def acquire_box(offers: List[dict], pubkey: str, rid: str,
                known_hosts: Optional[Path] = None) -> Tuple[int, List[str]]:
    """Rent boxes until one boots AND answers ssh; destroy the ones that don't.

    Individual boxes fail in ways that are nobody's bug: the image never
    finishes pulling, sshd never accepts the key, the host is firewalled. The
    cheap answer is to stop paying for that box and take the next offer --
    a bad box should cost one short rental, not the run.

    A box that fails here is destroyed immediately rather than left to the
    outer teardown, because we are about to rent another one and two live
    instances is how the burn rate doubles unnoticed.
    """
    last_error = None
    label = f"smoke-{rid}"
    for n, offer in enumerate(offers, 1):
        print(f"  offer {n}/{len(offers)}: {offer.get('id')} "
              f"${float(offer.get('dph_total') or 0):.3f}/hr "
              f"{offer.get('gpu_name')} cc={offer.get('compute_cap')} "
              f"cpu_eff={float(offer.get('cpu_cores_effective') or 0):.1f} "
              f"down={offer.get('inet_down')}Mbps {offer.get('geolocation')}")
        instance_id = None
        try:
            # create() is INSIDE the try for two reasons. Offers are live
            # marketplace inventory and one can vanish between the search and
            # the rental, which should cost us the next offer rather than the
            # run. And a create that fails locally does not prove nothing was
            # rented -- so we look for the label before moving on, or the lost
            # box bills unattended.
            try:
                instance_id = vast.create(int(offer["id"]), image=IMAGE,
                                          disk=DISK_GB, label=label, ssh=True,
                                          env=worker_env(rid))
            except Exception as ce:
                stray = vast.find_by_label(label)
                if stray:
                    instance_id = int(stray[0]["id"])
                    print(f"  create reported {type(ce).__name__} but "
                          f"instance {instance_id} exists under {label} -- "
                          f"adopting it", flush=True)
                else:
                    raise
            print(f"  instance {instance_id}")
            inst = wait_for_running(instance_id)
            print(f"  {inst.get('gpu_name')} cpu_eff="
                  f"{inst.get('cpu_cores_effective')}/{inst.get('cpu_cores')} "
                  f"${float(inst.get('dph_total') or 0):.3f}/hr")

            # Authorize our key on this box specifically. Team accounts cannot
            # hold account-wide keys, and this is the tighter grant anyway: it
            # covers exactly the instance we rented and dies with it.
            vast.attach_ssh_key(instance_id, pubkey)
            print("  ssh key attached")

            routes = vast.ssh_endpoints(instance_id)
            print(f"  routes: {[(l, f'{h}:{p}') for l, h, p in routes]}")
            target = wait_for_ssh(
                routes, known_hosts=known_hosts,
                on_retry=lambda: vast.attach_ssh_key(instance_id, pubkey))
            return instance_id, target
        # BaseException, deliberately. This block spans up to ten minutes of
        # provisioning plus ssh retries -- the likeliest moment for someone to
        # lose patience and hit Ctrl-C -- and it sits OUTSIDE the teardown
        # context, which is only entered once a box is usable. Catching only
        # Exception here would let a KeyboardInterrupt walk past a live
        # instance and leave it billing: precisely the orphan this module
        # exists to prevent.
        except BaseException as e:
            if instance_id is not None:
                print(f"  box {instance_id} unusable: {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)
                print("  destroying it before doing anything else", flush=True)
                try:
                    vast.destroy(instance_id)
                except Exception as de:
                    print(f"  DESTROY FAILED for {instance_id}: {de} -- "
                          f"run `smoke --abort`", flush=True)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            last_error = e
            print("  trying the next offer", flush=True)

    raise RuntimeError(f"no offer produced a usable box "
                       f"({len(offers)} tried); last: {last_error}")


def tail_remote_log(target: List[str]) -> None:
    try:
        out = _ssh(target, "tail -40 /workspace/extract.log", timeout=60,
                   label="log tail", check=False)
        print("---- extract.log (tail) ----")
        print(out)
        print("---------------------------")
    except Exception as e:
        print(f"  could not read remote log: {type(e).__name__}: {e}")


def rescue(user: str, host: str, port: int, out_dir: Path,
           known_hosts: Optional[Path] = None) -> None:
    """Pull anything worth keeping off a box that is about to be destroyed.

    The worker prints DO NOT DESTROY when its final sync fails, which is right
    for a production shard holding hours of work. This box holds one small
    bundle: rescuing it bounded and then destroying beats leaving an instance
    billing while someone decides.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for remote in ("/workspace/extract.log",
                   f"/workspace/results/{DATA_FLAG}_lp/extract_report_shard0.json",
                   f"/workspace/results/{DATA_FLAG}_lp/features/"):
        try:
            _sh(["scp", "-r",
                 *vast.scp_args(host, port, user, known_hosts=known_hosts),
                 f"{user}@{host}:{remote}", str(out_dir)],
                timeout=300, label=f"rescue {remote}")
        except Exception as e:
            print(f"  rescue of {remote} failed: {type(e).__name__}: {e}")


def go(rid: str, scratch: Path) -> int:
    prefix = smoke_prefix(rid)
    print(f"=== SMOKE RUN {rid} -> {prefix}\n")

    # -- Phase 1: everything that can fail for free ------------------------
    print("[1/4] pre-rental checks")
    rec = vast.reconcile()
    print(rec.render())
    if rec.orphans:
        print("ORPHANS are billing. Resolve them before renting more.")
        return 1

    ok, msg = s3.bucket_listable(f"s3://{BUCKET}")
    print(f"  bucket listable: {ok} {msg}")
    if not ok:
        return 1

    creds = aws_credentials()
    print(f"  local AWS creds: {sorted(creds)}")

    # Read the key BEFORE renting. A missing or malformed key means every box
    # refuses the login, and the run would burn its whole ssh deadline
    # discovering something that is free to check here -- which is exactly
    # what one earlier attempt paid for.
    if not vast.DEFAULT_SSH_KEY.exists():
        print(f"  no local private key at {vast.DEFAULT_SSH_KEY}")
        return 1
    pubkey = vast.public_key()
    print(f"  ssh key {vast.DEFAULT_SSH_KEY}.pub ({pubkey.split()[0]}), "
          f"attached per-instance after boot")

    tarball = code_tarball(scratch / "kprelogits.tgz")
    selection = scratch / "selection.json"
    selection.write_text(json.dumps(selection_doc(), indent=2))
    print(f"  code {tarball.stat().st_size // 1024} KB, selection {MODEL}")

    # Run the REAL preflight locally, against the exact prefix the box will
    # use. Preflight is the same code either way, so anything it rejects it
    # would reject on rented hardware -- after the boot, the pip install and
    # the 3.9 GB pull have all been paid for.
    local = scratch / f"preflight-{rid}"
    (local / "data").mkdir(parents=True, exist_ok=True)
    (local / "results").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **worker_env(rid),
           "DATA_DIR": str(local / "data"), "RESULTS_DIR": str(local / "results")}
    r = subprocess.run(
        [sys.executable, "-m", "kprelogits.extract",
         "--selection", str(selection), "--dry-run"],
        env=env, capture_output=True, text=True, timeout=180,
        cwd=str(Path(__file__).resolve().parents[2]))
    if r.returncode != 0:
        print("  local preflight FAILED -- not renting:")
        print((r.stdout or "")[-800:])
        print((r.stderr or "")[-800:])
        return 1
    print("  local preflight passed")

    offers = vast.search_offers(OFFER_QUERY, limit=3)
    if not offers:
        print("no offers matched; nothing rented")
        return 1
    print(f"  {len(offers)} offer(s) held")

    # -- Phase 2: rent ----------------------------------------------------
    print("\n[2/4] renting")
    # A known-hosts file per run. Vast recycles host:port across rentals,
    # so a new box can appear at an address seen before with a different
    # host key -- which OpenSSH reports as a possible man-in-the-middle and
    # refuses. Against ephemeral boxes that is a guaranteed false alarm, and
    # it would read here as "unusable box" and burn a replacement rental.
    known_hosts = scratch / f"known_hosts-{rid}"
    known_hosts.write_text("")
    instance_id, target = acquire_box(offers, pubkey, rid, known_hosts)
    host = target[-1].split("@")[-1]
    port = int(target[target.index("-p") + 1])
    user = target[-1].split("@")[0]

    # Everything past this point bills, so everything past this point is
    # inside teardown -- including the failures.
    with vast.teardown([instance_id]):

        print("\n[3/4] staging")
        # Built from the same options ssh just authenticated with. Rebuilding
        # them by hand is how you connect successfully and then fail to copy:
        # scp would fall back to default identities and skip the algorithm
        # settings that made the key work in the first place.
        _sh(["scp", *vast.scp_args(host, port, user, known_hosts=known_hosts),
             str(tarball), str(selection), f"{user}@{host}:/workspace/"],
            timeout=300, label="scp")
        _ssh(target, "cd /workspace && tar xzf kprelogits.tgz && "
                     "rm -f kprelogits.tgz && mkdir -p data results",
             timeout=120, label="untar")
        vast.stage_credentials(target, creds)

        cmds = remote_script(rid).splitlines()
        _ssh(target, cmds[0], timeout=SETUP_TIMEOUT, label="pip install")
        _ssh(target, cmds[1], timeout=DATA_TIMEOUT, label="data pull")
        print(_ssh(target, cmds[2], timeout=300, label="remote dry-run")[-1500:])

        print("\n[4/4] extracting")
        # A hung launch is not a failed launch. Detaching a process through
        # ssh is notoriously fiddly -- the channel stays open while anything
        # still holds it -- and the distinction that matters is whether the
        # worker started, which the state file answers directly. So: try to
        # launch, do not trust the exit, and let FIRST_STATE_DEADLINE deliver
        # the verdict a few minutes later.
        try:
            _ssh(target, cmds[3], timeout=60, label="launch", stdin_null=True)
            print("  launch returned cleanly")
        except subprocess.TimeoutExpired:
            print("  launch did not return (ssh held the channel); the job may "
                  "still be running -- the state file decides", flush=True)
        except Exception as e:
            print(f"  launch errored: {type(e).__name__}: {e}", flush=True)

        ok, reason = poll_to_completion(prefix)
        print(f"\n  worker: {reason}")

        # Ask S3 what is actually there before deciding anything. A worker can
        # report done and still have uploaded nothing, so "done" is a claim and
        # this is the check of it.
        check = vast.preteardown_check(prefix, expected_shards=1,
                                       expected_bundles=1 if ok else None)
        print(check.render())

        # Rescue on EITHER failure. The worker saying done while the bundle is
        # missing is the case most worth pulling artifacts for, and it is
        # exactly the case a `not ok` test would skip.
        if not ok or not check.ok:
            ok = False
            tail_remote_log(target)
            rescue(user, host, port, scratch / f"rescue-{rid}", known_hosts)

    # -- Phase 4: validate, box already destroyed -------------------------
    if not ok:
        print("\nFAILED. Rescued artifacts (if any) are in "
              f"{scratch / f'rescue-{rid}'}")
        return 1

    print("\n[validate] downloading bundles")
    smoke_local = scratch / f"smoke-{rid}.npz"
    ref_local = scratch / "reference.npz"
    s3.download(f"{prefix}/{DATA_FLAG}_{MODEL}_features.npz", smoke_local)
    try:
        s3.download(prod_bundle_uri(), ref_local)
    except Exception as e:
        print(f"  no reference bundle ({e}); validating alone")
        ref_local = None

    good = compare_bundles(smoke_local, ref_local)
    print(f"\n{'SMOKE TEST PASSED' if good else 'SMOKE TEST FAILED'}")
    print(f"  bundle: {prefix}/{DATA_FLAG}_{MODEL}_features.npz")
    return 0 if good else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--go", action="store_true",
                   help="actually rent hardware (default prints a plan)")
    p.add_argument("--abort", action="store_true",
                   help="destroy anything labelled smoke-*, then exit")
    p.add_argument("--compare", type=Path, metavar="BUNDLE",
                   help="validate a bundle offline; --reference for a second")
    p.add_argument("--reference", type=Path)
    p.add_argument("--scratch", type=Path, default=Path("out/smoke"))
    args = p.parse_args(argv)

    if args.abort:
        return abort()
    if args.compare:
        return 0 if compare_bundles(args.compare, args.reference) else 1

    rid = run_id()
    if not args.go:
        return plan(rid)

    args.scratch.mkdir(parents=True, exist_ok=True)
    try:
        return go(rid, args.scratch)
    except KeyboardInterrupt:
        print("\ninterrupted -- teardown ran if a box was rented; "
              "confirm with: python -m kprelogits.ops.smoke --abort")
        return 130


if __name__ == "__main__":
    sys.exit(main())
