"""Vast.ai instance lifecycle: rent, track, verify, tear down.

Instances bill until destroyed, and the failure mode is not a crash -- it is
silence. This project has already paid for a 4.3-hour orphan (rented ad hoc, so
never in the registry file) and nearly lost a set of fits that existed only on
a box about to be destroyed. Both incidents share a cause: the registry file
was treated as the truth. It is not. **The API is the truth**, and every helper
here reconciles against it.

Three guarantees this module provides:

  1. ``teardown()`` destroys in ``finally`` and then RE-LISTS, so "we asked it
     to die" is never confused with "it is dead". It reports the remaining burn
     rate either way.
  2. ``reconcile()`` reports instances the API knows about that the registry
     does not -- the orphan class.
  3. ``stage_credentials()`` is the only supported way to get a secret onto a
     box. Secrets go over stdin into a ``umask 077`` file; ``create()`` refuses
     credential-shaped keys in ``-e``, because Vast hosts are shared and ``ps``
     is world-readable.

Completion is never inferred from processes here -- see ``ops/state.py``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import check_no_secrets

DEFAULT_REGISTRY = Path(".vast_instances")

# Remote paths are interpolated into a shell command, so they are restricted to
# a boring alphabet rather than quoted-and-hoped-for.
_SAFE_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")


class VastError(RuntimeError):
    """A vastai CLI call failed, or its output was not what we expected."""


def have_vastai() -> bool:
    return shutil.which("vastai") is not None


def _vastai(args: Sequence[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    if not have_vastai():
        raise VastError("vastai CLI not found on PATH")
    return subprocess.run(["vastai", *args], capture_output=True, text=True,
                          timeout=timeout)


# ---------------------------------------------------------------------------
# Reading the world
# ---------------------------------------------------------------------------

def show_instances() -> List[Dict[str, Any]]:
    """Every instance the API says we own. This is the authoritative list."""
    r = _vastai(["show", "instances", "--raw"])
    if r.returncode != 0:
        raise VastError(f"show instances failed rc={r.returncode}: "
                        f"{r.stderr.strip()[:300]}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise VastError(f"show instances returned non-JSON: {e}") from e
    return data if isinstance(data, list) else []


def burn_rate(instances: Iterable[Dict[str, Any]]) -> float:
    """Total $/hr currently being spent."""
    return sum(float(i.get("dph_total") or 0.0) for i in instances)


def describe(instances: Iterable[Dict[str, Any]]) -> str:
    rows = list(instances)
    if not rows:
        return "  (no instances)"
    lines = [f"  {i.get('id')}  {str(i.get('label') or '-'):<20} "
             f"${float(i.get('dph_total') or 0):.3f}/hr  {i.get('actual_status')}"
             for i in rows]
    lines.append(f"  burn: ${burn_rate(rows):.3f}/hr")
    # dph_total is compute. A *stopped* instance reports little or none of it
    # and still bills for the disk it is holding, so a reassuring $0.000/hr
    # next to a stopped box is not the same as costing nothing. Only
    # destroying releases the storage.
    stopped = [i for i in rows if str(i.get("actual_status")) not in
               ("running", "loading", "created", "None", "none")]
    if stopped:
        lines.append(f"  NOTE {len(stopped)} instance(s) not running "
                     f"({', '.join(str(i.get('id')) for i in stopped)}) still "
                     f"bill for storage until destroyed -- $/hr above is "
                     f"compute only")
    return "\n".join(lines)


def search_offers(query: str, *, limit: int = 10,
                  order: str = "dph") -> List[Dict[str, Any]]:
    """Offers matching a vastai query string. Reads only -- rents nothing.

    Ordered by price ascending unless told otherwise. The API's natural order
    is not price, and taking its first row has cost 3x for the same GPU class;
    ``limit`` then truncates the *cheapest* rows rather than arbitrary ones.
    """
    r = _vastai(["search", "offers", query, "-o", order, "--raw"])
    if r.returncode != 0:
        raise VastError(f"search offers failed rc={r.returncode}: "
                        f"{r.stderr.strip()[:300]}")
    try:
        offers = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise VastError(f"search offers returned non-JSON: {e}") from e
    return offers[:limit] if isinstance(offers, list) else []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """A local record of what we rented, reconciled against the API.

    Its only job is to remember *intent* (labels, purpose, when) for instances
    the API reports. It is never trusted as the list of what exists.
    """

    def __init__(self, path: Path = DEFAULT_REGISTRY) -> None:
        self.path = Path(path)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            # Historic .vast_instances was a plain list of ids, one per line.
            return {ln.strip(): {} for ln in self.path.read_text().splitlines()
                    if ln.strip()}

    def _save(self, d: Dict[str, Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(d, indent=2, sort_keys=True))

    def ids(self) -> List[int]:
        return sorted(int(k) for k in self._load())

    def add(self, instance_id: int, **meta: Any) -> None:
        d = self._load()
        d[str(instance_id)] = meta
        self._save(d)

    def remove(self, instance_id: int) -> None:
        d = self._load()
        d.pop(str(instance_id), None)
        self._save(d)


@dataclass
class Reconciliation:
    """The difference between what we think we rented and what we are paying for."""
    orphans: List[Dict[str, Any]]     # live per the API, absent from the registry
    stale: List[int]                  # in the registry, gone per the API
    tracked: List[Dict[str, Any]]
    burn: float

    def render(self) -> str:
        out = [f"burn: ${self.burn:.3f}/hr across {len(self.tracked) + len(self.orphans)} instance(s)"]
        if self.orphans:
            out.append(f"ORPHANS ({len(self.orphans)}) -- billing, untracked:")
            out.append(describe(self.orphans))
        if self.stale:
            out.append(f"stale registry entries (no longer exist): {self.stale}")
        return "\n".join(out)


def reconcile(registry: Registry = None,
              instances: Optional[List[Dict[str, Any]]] = None) -> Reconciliation:
    """Compare the API's instance list against the registry file."""
    registry = registry or Registry()
    live = show_instances() if instances is None else instances
    known = set(registry.ids())
    live_ids = {int(i["id"]) for i in live if "id" in i}
    return Reconciliation(
        orphans=[i for i in live if int(i.get("id", -1)) not in known],
        stale=sorted(known - live_ids),
        tracked=[i for i in live if int(i.get("id", -1)) in known],
        burn=burn_rate(live),
    )


# ---------------------------------------------------------------------------
# Renting and destroying
# ---------------------------------------------------------------------------

def create(offer_id: int, *, image: str, disk: int = 100,
           label: str = "kprelogits", onstart_cmd: str = "",
           env: Optional[Dict[str, str]] = None, ssh: bool = False,
           registry: Registry = None, dry_run: bool = False) -> Optional[int]:
    """Rent one instance and record it BEFORE doing anything else with it.

    ``env`` is passed to ``vastai create -e`` and is therefore rejected if it
    looks credential-shaped: use ``stage_credentials`` after the box is up.
    With ``dry_run`` the command is printed and nothing is rented.

    ``ssh=True`` rents the box as an SSH instance: Vast runs sshd instead of
    the image's entrypoint. Two reasons to want that. A stock image whose CMD
    is a shell exits immediately in entrypoint mode, taking the instance with
    it; and our own worker image would start extracting at boot, which races
    ``stage_credentials`` -- the job would reach S3 before its keys did. With
    sshd holding the box open, credentials land first and the job is launched
    deliberately.
    """
    env = env or {}
    check_no_secrets(env)

    args = ["create", "instance", str(offer_id), "--image", image,
            "--disk", str(disk), "--label", label, "--raw"]
    if ssh:
        # --direct asks for a direct-port connection; Vast falls back to the
        # proxy on hosts that cannot offer one (direct_port_start == -1).
        args += ["--ssh", "--direct"]
    if env:
        # The CLI takes ONE --env argument holding the whole "-e K=V -e K=V"
        # string; repeated -e flags are rejected by its argument parser. A
        # value containing whitespace would silently split into a bogus extra
        # flag inside that string, so refuse it rather than mis-rent a box.
        for k, v in sorted(env.items()):
            if any(c.isspace() for c in str(v)):
                raise ValueError(
                    f"env value for {k!r} contains whitespace, which --env "
                    f"cannot represent: {v!r}")
        args += ["--env", " ".join(f"-e {k}={v}" for k, v in sorted(env.items()))]
    if onstart_cmd:
        args += ["--onstart-cmd", onstart_cmd]

    if dry_run:
        print("DRY RUN, would run: vastai " + " ".join(args))
        return None

    r = _vastai(args)
    if r.returncode != 0:
        raise VastError(f"create failed rc={r.returncode}: {r.stderr.strip()[:300]}")
    try:
        new_id = int(json.loads(r.stdout)["new_contract"])
    except Exception as e:
        raise VastError(f"could not read instance id from create output: "
                        f"{r.stdout.strip()[:200]}") from e

    # Register first, verify second: an instance that exists but is untracked
    # is the orphan class, and the window between the two must be minimal.
    (registry or Registry()).add(new_id, label=label, image=image, offer=offer_id)
    if new_id not in {int(i.get("id", -1)) for i in show_instances()}:
        print(f"  WARNING: instance {new_id} not yet visible in show instances "
              f"-- it may still be provisioning")
    return new_id


def find_by_label(label: str) -> List[Dict[str, Any]]:
    """Instances carrying a label. Recovery for an ambiguous ``create``.

    A create that fails locally -- CLI timeout, unparseable output -- does not
    prove nothing was rented. Vast may have accepted the contract and the
    answer got lost on the way back, leaving a box billing under an id we
    never saw. A unique per-attempt label is the only handle left, so it is
    worth searching for one before concluding nothing happened.
    """
    return [i for i in show_instances() if str(i.get("label") or "") == label]


def destroy(instance_id: int, *, registry: Registry = None,
            confirm_tries: int = 6, confirm_delay: float = 5.0,
            sleep=None) -> bool:
    """Destroy one instance and confirm against the API that it is gone.

    Deletion is asynchronous, so a single immediate check can still see the
    instance and cry STILL BILLING about a box that is on its way out. Poll to
    a deadline instead -- but keep the alarm, because the alternative failure
    (assuming a destroy worked) is the one that costs money.
    """
    import time as _time
    sleep = sleep or _time.sleep

    r = _vastai(["destroy", "instance", str(instance_id), "-y"])
    ok = r.returncode == 0
    if not ok:
        print(f"  DESTROY FAILED {instance_id} rc={r.returncode}: "
              f"{r.stderr.strip()[:200]}")

    for attempt in range(confirm_tries):
        if instance_id not in {int(i.get("id", -1)) for i in show_instances()}:
            (registry or Registry()).remove(instance_id)
            return ok
        if attempt < confirm_tries - 1:
            sleep(confirm_delay)

    print(f"  WARNING: instance {instance_id} still listed after destroy "
          f"-- STILL BILLING")
    return False


@contextlib.contextmanager
def teardown(ids: Sequence[int], *, registry: Registry = None,
             enabled: bool = True):
    """Run a block, then destroy ``ids`` in ``finally`` and re-list to confirm.

    The whole value is in ``finally`` plus the re-list: an exception, a
    keyboard interrupt, or a clean exit all end at the same place, and the
    remaining burn rate is printed from the API rather than assumed.
    """
    try:
        yield
    finally:
        if not enabled:
            print(f"teardown disabled -- {list(ids)} left running, still billing")
            return
        reg = registry or Registry()
        for i in ids:
            try:
                destroy(int(i), registry=reg)
            except Exception as e:
                print(f"  teardown error on {i}: {type(e).__name__}: {e}")
        try:
            live = show_instances()
            print(f"after teardown: ${burn_rate(live):.3f}/hr still burning")
            print(describe(live))
        except Exception as e:
            print(f"  could not confirm teardown against the API: {e}")


# ---------------------------------------------------------------------------
# Reaching the box
# ---------------------------------------------------------------------------

_SSH_URL = re.compile(r"^ssh://(?P<user>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/?$")


def ssh_routes(instance: Dict[str, Any]) -> List[tuple]:
    """Every way into a box, most reliable first, as ``(label, host, port)``.

    Vast offers two, and they fail differently. The **proxy**
    (``ssh_host``/``ssh_port``, an ``sshN.vast.ai`` relay) always accepts a
    connection. The **direct** route (``public_ipaddr``/``direct_port_start``)
    is faster but frequently unreachable: the host may be behind NAT or a
    firewall, and ``direct_port_start`` is ``-1`` on hosts that do not offer it
    at all -- in which case a connection attempt hangs until it times out
    rather than being refused.

    Proxy first, therefore. SSH here carries commands and a 59 KB tarball; the
    3.9 GB of data moves between the box and S3 and never touches this
    connection, so the proxy's lower bandwidth costs nothing that matters.
    """
    routes = []
    host, port = instance.get("ssh_host"), instance.get("ssh_port")
    if host and port:
        routes.append(("proxy", str(host), int(port)))
    ip, dport = instance.get("public_ipaddr"), instance.get("direct_port_start")
    if ip and dport and int(dport) > 0:
        routes.append(("direct", str(ip).strip(), int(dport)))
    return routes


def ssh_endpoints(instance_id: int, *, retries: int = 6, delay: float = 10.0,
                  sleep=None) -> List[tuple]:
    """Routes to a running instance, retrying while its networking settles.

    Retrying is not defensive padding: an instance that has just reached
    ``running`` may not have published its ssh fields yet, so the first empty
    answer says nothing about whether the box will ever be reachable.
    ``sleep`` is injectable so tests can exercise the loop without waiting.
    """
    import time
    sleep = sleep or time.sleep

    for attempt in range(retries):
        inst = next((i for i in show_instances()
                     if int(i.get("id", -1)) == instance_id), None)
        if inst:
            routes = ssh_routes(inst)
            if routes:
                return routes
        if attempt < retries - 1:
            sleep(delay)

    raise VastError(
        f"instance {instance_id} published no ssh endpoint after {retries} "
        f"attempts. If it is running, the key may not be attached: "
        f"vastai attach ssh {instance_id} \"$(cat ~/.ssh/id_rsa.pub)\"")


# Preference order, most modern first. Vast's own `create ssh-key` generates
# an ed25519 key, so hard-coding id_rsa would reject a user who followed
# Vast's setup instructions and has a perfectly good key.
SSH_KEY_CANDIDATES = ("id_ed25519", "id_ecdsa", "id_rsa")


def default_ssh_key() -> Optional[Path]:
    """The private key to authenticate with, or None if there is no usable one.

    ``KPRELOGITS_SSH_KEY`` overrides. Otherwise the first candidate that exists
    with its ``.pub`` beside it -- both halves are needed: the private key to
    connect, the public one to attach to the instance.
    """
    override = os.environ.get("KPRELOGITS_SSH_KEY")
    if override:
        return Path(override).expanduser()
    ssh = Path.home() / ".ssh"
    for name in SSH_KEY_CANDIDATES:
        k = ssh / name
        if k.exists() and Path(f"{k}.pub").exists():
            return k
    return None


# Kept as a name because callers and messages refer to it; resolved lazily so
# a machine with an ed25519 key is not told to look for an RSA one.
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "id_rsa"


def _ssh_options(port: int, *, key: Optional[Path] = None,
                 known_hosts: Optional[Path] = None) -> List[str]:
    """Options shared by ssh and scp, so the two cannot disagree.

    They must not disagree: authenticating over ssh and then failing to scp
    because the identity was rebuilt by hand is a confusing way to lose a box.
    """
    key = default_ssh_key() if key is None else Path(key)
    opts: List[str] = []
    if key and key.exists():
        opts += ["-i", str(key)]
        # OpenSSH 8.8+ refuses SHA-1 RSA signatures by default, and Vast hosts
        # run a range of sshd versions -- so an ssh-rsa key authenticates on
        # some boxes and is rejected on others, which reads as random,
        # box-dependent failure. Re-enable the algorithm (this does not
        # disable verification) only when the key actually is RSA.
        if key.name.endswith("rsa") or "rsa" in key.name:
            opts += ["-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
                     "-o", "HostKeyAlgorithms=+ssh-rsa"]
    if known_hosts is not None:
        # Vast recycles host:port across rentals, so a *new* box can appear at
        # an address we have seen before with a different host key -- which
        # OpenSSH correctly reports as a possible man-in-the-middle and
        # refuses. Against ephemeral boxes that is a guaranteed false alarm
        # that costs a rental, so each run gets its own known-hosts file:
        # still consistent within a run, with no stale entries to trip over.
        opts += ["-o", f"UserKnownHostsFile={known_hosts}"]
    return opts + ["-o", "StrictHostKeyChecking=accept-new",
                   "-o", "BatchMode=yes",
                   "-o", "ConnectTimeout=15",
                   "-o", "ServerAliveInterval=15",
                   "-o", "ServerAliveCountMax=4"]


def ssh_args(host: str, port: int, user: str = "root",
             key: Optional[Path] = None,
             known_hosts: Optional[Path] = None) -> List[str]:
    """The ssh flags this project always wants for a non-interactive call.

    ``BatchMode`` makes a missing key fail immediately instead of hanging on a
    password prompt that nothing will ever answer, and ``accept-new`` takes the
    host key of a box that by definition has never been seen before without
    disabling host-key checking outright.

    The identity is passed explicitly because there may be no agent running --
    with ``BatchMode`` and an empty agent, ssh has nothing to offer and the box
    rejects us while billing by the minute.
    """
    return [*_ssh_options(port, key=key, known_hosts=known_hosts),
            "-p", str(port), f"{user}@{host}"]


def scp_args(host: str, port: int, user: str = "root",
             key: Optional[Path] = None,
             known_hosts: Optional[Path] = None) -> List[str]:
    """The same connection settings as ``ssh_args``, spelled for scp.

    scp takes ``-P`` where ssh takes ``-p``, and that one letter is the entire
    reason to build both from one place rather than retyping the flags.
    """
    return [*_ssh_options(port, key=key, known_hosts=known_hosts),
            "-P", str(port)]


def account_ssh_keys() -> List[Dict[str, Any]]:
    """SSH keys registered on the vast.ai account.

    Informational only. Team accounts cannot register account-wide keys, so an
    empty list here is normal rather than a problem -- see ``attach_ssh_key``,
    which scopes a key to one instance instead.
    """
    r = _vastai(["show", "ssh-keys", "--raw"])
    if r.returncode != 0:
        raise VastError(f"show ssh-keys failed rc={r.returncode}: "
                        f"{r.stderr.strip()[:200]}")
    try:
        keys = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise VastError(f"show ssh-keys returned non-JSON: {e}") from e
    return keys if isinstance(keys, list) else []


def public_key(key: Optional[Path] = None) -> str:
    """Read the public half of an ssh key. Never touches the private half."""
    key = key or default_ssh_key()
    if key is None:
        raise VastError(
            "no ssh key found. Looked for "
            + ", ".join(f"~/.ssh/{n}" for n in SSH_KEY_CANDIDATES)
            + " (each needs its .pub beside it). Set KPRELOGITS_SSH_KEY to "
              "point at one, or generate: ssh-keygen -t ed25519")
    pub = Path(f"{key}.pub")
    if not pub.exists():
        raise VastError(f"no public key at {pub}")
    text = pub.read_text().strip()
    if not text.startswith(("ssh-", "ecdsa-")):
        raise VastError(f"{pub} does not look like an ssh public key")
    return text


def attach_ssh_key(instance_id: int, pubkey: str) -> None:
    """Authorize one key on one instance.

    The per-instance path exists because team accounts cannot hold
    account-wide keys. It is also the tighter grant: the key authorizes
    exactly the box we just rented and dies with it.

    Vast propagates a freshly attached key with a lag, so a login attempt
    immediately after this can still be refused -- callers should retry rather
    than conclude the box is broken.
    """
    r = _vastai(["attach", "ssh", str(instance_id), pubkey])
    if r.returncode != 0:
        raise VastError(f"attach ssh key to {instance_id} failed "
                        f"rc={r.returncode}: {r.stderr.strip()[:200]}")


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def stage_credentials(ssh_target, values: Dict[str, str],
                      remote_path: str = "/workspace/.env-secrets") -> None:
    """Write secrets to a mode-600 file on the box, via stdin.

    The payload never appears in argv, so it never appears in ``ps`` on a host
    we share with strangers. Remote scripts consume it with::

        set -a; . /workspace/.env-secrets; set +a

    This is the only sanctioned path for AWS keys and HF tokens.

    ``ssh_target`` is either a plain ``user@host`` (when the port comes from
    ``~/.ssh/config``) or the full flag list from ``ssh_args()`` -- Vast hands
    out a non-22 port, which cannot be expressed in a bare target string.
    """
    if not _SAFE_REMOTE_PATH.match(remote_path):
        raise ValueError(f"unsafe remote path {remote_path!r}")
    for k in values:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            raise ValueError(f"invalid env var name {k!r}")
    target = [ssh_target] if isinstance(ssh_target, str) else list(ssh_target)
    # The file is *sourced*, so its contents are shell code, not data. AWS keys
    # happen to be alphanumeric, but this helper takes arbitrary strings and a
    # value carrying a space, a quote, or `$(...)` would either corrupt the
    # credential or execute. Quoting is not paranoia here: the one function
    # whose whole purpose is handling secrets safely should not be the one that
    # evaluates them.
    payload = "".join(f"{k}={shlex.quote(v)}\n"
                      for k, v in sorted(values.items()))
    r = subprocess.run(
        ["ssh", *target, f"umask 077; cat > {remote_path}"],
        input=payload, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise VastError(f"staging credentials to {remote_path} "
                        f"failed rc={r.returncode}: {r.stderr.strip()[:200]}")


# ---------------------------------------------------------------------------
# Pre-teardown
# ---------------------------------------------------------------------------

@dataclass
class TeardownCheck:
    """Whether it is safe to destroy the boxes working on this prefix."""
    ok: bool
    blockers: List[str]
    notes: List[str]

    def render(self) -> str:
        head = "SAFE TO DESTROY" if self.ok else "DO NOT DESTROY"
        return "\n".join([head]
                         + [f"  BLOCKER {b}" for b in self.blockers]
                         + [f"  note    {n}" for n in self.notes])


def preteardown_check(s3_prefix: str, *, expected_shards: int,
                      expected_bundles: Optional[int] = None) -> TeardownCheck:
    """Replace the manual checklist: are all shards done and all bundles up?

    Both halves matter. A shard reporting ``done`` with bundles missing from
    S3 means the final sync lied or never ran; bundles present with a shard
    still ``running`` means work is in flight. Only agreement clears teardown.
    """
    from . import s3, state

    blockers: List[str] = []
    notes: List[str] = []

    # State files are keyed by rank alone, so a prefix that has seen an earlier
    # run with a different shard count still holds that run's files -- a
    # 1-shard refix into the DermaMNIST prefix reported DO NOT DESTROY over
    # shards 2-7 of a finished 8-shard run. They are not this run's, and must
    # neither block nor clear its teardown.
    all_states = state.read_states(s3_prefix)
    states = {r: st for r, st in all_states.items()
              if r < expected_shards
              and getattr(st, "shards", expected_shards) in (None, expected_shards)}
    if len(states) < len(all_states):
        notes.append(f"ignored {len(all_states) - len(states)} state file(s) "
                     f"from a run with a different shard count")
    missing = [r for r in range(expected_shards) if r not in states]
    if missing:
        blockers.append(f"no published state for shard(s) {missing}")
    for rank, st in sorted(states.items()):
        if st.phase not in state.TERMINAL:
            blockers.append(f"shard {rank} is {st.phase} (last update {st.updated_at})")
        elif st.phase == "failed":
            blockers.append(f"shard {rank} FAILED: {st.errors[-1] if st.errors else '?'}")
        elif not st.is_complete:
            blockers.append(
                f"shard {rank} says done but {st.extracted + st.skipped}/"
                f"{st.scheduled} accounted for, {st.uploaded} uploaded, "
                f"{st.failed} failed")

    bundles = s3.list_names(s3_prefix, suffix="_features.npz")
    notes.append(f"{len(bundles)} bundle(s) under {s3_prefix}")
    if expected_bundles is not None and len(bundles) < expected_bundles:
        blockers.append(f"only {len(bundles)} of {expected_bundles} bundles in S3")

    return TeardownCheck(ok=not blockers, blockers=blockers, notes=notes)
