"""Operational plumbing for running extraction on rented hardware.

Three concerns, deliberately separate from the extraction logic so that
``extract.py`` reads as "what we compute" and this package as "how we survive
doing it on a machine that bills by the hour and can vanish":

    s3.py     upload / list / sync, with the retry policy in ONE place
    state.py  per-shard task-state files -- the only supported way to ask
              "is that box done?"  (never `pgrep` over SSH; it self-matches)
    vast.py   instance lifecycle: registry, reconciliation against the API,
              teardown in `finally`, and credential staging that keeps secrets
              out of `ps`
"""
