#!/usr/bin/env python3
"""Resumable Hugging Face dataset fetch for the this recipe extra buckets (SPEC §1: vaani, spring_inx).

Retries until the gate is accepted (403 -> sleep) and uses allow_patterns so only the Bengali shards land.
Usage: hf_fetch.py <repo_id> <local_dir> [allow_pattern ...]
"""
import sys, time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError



def main():
    repo, out, pats = sys.argv[1], sys.argv[2], sys.argv[3:] or None
    for attempt in range(1, 400):
        try:
            p = snapshot_download(repo, repo_type="dataset", local_dir=out, allow_patterns=pats, max_workers=4)
            print("DONE", p, flush=True); sys.exit(0)
        except (GatedRepoError, HfHubHTTPError) as e:
            print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:120]} -- waiting for gate/network", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"attempt {attempt}: {type(e).__name__}: {str(e)[:120]}", flush=True)
        time.sleep(120)
    sys.exit(1)


if __name__ == "__main__":
    main()
