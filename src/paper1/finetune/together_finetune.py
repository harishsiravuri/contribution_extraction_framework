"""Submit, poll, and resolve a Together AI fine-tune job.

Workflow:
  1. Upload `train.jsonl` (and val.jsonl) via /v1/files
  2. Submit /v1/fine-tunes with hyperparameters
  3. Poll /v1/fine-tunes/{id} until status is "completed", "error", or "cancelled"
  4. Persist meta + the resulting fine-tuned model id to JSON

Resume support: if outputs/paper_data_v5/finetune_meta.json exists with a job_id,
re-poll instead of resubmitting.

Together's REST API is documented at https://docs.together.ai/docs/fine-tuning .
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

API_BASE = "https://api.together.xyz/v1"
META_PATH = Path("outputs/paper_data_v5/finetune_meta.json")
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"


def _api_key() -> str:
    load_dotenv()
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        print("ERROR: TOGETHER_API_KEY is not set in .env", file=sys.stderr)
        sys.exit(2)
    return key


def _client() -> httpx.Client:
    key = _api_key()
    return httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=60.0,
    )


def _sdk():
    """Lazy import of the Together SDK to keep test imports light."""
    _api_key()  # ensures .env is loaded; raises if key missing
    from together import Together  # type: ignore
    return Together()


def upload_file(path: Path, purpose: str = "fine-tune") -> str:
    """Upload a JSONL file to Together via the official SDK; returns file_id."""
    sdk = _sdk()
    res = sdk.files.upload(file=str(path), check=True)
    return res.id


def submit_finetune(
    train_file_id: str,
    val_file_id: str | None,
    base_model: str = DEFAULT_BASE_MODEL,
    n_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    suffix: str = "scirex",
) -> dict:
    sdk = _sdk()
    kwargs = dict(
        training_file=train_file_id,
        model=base_model,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora=True,
        lora_r=lora_rank,
        lora_alpha=lora_alpha,
        suffix=suffix,
    )
    if val_file_id:
        kwargs["validation_file"] = val_file_id
    res = sdk.fine_tuning.create(**kwargs)
    # Convert the pydantic model to a plain dict for storage
    if hasattr(res, "model_dump"):
        return res.model_dump()
    if hasattr(res, "dict"):
        return res.dict()
    return dict(res)


def poll_status(job_id: str) -> dict:
    sdk = _sdk()
    res = sdk.fine_tuning.retrieve(job_id)
    if hasattr(res, "model_dump"):
        return res.model_dump()
    if hasattr(res, "dict"):
        return res.dict()
    return dict(res)


def list_jobs() -> list:
    sdk = _sdk()
    res = sdk.fine_tuning.list()
    return res.model_dump() if hasattr(res, "model_dump") else res


def run() -> dict:
    """Submit (or resume) the SciREX fine-tune; poll to completion."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        if meta.get("job_id"):
            print(f"[resume] existing job {meta['job_id']}")
            return _poll_loop(meta)
    else:
        meta = {}

    train_path = Path("outputs/paper_data_v5/scirex_finetune_train.jsonl")
    val_path = Path("outputs/paper_data_v5/scirex_finetune_val.jsonl")
    if meta.get("train_file_id"):
        train_file_id = meta["train_file_id"]
        val_file_id = meta.get("val_file_id")
        print(f"[reuse] uploaded files train={train_file_id} val={val_file_id}")
    else:
        print(f"[upload] {train_path}")
        train_file_id = upload_file(train_path)
        print(f"  → train file_id = {train_file_id}")
        print(f"[upload] {val_path}")
        val_file_id = upload_file(val_path)
        print(f"  → val   file_id = {val_file_id}")

    print(f"[submit] base = {DEFAULT_BASE_MODEL}; LoRA r=16 α=32; 3 epochs; lr=2e-5; bs=4")
    submit = submit_finetune(train_file_id, val_file_id)
    job_id = submit.get("id") or submit.get("job_id")
    print(f"  → job_id = {job_id}")
    meta = {
        "job_id": job_id,
        "submit_response": submit,
        "train_file_id": train_file_id,
        "val_file_id": val_file_id,
        "base_model": DEFAULT_BASE_MODEL,
        "submitted_at": time.time(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str))
    return _poll_loop(meta)


def _poll_loop(meta: dict, *, interval_s: int = 60, max_hours: float = 12.0) -> dict:
    start = time.time()
    job_id = meta["job_id"]
    while True:
        elapsed = time.time() - start
        if elapsed > max_hours * 3600:
            print(f"[timeout] {elapsed/3600:.1f}h exceeded {max_hours}h cap; stopping poll.")
            meta["timed_out"] = True
            META_PATH.write_text(json.dumps(meta, indent=2, default=str))
            return meta
        try:
            status = poll_status(job_id)
        except Exception as e:
            print(f"[poll] error: {e}; retrying in {interval_s}s")
            time.sleep(interval_s)
            continue
        s = status.get("status", "?")
        ev = status.get("events") or []
        latest = ev[-1].get("message") if ev else ""
        print(f"[{int(elapsed):>5}s] status={s} latest={latest!r}")
        meta["last_status"] = status
        meta["last_polled_at"] = time.time()
        META_PATH.write_text(json.dumps(meta, indent=2, default=str))
        if s in ("completed", "error", "cancelled", "failed"):
            ft_model = (
                status.get("output_name")
                or status.get("model_output")
                or status.get("model")
            )
            meta["finetuned_model"] = ft_model
            meta["final_status"] = s
            META_PATH.write_text(json.dumps(meta, indent=2, default=str))
            print(f"[done] status={s} fine-tuned model = {ft_model}")
            return meta
        time.sleep(interval_s)


if __name__ == "__main__":
    run()
