"""Poker44 production miner entrypoint.

This file mirrors the official Poker44 neuron contract and routes validator
DetectionSynapse chunks through the local LightGBM inference pipeline.
"""

import logging
import sys
import time
import argparse
import threading
import os
from pathlib import Path
from typing import Any, ClassVar, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from miners.custom_miner import Poker44BotDetectionMiner
from src.utils.helpers import load_json
from src.utils.synapse_logger import (
    SynapseTracker,
    configure as configure_synapse_logger,
    install_axon_http_logging,
)

LOGGER = logging.getLogger(__name__)
LOG_FILE_HANDLE: Any = None


class TeeStream:
    def __init__(self, original: Any, file_handle: Any) -> None:
        self.original = original
        self.file_handle = file_handle

    def write(self, data: str) -> int:
        self.original.write(data)
        self.file_handle.write(data)
        return len(data)

    def flush(self) -> None:
        self.original.flush()
        self.file_handle.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def configure_project_logging() -> Path:
    global LOG_FILE_HANDLE

    load_env_file(REPO_ROOT / ".env")
    log_dir = Path(os.getenv("POKER44_LOG_DIR", str(REPO_ROOT / "logs")))
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, os.getenv("POKER44_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log_file = log_dir / "poker44-miner.log"

    import datetime as _dt
    _UTC8 = _dt.timezone(_dt.timedelta(hours=-8))

    class _UTC8Formatter(logging.Formatter):
        def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
            ct = _dt.datetime.fromtimestamp(record.created, tz=_UTC8)
            return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

    formatter = _UTC8Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
        for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Attach FileHandler directly to LOGGER so bt.logging resetting root to WARNING
    # doesn't silence miner's own INFO messages (e.g. "Serving miner axon").
    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
        for handler in LOGGER.handlers
    ):
        miner_fh = logging.FileHandler(log_file, encoding="utf-8")
        miner_fh.setLevel(log_level)
        miner_fh.setFormatter(formatter)
        LOGGER.addHandler(miner_fh)
    LOGGER.setLevel(log_level)

    if os.getenv("POKER44_LOG_TEE_STDIO", "1").lower() in {"1", "true", "yes", "on"}:
        if not isinstance(sys.stdout, TeeStream) and not isinstance(sys.stderr, TeeStream):
            LOG_FILE_HANDLE = log_file.open("a", encoding="utf-8", buffering=1)
            sys.stdout = TeeStream(sys.stdout, LOG_FILE_HANDLE)
            sys.stderr = TeeStream(sys.stderr, LOG_FILE_HANDLE)

    def log_uncaught_exception(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        LOGGER.critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = log_uncaught_exception
    logging.captureWarnings(True)
    configure_synapse_logger(log_dir)
    LOGGER.info("Project log recording enabled at %s", log_file)
    return log_file


PROJECT_LOG_FILE = configure_project_logging()

try:
    import bittensor as bt

    BITTENSOR_RUNTIME_AVAILABLE = True
except ImportError as exc:
    bt = None
    BITTENSOR_IMPORT_ERROR = exc
    DetectionSynapse = Any  # type: ignore[misc, assignment]
    BITTENSOR_RUNTIME_AVAILABLE = False
else:
    BITTENSOR_IMPORT_ERROR = None
    try:
        from poker44.validator.synapse import DetectionSynapse

        SYNAPSE_SOURCE = "poker44.validator.synapse"
    except ImportError:
        from pydantic import ConfigDict, Field

        class DetectionSynapse(bt.Synapse):  # type: ignore[misc]
            """Local copy of Poker44's validator synapse contract."""

            model_config = ConfigDict(arbitrary_types_allowed=True)

            chunks: list[list[dict]] = Field(default_factory=list)
            risk_scores: list[float] | None = None
            predictions: list[bool] | None = None
            model_manifest: dict[str, Any] | None = None
            required_hash_fields: ClassVar[list[str]] = ["chunks"]

            def deserialize(self) -> "DetectionSynapse":
                return self

        SYNAPSE_SOURCE = "local fallback"

try:
    from poker44.base.miner import BaseMinerNeuron
    from poker44.utils.model_manifest import evaluate_manifest_compliance, manifest_digest

    POKER44_BASE_AVAILABLE = True
except ImportError:
    BaseMinerNeuron = object  # type: ignore[assignment]
    POKER44_RUNTIME_AVAILABLE = False
else:
    POKER44_RUNTIME_AVAILABLE = BITTENSOR_RUNTIME_AVAILABLE and POKER44_BASE_AVAILABLE


def _save_hard_chunks(
    chunks: list,
    risk_scores: list,
    predictions: list,
    threshold: float,
) -> None:
    """Save raw chunks from hard-regime batches (mean < 0.85) for Hard-Case Model training.

    Called in a daemon thread — never raises, never blocks inference.
    Chunks predicted as human (score < threshold) saved with label=0,
    predicted as bot (score >= threshold) saved with label=1.
    Files land in data/hard_regime/ with one JSON per batch, named by UTC timestamp.
    """
    import json as _json, datetime as _dt, pathlib as _pl
    try:
        out_dir = _pl.Path(__file__).resolve().parents[1] / "data" / "hard_regime"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mean_score = sum(risk_scores) / len(risk_scores)
        n_human = sum(1 for p in predictions if not p)
        records = []
        for chunk, score, pred in zip(chunks, risk_scores, predictions):
            records.append({
                "label": 0 if not pred else 1,
                "score": round(score, 4),
                "chunk": chunk,
            })
        payload = {
            "saved_at": ts,
            "batch_mean": round(mean_score, 4),
            "threshold": round(threshold, 4),
            "n_chunks": len(chunks),
            "n_human": n_human,
            "n_bot": len(chunks) - n_human,
            "chunks": records,
        }
        out_file = out_dir / f"hard_{ts}.json"
        out_file.write_text(_json.dumps(payload, default=str))
    except Exception:
        pass  # never affect live inference


class Miner(BaseMinerNeuron):  # type: ignore[misc, valid-type]
    """Model-backed Poker44 miner.

    Validators send `DetectionSynapse(chunks=...)`. The miner must return one
    risk score per chunk, plus optional predictions and model manifest metadata.
    """

    def __init__(self, config: Any = None):
        if POKER44_RUNTIME_AVAILABLE:
            super(Miner, self).__init__(config=config)
            install_axon_http_logging(self.axon)
        elif BITTENSOR_RUNTIME_AVAILABLE:
            self._init_direct_bittensor(config=config)

        repo_root = Path(__file__).resolve().parents[1]
        self.detector = Poker44BotDetectionMiner(config_path=str(repo_root / "config" / "config.yaml"))
        self.model_manifest = self._load_model_manifest(repo_root)
        self.manifest_compliance = self._manifest_compliance()
        self.manifest_digest = self._manifest_digest()

        self._log("info", "Model-backed Poker44 miner started")
        self._log_manifest_startup(repo_root)

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Score each received chunk group independently and log the full lifecycle."""
        tracker = SynapseTracker(synapse)
        chunks = getattr(synapse, "chunks", None) or []
        tracker.log_received(len(chunks))

        # One-time dump of raw synapse for inspection
        import json as _json
        _dump = Path(__file__).resolve().parents[1] / "logs" / "synapse_dump.json"
        if not _dump.exists() and chunks:
            try:
                _data = {
                    "chunk_count": len(chunks),
                    "hands_per_chunk": [len(c) for c in chunks[:3]],
                    "sample_chunk_0_hand_0": chunks[0][0] if chunks[0] else None,
                    "sample_chunk_0_hand_1": chunks[0][1] if len(chunks[0]) > 1 else None,
                    "synapse_fields": [f for f in dir(synapse) if not f.startswith("_")],
                }
                _dump.write_text(_json.dumps(_data, indent=2, default=str))
                LOGGER.info("Synapse dump saved to %s", _dump)
            except Exception:
                pass
        try:
            result = self.detector.pipeline.score_synapse_chunks(chunks)
            risk_scores: list[float] = result["risk_scores"]
            predictions: list[bool] = result["predictions"]
            latency_ms: float = result.get("inference_latency_ms", 0.0)
            threshold_used: float = result.get("threshold_used", 0.5)
            tracker.log_inference(risk_scores, predictions, latency_ms, threshold=threshold_used)
            synapse.risk_scores = risk_scores
            synapse.predictions = predictions
            synapse.model_manifest = dict(self.model_manifest)
            tracker.log_response(risk_scores, predictions)
            # Save raw chunks from hard-regime batches (mean < 0.85) for Hard-Case Model training.
            # Runs in a fire-and-forget thread — zero impact on latency or inference result.
            raw_mean = result.get("raw_scores_mean", sum(risk_scores) / len(risk_scores) if risk_scores else 1.0)
            if risk_scores and raw_mean < 0.85:
                import threading as _thr
                _thr.Thread(
                    target=_save_hard_chunks,
                    args=(chunks, risk_scores, predictions, threshold_used),
                    daemon=True,
                ).start()
        except Exception as exc:
            tracker.log_error(exc)
            LOGGER.exception("forward() failed; returning neutral scores")
            n = len(chunks)
            synapse.risk_scores = [0.5] * n
            synapse.predictions = [False] * n
            synapse.model_manifest = dict(self.model_manifest)
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Use the official base miner blacklist policy and log the decision."""
        tracker = SynapseTracker(synapse)
        if hasattr(self, "common_blacklist"):
            blocked, reason = self.common_blacklist(synapse)  # type: ignore[attr-defined]
        else:
            blocked, reason = False, "Poker44 runtime not available; local fallback accepts request."
        tracker.log_blacklist(blocked, reason)
        return blocked, reason

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Use the official base miner caller priority and log the score."""
        tracker = SynapseTracker(synapse)
        if hasattr(self, "caller_priority"):
            score = self.caller_priority(synapse)  # type: ignore[attr-defined]
        else:
            hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
            if hotkey and hasattr(self, "metagraph") and hotkey in self.metagraph.hotkeys:
                uid = self.metagraph.hotkeys.index(hotkey)
                score = float(self.metagraph.S[uid])
            else:
                score = 0.0
        tracker.log_priority(score)
        return score

    def _init_direct_bittensor(self, config: Any = None) -> None:
        self.config = config or _build_direct_config()
        self.wallet = bt.Wallet(config=self.config)
        self.subtensor = bt.Subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        self.uid = _registered_uid(self.metagraph, self.wallet.hotkey.ss58_address)
        self.axon = bt.Axon(wallet=self.wallet, config=self.config)
        install_axon_http_logging(self.axon)
        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )
        self.should_exit = False
        self.is_running = False
        self.thread: threading.Thread | None = None

    def _load_model_manifest(self, repo_root: Path) -> dict[str, Any]:
        manifest = load_json(repo_root / "config" / "manifest.json", default={}) or {}
        manifest.setdefault("model_name", "poker44-bot-detector")
        manifest.setdefault("model_version", "1.0.0")
        manifest.setdefault("framework", "lightgbm")
        manifest.setdefault("inference_mode", "remote")
        return manifest

    def _manifest_compliance(self) -> dict[str, Any]:
        if POKER44_RUNTIME_AVAILABLE:
            return evaluate_manifest_compliance(self.model_manifest)
        required = [
            "open_source",
            "repo_url",
            "repo_commit",
            "model_name",
            "model_version",
            "training_data_statement",
            "private_data_attestation",
            "implementation_files",
            "implementation_sha256",
        ]
        missing = [field for field in required if not self.model_manifest.get(field)]
        return {"status": "pass" if not missing else "incomplete", "missing_fields": missing}

    def _manifest_digest(self) -> str:
        if POKER44_RUNTIME_AVAILABLE:
            return manifest_digest(self.model_manifest)
        return self.model_manifest.get("implementation_sha256", "")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        self._log("info", "Open-sourced miner manifest standard active for this miner.")
        self._log(
            "info",
            "Miner transparency status: "
            f"{self.manifest_compliance.get('status')} "
            f"(missing_fields={self.manifest_compliance.get('missing_fields')})",
        )
        self._log(
            "info",
            "Manifest summary | "
            f"model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}",
        )
        self._log(
            "info",
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}",
        )
        self._log("info", f"Miner prep docs available | miner_doc={repo_root / 'README.md'}")

    @staticmethod
    def _log(level: str, message: str) -> None:
        python_level = logging.INFO if level == "success" else getattr(logging, level.upper(), logging.INFO)
        LOGGER.log(python_level, message)
        if bt is not None:
            getattr(bt.logging, level)(message)

    def run(self) -> None:
        if POKER44_RUNTIME_AVAILABLE:
            return super().run()  # type: ignore[misc]
        if not BITTENSOR_RUNTIME_AVAILABLE:
            raise RuntimeError("Bittensor is not installed.")

        self._log(
            "info",
            f"Serving miner axon on netuid={self.config.netuid} "
            f"endpoint={self.subtensor.chain_endpoint}",
        )
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        self.axon.start()
        self._log("info", f"Miner running with UID: {self.uid}")
        # Re-register axon every ~100 blocks (~20 min) to keep active=1 on metagraph
        _SERVE_INTERVAL_BLOCKS = 100
        _last_serve_block = self.subtensor.block
        try:
            while not self.should_exit:
                time.sleep(60)
                try:
                    current_block = self.subtensor.block
                    if current_block - _last_serve_block >= _SERVE_INTERVAL_BLOCKS:
                        self.metagraph.sync(subtensor=self.subtensor)
                        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
                        _last_serve_block = current_block
                        self._log("info", f"Axon re-registered at block {current_block}")
                except Exception as _e:
                    self._log("warning", f"Periodic re-registration failed: {_e}")
        except KeyboardInterrupt:
            self.axon.stop()
            self._log("success", "Miner killed by keyboard interrupt.")

    def run_in_background_thread(self) -> None:
        if POKER44_RUNTIME_AVAILABLE:
            return super().run_in_background_thread()  # type: ignore[misc]
        if not self.is_running:
            self.should_exit = False
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            self.is_running = True

    def stop_run_thread(self) -> None:
        if POKER44_RUNTIME_AVAILABLE:
            return super().stop_run_thread()  # type: ignore[misc]
        if self.is_running:
            self.should_exit = True
            if self.thread is not None:
                self.thread.join(5)
            if hasattr(self, "axon"):
                self.axon.stop()
            self.is_running = False

    def __enter__(self) -> "Miner":
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop_run_thread()


def _build_direct_config() -> Any:
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Axon.add_args(parser)
    bt.logging.add_args(parser)
    parser.add_argument("--netuid", type=int, default=126)
    parser.add_argument("--blacklist.force_validator_permit", action="store_true", default=False)
    parser.add_argument("--blacklist.allow_non_registered", action="store_true", default=False)
    parser.add_argument("--blacklist.allowed_validator_hotkeys", nargs="*", default=[])
    parsed = vars(parser.parse_args(sys.argv[1:]))
    config = bt.Config(parser)
    for key, value in parsed.items():
        _set_config_value(config, key, value)
    return config


def _set_config_value(config: Any, dotted_key: str, value: Any) -> None:
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = target.get(part) if hasattr(target, "get") else getattr(target, part, None)
        if current is None:
            target[part] = {}
            current = target[part]
        target = current
    target[parts[-1]] = value


def _registered_uid(metagraph: Any, hotkey: str) -> int:
    if hotkey not in metagraph.hotkeys:
        raise SystemExit(
            f"Wallet hotkey {hotkey} is not registered on this subnet. "
            "Register it with `btcli subnet register --netuid 126` first."
        )
    return metagraph.hotkeys.index(hotkey)


if __name__ == "__main__":
    if not BITTENSOR_RUNTIME_AVAILABLE:
        raise SystemExit(
            "Bittensor could not be imported by this Python interpreter. "
            "Run `python -m pip install bittensor` in the same environment and retry. "
            f"Import error: {BITTENSOR_IMPORT_ERROR}"
        )

    with Miner() as miner:
        bt.logging.info("Poker44 model miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
