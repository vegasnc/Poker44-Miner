"""Structured per-synapse logger.

Every validator request goes through four phases logged here:
  RECEIVED  – synapse arrives (validator identity, chunk count)
  BLACKLIST – blacklist/allow decision with reason
  PRIORITY  – caller priority score
  INFERENCE – per-chunk scores, latency breakdown
  RESPONSE  – final scores, predictions, total round-trip latency
  ERROR     – any exception in the forward path

Records are written to two files:
  logs/synapses.jsonl  – one JSON object per line (machine-parseable)
  logs/synapses.log    – human-readable companion (same data, pretty-printed)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

_JSONL_LOGGER = logging.getLogger("poker44.synapses.jsonl")
_TEXT_LOGGER = logging.getLogger("poker44.synapses.text")
_configured = False


def configure(log_dir: str | Path = "logs") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt_text = logging.Formatter(
        "%(asctime)s %(levelname)s [synapse] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_jsonl = logging.Formatter("%(message)s")

    def _add(logger: logging.Logger, path: Path, fmt: logging.Formatter) -> None:
        if not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
            for h in logger.handlers
        ):
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

    _add(_JSONL_LOGGER, log_dir / "synapses.jsonl", fmt_jsonl)
    _add(_TEXT_LOGGER, log_dir / "synapses.log", fmt_text)


def _emit(record: dict[str, Any]) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _JSONL_LOGGER.info(json.dumps(record, default=str))


def _validator_info(synapse: Any) -> dict[str, str]:
    dendrite = getattr(synapse, "dendrite", None)
    return {
        "validator_hotkey": str(getattr(dendrite, "hotkey", "") or ""),
        "validator_ip": str(getattr(dendrite, "ip", "") or ""),
        "validator_port": str(getattr(dendrite, "port", "") or ""),
        "validator_version": str(getattr(dendrite, "version", "") or ""),
        "synapse_name": type(synapse).__name__,
    }


class SynapseTracker:
    """Context object created at the top of forward(); carries timing state."""

    def __init__(self, synapse: Any) -> None:
        self.synapse = synapse
        self.t_start = time.perf_counter()
        self._vinfo = _validator_info(synapse)

    # ------------------------------------------------------------------
    # Phase helpers (call in order inside forward / blacklist / priority)
    # ------------------------------------------------------------------

    def log_received(self, chunk_count: int) -> None:
        chunks = getattr(self.synapse, "chunks", None) or []
        hand_counts = [len(c) for c in chunks if isinstance(c, list)]
        record: dict[str, Any] = {
            "phase": "RECEIVED",
            **self._vinfo,
            "chunk_count": chunk_count,
            "hand_counts": hand_counts,
            "total_hands": sum(hand_counts),
        }
        _emit(record)
        _TEXT_LOGGER.info(
            "RECEIVED  validator=%s  ip=%s  chunks=%d  hands=%d",
            self._vinfo["validator_hotkey"] or "unknown",
            self._vinfo["validator_ip"] or "unknown",
            chunk_count,
            sum(hand_counts),
        )

    def log_blacklist(self, blocked: bool, reason: str) -> None:
        record: dict[str, Any] = {
            "phase": "BLACKLIST",
            **self._vinfo,
            "blocked": blocked,
            "reason": reason,
        }
        _emit(record)
        level = logging.WARNING if blocked else logging.DEBUG
        _TEXT_LOGGER.log(
            level,
            "BLACKLIST blocked=%s  reason=%s  validator=%s",
            blocked,
            reason,
            self._vinfo["validator_hotkey"] or "unknown",
        )

    def log_priority(self, score: float) -> None:
        record: dict[str, Any] = {
            "phase": "PRIORITY",
            **self._vinfo,
            "priority_score": round(score, 6),
        }
        _emit(record)
        _TEXT_LOGGER.debug(
            "PRIORITY  score=%.4f  validator=%s",
            score,
            self._vinfo["validator_hotkey"] or "unknown",
        )

    def log_inference(
        self,
        risk_scores: list[float],
        predictions: list[bool],
        latency_ms: float,
    ) -> None:
        bot_count = sum(predictions)
        human_count = len(predictions) - bot_count
        record: dict[str, Any] = {
            "phase": "INFERENCE",
            **self._vinfo,
            "chunk_count": len(risk_scores),
            "risk_scores": [round(s, 4) for s in risk_scores],
            "predictions": predictions,
            "bot_count": bot_count,
            "human_count": human_count,
            "score_min": round(min(risk_scores), 4) if risk_scores else None,
            "score_max": round(max(risk_scores), 4) if risk_scores else None,
            "score_mean": round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else None,
            "inference_latency_ms": round(latency_ms, 1),
        }
        _emit(record)
        scores_str = "  ".join(f"{s:.3f}" for s in risk_scores)
        _TEXT_LOGGER.info(
            "INFERENCE chunks=%d  bot=%d  human=%d  latency=%.0fms  scores=[%s]",
            len(risk_scores),
            bot_count,
            human_count,
            latency_ms,
            scores_str,
        )

    def log_response(self, risk_scores: list[float], predictions: list[bool]) -> None:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        record: dict[str, Any] = {
            "phase": "RESPONSE",
            **self._vinfo,
            "chunk_count": len(risk_scores),
            "risk_scores": [round(s, 4) for s in risk_scores],
            "predictions": predictions,
            "bot_count": sum(predictions),
            "total_latency_ms": round(total_ms, 1),
        }
        _emit(record)
        _TEXT_LOGGER.info(
            "RESPONSE  chunks=%d  bot=%d  total_latency=%.0fms  validator=%s",
            len(risk_scores),
            sum(predictions),
            total_ms,
            self._vinfo["validator_hotkey"] or "unknown",
        )

    def log_error(self, exc: Exception) -> None:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        record: dict[str, Any] = {
            "phase": "ERROR",
            **self._vinfo,
            "error": type(exc).__name__,
            "message": str(exc),
            "total_latency_ms": round(total_ms, 1),
        }
        _emit(record)
        _TEXT_LOGGER.error(
            "ERROR  %s: %s  validator=%s  latency=%.0fms",
            type(exc).__name__,
            exc,
            self._vinfo["validator_hotkey"] or "unknown",
            total_ms,
        )
