"""Structured per-synapse logger.

All synapse events are written to the root logger (poker44-miner.log).
No separate files are created.

Phases logged for every validator request:
  RECEIVED  – synapse arrives (validator identity, chunk count, hand counts)
  BLACKLIST – blacklist/allow decision with reason
  PRIORITY  – caller priority score
  INFERENCE – per-chunk scores, bot/human counts, inference latency
  RESPONSE  – final scores, predictions, total round-trip latency
  ERROR     – any exception in the forward path
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("poker44.synapse")


def configure() -> None:
    """No-op — synapse events propagate to the root logger (poker44-miner.log)."""
    logger.propagate = True


def _validator_info(synapse: Any) -> dict[str, str]:
    dendrite = getattr(synapse, "dendrite", None)
    return {
        "hotkey": str(getattr(dendrite, "hotkey", "") or "unknown"),
        "ip": str(getattr(dendrite, "ip", "") or "unknown"),
        "port": str(getattr(dendrite, "port", "") or ""),
        "version": str(getattr(dendrite, "version", "") or ""),
    }


class SynapseTracker:
    """Created at the top of forward(); tracks timing and logs each lifecycle phase."""

    def __init__(self, synapse: Any) -> None:
        self.synapse = synapse
        self.t_start = time.perf_counter()
        self._v = _validator_info(synapse)

    def log_received(self, chunk_count: int) -> None:
        chunks = getattr(self.synapse, "chunks", None) or []
        hand_counts = [len(c) for c in chunks if isinstance(c, list)]
        logger.info(
            "[SYNAPSE:RECEIVED] validator=%s ip=%s chunks=%d hands=%d hand_counts=%s",
            self._v["hotkey"], self._v["ip"], chunk_count, sum(hand_counts), hand_counts,
        )

    def log_blacklist(self, blocked: bool, reason: str) -> None:
        level = logging.WARNING if blocked else logging.INFO
        logger.log(
            level,
            "[SYNAPSE:BLACKLIST] blocked=%s reason=%s validator=%s",
            blocked, reason, self._v["hotkey"],
        )

    def log_priority(self, score: float) -> None:
        logger.info(
            "[SYNAPSE:PRIORITY] score=%.4f validator=%s",
            score, self._v["hotkey"],
        )

    def log_inference(self, risk_scores: list[float], predictions: list[bool], latency_ms: float) -> None:
        bot_count = sum(predictions)
        scores_str = " ".join(f"{s:.3f}" for s in risk_scores)
        logger.info(
            "[SYNAPSE:INFERENCE] chunks=%d bot=%d human=%d latency=%.0fms "
            "min=%.3f max=%.3f mean=%.3f scores=[%s]",
            len(risk_scores), bot_count, len(predictions) - bot_count, latency_ms,
            min(risk_scores) if risk_scores else 0,
            max(risk_scores) if risk_scores else 0,
            sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            scores_str,
        )

    def log_response(self, risk_scores: list[float], predictions: list[bool]) -> None:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        logger.info(
            "[SYNAPSE:RESPONSE] chunks=%d bot=%d total_latency=%.0fms validator=%s",
            len(risk_scores), sum(predictions), total_ms, self._v["hotkey"],
        )

    def log_error(self, exc: Exception) -> None:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        logger.error(
            "[SYNAPSE:ERROR] %s: %s validator=%s latency=%.0fms",
            type(exc).__name__, exc, self._v["hotkey"], total_ms,
        )
