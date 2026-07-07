"""Structured Axon and per-synapse logger.

Valid synapse lifecycle events and raw HTTP requests are written to the root
logger (`poker44-miner.log`). Raw HTTP logging captures requests that Bittensor
rejects before `forward()` is called.

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
import os
import time
from typing import Any

logger = logging.getLogger("poker44.synapse")
http_logger = logging.getLogger("poker44.axon_http")


def configure(log_dir: str | Any = "logs") -> None:
    """Configure synapse and raw Axon HTTP logging."""
    import pathlib, datetime as _dt

    log_path = pathlib.Path(str(log_dir)).resolve() / "poker44-miner.log"

    _UTC8 = _dt.timezone(_dt.timedelta(hours=-8))

    class _UTC8Formatter(logging.Formatter):
        def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
            ct = _dt.datetime.fromtimestamp(record.created, tz=_UTC8)
            return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S")

    formatter = _UTC8Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for lg in (logger, http_logger):
        lg.setLevel(logging.INFO)
        lg.propagate = False  # don't propagate — write directly, immune to root level
        # Remove stale handlers pointing to same file (avoids duplicates on reload)
        lg.handlers = [
            h for h in lg.handlers
            if not (isinstance(h, logging.FileHandler)
                    and pathlib.Path(h.baseFilename).resolve() == log_path)
        ]
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        lg.addHandler(fh)

    logger.info("Synapse log recording enabled")
    http_logger.info("Raw Axon HTTP log recording enabled in project log")


def _write_direct(msg: str) -> None:
    """Write directly to the log file, bypassing Python logging entirely."""
    import pathlib, datetime
    _UTC8 = datetime.timezone(datetime.timedelta(hours=-8))
    log_file = pathlib.Path(os.getenv("POKER44_LOG_DIR", "/home/poker44/logs")) / "poker44-miner.log"
    try:
        ts = datetime.datetime.now(tz=_UTC8).strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{ts} INFO [poker44.axon_http] {msg}\n")
    except Exception:
        pass


def install_axon_http_logging(axon: Any) -> None:
    """Install FastAPI middleware to log every inbound Axon HTTP request."""
    if getattr(axon, "_poker44_http_logging_installed", False):
        return

    @axon.app.middleware("http")
    async def poker44_http_logger(request: Any, call_next: Any) -> Any:
        started = time.perf_counter()
        client = getattr(request, "client", None)
        client_host = getattr(client, "host", "unknown")
        client_port = getattr(client, "port", "")
        headers = request.headers
        synapse_name = _header(headers, "bt_header_input_obj_name") or _header(headers, "name")
        dendrite_hotkey = _header(headers, "bt_header_dendrite_hotkey") or _header(headers, "dendrite_hotkey")
        uuid = _header(headers, "bt_header_uuid") or _header(headers, "uuid")
        msg = (
            f"[AXON:REQUEST] method={request.method} path={request.url.path} "
            f"client={client_host}:{client_port} synapse={synapse_name or 'unknown'} "
            f"hotkey={dendrite_hotkey or 'unknown'} uuid={uuid or 'unknown'}"
        )
        _write_direct(msg)
        http_logger.info(msg)
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            status_code = getattr(response, "status_code", "error")
            elapsed_ms = (time.perf_counter() - started) * 1000
            resp_msg = (
                f"[AXON:RESPONSE] method={request.method} path={request.url.path} "
                f"status={status_code} elapsed={elapsed_ms:.0f}ms "
                f"synapse={synapse_name or 'unknown'} hotkey={dendrite_hotkey or 'unknown'}"
            )
            _write_direct(resp_msg)
            http_logger.info(resp_msg)

    axon._poker44_http_logging_installed = True


def _header(headers: Any, name: str) -> str:
    try:
        return str(headers.get(name, "") or headers.get(name.replace("_", "-"), "") or "")
    except Exception:
        return ""


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

    def _log(self, msg: str) -> None:
        logger.info(msg)
        _write_direct(msg)

    def log_received(self, chunk_count: int) -> None:
        chunks = getattr(self.synapse, "chunks", None) or []
        hand_counts = [len(c) for c in chunks if isinstance(c, list)]
        self._log(
            f"[SYNAPSE:RECEIVED] validator={self._v['hotkey']} ip={self._v['ip']} "
            f"chunks={chunk_count} hands={sum(hand_counts)} hand_counts={hand_counts}"
        )

    def log_blacklist(self, blocked: bool, reason: str) -> None:
        msg = f"[SYNAPSE:BLACKLIST] blocked={blocked} reason={reason} validator={self._v['hotkey']}"
        self._log(msg)

    def log_priority(self, score: float) -> None:
        self._log(f"[SYNAPSE:PRIORITY] score={score:.4f} validator={self._v['hotkey']}")

    def log_inference(self, risk_scores: list[float], predictions: list[bool], latency_ms: float, threshold: float = 0.5) -> None:
        bot_count = sum(predictions)
        scores_str = " ".join(f"{s:.3f}" for s in risk_scores)
        self._log(
            f"[SYNAPSE:INFERENCE] chunks={len(risk_scores)} bot={bot_count} "
            f"human={len(predictions)-bot_count} latency={latency_ms:.0f}ms "
            f"threshold={threshold:.3f} "
            f"min={min(risk_scores) if risk_scores else 0:.3f} "
            f"max={max(risk_scores) if risk_scores else 0:.3f} "
            f"mean={sum(risk_scores)/len(risk_scores) if risk_scores else 0:.3f} "
            f"scores=[{scores_str}]"
        )

    def log_response(self, risk_scores: list[float], predictions: list[bool]) -> None:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        self._log(
            f"[SYNAPSE:RESPONSE] chunks={len(risk_scores)} bot={sum(predictions)} "
            f"total_latency={total_ms:.0f}ms validator={self._v['hotkey']}"
        )

    def log_error(self, exc: Exception) -> None:
        total_ms = (time.perf_counter() - self.t_start) * 1000
        msg = (
            f"[SYNAPSE:ERROR] {type(exc).__name__}: {exc} "
            f"validator={self._v['hotkey']} latency={total_ms:.0f}ms"
        )
        logger.error(msg)
        _write_direct(f"ERROR {msg}")
