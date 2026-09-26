"""
Crypto Signal / Paper-Trading Bot -- single-file version.

Everything (config, exchange wrapper, strategy, backtester, paper trader,
background scheduler, FastAPI routes) lives in this one file so it can be
uploaded to GitHub as a single flat file from a phone, with zero folder
structure required.

=====================================================================
IMPORTANT: No trading strategy has a guaranteed win rate. Nobody can
promise "75%+ profitable." Use the /backtest endpoint below to see this
strategy's ACTUAL historical performance on real data before trusting it
with real money. This is educational software, not financial advice.
=====================================================================

--- Deploying on Render (no other files needed) ---
1. Push this single main.py to a new GitHub repo (just upload the file).
2. On render.com: New -> Web Service -> connect that repo.
3. Environment: Python 3
   Build Command:  pip install fastapi "uvicorn[standard]" pydantic ccxt pandas numpy
   Start Command:  uvicorn main:app --host 0.0.0.0 --port $PORT
4. Add environment variables in the Render dashboard (all optional, sensible
   defaults are baked in below): TRADING_MODE, DEFAULT_SYMBOL,
   DEFAULT_TIMEFRAME, EXCHANGE_API_KEY, EXCHANGE_API_SECRET, API_CONTROL_KEY.
5. Deploy. Visit https://<your-app>.onrender.com/ for the dashboard
   (or /docs for the raw interactive API reference).
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from urllib.parse import unquote

import ccxt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger("cryptobot")
logging.basicConfig(level=logging.INFO)

# =====================================================================
# CONFIG -- all overridable via environment variables in Render's dashboard
# =====================================================================


class Settings(BaseModel):
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    api_key: str = os.getenv("EXCHANGE_API_KEY", "")
    api_secret: str = os.getenv("EXCHANGE_API_SECRET", "")

    trading_mode: str = os.getenv("TRADING_MODE", "PAPER")  # PAPER or LIVE

    default_symbol: str = os.getenv("DEFAULT_SYMBOL", "BTC/USDT")
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "1h")

    ema_fast: int = int(os.getenv("EMA_FAST", "12"))
    ema_slow: int = int(os.getenv("EMA_SLOW", "26"))
    rsi_period: int = int(os.getenv("RSI_PERIOD", "14"))
    rsi_long_max: float = float(os.getenv("RSI_LONG_MAX", "70"))
    rsi_short_min: float = float(os.getenv("RSI_SHORT_MIN", "30"))
    atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
    atr_stop_mult: float = float(os.getenv("ATR_STOP_MULT", "2.0"))
    atr_target_mult: float = float(os.getenv("ATR_TARGET_MULT", "3.0"))

    paper_starting_balance: float = float(os.getenv("PAPER_STARTING_BALANCE", "10000"))
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

    poll_seconds: int = int(os.getenv("POLL_SECONDS", "60"))
    api_control_key: str = os.getenv("API_CONTROL_KEY", "")


settings = Settings()

# =====================================================================
# EXCHANGE -- ccxt wrapper for fetching candles and (LIVE mode only) orders
# =====================================================================


def get_exchange():
    exchange_class = getattr(ccxt, settings.exchange_id)
    return exchange_class({
        "apiKey": settings.api_key,
        "secret": settings.api_secret,
        "enableRateLimit": True,
    })


def _friendly_exchange_error(e: Exception) -> Exception:
    """
    Binance (and hosts like Render that share outbound IPs across many
    customers' apps) sometimes returns a short-lived HTTP 418/429 ban on
    the whole IP, not just this app. Give a clear, honest message instead
    of the raw ccxt exception text.
    """
    msg = str(e)
    if "418" in msg or "-1003" in msg or isinstance(e, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)):
        return RuntimeError(
            "Binance is temporarily rate-limiting requests from this server's "
            "shared hosting IP (common on free-tier hosts, and not specific to "
            "your usage). It self-clears within a few minutes -- just try again shortly."
        )
    return e


def _with_retry(fn, *args, retries: int = 3, base_delay: float = 1.5, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable, ccxt.NetworkError) as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
        except ccxt.ExchangeError as e:
            # Binance's 418 ban surfaces as a generic ExchangeError with "418"/"-1003" in the message
            if "418" in str(e) or "-1003" in str(e):
                last_err = e
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise _friendly_exchange_error(e)
    raise _friendly_exchange_error(last_err)


def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    exchange = get_exchange()
    raw = _with_retry(exchange.fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def fetch_ohlcv_range(symbol: str, timeframe: str, start_iso: str, end_iso: str | None = None) -> pd.DataFrame:
    exchange = get_exchange()
    since = exchange.parse8601(f"{start_iso}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end_iso}T00:00:00Z") if end_iso else exchange.milliseconds()

    all_rows = []
    limit = 1000
    while since < end_ts:
        batch = _with_retry(exchange.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == since:
            break
        since = last_ts + 1
        if len(batch) < limit:
            break

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    if end_iso:
        df = df[df.index <= pd.to_datetime(end_iso, utc=True)]
    return df


def place_live_market_order(symbol: str, side: str, amount: float):
    if settings.trading_mode != "LIVE":
        raise RuntimeError("Refusing to place a live order while TRADING_MODE != LIVE")
    exchange = get_exchange()
    return exchange.create_order(symbol=symbol, type="market", side=side, amount=amount)


# =====================================================================
# STRATEGY -- EMA crossover + RSI filter + ATR-based stop/target
# =====================================================================


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], settings.ema_fast)
    out["ema_slow"] = ema(out["close"], settings.ema_slow)
    out["rsi"] = rsi(out["close"], settings.rsi_period)
    out["atr"] = atr(out, settings.atr_period)
    return out


def generate_signals(df: pd.DataFrame, allow_short: bool = False) -> pd.DataFrame:
    out = add_indicators(df)

    cross_up = (out["ema_fast"] > out["ema_slow"]) & (out["ema_fast"].shift(1) <= out["ema_slow"].shift(1))
    cross_down = (out["ema_fast"] < out["ema_slow"]) & (out["ema_fast"].shift(1) >= out["ema_slow"].shift(1))

    long_signal = cross_up & (out["rsi"] < settings.rsi_long_max)
    short_signal = cross_down & (out["rsi"] > settings.rsi_short_min) if allow_short else cross_down

    out["signal"] = 0
    out.loc[long_signal, "signal"] = 1
    out.loc[short_signal, "signal"] = -1

    out["stop_price"] = np.where(
        out["signal"] == 1, out["close"] - settings.atr_stop_mult * out["atr"],
        np.where(out["signal"] == -1, out["close"] + settings.atr_stop_mult * out["atr"], np.nan),
    )
    out["target_price"] = np.where(
        out["signal"] == 1, out["close"] + settings.atr_target_mult * out["atr"],
        np.where(out["signal"] == -1, out["close"] - settings.atr_target_mult * out["atr"], np.nan),
    )
    return out


def latest_signal(df_with_indicators: pd.DataFrame) -> dict:
    last = df_with_indicators.iloc[-1]
    return {
        "timestamp": df_with_indicators.index[-1].isoformat(),
        "close": float(last["close"]),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]),
        "signal": int(last["signal"]) if "signal" in last and not pd.isna(last["signal"]) else 0,
        "trend": "up" if last["ema_fast"] > last["ema_slow"] else "down",
    }


# =====================================================================
# BACKTESTER -- run the strategy over real history, report real metrics
# =====================================================================


@dataclass
class Trade:
    side: str
    entry_time: str
    entry_price: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    pnl_quote: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    timeframe: str
    start: str
    end: str
    starting_balance: float
    ending_balance: float
    total_return_pct: float
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    trades: list = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    starting_balance: float = 10000.0,
    risk_per_trade_pct: float = 1.0,
    fee_pct: float = 0.1,
    slippage_pct: float = 0.05,
    allow_short: bool = False,
) -> BacktestResult:
    sig_df = generate_signals(df, allow_short=allow_short)

    balance = starting_balance
    equity_curve = [balance]
    position = None
    trades: list[Trade] = []
    cost_pct = (fee_pct + slippage_pct) / 100.0

    for ts, row in sig_df.iterrows():
        price = row["close"]

        if position is not None:
            hit_stop = (price <= position["stop"]) if position["side"] == "long" else (price >= position["stop"])
            hit_target = (price >= position["target"]) if position["side"] == "long" else (price <= position["target"])
            opposite_signal = (row["signal"] == -1 and position["side"] == "long") or \
                               (row["signal"] == 1 and position["side"] == "short")

            if hit_stop or hit_target or opposite_signal:
                exit_price = price * (1 - cost_pct) if position["side"] == "long" else price * (1 + cost_pct)
                direction = 1 if position["side"] == "long" else -1
                pnl_pct = direction * (exit_price - position["entry_price"]) / position["entry_price"] * 100
                pnl_quote = position["size"] * pnl_pct / 100
                balance += pnl_quote

                reason = "stop" if hit_stop else ("target" if hit_target else "signal_flip")
                trades.append(Trade(
                    side=position["side"], entry_time=position["entry_time"],
                    entry_price=float(position["entry_price"]), exit_time=ts.isoformat(),
                    exit_price=float(exit_price), exit_reason=reason,
                    pnl_pct=float(pnl_pct), pnl_quote=float(pnl_quote),
                ))
                position = None

        if position is None and row["signal"] in (1, -1) and not np.isnan(row.get("stop_price", np.nan)):
            if row["signal"] == -1 and not allow_short:
                pass
            else:
                side = "long" if row["signal"] == 1 else "short"
                entry_price = price * (1 + cost_pct) if side == "long" else price * (1 - cost_pct)
                risk_amount = balance * risk_per_trade_pct / 100
                stop_dist = abs(entry_price - row["stop_price"])
                size = (risk_amount / stop_dist) * entry_price if stop_dist > 0 else 0
                size = min(size, balance)
                position = {
                    "side": side, "entry_price": entry_price, "entry_time": ts.isoformat(),
                    "stop": row["stop_price"], "target": row["target_price"], "size": size,
                }

        equity_curve.append(balance)

    eq = pd.Series(equity_curve)
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max * 100
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    wins = [t for t in trades if t.pnl_quote > 0]
    losses = [t for t in trades if t.pnl_quote <= 0]
    win_rate = (len(wins) / len(trades) * 100) if trades else 0.0
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    gross_profit = sum(t.pnl_quote for t in wins)
    gross_loss = abs(sum(t.pnl_quote for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    return BacktestResult(
        symbol=symbol, timeframe=timeframe,
        start=df.index[0].isoformat() if len(df) else "",
        end=df.index[-1].isoformat() if len(df) else "",
        starting_balance=starting_balance, ending_balance=round(balance, 2),
        total_return_pct=round((balance - starting_balance) / starting_balance * 100, 2),
        num_trades=len(trades), win_rate_pct=round(win_rate, 2),
        avg_win_pct=round(avg_win, 2), avg_loss_pct=round(avg_loss, 2),
        profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else -1,
        max_drawdown_pct=round(max_dd, 2),
        trades=[t.__dict__ for t in trades],
    )


# =====================================================================
# PAPER TRADER -- in-memory simulated portfolio, fills against live prices
# =====================================================================


@dataclass
class PaperPosition:
    symbol: str
    side: str
    entry_price: float
    size: float
    stop: float
    target: float
    opened_at: float


@dataclass
class PaperAccount:
    balance: float = settings.paper_starting_balance
    equity: float = settings.paper_starting_balance
    position: PaperPosition | None = None
    trade_log: list = field(default_factory=list)
    running: dict = field(default_factory=dict)


paper_account = PaperAccount()


def _fee_adjusted(price: float, side: str, opening: bool) -> float:
    cost_pct = 0.0015
    if (side == "long" and opening) or (side == "short" and not opening):
        return price * (1 + cost_pct)
    return price * (1 - cost_pct)


def paper_open(symbol: str, side: str, price: float, stop: float, target: float):
    if paper_account.position is not None:
        return {"status": "skipped", "reason": "position already open"}
    risk_amount = paper_account.balance * settings.risk_per_trade_pct / 100
    stop_dist = abs(price - stop)
    size = (risk_amount / stop_dist) * price if stop_dist > 0 else 0
    size = min(size, paper_account.balance)
    entry_price = _fee_adjusted(price, side, opening=True)
    paper_account.position = PaperPosition(
        symbol=symbol, side=side, entry_price=entry_price, size=size,
        stop=stop, target=target, opened_at=time.time(),
    )
    return {"status": "opened", "position": paper_account.position.__dict__}


def paper_close(price: float, reason: str):
    pos = paper_account.position
    if pos is None:
        return {"status": "skipped", "reason": "no open position"}
    exit_price = _fee_adjusted(price, pos.side, opening=False)
    direction = 1 if pos.side == "long" else -1
    pnl_pct = direction * (exit_price - pos.entry_price) / pos.entry_price * 100
    pnl_quote = pos.size * pnl_pct / 100
    paper_account.balance += pnl_quote
    paper_account.equity = paper_account.balance
    record = {
        "symbol": pos.symbol, "side": pos.side, "entry_price": pos.entry_price,
        "exit_price": exit_price, "size": pos.size, "pnl_pct": round(pnl_pct, 3),
        "pnl_quote": round(pnl_quote, 2), "reason": reason,
        "opened_at": pos.opened_at, "closed_at": time.time(),
    }
    paper_account.trade_log.append(record)
    paper_account.position = None
    return {"status": "closed", "trade": record}


def paper_mark_to_market(price: float):
    if paper_account.position is None:
        paper_account.equity = paper_account.balance
        return
    pos = paper_account.position
    direction = 1 if pos.side == "long" else -1
    unrealized_pct = direction * (price - pos.entry_price) / pos.entry_price * 100
    paper_account.equity = paper_account.balance + pos.size * unrealized_pct / 100


def paper_status() -> dict:
    return {
        "balance": round(paper_account.balance, 2),
        "equity": round(paper_account.equity, 2),
        "open_position": paper_account.position.__dict__ if paper_account.position else None,
        "num_trades": len(paper_account.trade_log),
        "trade_log": paper_account.trade_log[-50:],
        "running_symbols": [s for s, r in paper_account.running.items() if r],
    }


# =====================================================================
# SCHEDULER -- background loop polling prices and driving the paper trader
# =====================================================================

_tasks: dict[str, asyncio.Task] = {}


async def _loop(symbol: str, timeframe: str):
    paper_account.running[symbol] = True
    logger.info(f"Started signal loop for {symbol} ({timeframe}) in {settings.trading_mode} mode")
    try:
        while True:
            try:
                df = fetch_ohlcv_df(symbol, timeframe, limit=300)
                sig_df = generate_signals(df)
                last = sig_df.iloc[-1]
                price = float(last["close"])

                paper_mark_to_market(price)
                pos = paper_account.position

                if pos is not None:
                    hit_stop = price <= pos.stop if pos.side == "long" else price >= pos.stop
                    hit_target = price >= pos.target if pos.side == "long" else price <= pos.target
                    opposite = (last["signal"] == -1 and pos.side == "long") or \
                               (last["signal"] == 1 and pos.side == "short")
                    if hit_stop or hit_target or opposite:
                        reason = "stop" if hit_stop else ("target" if hit_target else "signal_flip")
                        if settings.trading_mode == "LIVE":
                            place_live_market_order(symbol, "sell" if pos.side == "long" else "buy", pos.size / price)
                        paper_close(price, reason)

                elif last["signal"] == 1:
                    paper_open(symbol, "long", price, float(last["stop_price"]), float(last["target_price"]))

            except Exception as e:
                logger.exception(f"Loop error for {symbol}: {e}")

            await asyncio.sleep(settings.poll_seconds)
    finally:
        paper_account.running[symbol] = False


def scheduler_start(symbol: str, timeframe: str) -> bool:
    if symbol in _tasks and not _tasks[symbol].done():
        return False
    _tasks[symbol] = asyncio.create_task(_loop(symbol, timeframe))
    return True


def scheduler_stop(symbol: str) -> bool:
    task = _tasks.get(symbol)
    if task and not task.done():
        task.cancel()
        return True
    return False


# =====================================================================
# FASTAPI APP
# =====================================================================

app = FastAPI(
    title="Crypto Signal / Paper-Trading Bot",
    description="EMA/RSI/ATR trend-following bot with a real backtester. "
                "Paper trading by default. Not financial advice.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Always return JSON, never a bare "Internal Server Error" text response --
    # the dashboard's JS always expects r.json() to succeed.
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


def _check_key(x_api_key: str | None):
    if settings.api_control_key and x_api_key != settings.api_control_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Ledger &mdash; Signal Terminal</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect width='24' height='24' fill='%230a0e14'/%3E%3Cpath d='M4 17l4-6 4 3 5-8 3 4' stroke='%23e2a33d' stroke-width='2' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600;700&amp;display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0a0e14; --bg-raised:#0d131b; --line:#1c2530; --line-soft:#161e28;
    --text:#e6ecf2; --muted:#6e7a88; --muted-2:#4a545f;
    --accent:#e2a33d; --accent-dim:#7a5a26;
    --pos:#23c586; --neg:#ff6b5f;
    --sans:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box;}
  ::selection{background:var(--accent-dim); color:#fff;}
  body{
    margin:0; background:var(--bg); color:var(--text); font-family:var(--sans);
    padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
    -webkit-font-smoothing:antialiased;
  }
  a{color:var(--accent);}

  /* ---------- Header / brand ---------- */
  header{
    display:flex; align-items:center; justify-content:space-between; gap:12px;
    padding:16px 18px; border-bottom:1px solid var(--line);
  }
  .brand{display:flex; align-items:center; gap:10px; min-width:0;}
  .brand svg{flex:none;}
  .brand-name{font-weight:600; font-size:16px; letter-spacing:.01em; white-space:nowrap;}
  .brand-sub{color:var(--muted); font-size:11px; margin-top:1px;}
  .status{display:flex; align-items:center; gap:8px; flex:none;}
  .dot{width:7px; height:7px; border-radius:50%; background:var(--muted-2); transition:background .3s;}
  .dot.ok{background:var(--pos); box-shadow:0 0 0 3px rgba(35,197,134,.15);}
  .dot.bad{background:var(--neg); box-shadow:0 0 0 3px rgba(255,107,95,.15);}
  .mode{
    font-family:var(--mono); font-size:11px; padding:3px 8px; border:1px solid var(--line);
    color:var(--muted); letter-spacing:.02em;
  }
  .mode.live{color:var(--neg); border-color:rgba(255,107,95,.35);}
  .mode.paper{color:var(--accent); border-color:rgba(226,163,61,.35);}

  /* ---------- Ticker strip ---------- */
  .ticker{
    display:grid; grid-template-columns:repeat(4,1fr);
    border-bottom:1px solid var(--line);
  }
  .ticker .cell{
    padding:14px 16px; border-right:1px solid var(--line);
  }
  .ticker .cell:last-child{border-right:none;}
  .ticker .cell .label{color:var(--muted); font-size:11px; margin-bottom:5px;}
  .ticker .cell .num{font-family:var(--mono); font-size:19px; font-weight:500; letter-spacing:-.01em;}
  .num.pos{color:var(--pos);} .num.neg{color:var(--neg);}

  /* ---------- Content sections ---------- */
  main{max-width:760px; margin:0 auto; padding:0 0 32px;}
  section{padding:22px 18px; border-bottom:1px solid var(--line);}
  section h2{
    font-size:13px; font-weight:600; margin:0 0 14px; color:var(--accent);
    display:flex; align-items:center; gap:7px;
  }
  section h2 .glyph{font-family:var(--mono); font-weight:400; color:var(--muted-2);}
  .hint{color:var(--muted); font-size:12.5px; line-height:1.5;}

  .row{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;}
  input,select{
    background:var(--bg-raised); border:1px solid var(--line); color:var(--text);
    padding:9px 10px; font-size:13px; flex:1; min-width:96px; font-family:var(--sans);
  }
  input:focus,select:focus{outline:none; border-color:var(--accent);}
  input::placeholder{color:var(--muted-2);}

  button{
    background:var(--accent); color:#1a1206; border:none; padding:9px 16px;
    font-size:13px; font-weight:600; cursor:pointer; font-family:var(--sans);
  }
  button:hover{background:#eeb45c;}
  button.ghost{background:transparent; border:1px solid var(--line); color:var(--text);}
  button.ghost:hover{border-color:var(--muted);}
  button.stop{background:transparent; border:1px solid rgba(255,107,95,.4); color:var(--neg);}
  button.stop:hover{background:rgba(255,107,95,.08);}
  button:disabled{opacity:.45; cursor:default;}

  table{width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12px;}
  th,td{text-align:left; padding:8px 10px; border-bottom:1px solid var(--line-soft); white-space:nowrap;}
  th{color:var(--muted); font-weight:400; font-size:10.5px; letter-spacing:.03em; font-family:var(--sans);}

  .position-line{
    font-family:var(--mono); font-size:13px; padding:12px 14px; background:var(--bg-raised);
    border-left:2px solid var(--accent);
  }
  .position-empty{color:var(--muted); font-size:13px;}

  .result-strip{display:flex; flex-wrap:wrap; border:1px solid var(--line); margin:14px 0;}
  .result-strip .cell{flex:1; min-width:100px; padding:12px 14px; border-right:1px solid var(--line);}
  .result-strip .cell:last-child{border-right:none;}
  .result-strip .label{color:var(--muted); font-size:10.5px; margin-bottom:4px;}
  .result-strip .num{font-family:var(--mono); font-size:16px;}

  canvas{max-height:180px; margin-top:6px;}
  .scroll-x{overflow-x:auto;}

  .disclaimer{
    color:var(--muted-2); font-size:11px; text-align:center; padding:20px 18px 28px; line-height:1.6;
  }

  .toast{
    position:fixed; bottom:calc(16px + env(safe-area-inset-bottom,0px)); left:50%;
    transform:translateX(-50%); background:var(--bg-raised); border:1px solid var(--line);
    padding:10px 16px; font-size:13px; display:none; z-index:50; font-family:var(--mono);
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
      <path d="M4 17l4-6 4 3 5-8 3 4" stroke="#e2a33d" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <div>
      <div class="brand-name">Ledger</div>
      <div class="brand-sub">Signal terminal</div>
    </div>
  </div>
  <div class="status">
    <span id="health-dot" class="dot bad"></span>
    <span id="mode-badge" class="mode paper">&mdash;</span>
  </div>
</header>

<div class="ticker">
  <div class="cell"><div class="label">Balance</div><div class="num" id="balance">&mdash;</div></div>
  <div class="cell"><div class="label">Equity</div><div class="num" id="equity">&mdash;</div></div>
  <div class="cell"><div class="label">Open P&amp;L</div><div class="num" id="open-pnl">&mdash;</div></div>
  <div class="cell"><div class="label">Trades</div><div class="num" id="num-trades">&mdash;</div></div>
</div>

<main>

  <section>
    <h2><span class="glyph">01</span> Live signal</h2>
    <div class="row">
      <input id="sig-symbol" value="BTC/USDT" placeholder="BTC/USDT">
      <select id="sig-timeframe">
        <option value="15m">15m</option><option value="1h" selected>1h</option>
        <option value="4h">4h</option><option value="1d">1d</option>
      </select>
      <button onclick="checkSignal()">Check</button>
    </div>
    <div id="signal-out" class="hint">Tap Check to fetch the current signal.</div>
  </section>

  <section>
    <h2><span class="glyph">02</span> Paper trading</h2>
    <div class="row">
      <input id="ctl-symbol" value="BTC/USDT" placeholder="BTC/USDT">
      <select id="ctl-timeframe">
        <option value="15m">15m</option><option value="1h" selected>1h</option>
        <option value="4h">4h</option><option value="1d">1d</option>
      </select>
    </div>
    <div class="row">
      <input id="api-key" type="password" placeholder="API control key (if set on server)">
    </div>
    <div class="row">
      <button onclick="startPaper()">Start</button>
      <button class="stop" onclick="stopPaper()">Stop</button>
      <button class="ghost" onclick="refreshStatus()">Refresh now</button>
    </div>
    <div id="ctl-out" class="hint"></div>
    <div class="hint" id="running-list" style="margin-top:4px;"></div>
  </section>

  <section>
    <h2><span class="glyph">03</span> Open position</h2>
    <div id="position-out" class="position-empty">No open position.</div>
  </section>

  <section>
    <h2><span class="glyph">04</span> Trade log</h2>
    <div class="scroll-x">
      <table id="trade-table">
        <thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&amp;L %</th><th>P&amp;L</th><th>Reason</th></tr></thead>
        <tbody><tr><td colspan="7" class="hint">No trades yet.</td></tr></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2><span class="glyph">05</span> Backtest</h2>
    <div class="row">
      <input id="bt-symbol" value="BTC/USDT" placeholder="BTC/USDT">
      <select id="bt-timeframe">
        <option value="15m">15m</option><option value="1h" selected>1h</option>
        <option value="4h">4h</option><option value="1d">1d</option>
      </select>
    </div>
    <div class="row">
      <input id="bt-start" type="date" value="2024-01-01">
      <input id="bt-end" type="date" placeholder="End (optional)">
    </div>
    <div class="row">
      <input id="bt-balance" type="number" value="10000" placeholder="Starting balance">
      <input id="bt-risk" type="number" value="1" step="0.1" placeholder="Risk % / trade">
    </div>
    <button onclick="runBacktest()" id="bt-btn">Run backtest</button>
    <div id="bt-status" class="hint" style="margin-top:10px;"></div>
    <div id="bt-results" style="display:none;">
      <div class="result-strip">
        <div class="cell"><div class="label">Win rate</div><div class="num" id="bt-winrate">&mdash;</div></div>
        <div class="cell"><div class="label">Return</div><div class="num" id="bt-return">&mdash;</div></div>
        <div class="cell"><div class="label">Max drawdown</div><div class="num" id="bt-dd">&mdash;</div></div>
        <div class="cell"><div class="label">Profit factor</div><div class="num" id="bt-pf">&mdash;</div></div>
        <div class="cell"><div class="label">Trades</div><div class="num" id="bt-num">&mdash;</div></div>
      </div>
      <canvas id="bt-chart"></canvas>
    </div>
  </section>

  <div class="disclaimer">
    Paper trading / educational tool. No strategy has a guaranteed win rate.<br>Not financial advice.
  </div>
</main>

<div class="toast" id="toast"></div>

<script>
const fmt = n => (n === null || n === undefined || isNaN(n)) ? '--' : Number(n).toLocaleString(undefined,{maximumFractionDigits:2});
let btChart = null;

function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display = 'block';
  clearTimeout(t._h); t._h = setTimeout(()=>t.style.display='none', 3000);
}

async function refreshHealth(){
  try{
    const r = await fetch('/health'); const d = await r.json();
    document.getElementById('health-dot').className = 'dot ok';
    const badge = document.getElementById('mode-badge');
    badge.textContent = d.mode;
    badge.className = 'mode ' + (d.mode === 'LIVE' ? 'live' : 'paper');
  }catch(e){
    document.getElementById('health-dot').className = 'dot bad';
  }
}

async function refreshStatus(){
  try{
    const r = await fetch('/paper/status'); const d = await r.json();
    document.getElementById('balance').textContent = '$' + fmt(d.balance);
    document.getElementById('equity').textContent = '$' + fmt(d.equity);
    document.getElementById('num-trades').textContent = d.num_trades;

    const openPnl = d.equity - d.balance;
    const pnlEl = document.getElementById('open-pnl');
    pnlEl.textContent = (openPnl >= 0 ? '+' : '') + '$' + fmt(openPnl);
    pnlEl.className = 'num ' + (openPnl > 0 ? 'pos' : (openPnl < 0 ? 'neg' : ''));

    const posOut = document.getElementById('position-out');
    if(d.open_position){
      const p = d.open_position;
      posOut.className = 'position-line';
      posOut.innerHTML = `${p.symbol} &nbsp;${p.side.toUpperCase()}&nbsp;
        &nbsp;| entry $${fmt(p.entry_price)} &nbsp;stop $${fmt(p.stop)} &nbsp;target $${fmt(p.target)}
        &nbsp;| size $${fmt(p.size)}`;
    } else {
      posOut.className = 'position-empty';
      posOut.textContent = 'No open position.';
    }

    const tbody = document.querySelector('#trade-table tbody');
    if(d.trade_log && d.trade_log.length){
      tbody.innerHTML = d.trade_log.slice().reverse().map(t => `
        <tr>
          <td>${t.symbol}</td>
          <td>${t.side}</td>
          <td>$${fmt(t.entry_price)}</td>
          <td>$${fmt(t.exit_price)}</td>
          <td style="color:${t.pnl_pct>=0?'var(--pos)':'var(--neg)'}">${t.pnl_pct>=0?'+':''}${fmt(t.pnl_pct)}%</td>
          <td style="color:${t.pnl_quote>=0?'var(--pos)':'var(--neg)'}">${t.pnl_quote>=0?'+':''}$${fmt(t.pnl_quote)}</td>
          <td>${t.reason}</td>
        </tr>`).join('');
    } else {
      tbody.innerHTML = '<tr><td colspan="7" class="hint">No trades yet.</td></tr>';
    }

    document.getElementById('running-list').textContent =
      (d.running_symbols && d.running_symbols.length)
        ? 'Running: ' + d.running_symbols.join(', ')
        : 'Not currently running.';
  }catch(e){
    console.error(e);
  }
}

async function checkSignal(){
  const symbol = document.getElementById('sig-symbol').value.trim();
  const timeframe = document.getElementById('sig-timeframe').value;
  const out = document.getElementById('signal-out');
  out.textContent = 'Loading...';
  try{
    const r = await fetch(`/signal?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}`);
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'error');
    const sigLabel = d.signal === 1 ? 'LONG' : (d.signal === -1 ? 'EXIT / SHORT' : 'none');
    const sigColor = d.signal === 1 ? 'var(--pos)' : (d.signal === -1 ? 'var(--neg)' : 'var(--muted)');
    out.innerHTML = `price $${fmt(d.close)} &nbsp;&nbsp;trend ${d.trend} &nbsp;&nbsp;
      rsi ${fmt(d.rsi)} &nbsp;&nbsp;signal <span style="color:${sigColor}">${sigLabel}</span>`;
  }catch(e){
    out.textContent = 'Error: ' + e.message;
  }
}

async function startPaper(){
  const symbol = document.getElementById('ctl-symbol').value.trim();
  const timeframe = document.getElementById('ctl-timeframe').value;
  const key = document.getElementById('api-key').value;
  const out = document.getElementById('ctl-out');
  try{
    const r = await fetch('/paper/start', {
      method:'POST',
      headers:{'Content-Type':'application/json', 'X-API-Key': key},
      body: JSON.stringify({symbol, timeframe})
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'error');
    out.textContent = d.started ? `Started ${d.symbol} (${d.timeframe}).` : `${d.symbol} already running.`;
    toast(out.textContent);
    refreshStatus();
  }catch(e){ out.textContent = 'Error: ' + e.message; toast('Failed to start'); }
}

async function stopPaper(){
  const symbol = document.getElementById('ctl-symbol').value.trim();
  const key = document.getElementById('api-key').value;
  const out = document.getElementById('ctl-out');
  try{
    const r = await fetch(`/paper/stop?symbol=${encodeURIComponent(symbol)}`, {
      method:'POST', headers:{'X-API-Key': key}
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'error');
    out.textContent = d.stopped ? `Stopped ${d.symbol}.` : `${d.symbol} was not running.`;
    toast(out.textContent);
    refreshStatus();
  }catch(e){ out.textContent = 'Error: ' + e.message; toast('Failed to stop'); }
}

async function runBacktest(){
  const btn = document.getElementById('bt-btn');
  const status = document.getElementById('bt-status');
  btn.disabled = true; status.textContent = 'Running backtest against real historical data...';
  document.getElementById('bt-results').style.display = 'none';

  const body = {
    symbol: document.getElementById('bt-symbol').value.trim(),
    timeframe: document.getElementById('bt-timeframe').value,
    start: document.getElementById('bt-start').value,
    end: document.getElementById('bt-end').value || null,
    starting_balance: parseFloat(document.getElementById('bt-balance').value) || 10000,
    risk_per_trade_pct: parseFloat(document.getElementById('bt-risk').value) || 1,
  };

  try{
    const r = await fetch('/backtest', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'error');

    document.getElementById('bt-winrate').textContent = fmt(d.win_rate_pct) + '%';
    const retEl = document.getElementById('bt-return');
    retEl.textContent = (d.total_return_pct>=0?'+':'') + fmt(d.total_return_pct) + '%';
    retEl.className = 'num ' + (d.total_return_pct>=0?'pos':'neg');
    document.getElementById('bt-dd').textContent = fmt(d.max_drawdown_pct) + '%';
    document.getElementById('bt-pf').textContent = d.profit_factor === -1 ? '&#8734;' : fmt(d.profit_factor);
    document.getElementById('bt-num').textContent = d.num_trades;
    document.getElementById('bt-results').style.display = 'block';
    status.textContent = `${d.symbol} ${d.timeframe}, ${d.start.slice(0,10)} to ${d.end.slice(0,10)}.`;

    let bal = d.starting_balance;
    const points = [bal];
    const labels = ['start'];
    (d.trades || []).forEach((t,i) => { bal += t.pnl_quote; points.push(bal); labels.push(i+1); });

    const ctx = document.getElementById('bt-chart').getContext('2d');
    if(btChart) btChart.destroy();
    btChart = new Chart(ctx, {
      type:'line',
      data:{ labels, datasets:[{ data: points, borderColor:'#e2a33d', backgroundColor:'rgba(226,163,61,.08)',
        fill:true, tension:0.1, pointRadius:0, borderWidth:1.5 }]},
      options:{ responsive:true, plugins:{legend:{display:false}},
        scales:{ x:{display:false}, y:{ticks:{color:'#6e7a88', font:{family:'IBM Plex Mono', size:10}}, grid:{color:'#1c2530'}} } }
    });
  }catch(e){
    status.textContent = 'Error: ' + e.message;
  }finally{
    btn.disabled = false;
  }
}

refreshHealth();
refreshStatus();
setInterval(refreshHealth, 15000);
setInterval(refreshStatus, 10000);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "ok", "mode": settings.trading_mode}


@app.get("/signal")
def get_signal(symbol: str = Query(...), timeframe: str = Query(default=None)):
    symbol = unquote(symbol)
    tf = timeframe or settings.default_timeframe
    try:
        df = fetch_ohlcv_df(symbol, tf, limit=300)
        sig_df = generate_signals(df)
        return latest_signal(sig_df)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Exchange/data error: {e}")


class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start: str
    end: str | None = None
    starting_balance: float = 10000.0
    risk_per_trade_pct: float = 1.0
    allow_short: bool = False


@app.post("/backtest")
def backtest(req: BacktestRequest):
    try:
        df = fetch_ohlcv_range(req.symbol, req.timeframe, req.start, req.end)
        if len(df) < 50:
            raise HTTPException(status_code=400, detail="Not enough candles returned for that range")
        result = run_backtest(
            df, req.symbol, req.timeframe,
            starting_balance=req.starting_balance,
            risk_per_trade_pct=req.risk_per_trade_pct,
            allow_short=req.allow_short,
        )
        return result.__dict__
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backtest error: {e}")


class PaperStartRequest(BaseModel):
    symbol: str | None = None
    timeframe: str | None = None


@app.post("/paper/start")
async def paper_start_route(req: PaperStartRequest, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    symbol = req.symbol or settings.default_symbol
    timeframe = req.timeframe or settings.default_timeframe
    started = scheduler_start(symbol, timeframe)
    return {"started": started, "symbol": symbol, "timeframe": timeframe}


@app.post("/paper/stop")
async def paper_stop_route(symbol: str = Query(default=None), x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    sym = symbol or settings.default_symbol
    stopped = scheduler_stop(sym)
    return {"stopped": stopped, "symbol": sym}


@app.get("/paper/status")
def paper_status_route():
    return paper_status()
