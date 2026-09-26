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


def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    exchange = get_exchange()
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
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
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
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


def _check_key(x_api_key: str | None):
    if settings.api_control_key and x_api_key != settings.api_control_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Crypto Bot Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0b0f14; --panel:#121820; --panel-border:#232b36; --text:#e6edf3;
    --muted:#8b98a5; --accent:#3fb950; --accent2:#58a6ff; --danger:#f85149;
    --warn:#d29922;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
  }
  header{
    padding:16px 16px 12px; border-bottom:1px solid var(--panel-border);
    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
  }
  header h1{font-size:17px; margin:0; font-weight:600;}
  .badge{
    font-size:11px; padding:3px 9px; border-radius:20px; font-weight:600;
    text-transform:uppercase; letter-spacing:.03em;
  }
  .badge.paper{background:rgba(88,166,255,.15); color:var(--accent2);}
  .badge.live{background:rgba(248,81,73,.15); color:var(--danger);}
  .dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px;}
  .dot.ok{background:var(--accent);} .dot.bad{background:var(--danger);}
  main{padding:14px; max-width:900px; margin:0 auto;}
  .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:16px;}
  .card{background:var(--panel); border:1px solid var(--panel-border); border-radius:10px; padding:12px;}
  .card .label{color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px;}
  .card .value{font-size:20px; font-weight:700;}
  .value.pos{color:var(--accent);} .value.neg{color:var(--danger);}
  section{background:var(--panel); border:1px solid var(--panel-border); border-radius:10px; padding:14px; margin-bottom:14px;}
  section h2{font-size:14px; margin:0 0 10px; color:var(--text); display:flex; align-items:center; gap:8px;}
  .row{display:flex; gap:8px; flex-wrap:wrap; margin-bottom:8px;}
  input,select{
    background:#0d1117; border:1px solid var(--panel-border); color:var(--text);
    padding:8px 10px; border-radius:6px; font-size:13px; flex:1; min-width:100px;
  }
  button{
    background:var(--accent2); color:#04101c; border:none; padding:9px 14px;
    border-radius:6px; font-size:13px; font-weight:600; cursor:pointer;
  }
  button.secondary{background:transparent; border:1px solid var(--panel-border); color:var(--text);}
  button.danger{background:var(--danger); color:#fff;}
  button:disabled{opacity:.5;}
  table{width:100%; border-collapse:collapse; font-size:12px; margin-top:8px;}
  th,td{text-align:left; padding:6px 8px; border-bottom:1px solid var(--panel-border); white-space:nowrap;}
  th{color:var(--muted); font-weight:600; font-size:11px; text-transform:uppercase;}
  .muted{color:var(--muted); font-size:12px;}
  .result-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:8px; margin:10px 0;}
  .result-grid .card{padding:8px;}
  .result-grid .value{font-size:16px;}
  canvas{max-height:200px;}
  .scroll-x{overflow-x:auto;}
  .toast{
    position:fixed; bottom:calc(16px + env(safe-area-inset-bottom,0px)); left:50%;
    transform:translateX(-50%); background:#1c2530; border:1px solid var(--panel-border);
    padding:10px 16px; border-radius:8px; font-size:13px; display:none; z-index:50;
  }
  .disclaimer{color:var(--muted); font-size:11px; text-align:center; padding:10px 16px 20px;}
</style>
</head>
<body>

<header>
  <h1>&#128200; Crypto Signal Bot</h1>
  <div>
    <span id="health-dot" class="dot bad"></span>
    <span id="mode-badge" class="badge paper">--</span>
  </div>
</header>

<main>

  <div class="grid">
    <div class="card"><div class="label">Balance</div><div class="value" id="balance">--</div></div>
    <div class="card"><div class="label">Equity</div><div class="value" id="equity">--</div></div>
    <div class="card"><div class="label">Open P&amp;L</div><div class="value" id="open-pnl">--</div></div>
    <div class="card"><div class="label">Trades</div><div class="value" id="num-trades">--</div></div>
  </div>

  <section>
    <h2>&#128225; Live Signal</h2>
    <div class="row">
      <input id="sig-symbol" value="BTC/USDT" placeholder="BTC/USDT">
      <select id="sig-timeframe">
        <option value="15m">15m</option><option value="1h" selected>1h</option>
        <option value="4h">4h</option><option value="1d">1d</option>
      </select>
      <button onclick="checkSignal()">Check</button>
    </div>
    <div id="signal-out" class="muted">Tap Check to fetch the current signal.</div>
  </section>

  <section>
    <h2>&#9881;&#65039; Paper Trading</h2>
    <div class="row">
      <input id="ctl-symbol" value="BTC/USDT" placeholder="BTC/USDT">
      <select id="ctl-timeframe">
        <option value="15m">15m</option><option value="1h" selected>1h</option>
        <option value="4h">4h</option><option value="1d">1d</option>
      </select>
    </div>
    <div class="row">
      <input id="api-key" type="password" placeholder="API_CONTROL_KEY (if set on server)">
    </div>
    <div class="row">
      <button onclick="startPaper()">Start</button>
      <button class="danger" onclick="stopPaper()">Stop</button>
      <button class="secondary" onclick="refreshStatus()">Refresh now</button>
    </div>
    <div id="ctl-out" class="muted"></div>
    <div class="muted" id="running-list" style="margin-top:6px;"></div>
  </section>

  <section>
    <h2>&#128202; Open Position</h2>
    <div id="position-out" class="muted">No open position.</div>
  </section>

  <section>
    <h2>&#128203; Trade Log</h2>
    <div class="scroll-x">
      <table id="trade-table">
        <thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&amp;L %</th><th>P&amp;L</th><th>Reason</th></tr></thead>
        <tbody><tr><td colspan="7" class="muted">No trades yet.</td></tr></tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>&#129514; Backtest</h2>
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
    <button onclick="runBacktest()" id="bt-btn">Run Backtest</button>
    <div id="bt-status" class="muted" style="margin-top:8px;"></div>
    <div id="bt-results" style="display:none;">
      <div class="result-grid">
        <div class="card"><div class="label">Win Rate</div><div class="value" id="bt-winrate">--</div></div>
        <div class="card"><div class="label">Return</div><div class="value" id="bt-return">--</div></div>
        <div class="card"><div class="label">Max Drawdown</div><div class="value" id="bt-dd">--</div></div>
        <div class="card"><div class="label">Profit Factor</div><div class="value" id="bt-pf">--</div></div>
        <div class="card"><div class="label"># Trades</div><div class="value" id="bt-num">--</div></div>
      </div>
      <canvas id="bt-chart"></canvas>
    </div>
  </section>

  <div class="disclaimer">
    Paper trading / educational tool. No strategy has a guaranteed win rate.
    Not financial advice.
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
    badge.className = 'badge ' + (d.mode === 'LIVE' ? 'live' : 'paper');
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
    pnlEl.className = 'value ' + (openPnl > 0 ? 'pos' : (openPnl < 0 ? 'neg' : ''));

    const posOut = document.getElementById('position-out');
    if(d.open_position){
      const p = d.open_position;
      posOut.innerHTML = `<strong>${p.symbol}</strong> &mdash; ${p.side.toUpperCase()}
        &nbsp;| Entry: $${fmt(p.entry_price)} | Stop: $${fmt(p.stop)} | Target: $${fmt(p.target)}
        | Size: $${fmt(p.size)}`;
    } else {
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
          <td style="color:${t.pnl_pct>=0?'var(--accent)':'var(--danger)'}">${t.pnl_pct>=0?'+':''}${fmt(t.pnl_pct)}%</td>
          <td style="color:${t.pnl_quote>=0?'var(--accent)':'var(--danger)'}">${t.pnl_quote>=0?'+':''}$${fmt(t.pnl_quote)}</td>
          <td>${t.reason}</td>
        </tr>`).join('');
    } else {
      tbody.innerHTML = '<tr><td colspan="7" class="muted">No trades yet.</td></tr>';
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
    const r = await fetch(`/signal/${encodeURIComponent(symbol)}?timeframe=${timeframe}`);
    const d = await r.json();
    if(!r.ok) throw new Error(d.detail || 'error');
    const sigLabel = d.signal === 1 ? '&#128994; LONG' : (d.signal === -1 ? '&#128308; EXIT/SHORT' : '&#9898; none');
    out.innerHTML = `Price: $${fmt(d.close)} &nbsp;|&nbsp; Trend: ${d.trend} &nbsp;|&nbsp;
      RSI: ${fmt(d.rsi)} &nbsp;|&nbsp; Signal: ${sigLabel}`;
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
    retEl.className = 'value ' + (d.total_return_pct>=0?'pos':'neg');
    document.getElementById('bt-dd').textContent = fmt(d.max_drawdown_pct) + '%';
    document.getElementById('bt-pf').textContent = d.profit_factor === -1 ? '&#8734;' : fmt(d.profit_factor);
    document.getElementById('bt-num').textContent = d.num_trades;
    document.getElementById('bt-results').style.display = 'block';
    status.textContent = `${d.symbol} ${d.timeframe}, ${d.start.slice(0,10)} to ${d.end.slice(0,10)}.`;

    // Build equity curve from the trade list
    let bal = d.starting_balance;
    const points = [bal];
    const labels = ['start'];
    (d.trades || []).forEach((t,i) => { bal += t.pnl_quote; points.push(bal); labels.push(i+1); });

    const ctx = document.getElementById('bt-chart').getContext('2d');
    if(btChart) btChart.destroy();
    btChart = new Chart(ctx, {
      type:'line',
      data:{ labels, datasets:[{ data: points, borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,.1)',
        fill:true, tension:0.15, pointRadius:0, borderWidth:2 }]},
      options:{ responsive:true, plugins:{legend:{display:false}},
        scales:{ x:{display:false}, y:{ticks:{color:'#8b98a5'}, grid:{color:'#232b36'}} } }
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


@app.get("/signal/{symbol}")
def get_signal(symbol: str, timeframe: str = Query(default=None)):
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
def paper_start_route(req: PaperStartRequest, x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    symbol = req.symbol or settings.default_symbol
    timeframe = req.timeframe or settings.default_timeframe
    started = scheduler_start(symbol, timeframe)
    return {"started": started, "symbol": symbol, "timeframe": timeframe}


@app.post("/paper/stop")
def paper_stop_route(symbol: str = Query(default=None), x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)
    sym = symbol or settings.default_symbol
    stopped = scheduler_stop(sym)
    return {"stopped": stopped, "symbol": sym}


@app.get("/paper/status")
def paper_status_route():
    return paper_status()
