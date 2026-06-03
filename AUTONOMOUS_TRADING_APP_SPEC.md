# Autonomous Trading Application — Technical Specification

> **Capital Target:** £100 → £10,000 (30 days) → £1,000,000 (12 months)
> **Mode:** Fully autonomous, locally hosted, legally compliant

---

## 1. Reality Check & Risk Framework

Before architecture: understand what the targets demand mathematically.

| Phase | Start | End | Duration | Required Daily Return |
|-------|-------|-----|----------|-----------------------|
| 1 | £100 | £10,000 | 30 days | ~16.6% per day (compounded) |
| 2 | £10,000 | £1,000,000 | ~335 days | ~1.4% per day (compounded) |

**Phase 1 is essentially a lottery ticket operation.** A 16.6% daily return requires high-leverage instruments (crypto, options, leveraged tokens) and significant luck alongside skill. No deterministic strategy guarantees this. The app must be designed to attempt it while strictly managing ruin risk — losing everything and having nothing to compound is the only unrecoverable outcome.

**Phase 2 is aggressive but achievable** with disciplined leverage, options strategies, and momentum systems. Consistent 1.4%/day (~400% annualised) is in the range of documented top-tier quant funds and aggressive retail traders.

**Approach:** Use a two-phase engine. Phase 1 uses high-risk/high-reward instruments with hard ruin prevention. Phase 2 switches to systematic, diversified momentum and mean-reversion strategies once capital reaches £10,000.

---

## 2. Legal & Compliance Constraints

| Requirement | Detail |
|-------------|--------|
| Broker API | Must use a regulated broker (Alpaca, Interactive Brokers, eToro, Saxo, Kraken for crypto) |
| PDT Rule | US Pattern Day Trader rule requires $25K minimum for >3 day trades/week on US equities — use UK/EU brokers or crypto to avoid this |
| Tax | All gains are subject to UK Capital Gains Tax (CGT). System must export a full trade log for HMRC reporting |
| Market Manipulation | No spoofing, layering, or wash trading — strictly illegal under FCA/MAR |
| Short Selling | Requires margin account and broker permission — must declare intent at account opening |
| Leverage Limits | UK/EU retail CFD leverage capped at 2:1 (stocks), 5:1 (indices), 2:1 (crypto) under ESMA rules — use professional account if eligible or crypto spot with own leverage logic |
| Data | Use licensed market data feeds only (Polygon.io, Alpha Vantage, Yahoo Finance, Binance API) |

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS TRADING ENGINE                   │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │  Data Layer  │  │ Strategy     │  │  Risk Manager        │ │
│  │              │  │ Engine       │  │                      │ │
│  │ • Market feed│→ │ • Signal gen │→ │ • Position sizing    │ │
│  │ • News/NLP   │  │ • Alpha calc │  │ • Stop-loss engine   │ │
│  │ • Order book │  │ • Backtest   │  │ • Drawdown circuit   │ │
│  │ • Alt data   │  │   validator  │  │   breaker            │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│           │                │                    │              │
│           └────────────────┴────────────────────┘             │
│                            │                                   │
│                   ┌────────────────┐                           │
│                   │ Execution      │                           │
│                   │ Engine         │                           │
│                   │ • Order router │                           │
│                   │ • Smart order  │                           │
│                   │   routing      │                           │
│                   │ • Slippage est │                           │
│                   └────────────────┘                           │
│                            │                                   │
│           ┌────────────────┼────────────────┐                  │
│           │                │                │                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  Portfolio  │  │  Monitoring  │  │  Tax & Reporting    │   │
│  │  Manager   │  │  & Alerting  │  │  • Trade log export  │   │
│  │  • P&L     │  │  • Telegram  │  │  • CGT calculator   │   │
│  │  • Exposure│  │  • Dashboard │  │  • HMRC format      │   │
│  └─────────────┘  └──────────────┘  └─────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12 | Fastest ecosystem for quant/ML |
| Async runtime | `asyncio` + `aiohttp` | Low-latency concurrent I/O |
| Data storage | TimescaleDB (Postgres extension) | Time-series optimised, SQL queryable |
| Cache | Redis | Sub-millisecond signal state & order dedup |
| Strategy backtest | `backtrader` / `vectorbt` | Vectorised backtesting with realistic fees |
| ML signals | `scikit-learn`, `lightgbm`, `ta-lib` | Feature engineering + gradient boosting |
| NLP/Sentiment | `transformers` (FinBERT) | Financial news sentiment |
| Broker API — stocks | Alpaca Markets (paper + live) | Commission-free, REST + WebSocket, UK accessible |
| Broker API — crypto | Binance / Kraken | Deep liquidity, full API, UK legal |
| Market data | Polygon.io (stocks), Binance WS (crypto) | Real-time tick data |
| News | NewsAPI + RSS scraper | Free tier for signal enrichment |
| Dashboard | FastAPI + React (Recharts) | Local web UI for monitoring |
| Notifications | Telegram Bot API | Instant mobile alerts |
| Containerisation | Docker Compose | Reproducible local deployment |
| Config | `.env` + `pydantic-settings` | Secrets out of code |

---

## 5. Phase 1 Strategy: Aggressive Growth (£100 → £10,000)

### 5.1 Target Instruments

> **UK Note:** Crypto derivatives (futures, perpetual swaps, leveraged tokens) are banned for UK retail clients by the FCA. Phase 1 uses **spot crypto only** via Kraken (FCA registered). The compounding still works — high frequency spot scalping at 1–3% per trade × 5–15 trades/day compounds rapidly.

- **Crypto spot** (Kraken): BTC/USD, ETH/USD, SOL/USD — 24/7, high volatility, no PDT rule, FCA compliant
- **Micro-cap momentum stocks** (IBKR, Phase 1 supplement if balance > £500): 5–20% single-day moves
- **Options** (IBKR, Phase 1 supplement if capital > £1,000): defined-risk leverage on US stocks

### 5.2 Core Phase 1 Strategies

#### A. Crypto Momentum Scalping (Primary)
- **Timeframe:** 5-minute and 15-minute candles
- **Universe:** Top 20 crypto by volume (BTC, ETH, SOL, BNB, etc.)
- **Entry signal:** 
  - RSI(14) crosses above 55 from below (bullish) / below 45 (bearish short)
  - MACD histogram turns positive with increasing slope
  - Volume > 1.5× 20-period average (confirms move)
  - Price breaks above/below Bollinger Band midline with momentum
- **Exit signal:**
  - Target: +3% to +5% per trade
  - Stop: −1.5% (asymmetric 2:1 minimum R:R)
  - Time stop: exit at candle close if signal invalidated
- **Position size:** 20% of capital per trade (Kelly-bounded, see §7)
- **Max concurrent positions:** 3

#### B. Breakout Momentum (Secondary)
- **Timeframe:** 1-hour candles
- **Entry:** Price closes above 20-period high with volume expansion (> 2× avg)
- **Confirmation:** ADX > 25 (trending, not ranging)
- **Target:** 5–15% move (ride the trend with trailing stop)
- **Stop:** Below breakout candle low
- **Position size:** 15–25% of capital

#### C. News Catalyst Trading
- **Trigger:** FinBERT sentiment score > 0.85 positive on asset-specific news
- **Filter:** Only trade within 2 minutes of news publication
- **Entry:** Market order on trigger with hard 2% stop
- **Target:** 5% quick scalp, exit within 15 minutes regardless

### 5.3 Phase 1 Risk Rules (Hard Limits)

```python
MAX_DAILY_LOSS_PCT = 0.20        # Stop trading if down 20% on day
MAX_POSITION_SIZE_PCT = 0.25     # Never more than 25% in one trade
MAX_DRAWDOWN_FROM_PEAK_PCT = 0.35  # Circuit breaker: halt all trading
MIN_RR_RATIO = 2.0               # Never take a trade with R:R < 2:1
RUIN_FLOOR_GBP = 20.0            # Never risk below £20 total — preserve seed
```

---

## 6. Phase 2 Strategy: Systematic Growth (£10,000 → £1,000,000)

Phase 2 activates once equity exceeds £10,000. Focus shifts from lottery-style bets to systematic, diversified, multi-strategy operation targeting consistent 1–2% daily.

### 6.1 Strategy Portfolio (Diversified)

#### A. Dual Momentum System (Core, 40% of capital)
- **Universe:** ETFs, indices, top 50 crypto, large-cap tech stocks
- **Absolute momentum:** Only long assets with positive 3-month return
- **Relative momentum:** Rank assets; hold top quartile
- **Rebalance:** Weekly
- **Leverage:** 1–2× via CFDs or futures when conviction high (Sharpe > 1.5)

#### B. Statistical Arbitrage / Pairs Trading (20% of capital)
- **Method:** Find cointegrated pairs (Engle-Granger test, p < 0.05)
- **Universe:** Sector ETFs, crypto pairs (BTC/ETH, SOL/AVAX)
- **Entry:** Z-score of spread > 2.0 standard deviations
- **Exit:** Z-score returns to 0.5 or less
- **Hedge ratio:** Kalman filter dynamic beta
- **Expected return:** 0.3–0.8% per trade, very high win rate (65–70%)

#### C. Options Premium Harvesting (20% of capital — requires £5K+ per position)
- **Strategy:** Cash-secured puts + covered calls (Wheel strategy)
- **Instruments:** High IV stocks/ETFs (QQQ, SPY, TSLA, NVDA)
- **Delta target:** Short puts at 0.20–0.30 delta (OTM, high probability of expiry worthless)
- **DTE:** 30–45 days to expiration at entry, close at 50% profit
- **Expected return:** 2–5% per month on capital deployed

#### D. Intraday Mean Reversion (20% of capital)
- **Timeframe:** 5-minute bars
- **Signal:** Price deviates > 2σ from VWAP + RSI(5) < 25 (oversold) or > 75 (overbought)
- **Entry:** Limit order at reversion point
- **Target:** Return to VWAP (typically 0.5–1.5%)
- **Stop:** 1% beyond entry (away from mean)
- **Filter:** Only trade first 90 minutes and last 60 minutes of session (highest mean-reversion tendency)

### 6.2 Phase 2 Risk Rules

```python
MAX_STRATEGY_ALLOCATION_PCT = 0.40  # No single strategy > 40%
MAX_SECTOR_EXPOSURE_PCT = 0.30      # Diversification across sectors
MAX_DAILY_LOSS_PCT = 0.05           # More conservative — protecting large capital
MAX_POSITION_SIZE_PCT = 0.10        # Never > 10% in one position
VOLATILITY_TARGET_ANNUAL = 0.25     # Scale position sizes to target 25% annual vol
MAX_LEVERAGE = 3.0                  # Never exceed 3× leverage
CORRELATION_LIMIT = 0.70            # Reject new position if > 0.70 corr to existing
```

---

## 7. Position Sizing: Kelly Criterion with Half-Kelly Safety

Full Kelly maximises long-run growth but causes catastrophic drawdowns. Use Half-Kelly:

```
f* = (p × b - q) / b     # Full Kelly fraction
f_half = f* / 2           # Half Kelly (practical)

Where:
  p = win rate (estimated from backtest, updated live)
  q = 1 - p (loss rate)
  b = average win / average loss ratio
```

**Dynamic adjustment:** Scale `f_half` down by the ratio `(current_equity / peak_equity)^2` when in drawdown. This auto-reduces size as you lose, reducing ruin probability.

```python
def kelly_position_size(equity, peak_equity, win_rate, avg_win, avg_loss):
    b = avg_win / avg_loss
    q = 1 - win_rate
    full_kelly = (win_rate * b - q) / b
    half_kelly = full_kelly / 2
    drawdown_scalar = (equity / peak_equity) ** 2
    return min(half_kelly * drawdown_scalar, 0.25)  # Cap at 25%
```

---

## 8. Technical Indicators & Signal Library

| Indicator | Use Case | Parameters |
|-----------|----------|------------|
| EMA crossover | Trend direction | 9/21 EMA, 50/200 EMA |
| RSI | Overbought/oversold | Period 14, thresholds 30/70 |
| MACD | Momentum + divergence | 12/26/9 |
| Bollinger Bands | Volatility + reversion | 20 period, 2σ |
| ATR | Stop distance, position sizing | Period 14 |
| VWAP | Intraday fair value | Rolling daily |
| ADX | Trend strength filter | Period 14, threshold 25 |
| OBV | Volume/price divergence | Cumulative |
| Stochastic | Short-term momentum | 14,3,3 |
| Ichimoku Cloud | Multi-factor trend | Standard (9,26,52) |
| Heikin Ashi | Noise-filtered candles | Smoothed OHLC |
| Keltner Channel | Breakout identification | EMA20, 1.5×ATR |

---

## 9. Backtesting & Validation Framework

Before any strategy goes live, it must pass:

```
Minimum Backtest Requirements:
├── Period: 3+ years of historical data
├── Out-of-sample: Last 20% of data held out (walk-forward)
├── Sharpe Ratio: > 1.5 (annualised, after fees)
├── Max Drawdown: < 30%
├── Win Rate: > 45% (or R:R compensates if lower)
├── Profit Factor: > 1.5 (gross profit / gross loss)
├── Number of trades: > 200 (statistical significance)
└── Realistic fees: Include spread, commission, slippage (0.1% per side min)
```

**Walk-forward optimisation:** Re-optimise parameters every 30 days on trailing 6-month data. This prevents overfitting to stale market regimes.

---

## 10. Data Pipeline

```
┌─────────────────────────────────────────────────┐
│ Real-Time Feeds (WebSocket)                      │
│  • Binance WS: tick data, order book, trade feed │
│  • Polygon.io WS: US equities NBBO + trades      │
│  • NewsAPI polling: 60-second interval           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│ Normalisation Layer                              │
│  • Unified OHLCV schema                          │
│  • Timezone normalisation (UTC)                  │
│  • Duplicate/gap detection                       │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│ TimescaleDB (local)                              │
│  • ohlcv_1m, ohlcv_5m, ohlcv_1h, ohlcv_1d      │
│  • order_book_snapshots                          │
│  • news_events                                   │
│  • trades_executed                               │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│ Feature Store (Redis cache)                      │
│  • Pre-computed indicators (TTL = candle period) │
│  • Current positions state                       │
│  • Signal scores per asset                       │
└─────────────────────────────────────────────────┘
```

---

## 11. Execution Engine

### 11.1 Order Types
- **Market order:** Used for momentum entries and news catalyst trades (speed > price)
- **Limit order:** Used for mean-reversion entries (price > speed)
- **Stop-market:** Exit stops always stop-market to guarantee execution
- **Trailing stop:** Momentum trades lock in profit dynamically

### 11.2 Smart Order Routing
```python
def select_order_type(strategy_type, urgency, spread_bps):
    if urgency == 'HIGH' or spread_bps < 10:
        return 'MARKET'
    elif strategy_type == 'MEAN_REVERSION':
        return 'LIMIT'
    else:
        return 'LIMIT' if spread_bps > 20 else 'MARKET'
```

### 11.3 Slippage Estimation
- Model slippage as: `slippage = k × sqrt(order_size / avg_daily_volume)`
- k = 0.1 for large-cap, 0.5 for mid-cap, 2.0 for small-cap/altcoin
- Reject trade if estimated slippage > 0.5% on a 2% target (kills edge)

---

## 12. Risk Management Module

### 12.1 Real-Time Monitoring
```python
class RiskManager:
    checks = [
        DailyLossLimitCheck(),      # Halt if daily P&L < threshold
        DrawdownCircuitBreaker(),   # Halt if equity < peak × (1 - max_dd)
        ConcentrationCheck(),       # Block if position > max_pct
        CorrelationCheck(),         # Block if new trade correlated to existing
        LiquidityCheck(),           # Block if spread > threshold
        NewsBlackoutCheck(),        # Halt 5min before/after major macro events
        ExchangeStatusCheck(),      # Halt if exchange API latency > 500ms
    ]
```

### 12.2 Macro Event Calendar
- Pull economic calendar (Forex Factory, Investing.com API)
- Halt all new positions 5 minutes before and after:
  - Interest rate decisions (BoE, Fed, ECB)
  - CPI/NFP/GDP releases
  - Major earnings (if trading that stock)
- Resume after volatility normalises (ATR returns to baseline)

### 12.3 Drawdown Recovery Mode
| Drawdown from Peak | Action |
|--------------------|--------|
| 0–15% | Normal operation |
| 15–25% | Reduce position sizes by 50%, increase R:R minimum to 3:1 |
| 25–35% | Paper-trade only mode — no real orders |
| > 35% | Full halt, require manual restart after review |

---

## 13. Portfolio Manager

### 13.1 Portfolio State
```python
@dataclass
class Portfolio:
    cash_gbp: float
    positions: dict[str, Position]
    open_orders: dict[str, Order]
    realised_pnl: float
    unrealised_pnl: float
    peak_equity: float
    phase: Literal['PHASE_1', 'PHASE_2']

    @property
    def total_equity(self):
        return self.cash_gbp + self.unrealised_pnl

    @property
    def drawdown_pct(self):
        return (self.peak_equity - self.total_equity) / self.peak_equity
```

### 13.2 Phase Transition Logic
```python
def check_phase_transition(portfolio: Portfolio) -> None:
    if portfolio.phase == 'PHASE_1' and portfolio.total_equity >= 10_000:
        portfolio.phase = 'PHASE_2'
        # Close all Phase 1 positions
        # Rebalance into Phase 2 strategy allocation
        # Reduce max position size from 25% to 10%
        notify("PHASE TRANSITION: Entering Phase 2 systematic trading")
```

---

## 14. Monitoring Dashboard & Alerting

### 14.1 Local Web Dashboard (FastAPI + React)
```
Dashboard Panels:
├── Equity Curve (real-time chart vs. target curve)
├── Open Positions (symbol, entry, current, P&L, stop distance)
├── Today's Trades (time, symbol, side, size, entry, exit, P&L)
├── Strategy Performance (Sharpe, win rate, profit factor per strategy)
├── Risk Meters (daily loss %, drawdown %, exposure %)
├── Market Regime (trending / ranging / volatile indicator)
└── System Health (API latency, data feed status, last heartbeat)
```

### 14.2 Telegram Notifications
```
Alert Types:
• Trade opened: "BUY BTC/USDT £250 @ 65,420 | Stop: 64,400 | Target: 68,600"
• Trade closed: "CLOSED BTC/USDT +£87 (+34.8%) | Running P&L today: +£241"
• Risk alert: "⚠️ Daily loss limit 15% reached — reducing sizes"
• Circuit breaker: "🛑 HALT: Drawdown exceeded 25%. Manual review required"
• Phase transition: "🎯 PHASE 2 ACTIVATED — Equity: £10,247"
• Daily summary: "Daily PnL: +£412 | Equity: £4,823 | Trades: 7 | Win rate: 71%"
```

---

## 15. Tax & Reporting (UK HMRC)

```python
class TaxReporter:
    def generate_cgt_report(self, tax_year: str) -> pd.DataFrame:
        # UK CGT: Section 104 pooling for same-asset trades
        # Bed & breakfast rule: 30-day matching
        # Report: disposal date, proceeds, cost basis, gain/loss
        pass

    def export_trade_log(self, format: Literal['CSV', 'JSON']) -> str:
        # All trades with: timestamp, symbol, side, qty, price, fee, fx_rate
        pass
```

- All trades stored with GBP equivalent at time of execution
- Crypto taxed as capital asset (not currency) under HMRC guidance
- Annual CGT allowance (£3,000 in 2025/26) tracked and optimised

---

## 16. Directory Structure

```
trading-app/
├── core/
│   ├── engine.py              # Main event loop
│   ├── portfolio.py           # Portfolio state manager
│   └── phase_controller.py   # Phase 1/2 switching logic
├── data/
│   ├── feeds/
│   │   ├── binance_ws.py      # Binance WebSocket feed
│   │   ├── polygon_ws.py      # Polygon equities feed
│   │   └── news_scraper.py    # News + NLP pipeline
│   ├── storage/
│   │   ├── timescale.py       # TimescaleDB interface
│   │   └── redis_cache.py     # Feature cache
│   └── indicators/
│       ├── technical.py       # RSI, MACD, BB, ATR, etc.
│       └── sentiment.py       # FinBERT NLP scoring
├── strategies/
│   ├── base.py                # Abstract strategy interface
│   ├── phase1/
│   │   ├── crypto_scalp.py    # Crypto momentum scalping
│   │   ├── breakout.py        # Breakout momentum
│   │   └── news_catalyst.py   # News-driven entries
│   └── phase2/
│       ├── dual_momentum.py   # Dual momentum system
│       ├── pairs_trading.py   # Statistical arbitrage
│       ├── options_wheel.py   # Cash-secured puts + calls
│       └── mean_reversion.py  # Intraday VWAP reversion
├── risk/
│   ├── manager.py             # Real-time risk checks
│   ├── position_sizer.py      # Kelly criterion sizing
│   └── circuit_breaker.py    # Drawdown halt logic
├── execution/
│   ├── order_router.py        # Smart order routing
│   ├── brokers/
│   │   ├── alpaca.py          # Alpaca API wrapper
│   │   ├── binance.py         # Binance API wrapper
│   │   └── kraken.py          # Kraken API wrapper
│   └── slippage.py            # Slippage estimation
├── backtest/
│   ├── runner.py              # Vectorbt/backtrader runner
│   ├── walk_forward.py        # Walk-forward optimiser
│   └── metrics.py             # Sharpe, drawdown, profit factor
├── monitoring/
│   ├── metrics.py             # Prometheus metrics definitions
│   ├── logger.py              # Structured JSON logging (structlog)
│   ├── tracing.py             # OpenTelemetry setup
│   ├── dashboard/             # FastAPI + React UI
│   ├── telegram_bot.py        # Alert notifications
│   ├── health_check.py        # System heartbeat
│   ├── prometheus.yml         # Scrape config
│   ├── alert_rules.yml        # Alertmanager rules
│   ├── alertmanager.yml       # Routing config (Telegram/email)
│   ├── promtail.yml           # Log shipping to Loki
│   └── grafana/
│       ├── dashboards/        # JSON dashboard exports
│       │   ├── trading_overview.json
│       │   ├── market_monitor.json
│       │   ├── system_health.json
│       │   └── trade_audit.json
│       └── provisioning/      # Auto-load datasources + dashboards
├── reporting/
│   ├── tax_reporter.py        # HMRC CGT reports
│   └── trade_log.py           # Full trade history export
├── config/
│   ├── settings.py            # Pydantic settings model
│   ├── phase1_config.yaml     # Phase 1 strategy params
│   └── phase2_config.yaml     # Phase 2 strategy params
├── tests/
│   ├── unit/                  # Strategy logic tests
│   ├── integration/           # Broker API mock tests
│   └── backtest_validation/   # Strategy performance tests
├── docker-compose.yml         # TimescaleDB + Redis + app
├── Dockerfile
├── requirements.txt
└── .env.example               # All API keys template
```

---

## 17. Configuration (`.env.example`)

```bash
# Broker APIs
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # paper first!
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
KRAKEN_API_KEY=
KRAKEN_PRIVATE_KEY=

# Market Data
POLYGON_API_KEY=
NEWS_API_KEY=

# Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Database
TIMESCALEDB_URL=postgresql://trader:password@localhost:5432/trading
REDIS_URL=redis://localhost:6379

# Safety
PAPER_TRADING=true          # MUST be true until backtesting complete
INITIAL_CAPITAL_GBP=100
PHASE1_TARGET_GBP=10000
PHASE2_TARGET_GBP=1000000
MAX_DAILY_LOSS_PCT=0.20
MAX_DRAWDOWN_PCT=0.35
```

---

## 18. Supported Brokers & Platform Selection

> **Note:** Trading212 does **not** expose a public API for automated/programmatic trading — it only supports manual trading and copy-trading. The platforms below all provide REST + WebSocket APIs suitable for autonomous operation.

### 18.1 UK Regulatory Note on Crypto

> **Important:** Binance withdrew from the UK market in August 2023 and is no longer available to UK residents. Additionally, the FCA **banned crypto derivatives (futures, leveraged CFDs) for retail UK clients in January 2021**. This means:
> - No Binance
> - No crypto futures or leveraged derivatives for retail accounts
> - Crypto must be traded **spot only** unless you qualify as a professional client
>
> The Phase 1 strategy is adjusted accordingly — high-frequency spot scalping on volatile assets (BTC, ETH, SOL) still compounds rapidly at smaller per-trade targets.

### 18.2 Broker Comparison Matrix

| Broker | Asset Classes | API Type | UK FCA Status | Min Deposit | Python SDK | Paper/Demo | Best For |
|--------|--------------|----------|---------------|-------------|------------|------------|----------|
| **Kraken** | Crypto spot | REST + WS v2 | FCA registered (Payward Ltd) | £0 | `krakenex` | No (use app paper mode) | **Phase 1 primary** |
| **Coinbase Advanced Trade** | Crypto spot | REST + WS | FCA registered | £0 | `coinbase-advanced-py` | No (use app paper mode) | Phase 1 backup |
| **Bitstamp** | Crypto spot | REST + WS | FCA registered (Bitstamp Europe) | £0 | `bitstamp` | No | Phase 1 alt |
| **Interactive Brokers (IBKR)** | Stocks, ETFs, options, futures, forex | TWS API + REST | FCA regulated | £0 (£2K for margin) | `ib_insync` | Yes (paper account) | **Phase 2 primary** |
| **Alpaca Markets** | US stocks, ETFs | REST + WS | FCA passporting | £0 | `alpaca-py` | Yes (free) | Phase 2 stocks |
| **Saxo Bank** | Stocks, ETFs, options, CFDs, forex | OpenAPI REST | FCA regulated | £500 | REST via `httpx` | No | Phase 2 multi-asset |
| **IG Group** | Stocks, spread betting, CFDs | REST API | FCA regulated | £250 | REST via `httpx` | Yes (demo) | Phase 2 tax-free (spread betting) |
| **Oanda** | Forex, indices CFDs | REST + Stream API | FCA regulated | £0 | `oandapy3` | Yes (demo) | Phase 2 forex |

### 18.3 Recommended Broker Stack

```
Phase 1 (£100 → £10K) — Crypto spot only:
  Primary:  Kraken (FCA registered, deep BTC/ETH/SOL liquidity, WS API v2)
  Backup:   Coinbase Advanced Trade (FCA registered, good API, higher fees)
  Strategy: High-frequency spot scalping — no leverage, rely on position sizing
            and compounding. Smaller per-trade targets (1–3%) compensated by
            higher trade frequency (5–15 trades/day).

Phase 2 (£10K → £1M):
  Stocks/ETFs/Options: Interactive Brokers (best API, global markets, FCA)
  Crypto:              Kraken (maintain 20–30% crypto allocation, spot)
  Tax-free profits:    IG Group Spread Betting account (no CGT on winnings!)
```

### 18.4 Why Not Trading212 or Binance
- **Trading212:** No public API — automation requires UI scraping which violates ToS
- **Binance:** Exited UK market August 2023; no longer available to UK residents
- **Crypto futures/leverage:** FCA-banned for UK retail since January 2021 — spot only

### 18.4 Broker API Setup

```python
# config/brokers.py — unified broker interface
from abc import ABC, abstractmethod

class BrokerBase(ABC):
    @abstractmethod
    async def place_order(self, symbol, side, qty, order_type, **kwargs): ...
    @abstractmethod
    async def get_positions(self): ...
    @abstractmethod
    async def get_account(self): ...
    @abstractmethod
    async def cancel_order(self, order_id): ...
    @abstractmethod
    async def stream_quotes(self, symbols, callback): ...

# Concrete implementations in execution/brokers/
# All use the same BrokerBase interface — swap broker without changing strategy code
```

---

## 19. Observability Layer

The observability layer provides full visibility into three domains: **system health**, **market state**, and **trading actions**. It follows the OpenTelemetry three-pillar model: Metrics, Logs, Traces.

### 19.1 Observability Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     OBSERVABILITY STACK                           │
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │   METRICS        │  │    LOGGING        │  │   TRACING      │   │
│  │  Prometheus      │  │  Structured JSON  │  │  OpenTelemetry │   │
│  │  + Grafana       │  │  → Loki          │  │  → Jaeger      │   │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘   │
│           │                     │                     │            │
│           └─────────────────────┴─────────────────────┘            │
│                                 │                                   │
│                    ┌────────────▼────────────┐                     │
│                    │   Grafana Unified UI     │                     │
│                    │   (localhost:3000)        │                     │
│                    │                          │                     │
│                    │  Dashboards:             │                     │
│                    │  • Trading Overview      │                     │
│                    │  • Market Monitor        │                     │
│                    │  • System Health         │                     │
│                    │  • Trade Audit Trail     │                     │
│                    └────────────┬────────────┘                     │
│                                 │                                   │
│                    ┌────────────▼────────────┐                     │
│                    │   Alertmanager           │                     │
│                    │   → Telegram            │                     │
│                    │   → Email (optional)    │                     │
│                    └─────────────────────────┘                     │
└───────────────────────────────────────────────────────────────────┘
```

### 19.2 Metrics (Prometheus)

All metrics are exposed on `/metrics` and scraped by Prometheus every 15 seconds.

```python
# monitoring/metrics.py
from prometheus_client import Counter, Gauge, Histogram, Summary

# --- Trading Metrics ---
trades_total = Counter(
    'trades_total', 'Total trades executed',
    ['strategy', 'symbol', 'side', 'phase']
)
trade_pnl = Histogram(
    'trade_pnl_gbp', 'P&L per trade in GBP',
    ['strategy', 'symbol'],
    buckets=[-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500]
)
portfolio_equity = Gauge('portfolio_equity_gbp', 'Total portfolio equity in GBP')
portfolio_drawdown = Gauge('portfolio_drawdown_pct', 'Current drawdown from peak')
open_positions = Gauge('open_positions_count', 'Number of open positions', ['strategy'])
daily_pnl = Gauge('daily_pnl_gbp', 'Realised + unrealised P&L today')
win_rate = Gauge('strategy_win_rate', 'Rolling 50-trade win rate', ['strategy'])
sharpe_ratio = Gauge('strategy_sharpe_ratio', 'Rolling 30-day Sharpe', ['strategy'])

# --- Market Metrics ---
market_spread_bps = Gauge(
    'market_spread_bps', 'Bid-ask spread in basis points',
    ['symbol', 'exchange']
)
market_volatility = Gauge(
    'market_volatility_atr', 'ATR(14) as % of price',
    ['symbol', 'timeframe']
)
market_regime = Gauge(
    'market_regime', 'Market regime: 1=trending, 0=ranging, -1=volatile',
    ['symbol']
)
signal_strength = Gauge(
    'signal_strength', 'Strategy signal score [-1, 1]',
    ['strategy', 'symbol']
)

# --- System Metrics ---
data_feed_latency_ms = Histogram(
    'data_feed_latency_ms', 'Time from market event to processed candle',
    ['feed', 'symbol'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)
order_execution_latency_ms = Histogram(
    'order_execution_latency_ms', 'Time from signal to order acknowledgement',
    ['broker', 'order_type'],
    buckets=[10, 25, 50, 100, 250, 500, 1000, 2500]
)
broker_api_errors = Counter(
    'broker_api_errors_total', 'Broker API errors',
    ['broker', 'error_type']
)
risk_checks_triggered = Counter(
    'risk_checks_triggered_total', 'Risk checks that fired',
    ['check_name', 'action']
)
```

### 19.3 Grafana Dashboards

#### Dashboard 1: Trading Overview
```
Row 1 — Equity & Performance
  • Equity curve (line) vs. target curve (dashed)     [portfolio_equity_gbp]
  • Daily P&L bar chart (green/red)                   [daily_pnl_gbp]
  • Drawdown meter (gauge, red > 20%)                 [portfolio_drawdown_pct]
  • Phase indicator (PHASE 1 / PHASE 2)               [text panel]

Row 2 — Active Positions
  • Table: symbol | side | qty | entry | current | unrealised P&L | stop distance
  • Exposure by asset class (pie chart)

Row 3 — Strategy Performance
  • Win rate per strategy (bar chart)                 [strategy_win_rate]
  • Sharpe ratio per strategy (bar chart)             [strategy_sharpe_ratio]
  • Trade count today (stat panels)                   [trades_total]
  • Profit factor rolling (line chart)

Row 4 — Recent Trades
  • Trade log table (last 50 trades): time | sym | side | entry | exit | P&L | strategy
```

#### Dashboard 2: Market Monitor
```
Row 1 — Market Regime
  • Regime status per symbol (heatmap: green=trending, yellow=ranging, red=volatile)
  • ATR % (volatility) sparklines per symbol

Row 2 — Signal Scoreboard
  • Signal strength heatmap: symbols × strategies, colour = score [-1..1]
  • Top 5 bullish signals (stat)
  • Top 5 bearish signals (stat)

Row 3 — Data Feed Health
  • Feed latency p50/p95/p99 (line chart)            [data_feed_latency_ms]
  • Last heartbeat per feed (status panels)
  • Missing candle alerts (red if gap > 2× period)

Row 4 — Order Book Depth (Crypto)
  • Bid/ask depth chart (bar, refreshed every 5s)
  • Spread in bps (line chart)                       [market_spread_bps]
```

#### Dashboard 3: System Health
```
Row 1 — Infrastructure
  • CPU / Memory / Disk (standard node exporter)
  • Redis memory usage + hit rate
  • TimescaleDB connections + query latency

Row 2 — Broker Connectivity
  • API latency per broker (line chart)              [order_execution_latency_ms]
  • Error rate per broker (counter)                  [broker_api_errors_total]
  • WebSocket reconnect events

Row 3 — Risk System
  • Risk checks triggered (bar chart by type)        [risk_checks_triggered_total]
  • Circuit breaker status (green/red)
  • Daily loss consumed % (gauge)
```

#### Dashboard 4: Trade Audit Trail
```
• Full trade history table with search/filter
• Trade P&L attribution: strategy, symbol, time-of-day
• Slippage analysis: expected vs actual execution price
• Fee analysis: cumulative fees vs gross P&L
• Tax events: realised gains/losses for CGT tracking
```

### 19.4 Structured Logging (Loki)

Every log line is JSON with consistent fields:

```python
# monitoring/logger.py
import structlog

log = structlog.get_logger()

# Example trade log
log.info("trade_opened",
    event_type="TRADE_OPEN",
    trade_id="T-20260602-001",
    strategy="crypto_scalp",
    symbol="BTC/USDT",
    side="BUY",
    qty=0.0038,
    price=65420.0,
    price_gbp=51820.0,
    stop_price=64400.0,
    target_price=68600.0,
    rr_ratio=2.3,
    kelly_fraction=0.18,
    phase="PHASE_1",
    equity_before=1050.0,
    signal_rsi=58.2,
    signal_macd_hist=142.3,
    signal_volume_ratio=1.82,
)

log.info("trade_closed",
    event_type="TRADE_CLOSE",
    trade_id="T-20260602-001",
    exit_reason="TARGET_HIT",  # TARGET_HIT | STOP_HIT | TIME_STOP | MANUAL
    entry_price=65420.0,
    exit_price=68594.0,
    pnl_gbp=87.40,
    pnl_pct=3.09,
    duration_minutes=47,
    slippage_bps=3.2,
    fee_gbp=1.12,
)

log.warning("risk_check_triggered",
    event_type="RISK_EVENT",
    check="DailyLossLimitCheck",
    action="REDUCE_SIZES",
    daily_loss_pct=-15.2,
    threshold_pct=-15.0,
)
```

### 19.5 Distributed Tracing (OpenTelemetry + Jaeger)

Each trade lifecycle is traced end-to-end:

```
Trace: "process_signal → execute_trade"
  │
  ├── span: data_feed.get_candles         [2ms]
  ├── span: indicators.compute_all        [8ms]
  ├── span: strategy.generate_signal      [3ms]
  ├── span: risk_manager.check_all        [1ms]
  ├── span: position_sizer.calculate      [0.5ms]
  ├── span: order_router.place_order      [45ms]  ← broker round-trip
  └── span: portfolio.update_state        [2ms]
  
Total: ~62ms signal-to-order
```

Jaeger UI accessible at `localhost:16686`.

### 19.6 Alerting Rules (Alertmanager → Telegram)

```yaml
# monitoring/alert_rules.yml
groups:
  - name: trading_critical
    rules:
      - alert: CircuitBreakerTripped
        expr: portfolio_drawdown_pct > 0.25
        for: 0m
        labels: { severity: critical }
        annotations:
          summary: "HALT: Drawdown {{ $value | humanizePercentage }} exceeded 25%"

      - alert: DailyLossLimit
        expr: daily_pnl_gbp / portfolio_equity_gbp < -0.15
        for: 0m
        labels: { severity: warning }
        annotations:
          summary: "Daily loss {{ $value | humanizePercentage }} — reducing sizes"

      - alert: DataFeedDown
        expr: time() - data_feed_last_heartbeat > 60
        for: 1m
        labels: { severity: critical }
        annotations:
          summary: "Data feed {{ $labels.feed }} silent for >60s — halting trading"

      - alert: BrokerAPIErrors
        expr: rate(broker_api_errors_total[5m]) > 0.5
        for: 2m
        labels: { severity: warning }
        annotations:
          summary: "Broker {{ $labels.broker }} errors: {{ $value }}/s"

      - alert: HighSlippage
        expr: histogram_quantile(0.95, order_execution_latency_ms) > 500
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "Order latency p95={{ $value }}ms — execution degraded"
```

### 19.7 Docker Compose — Full Observability Stack

```yaml
# docker-compose.yml (observability services)
services:
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=trading123
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning

  loki:
    image: grafana/loki:latest
    ports: ["3100:3100"]

  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./logs:/app/logs
      - ./monitoring/promtail.yml:/etc/promtail/config.yml

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"   # Jaeger UI
      - "4317:4317"     # OTLP gRPC
      - "4318:4318"     # OTLP HTTP

  alertmanager:
    image: prom/alertmanager:latest
    ports: ["9093:9093"]
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

  node-exporter:
    image: prom/node-exporter:latest
    ports: ["9100:9100"]

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    ports: ["5432:5432"]
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
```

### 19.8 Observability URLs (Local)

| Service | URL | Purpose |
|---------|-----|---------|
| Grafana | `http://localhost:3000` | All dashboards |
| Prometheus | `http://localhost:9090` | Raw metrics query |
| Jaeger | `http://localhost:16686` | Trace explorer |
| Alertmanager | `http://localhost:9093` | Alert routing |
| Trading API | `http://localhost:8000` | FastAPI app + `/metrics` |
| Trading UI | `http://localhost:8000/dashboard` | Live trading dashboard |

---

## 21. Development & Go-Live Checklist

### Phase 0: Paper Trading Validation (Week 1–2)
- [ ] Set up local Docker environment (TimescaleDB + Redis)
- [ ] Connect broker APIs in paper/sandbox mode
- [ ] Implement data feeds and verify OHLCV storage
- [ ] Implement all technical indicators with unit tests
- [ ] Backtest Phase 1 crypto scalp strategy (must pass §9 criteria)
- [ ] Run paper trading for 5+ days — verify Sharpe > 1.5 on paper
- [ ] Implement and test all risk checks
- [ ] Set up Telegram alerts and dashboard

### Phase 1: Live with £100 (Week 2–4)
- [ ] Switch `PAPER_TRADING=false`
- [ ] Fund broker account with £100
- [ ] Enable Phase 1 strategies only
- [ ] Monitor every trade for first 48 hours
- [ ] Review daily: adjust parameters if win rate < 40%
- [ ] Hard rule: never manually override the system

### Phase 2: Systematic (Month 2–12)
- [ ] Auto-transition when equity hits £10,000
- [ ] Backtest and validate all Phase 2 strategies
- [ ] Review and re-optimise parameters monthly (walk-forward)
- [ ] Diversify across multiple brokers at > £50,000 (FSCS limit)
- [ ] Consult accountant when realised gains approach CGT threshold

---

## 22. Honest Probability Assessment

| Scenario | Probability | Notes |
|----------|-------------|-------|
| Phase 1 success (£100 → £10K in 30 days) | 5–15% | Requires luck + skill. High leverage needed |
| Phase 1 partial (£100 → £1K in 30 days) | 40–60% | 10x more achievable with disciplined scalping |
| Phase 2 success (£10K → £1M in 11 months) | 15–30% | Achievable with strict discipline |
| Full target hit (£100 → £1M in 12 months) | 3–8% | Elite outcome — but the system maximises this |
| Total loss Phase 1 | 30–50% | Real risk — size the £100 as money you can lose |

**The system is designed to maximise the probability of the low-probability outcome while ensuring that failure is graceful (you never lose more than you put in, you always have data to learn from).**

---

## 24. Step-by-Step Getting Started Guide

Follow this in order. Do **not** skip the paper trading phase.

---

### STEP 1 — Install Prerequisites (30 minutes)

**On your local machine (Windows/Mac/Linux):**

```bash
# 1a. Install Python 3.12
# Windows: download from python.org, tick "Add to PATH"
# Mac:
brew install python@3.12
# Ubuntu/Debian:
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip -y

# 1b. Install Docker Desktop
# Download from docker.com/products/docker-desktop
# Start Docker Desktop and ensure it's running

# 1c. Install Git
# Windows: git-scm.com  |  Mac: brew install git  |  Ubuntu: sudo apt install git

# 1d. Verify everything works
python3.12 --version   # Should print Python 3.12.x
docker --version       # Should print Docker 24.x or higher
git --version
```

---

### STEP 2 — Create Broker Accounts (1–2 hours)

Do this while code is being set up. Both accounts can run in parallel.

#### 2a. Kraken (Phase 1 — Crypto, UK compliant)

> Binance exited the UK market in August 2023 and is no longer available to UK residents. Kraken (Payward Ltd) is FCA registered and fully legal for UK users.

1. Go to kraken.com → Create Account with your email
2. Complete KYC (Starter tier: email + phone; Intermediate tier: passport/driving licence — required for API trading)
3. Enable 2FA under **Security → Two-Factor Authentication**
4. Go to **Security → API → Add Key**
   - Key name: `autonomous-trader`
   - Permissions: ✅ Query Funds, ✅ Query Open Orders & Trades, ✅ Create & Modify Orders, ✅ Cancel/Close Orders
   - Restrict to your home IP (optional but recommended)
5. **Save the API Key and Private Key** — private key shown only once
6. Deposit GBP: Funding → Deposit → GBP (via UK Faster Payments — usually instant, no fees)
7. **No testnet on Kraken** — use the app's built-in paper trading mode (set `PAPER_TRADING=true`) to validate before going live

#### 2b. Interactive Brokers (Phase 2 — Stocks/Options)
1. Go to interactivebrokers.co.uk → Open Account
2. Choose **Individual** account, select **Stocks + Options + Futures**
3. Complete KYC (takes 1–3 business days for approval)
4. Once approved: log into Client Portal → Settings → API → **Enable Paper Trading**
5. Download **TWS (Trader Workstation)** — required for the API to connect locally
6. In TWS: Edit → Global Configuration → API → Settings → ✅ Enable ActiveX and Socket Clients, port `7497`

> **Start with IBKR paper trading account** — no real money, identical API to live.

---

### STEP 3 — Get Market Data API Keys (20 minutes)

#### 3a. Polygon.io (stock data)
1. Go to polygon.io → Sign Up (free tier: 15-min delayed data, paid: real-time)
2. Dashboard → API Keys → Copy your key
3. Free tier is fine for backtesting and initial paper trading

#### 3b. NewsAPI (news sentiment)
1. Go to newsapi.org → Get API Key (free: 100 requests/day)
2. Copy the API key from your account dashboard

---

### STEP 4 — Set Up the Project (15 minutes)

```bash
# 4a. Create your project directory
mkdir trading-app && cd trading-app

# 4b. Create Python virtual environment
python3.12 -m venv venv
source venv/bin/activate          # Mac/Linux
# OR: venv\Scripts\activate       # Windows

# 4c. Install core dependencies
pip install --upgrade pip
pip install \
  alpaca-py \
  python-binance \
  backtrader vectorbt \
  pandas numpy scipy \
  ta-lib \
  scikit-learn lightgbm \
  transformers torch \
  sqlalchemy asyncpg \
  redis aiohttp \
  prometheus-client \
  structlog \
  opentelemetry-sdk opentelemetry-exporter-otlp \
  fastapi uvicorn \
  pydantic-settings python-dotenv \
  requests websockets \
  python-telegram-bot

# Note: ta-lib requires the C library first:
# Mac: brew install ta-lib
# Ubuntu: sudo apt-get install libta-lib-dev
# Windows: download .whl from https://github.com/cgohlke/talib-build/releases
```

---

### STEP 5 — Configure Environment Variables (10 minutes)

```bash
# 5a. Create your .env file in the project root
touch .env
```

Open `.env` in any text editor and fill in:

```bash
# ===== SAFETY — ALWAYS START WITH PAPER =====
PAPER_TRADING=true
INITIAL_CAPITAL_GBP=100

# ===== KRAKEN (Primary crypto broker — FCA registered, UK legal) =====
KRAKEN_API_KEY=your_kraken_api_key_here
KRAKEN_PRIVATE_KEY=your_kraken_private_key_here

# ===== COINBASE ADVANCED TRADE (Backup, optional) =====
COINBASE_API_KEY=
COINBASE_PRIVATE_KEY=

# ===== INTERACTIVE BROKERS =====
IBKR_HOST=127.0.0.1
IBKR_PORT=7497                 # 7497 = paper trading, 7496 = live
IBKR_CLIENT_ID=1

# ===== MARKET DATA =====
POLYGON_API_KEY=your_polygon_key_here
NEWS_API_KEY=your_newsapi_key_here

# ===== NOTIFICATIONS =====
# Create a Telegram bot: search @BotFather on Telegram, /newbot, copy token
TELEGRAM_BOT_TOKEN=your_bot_token_here
# Get your chat ID: message @userinfobot on Telegram
TELEGRAM_CHAT_ID=your_chat_id_here

# ===== DATABASE =====
DB_PASSWORD=choose_a_strong_password_here
TIMESCALEDB_URL=postgresql://trader:choose_a_strong_password_here@localhost:5432/trading
REDIS_URL=redis://localhost:6379

# ===== RISK LIMITS — DO NOT CHANGE UNTIL EXPERIENCED =====
MAX_DAILY_LOSS_PCT=0.20
MAX_DRAWDOWN_PCT=0.35
MAX_POSITION_SIZE_PCT=0.25
PHASE1_TARGET_GBP=10000
PHASE2_TARGET_GBP=1000000
```

---

### STEP 6 — Start the Infrastructure (5 minutes)

```bash
# 6a. Start TimescaleDB + Redis + observability stack
docker compose up -d

# 6b. Verify all containers are running
docker compose ps
# Should show: timescaledb, redis, prometheus, grafana, loki, promtail, jaeger, alertmanager

# 6c. Check Grafana is accessible
# Open browser: http://localhost:3000
# Login: admin / trading123
# You should see the Grafana home screen — dashboards load after first data arrives

# 6d. Initialise the database schema
python scripts/init_db.py
# Creates: ohlcv_1m, ohlcv_5m, ohlcv_1h, ohlcv_1d, trades, orders, positions tables
```

---

### STEP 7 — Download Historical Data for Backtesting (1–2 hours)

```bash
# 7a. Download 3 years of BTC/USDT 15-minute candles from Binance
python scripts/download_historical.py \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2022-01-01

# 7b. Download ETH, SOL, BNB too (for strategy diversification)
python scripts/download_historical.py --symbol ETHUSDT --interval 15m --start 2022-01-01
python scripts/download_historical.py --symbol SOLUSDT --interval 15m --start 2022-01-01

# Expected download time: ~30–60 minutes per symbol
# Data stored in TimescaleDB automatically
```

---

### STEP 8 — Run Backtests (2–4 hours)

```bash
# 8a. Backtest the Phase 1 crypto scalp strategy
python backtest/runner.py \
  --strategy crypto_scalp \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --initial-capital 100

# 8b. Review the results — must meet ALL criteria before proceeding:
# ✅ Sharpe Ratio > 1.5
# ✅ Max Drawdown < 30%
# ✅ Win Rate > 45%
# ✅ Profit Factor > 1.5
# ✅ Total trades > 200

# 8c. Walk-forward validation (tests against unseen data)
python backtest/walk_forward.py --strategy crypto_scalp --folds 6

# Results saved to: backtest/results/crypto_scalp_YYYYMMDD.html
# Open in browser to see equity curve, trade log, and statistics
```

> If any criterion fails, adjust strategy parameters in `config/phase1_config.yaml` and re-run. **Do not proceed to paper trading if backtest fails.**

---

### STEP 9 — Paper Trade for 5–7 Days (1 week)

```bash
# 9a. Start the trading engine in paper mode
# Confirm PAPER_TRADING=true in your .env
python core/engine.py --phase 1

# 9b. Monitor in real-time
# Open http://localhost:3000 → "Trading Overview" dashboard
# You should see:
# • Equity curve updating (starts at £100 simulated)
# • Trades appearing in the trade log
# • Telegram notifications arriving on your phone

# 9c. Check paper trading performance daily:
python reporting/daily_summary.py
# Shows: trades today, P&L, win rate, any risk events triggered
```

**Paper trading pass criteria (after 5+ days):**
- [ ] At least 20 paper trades executed
- [ ] Positive P&L overall (edge confirmed)
- [ ] No unexpected errors or crashes
- [ ] All Telegram alerts received correctly
- [ ] Grafana dashboards showing data correctly
- [ ] Risk checks triggered correctly on simulated loss scenarios

---

### STEP 10 — Go Live with £100 (The Moment of Truth)

```bash
# Only proceed if Step 9 criteria are ALL met.

# 10a. Fund your Kraken account
# Kraken → Funding → Deposit → GBP (Faster Payments)
# Most UK banks: instant, no fees
# Deposit £100

# 10b. Kraken holds GBP — the feed converts to crypto as needed per trade
# No manual conversion required; the bot buys/sells crypto spot directly

# 10c. Switch to live mode
# Edit .env:
PAPER_TRADING=false

# 10d. Start the live engine
python core/engine.py --phase 1

# 10e. Verify first real trade
# Watch the dashboard — within the first market session you should see:
# • Real account balance reflected in Grafana
# • First trade notification on Telegram with real order ID
# • Position appearing in Binance app
```

---

### STEP 11 — Daily Operating Routine

Once live, your daily checklist:

```
Morning (before market session):
  □ Check Grafana "System Health" — all feeds green
  □ Check no overnight risk events in Telegram
  □ Review yesterday's trade log: python reporting/daily_summary.py --date yesterday

During trading (automated — just monitor):
  □ Grafana "Trading Overview" open in browser
  □ Telegram notifications arriving
  □ Intervene ONLY if circuit breaker fires (system halts itself)

Evening:
  □ Run daily summary report
  □ Note equity vs. target curve
  □ Review any trades with loss > 10% for signal quality
```

---

### STEP 12 — Phase Transition to £10,000

The system handles this automatically, but you should:

```bash
# When equity hits £10,000, the engine logs:
# "PHASE TRANSITION: Entering Phase 2 systematic trading"
# And sends Telegram alert

# What changes automatically:
# • Max position size drops from 25% → 10%
# • Phase 1 strategies disabled
# • Phase 2 strategies (dual momentum, pairs, mean reversion) enabled
# • Daily loss limit tightens from 20% → 5%

# What YOU should do at this point:
# 1. Open an Interactive Brokers account if not done (for stocks/options)
# 2. Consider IG Group spread betting account for UK tax-free gains
# 3. Transfer £5,000 to IBKR to activate stock/ETF strategies
# 4. Keep £5,000 on Binance for continued crypto allocation
# 5. Run Phase 2 backtests (should already be done — do it now if not):
python backtest/runner.py --strategy dual_momentum --start 2022-01-01 --end 2024-12-31
python backtest/runner.py --strategy pairs_trading --start 2022-01-01 --end 2024-12-31
python backtest/runner.py --strategy mean_reversion --start 2022-01-01 --end 2024-12-31
```

---

### STEP 13 — Tax Reporting (Ongoing)

```bash
# Export trade log at any time
python reporting/trade_log.py --format CSV --output trades_export.csv

# Generate HMRC CGT report at tax year end (5 April)
python reporting/tax_reporter.py --tax-year 2026-27 --output cgt_report_2627.csv

# The report includes:
# • Each disposal (sell): date, asset, proceeds, cost basis, gain/loss
# • Section 104 pool calculations for same-asset buys/sells
# • Bed & breakfast rule adjustments (30-day matching)
# • Running total vs. CGT annual allowance (£3,000 for 2025/26)
```

> **Important:** Keep ALL trade records. Crypto is a taxable asset under HMRC. Consult a UK tax accountant once annual gains exceed £3,000.

---

### Troubleshooting Quick Reference

| Problem | Check | Fix |
|---------|-------|-----|
| No trades executing | Grafana signal strength | Signals below threshold — market may be ranging |
| Binance API error 403 | IP whitelist in Binance API settings | Add your current IP or remove IP restriction |
| Database connection failed | `docker compose ps` | Restart: `docker compose restart timescaledb` |
| Feed latency > 500ms | System Health dashboard | Check internet connection; restart feed: `python data/feeds/binance_ws.py --restart` |
| Telegram alerts not arriving | Bot token in `.env` | Re-run: `python monitoring/telegram_bot.py --test` |
| Circuit breaker triggered | Drawdown > 25% | Review losing trades, fix strategy params, manually restart: `python core/engine.py --reset-circuit-breaker` |

---

## 23. Key Principles & Wisdom Embedded

1. **Cut losses fast, let winners run** — Asymmetric R:R is non-negotiable. Every strategy enforces 2:1 minimum.
2. **The market is always right** — Signals override opinions. No overriding the system.
3. **Capital preservation first** — The goal of every risk rule is to ensure there's always capital to compound.
4. **Compound is the real engine** — Even 1.4%/day becomes 100× in a year. The maths is unforgiving of drawdowns, not losses.
5. **Volatility is inventory** — Phase 1 seeks high volatility assets because volatility = opportunity for leveraged positions.
6. **Market regime awareness** — Momentum strategies in trending markets, mean-reversion in ranging. The engine detects regime and routes strategies accordingly.
7. **Never average down a loser** — Stops are stops. Adding to a losing position is the fastest path to ruin.
8. **Paper trade first, always** — No strategy goes live without backtesting + paper validation. No exceptions.
9. **Fees compound against you** — Every strategy must show positive expectancy *after* realistic fees. 0.1% per side on 10 trades/day = 2% daily drag.
10. **Diversification across uncorrelated strategies** — In Phase 2, no single strategy collapse should cause overall ruin.
