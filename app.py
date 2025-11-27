import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from PIL import Image
from datetime import datetime, timedelta
import numpy as np
import json
import sqlite3

# Auto-refresh için
# pip install streamlit-autorefresh
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# 1. GENEL AYARLAR
# =============================================================================
st.set_page_config(
    page_title="AI Kripto Analist Pro",
    layout="wide",
    page_icon="🛡️"
)

# ----------------- SESSION STATE BAŞLANGIÇ ----------------- #
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini"
if "api_status" not in st.session_state:
    st.session_state.api_status = False
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "trader_mode" not in st.session_state:
    st.session_state.trader_mode = "Dengeli"
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "plan_history" not in st.session_state:
    st.session_state.plan_history = []
if "live_refresh_mode" not in st.session_state:
    st.session_state.live_refresh_mode = "Manuel"

MAX_REQUESTS = 50  # Bir session'da maksimum analiz isteği

# =============================================================================
# 1.1. TEMA / CSS
# =============================================================================
st.markdown(
    """
    <style>
        .stApp { 
            background-color: #05060a; 
            color: #e6edf3;
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1300px;
        }
        .stFileUploader { 
            border: 2px dashed #4CAF50 !important; 
            border-radius: 10px; 
            padding: 20px; 
        }
        .risk-card, .section-card, .ai-card {
            background: radial-gradient(circle at top left, #161b22 0, #05060a 60%);
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(99,110,123,0.6);
            color: #e6edf3;
            margin-bottom: 12px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.55);
        }
        .risk-highlight {
            background: rgba(22,27,34,0.9);
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #30363d;
            font-size: 14px;
            margin-top: 8px;
        }
        .small-muted {
            font-size: 12px;
            color: #8b949e;
        }
        .history-badge {
            background: #161b22;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 11px;
            border: 1px solid #30363d;
            display: inline-block;
            margin-right: 6px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# D. VERİTABANI (SQLite) – PERSISTENCE
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_db_connection():
    conn = sqlite3.connect("ai_kripto_analist.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            exchange TEXT,
            mode TEXT,
            direction TEXT,
            balance REAL,
            risk_type TEXT,
            risk_pct REAL,
            risk_amount REAL,
            leverage REAL,
            entry REAL,
            stop REAL,
            tp1 REAL,
            tp2 REAL,
            tp3 REAL,
            quantity REAL,
            margin REAL,
            liq_price REAL
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_plan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            timeframe TEXT,
            direction TEXT,
            mode TEXT,
            balance REAL,
            risk_pct REAL,
            risk_amount REAL,
            notes TEXT,
            plan_text TEXT
        )
        """
    )

    conn.commit()

def save_risk_record_db(rec: dict):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO risk_history 
            (timestamp, symbol, exchange, mode, direction, balance, risk_type, 
             risk_pct, risk_amount, leverage, entry, stop, tp1, tp2, tp3, 
             quantity, margin, liq_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("timestamp"),
                rec.get("symbol"),
                rec.get("exchange"),
                rec.get("mode"),
                rec.get("direction"),
                rec.get("balance"),
                rec.get("risk_type"),
                rec.get("risk_pct"),
                rec.get("risk_amount"),
                rec.get("leverage"),
                rec.get("entry"),
                rec.get("stop"),
                rec.get("tp1"),
                rec.get("tp2"),
                rec.get("tp3"),
                rec.get("quantity"),
                rec.get("margin"),
                rec.get("liq_price"),
            ),
        )
        conn.commit()
    except Exception as e:
        st.warning(f"Risk kaydı veritabanına yazılamadı: {e}")

def save_plan_record_db(rec: dict):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO trade_plan_history
            (timestamp, symbol, timeframe, direction, mode, balance, risk_pct,
             risk_amount, notes, plan_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.get("timestamp"),
                rec.get("symbol"),
                rec.get("timeframe"),
                rec.get("direction"),
                rec.get("mode"),
                rec.get("balance"),
                rec.get("risk_pct"),
                rec.get("risk_amount"),
                rec.get("notes"),
                rec.get("plan_text"),
            ),
        )
        conn.commit()
    except Exception as e:
        st.warning(f"Trade plan kaydı veritabanına yazılamadı: {e}")

def load_risk_history_db(limit: int = 200) -> pd.DataFrame | None:
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "SELECT * FROM risk_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()
        if not rows:
            return None
        return pd.DataFrame([dict(r) for r in rows])
    except Exception as e:
        st.warning(f"Risk history DB'den okunamadı: {e}")
        return None

def load_plan_history_db(limit: int = 200) -> pd.DataFrame | None:
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "SELECT * FROM trade_plan_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = c.fetchall()
        if not rows:
            return None
        return pd.DataFrame([dict(r) for r in rows])
    except Exception as e:
        st.warning(f"Trade plan history DB'den okunamadı: {e}")
        return None

def clear_risk_history_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM risk_history")
        conn.commit()
    except Exception as e:
        st.warning(f"Risk history DB temizlenemedi: {e}")

def clear_plan_history_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM trade_plan_history")
        conn.commit()
    except Exception as e:
        st.warning(f"Trade plan history DB temizlenemedi: {e}")

init_db()

# =============================================================================
# 2. YARDIMCI FONKSİYONLAR & GÜVENLİK
# =============================================================================
def mask_error(err) -> str:
    text = str(err)
    key = st.session_state.api_key
    if key:
        tail = key[-8:]
        text = text.replace(tail, "********")
    return text

def validate_image(file) -> bool:
    allowed_types = {"image/png", "image/jpeg", "image/jpg"}
    if file.type not in allowed_types:
        return False

    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > 10 * 1024 * 1024:
        return False

    try:
        img = Image.open(file)
        img.verify()
    except Exception:
        return False
    finally:
        file.seek(0)

    return True

@st.cache_data(ttl=900)
def get_fear_and_greed_index():
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5)
        response.raise_for_status()
        data = response.json()
        value = int(data["data"][0]["value"])
        label = data["data"][0]["value_classification"]
    except Exception:
        value, label = 50, "Neutral"
    fetched_at = datetime.utcnow()
    return value, label, fetched_at

@st.cache_data(ttl=3600)
def get_mock_macro_events():
    today = datetime.now()
    events_data = [
        {
            "date": today + timedelta(days=1),
            "time": "15:30",
            "currency": "USD",
            "event": "ABD Çekirdek TÜFE",
            "impact": "high",
            "forecast": "%3.2",
        },
        {
            "date": today + timedelta(days=2),
            "time": "21:00",
            "currency": "USD",
            "event": "Fed Faiz Kararı",
            "impact": "high",
            "forecast": "%4.50",
        },
    ]
    return pd.DataFrame(events_data)

def create_gauge_chart(value, label):
    if value < 25:
        color = "#ff4b4b"
    elif value < 45:
        color = "#ffa500"
    elif value < 55:
        color = "#f0e68c"
    elif value < 75:
        color = "#90ee90"
    else:
        color = "#32cd32"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"<b>{label}</b>", "font": {"size": 18, "color": "white"}},
            number={"font": {"size": 30, "color": color}},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

def configure_gemini(api_key: str):
    clean_key = api_key.strip()
    genai.configure(api_key=clean_key, transport="rest")
    return clean_key

def _supports_generate_content(m) -> bool:
    methods = getattr(m, "supported_generation_methods", None)
    if not methods:
        return False
    return "generateContent" in methods

@st.cache_resource(show_spinner=False)
def get_gemini_model(api_key: str, preferred_pattern: str):
    clean_key = configure_gemini(api_key)
    try:
        all_models = list(genai.list_models())
    except Exception as e:
        return None, f"Model listesi alınamadı: {e}", None

    if not all_models:
        return None, "Bu API anahtarıyla erişilebilir model bulunamadı.", None

    generative_models = [m for m in all_models if _supports_generate_content(m)]
    if not generative_models:
        return None, "generateContent destekleyen model bulunamadı.", None

    candidates = []
    if preferred_pattern:
        for m in generative_models:
            if preferred_pattern in m.name:
                candidates.append(m.name)

    if not candidates:
        for m in generative_models:
            if "gemini" in m.name and "vision" in m.name:
                candidates.append(m.name)

    if not candidates:
        for m in generative_models:
            if "gemini" in m.name:
                candidates.append(m.name)

    if not candidates:
        candidates = [m.name for m in generative_models]

    last_err = None
    tried = []
    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            _ = model.generate_content("Test")
            return model, None, name
        except Exception as e:
            tried.append(name)
            last_err = e
            continue

    err_msg = f"Şu modeller denendi ama çalışmadı: {tried}. Son hata: {last_err}"
    return None, err_msg, None

def build_global_market_context():
    fg_val, fg_lbl, fg_time = get_fear_and_greed_index()
    lines = [
        f"Global Crypto Fear & Greed Index şu anda {fg_val} ({fg_lbl}).",
        "Piyasa verileri analizde ağırlıklı olarak Binance, OKX, Bybit, Coinbase ve Upbit spot/futures verileri üzerinden düşünülmelidir.",
        "Her analiz, borsa verilerinin gecikme ve API limitleri olabileceği varsayımıyla temkinli yapılmalıdır."
    ]
    return "\n".join(lines)

def get_trader_mode_description(mode: str) -> str:
    if mode == "Scalper":
        return (
            "Çok kısa vadeli (1–5–15dk) zaman dilimlerinde, hızlı giriş-çıkış yapan bir scalper gibi düşün. "
            "Dar stoplar, küçük ama sık alınan karlar, yüksek volatiliteye dikkat. Likidite, spread ve wick riskine vurgu yap."
        )
    elif mode == "Swing":
        return (
            "Orta vadeli (4H / 1D) zaman dilimlerinde, 3–15 gün arası elde tutulabilen swing işlemler. "
            "Ana trend, güçlü destek/direnç bölgeleri ve R/R dengesini ön plana çıkar."
        )
    elif mode == "Pozisyon":
        return (
            "Uzun vadeli (1D/1W) pozisyonlar, haftalar-aylar sürebilecek işlemler. "
            "Makro trend, döngüsel yapılar ve sermaye korunması kritik."
        )
    else:
        return (
            "Kısa ve orta vadenin dengeli karışımı. Hem intraday hem birkaç günlük işlemlere uygun, "
            "nötr risk yaklaşımı."
        )

# =============================================================================
# OHLC VERİ ÇEKME – SADECE BORSALAR (Binance / OKX / Bybit / Coinbase / Upbit)
# =============================================================================
@st.cache_data(ttl=60)
def get_ohlc_binance(symbol: str, interval: str = "1h", limit: int = 200):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if not raw:
            return None, "Binance API boş veri döndürdü."
        data = []
        for k in raw:
            data.append({
                "time": pd.to_datetime(k[0], unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(data).sort_values("time")
        return df, None
    except Exception as e:
        return None, f"Binance hatası: {e}"

@st.cache_data(ttl=60)
def get_ohlc_okx(inst_id: str, bar: str = "1H", limit: int = 200):
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json().get("data", [])
        if not raw:
            return None, "OKX API boş veri döndürdü."
        data = []
        for k in raw:
            data.append({
                "time": pd.to_datetime(int(k[0]), unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(data).sort_values("time")
        return df, None
    except Exception as e:
        return None, f"OKX hatası: {e}"

@st.cache_data(ttl=60)
def get_ohlc_bybit(symbol: str, interval: str = "1h", limit: int = 200):
    interval_map = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }
    bybit_interval = interval_map.get(interval, "60")
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",  # USDT perpetual
        "symbol": symbol,
        "interval": bybit_interval,
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        res = r.json()
        raw = (res.get("result") or {}).get("list", []) or []
        if not raw:
            return None, "Bybit API boş veri döndürdü."
        # Bybit genelde son kaydı ilk verir, ters çeviriyoruz
        raw = raw[::-1]
        data = []
        for k in raw:
            # [startTime, open, high, low, close, volume, turnOver, ...]
            data.append({
                "time": pd.to_datetime(int(k[0]), unit="ms"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(data).sort_values("time")
        return df, None
    except Exception as e:
        return None, f"Bybit hatası: {e}"

@st.cache_data(ttl=60)
def get_ohlc_coinbase(product_id: str, interval: str = "1h", limit: int = 200):
    granularity_map = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 21600,
        "1d": 86400,
    }
    granularity = granularity_map.get(interval, 3600)
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "granularity": granularity,
        "limit": limit,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if not raw:
            return None, "Coinbase API boş veri döndürdü."
        # Coinbase format: [ time, low, high, open, close, volume ]
        data = []
        for k in raw:
            data.append({
                "time": pd.to_datetime(int(k[0]), unit="s"),
                "open": float(k[3]),
                "high": float(k[2]),
                "low": float(k[1]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(data).sort_values("time")
        return df, None
    except Exception as e:
        return None, f"Coinbase hatası: {e}"

@st.cache_data(ttl=60)
def get_ohlc_upbit(market: str, interval: str = "1h", limit: int = 200):
    base_url = "https://api.upbit.com/v1/candles"
    if interval == "1d":
        url = f"{base_url}/days"
    else:
        unit_map = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
        }
        unit = unit_map.get(interval, 60)
        url = f"{base_url}/minutes/{unit}"

    params = {"market": market, "count": limit}
    headers = {"Accept": "application/json"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        raw = r.json() or []
        if not raw:
            return None, "Upbit API boş veri döndürdü."
        data = []
        # Upbit en son veriyi ilk verir; ters çeviriyoruz
        raw = raw[::-1]
        for k in raw:
            # candle_date_time_utc, opening_price, high_price, low_price, trade_price, candle_acc_trade_volume
            data.append({
                "time": pd.to_datetime(k["candle_date_time_utc"]),
                "open": float(k["opening_price"]),
                "high": float(k["high_price"]),
                "low": float(k["low_price"]),
                "close": float(k["trade_price"]),
                "volume": float(k["candle_acc_trade_volume"]),
            })
        df = pd.DataFrame(data).sort_values("time")
        return df, None
    except Exception as e:
        return None, f"Upbit hatası: {e}"

def compute_indicators(df: pd.DataFrame):
    df = df.copy().sort_values("time")
    close = df["close"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    roll_down = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi14"] = rsi.values

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    ma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    df["bb_mid"] = ma20
    df["bb_upper"] = ma20 + 2 * std20
    df["bb_lower"] = ma20 - 2 * std20

    return df

def create_live_market_figure(df: pd.DataFrame):
    df = df.copy().sort_values("time")
    last = df.iloc[-1]
    last_price = float(last["close"])
    last_time = last["time"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Fiyat + EMA + Bollinger + Anlık Fiyat", "RSI (14)", "MACD (12,26,9)")
    )

    # Candle
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#00e676",
            increasing_fillcolor="rgba(0,230,118,0.55)",
            decreasing_line_color="#ff5252",
            decreasing_fillcolor="rgba(255,82,82,0.55)",
        ),
        row=1, col=1
    )

    # EMA’ler
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema20"],
            mode="lines",
            name="EMA 20",
            line=dict(color="#00bcd4", width=1.6)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["ema50"],
            mode="lines",
            name="EMA 50",
            line=dict(color="#ffb300", width=1.6)
        ),
        row=1, col=1
    )

    # Bollinger
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["bb_upper"],
            mode="lines",
            name="BB Upper",
            line=dict(color="rgba(189,189,189,0.9)", width=1, dash="dot")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["bb_mid"],
            mode="lines",
            name="BB Mid",
            line=dict(color="rgba(158,158,158,0.8)", width=1)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["bb_lower"],
            mode="lines",
            name="BB Lower",
            line=dict(color="rgba(189,189,189,0.9)", width=1, dash="dot")
        ),
        row=1, col=1
    )

    # Anlık fiyat çizgisi + nokta
    fig.add_hline(
        y=last_price,
        line_dash="dot",
        line_color="#ffffff",
        line_width=1.3,
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[last_time],
            y=[last_price],
            mode="markers+text",
            name="Anlık Fiyat",
            marker=dict(
                color="#ffffff",
                size=9,
                line=dict(color="#00e676", width=1.5)
            ),
            text=[f"{last_price:.4f}"],
            textposition="middle right",
            textfont=dict(color="#ffffff", size=11),
            showlegend=False
        ),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["rsi14"],
            mode="lines",
            name="RSI 14",
            line=dict(color="#fdd835", width=1.5)
        ),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#42a5f5", row=2, col=1)

    # MACD
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["macd_hist"],
            name="MACD Hist",
            marker_color="#26c6da",
            opacity=0.8
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["macd"],
            mode="lines",
            name="MACD",
            line=dict(color="#ab47bc", width=1.4)
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["macd_signal"],
            mode="lines",
            name="Signal",
            line=dict(color="#ff7043", width=1.3)
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        paper_bgcolor="#05060a",
        plot_bgcolor="#05060a",
        font={"color": "#e6edf3"},
        margin=dict(l=10, r=10, t=40, b=20),
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont=dict(color="#9ea7b3")
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(80,80,80,0.3)",
        tickfont=dict(color="#9ea7b3")
    )

    return fig

# =============================================================================
# STRUCTURED OUTPUT HELPER’LARI
# =============================================================================
def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

def render_trade_plan_structured(plan: dict):
    st.markdown("#### 1️⃣ Genel Bakış")
    overview = plan.get("overview", {})
    if overview:
        st.write(overview.get("summary", ""))
        col1, col2, col3 = st.columns(3)
        if "trend" in overview:
            col1.metric("Trend", overview.get("trend"))
        if "volatility" in overview:
            col2.metric("Volatilite", overview.get("volatility"))
        if "context" in overview:
            col3.write(f"**Bağlam:** {overview.get('context')}")
    else:
        st.info("Genel bakış bilgisi yok.")

    st.markdown("#### 2️⃣ Long Senaryosu")
    long_s = plan.get("long_scenario")
    if long_s:
        st.write(long_s.get("description", ""))
        c1, c2, c3, c4 = st.columns(4)
        if "entry_zone" in long_s:
            c1.write(f"**Giriş Bölgesi:** {long_s.get('entry_zone')}")
        if "stop_loss" in long_s:
            c2.write(f"**Stop:** {long_s.get('stop_loss')}")
        if "tp1" in long_s:
            c3.write(f"**TP1:** {long_s.get('tp1')}")
        if "tp2" in long_s:
            c4.write(f"**TP2:** {long_s.get('tp2')}")
        extra_tps = long_s.get("extra_targets")
        if extra_tps:
            st.write("Ek Hedefler:")
            for t in extra_tps:
                st.write(f"- {t}")
    else:
        st.info("Long senaryosu belirtilmemiş.")

    st.markdown("#### 3️⃣ Short Senaryosu")
    short_s = plan.get("short_scenario")
    if short_s:
        st.write(short_s.get("description", ""))
        c1, c2, c3, c4 = st.columns(4)
        if "entry_zone" in short_s:
            c1.write(f"**Giriş Bölgesi:** {short_s.get('entry_zone')}")
        if "stop_loss" in short_s:
            c2.write(f"**Stop:** {short_s.get('stop_loss')}")
        if "tp1" in short_s:
            c3.write(f"**TP1:** {short_s.get('tp1')}")
        if "tp2" in short_s:
            c4.write(f"**TP2:** {short_s.get('tp2')}")
        extra_tps = short_s.get("extra_targets")
        if extra_tps:
            st.write("Ek Hedefler:")
            for t in extra_tps:
                st.write(f"- {t}")
    else:
        st.info("Short senaryosu belirtilmemiş veya şu anda zayıf olabilir.")

    st.markdown("#### 4️⃣ Risk Yönetimi")
    rm = plan.get("risk_management")
    if rm:
        st.write(rm.get("summary", ""))
        c1, c2, c3 = st.columns(3)
        if "position_size_logic" in rm:
            c1.write("**Pozisyon Büyüklüğü:**")
            c1.write(rm.get("position_size_logic"))
        if "rr_ratios" in rm:
            c2.write("**R/R Oranları:**")
            for k, v in rm.get("rr_ratios", {}).items():
                c2.write(f"- {k}: {v}")
        if "max_risk_comment" in rm:
            c3.write("**Max Risk Yorumu:**")
            c3.write(rm.get("max_risk_comment"))
    else:
        st.info("Risk yönetimi bölümüne dair structured veri yok.")

    st.markdown("#### 5️⃣ Zamanlama")
    timing = plan.get("timing")
    if timing:
        st.write(timing.get("summary", ""))
        details = timing.get("details", [])
        if details:
            for d in details:
                st.write(f"- {d}")
    else:
        st.info("Zamanlama bilgisi yok.")

    st.markdown("#### 6️⃣ Dikkat Edilmesi Gerekenler")
    warnings_list = plan.get("warnings")
    if warnings_list:
        for w in warnings_list:
            st.warning(w)
    else:
        st.info("Ekstra uyarı bulunmuyor.")

# =============================================================================
# AI ANALİZ FONKSİYONLARI
# =============================================================================
def analyze_chart_with_gemini(model, image: Image.Image, extra_context: str = "", trader_mode: str = "Dengeli") -> str:
    safety_header = """
    ÇOK ÖNEMLİ TALİMATLAR:
    - Kesin "al" veya "sat" sinyali verme.
    - Kaldıraçlı işlem açmayı doğrudan önermemelisin.
    - Cevaplarının yatırım tavsiyesi değil, eğitim amaçlı bir analiz örneği olduğunu belirt.
    """
    methodology_block = """
    Analiz yaparken, küresel olarak kabul görmüş teknik analiz prensiplerini kullan:
    - Dow Teorisi, trend yapısı (yükselen/düşen tepe-dip)
    - Destek/direnç ve arz-talep bölgeleri
    - Momentum (RSI, MACD, Stokastik) mantığıyla aşırı alım/satım yorumları
    - Volatilite (Bollinger, ATR) ile stop ve hedef mesafelerini değerlendirme
    - Hacim analizi: kırılımların hacimle desteklenip desteklenmediği
    - Risk/Ödül oranı (R/R) – en az 1:2 hedefle
    - Pozisyon büyüklüğü ve max sermaye riski gibi risk yönetimi prensipleri
    """

    mode_desc = get_trader_mode_description(trader_mode)

    base_prompt = f"""
    {safety_header}

    Sen deneyimli bir Türk teknik analist ve kripto trader'sın.

    {methodology_block}

    Trader modu: {trader_mode}
    Bu modun anlamı:
    {mode_desc}

    Analizini özellikle bu trader modunun bakış açısından yap.

    Ek bağlam (kullanıcı notu + piyasa verileri):
    {extra_context}

    Cevap formatı:

    1️⃣ Trend:
    - Genel trend yönü (Boğa / Ayı / Yatay)
    - Kısa, orta ve uzun vade için yorum
    - Dow teorisine göre tepe/dip yapısı

    2️⃣ Destek & Direnç:
    - En az 3 destek ve 3 direnç seviyesi (mümkünse sayısal)
    - Bu seviyelerin neden önemli olduğuna dair kısa açıklama

    3️⃣ Formasyonlar:
    - Olası formasyon(lar) (üçgen, OBO, TOBO, çift dip/tepe vs.)
    - Hedef fiyat bölgesi ve formasyon aşaması (oluşum/kırılım/retest)

    4️⃣ Momentum & Volatilite:
    - RSI/MACD mantığıyla aşırı alım/aşırı satım değerlendirmesi
    - Volatilite durumu (yüksek/düşük) ve stop/TP mesafelerine etkisi

    5️⃣ İşlem Stratejisi:
    - Olası AL (long) veya SAT (short) stratejisi (giriş bölgesi, stop, TP1/TP2)
    - Risk yönetimi (max risk %, R/R, pozisyon küçültme vb.)

    6️⃣ Risk Uyarıları:
    - Ani spike, likidite boşluğu vb. anormallikler
    - Makro/haber/FED gibi dış etkenlere karşı genel uyarı

    Yanıtı zengin biçimlendirilmiş Türkçe markdown olarak verebilirsin.
    """

    response = model.generate_content([base_prompt, image])
    return response.text if hasattr(response, "text") else str(response)

def generate_ai_trade_plan(model, symbol: str, timeframe: str, balance: float,
                           risk_amount: float, direction: str, trader_mode: str,
                           extra_notes: str, global_ctx: str) -> str:
    safety_header = """
    ÇOK ÖNEMLİ:
    - Kesin al/sat emri verme, sadece senaryo ve plan üret.
    - Kaldıraç ve yüksek risk konusunda mutlaka uyarı yap.
    - Bu çıktı yatırım tavsiyesi değildir, sadece eğitim amaçlı bir örnek trade planıdır.
    """

    json_schema_description = """
    Yanıtını mutlaka SADECE aşağıdaki JSON şemasına uygun şekilde ver.
    JSON dışında hiçbir açıklama, metin veya formatlama ekleme.

    {
      "overview": {
        "summary": "kısa paragraf",
        "trend": "boğa/ayı/yatay",
        "volatility": "yüksek/orta/düşük",
        "context": "BTC, makro vs ile ilişkili kısa metin"
      },
      "long_scenario": {
        "description": "long senaryosunun açıklaması",
        "entry_zone": "örneğin: 42000-42200",
        "stop_loss": "örneğin: 41500",
        "tp1": "örneğin: 43000",
        "tp2": "örneğin: 44000",
        "extra_targets": ["opsiyonel hedef1", "opsiyonel hedef2"]
      },
      "short_scenario": {
        "description": "short senaryosunun açıklaması (eğer mantıklıysa)",
        "entry_zone": "örneğin: 44000-44200",
        "stop_loss": "örneğin: 44700",
        "tp1": "örneğin: 43000",
        "tp2": "örneğin: 42000",
        "extra_targets": ["opsiyonel hedef1", "opsiyonel hedef2"]
      },
      "risk_management": {
        "summary": "risk yönetimi hakkında genel yorum",
        "position_size_logic": "örnek pozisyon büyüklüğü mantığı",
        "rr_ratios": {
          "long_tp1": "örneğin 1:2",
          "long_tp2": "örneğin 1:3",
          "short_tp1": "örneğin 1:2",
          "short_tp2": "örneğin 1:3"
        },
        "max_risk_comment": "Seçilen risk tutarı/ yüzdesinin mantıklılığı hakkında kısa yorum"
      },
      "timing": {
        "summary": "zamanlama mantığının genel özeti",
        "details": [
          "seçilen moda göre 1-3 madde"
        ]
      },
      "warnings": [
        "Haber akışı ve volatilite patlamalarına dair uyarı",
        "Likidite ve kaldıraç riskleri hakkında uyarı"
      ]
    }
    """

    prompt = f"""
    {safety_header}

    Aşağıdaki parametrelere göre örnek bir trade planı hazırla:

    Sembol: {symbol}
    Zaman dilimi: {timeframe}
    Hesap büyüklüğü: {balance} USD
    Bu trade'de riske edilen tutar: {risk_amount} USD
    Yön tercihi: {direction} (Long, Short veya Her İkisi)
    Trader modu: {trader_mode}

    Kullanıcı notları:
    {extra_notes}

    Global piyasa bağlamı:
    {global_ctx}

    {json_schema_description}

    TEKRAR: Yanıtın GEÇERLİ bir JSON olmak zorunda.
    JSON dışında açıklama yazma, markdown kullanma, yorum ekleme.
    """

    resp = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

# =============================================================================
# 3. BASİT OTURUM AÇMA (APP_PASSWORD VARSA)
# =============================================================================
def has_app_password() -> bool:
    try:
        return "APP_PASSWORD" in st.secrets
    except Exception:
        return False

def login_ui():
    with st.sidebar:
        st.subheader("🔑 Giriş")
        password = st.text_input("Uygulama Şifresi", type="password")
        if st.button("Giriş Yap"):
            real = st.secrets.get("APP_PASSWORD", "")
            if password and password == real:
                st.session_state.authenticated = True
                st.success("Giriş başarılı.")
            else:
                st.error("Yanlış şifre.")

require_auth = has_app_password()
if require_auth and not st.session_state.authenticated:
    login_ui()
    st.stop()

# =============================================================================
# 4. SIDEBAR: API, MODEL VE TRADER MODU
# =============================================================================
with st.sidebar:
    st.header("🔐 API Bağlantısı")

    cloud_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            cloud_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        cloud_key = None

    st.subheader("🤖 Model Tercihi (Pattern)")
    preferred_pattern = st.selectbox(
        "Tercih edilen model tipi",
        options=[
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro-vision",
            "gemini",
            "chat-bison",
            "text-bison",
        ],
        index=3
    )

    if st.button("🔓 API Anahtarını Temizle"):
        st.session_state.api_key = ""
        st.session_state.api_status = False
        st.session_state.model_name = "gemini"
        st.session_state.last_error = None
        st.success("API anahtarı hafızadan temizlendi.")

    if cloud_key:
        st.success("☁️ Cloud API Key kullanılıyor")
        st.session_state.api_key = cloud_key
        st.session_state.model_name = preferred_pattern

        model, err, resolved_name = get_gemini_model(
            st.session_state.api_key,
            st.session_state.model_name
        )
        if model:
            st.session_state.api_status = True
            st.session_state.last_error = None
            if resolved_name and resolved_name != st.session_state.model_name:
                st.info(f"Pattern: `{st.session_state.model_name}` → Gerçek model: **{resolved_name}**")
            st.session_state.model_name = resolved_name
        else:
            st.session_state.api_status = False
            st.session_state.last_error = err
            safe_err = mask_error(err)
            st.error(f"❌ Bağlantı hatası: {safe_err}")
    else:
        user_key_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.api_key,
            type="password",
            help="API anahtarını Google AI Studio / MakerSuite'ten alabilirsin."
        )
        st.session_state.model_name = preferred_pattern

        if st.button("Bağlan ve Test Et"):
            if user_key_input.strip():
                with st.spinner("Gemini REST API'ye bağlanılıyor..."):
                    model, err, resolved_name = get_gemini_model(
                        user_key_input,
                        st.session_state.model_name
                    )
                    if model:
                        st.session_state.api_key = user_key_input.strip()
                        st.session_state.api_status = True
                        st.session_state.last_error = None
                        if resolved_name and resolved_name != st.session_state.model_name:
                            st.info(f"Pattern: `{st.session_state.model_name}` → Gerçek model: **{resolved_name}**")
                        st.session_state.model_name = resolved_name
                        st.success(f"✅ Bağlantı başarılı! Aktif model: {resolved_name}")
                    else:
                        st.session_state.api_status = False
                        st.session_state.last_error = err
                        safe_err = mask_error(err)
                        st.error(f"❌ Bağlantı hatası: {safe_err}")
            else:
                st.warning("Lütfen API anahtarını giriniz.")

    if st.session_state.api_status:
        st.caption(f"🔌 API durumu: **Bağlı** | Model: `{st.session_state.model_name}`")
    else:
        st.caption("🔌 API durumu: **Bağlı değil**")
        if st.session_state.last_error:
            st.caption(f"Son hata: `{mask_error(st.session_state.last_error)}`")

    st.markdown("---")
    st.subheader("🎯 Trader Modu")
    mode_options = ["Dengeli", "Scalper", "Swing", "Pozisyon"]
    current_index = mode_options.index(st.session_state.trader_mode) if st.session_state.trader_mode in mode_options else 0
    selected_mode = st.radio(
        "Stilini seç",
        options=mode_options,
        index=current_index
    )
    st.session_state.trader_mode = selected_mode
    st.caption(get_trader_mode_description(selected_mode))

# =============================================================================
# 5. ANA GÖVDE – TAB YAPISI
# =============================================================================
st.title("🧠 AI Kripto Analist Pro")

tab_analysis, tab_tools, tab_live, tab_planner, tab_history = st.tabs(
    ["📊 Grafik Analizi", "🛠 Araçlar", "📈 Canlı Market", "🤖 Trade Planlayıcı", "📚 History"]
)

# ------------------------ TAB 1: GRAFİK ANALİZİ ------------------------ #
with tab_analysis:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### 📤 Grafik Yükle")
        uploaded_files = st.file_uploader(
            "TradingView / borsa grafiği ekran görüntüsü (Max 15 görsel)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="chart_upload"
        )

        extra_notes = st.text_area(
            "İsteğe bağlı not / ek bilgi",
            help="Örn: 'BTCUSDT 4H, son düşüş sonrası durum' gibi.",
            key="chart_notes"
        )

    with col_right:
        st.markdown("### ℹ️ Kullanım Notları")
        st.markdown(
            """
            <div class="section-card">
            • Birden fazla grafiği aynı anda yükleyebilirsin.  
            • Her grafik için ayrı teknik analiz üretilir.  
            • Analizler <b>öğretici ve temkinli</b> olarak tasarlanmıştır.  
            • Çıkan sonuçlar yatırım tavsiyesi değildir.
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_files:
        if len(uploaded_files) > 15:
            st.error("⚠️ Maksimum 15 dosya yükleyebilirsiniz.")
        else:
            start_analysis = st.button("🔍 Analizi Başlat", type="primary")
            if start_analysis:
                if st.session_state.request_count + len(uploaded_files) > MAX_REQUESTS:
                    st.error("⚠️ Maksimum istek limitine ulaştınız. Sayfayı yenileyip yeni oturum başlatın.")
                else:
                    if not st.session_state.api_status:
                        st.error("⚠️ Önce sol menüden API bağlantısını yapmalısınız.")
                    else:
                        model, err, resolved_name = get_gemini_model(
                            st.session_state.api_key,
                            st.session_state.model_name
                        )
                        if not model:
                            safe_err = mask_error(err)
                            st.error(f"Model oluşturulamadı: {safe_err}")
                        else:
                            if resolved_name and resolved_name != st.session_state.model_name:
                                st.session_state.model_name = resolved_name
                                st.info(f"Analiz modeli **{resolved_name}** olarak güncellendi.")

                            st.session_state.request_count += len(uploaded_files)
                            global_ctx = build_global_market_context()
                            combined_extra = (extra_notes or "") + "\n\n" + global_ctx
                            trader_mode = st.session_state.get("trader_mode", "Dengeli")

                            st.markdown("---")
                            st.subheader("🧠 Yapay Zeka Grafik Analizleri")
                            progress_bar = st.progress(0)
                            total = len(uploaded_files)

                            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                                progress_bar.progress(idx / total)

                                if not validate_image(uploaded_file):
                                    st.error(f"❌ Geçersiz dosya: {uploaded_file.name}")
                                    continue

                                try:
                                    image = Image.open(uploaded_file).convert("RGB")
                                except Exception as e:
                                    st.error(f"📁 {uploaded_file.name} açılamadı: {e}")
                                    continue

                                col_img, col_txt = st.columns([1, 2])
                                with col_img:
                                    st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)

                                with col_txt:
                                    with st.spinner("Grafik analiz ediliyor..."):
                                        try:
                                            text = analyze_chart_with_gemini(
                                                model=model,
                                                image=image,
                                                extra_context=combined_extra,
                                                trader_mode=trader_mode
                                            )
                                            st.markdown(text)
                                        except Exception as e:
                                            st.error(f"Analiz sırasında hata: {e}")

                                st.markdown("---")

                            progress_bar.empty()

# ------------------------ TAB 2: ARAÇLAR (RİSK + PİYASA PANELİ + MANUEL HABER/SENTIMENT) ------------------------ #
with tab_tools:
    st.markdown("### 🧮 Akıllı Risk, Marjin & Likidasyon Hesaplayıcı")

    with st.expander("Akıllı Risk Hesaplayıcı", expanded=True):
        trader_mode = st.session_state.trader_mode

        mode_recommendations = {
            "Scalper": "Önerilen risk: %0.2 – %0.5 • Çok dar stop • 1–5dk volatilitesine dikkat • Spread ve wick’e karşı tetikte ol.",
            "Swing": "Önerilen risk: %0.5 – %1.5 • Daha geniş stop • 2–3 TP’li yapı mantıklı.",
            "Pozisyon": "Önerilen risk: %0.25 – %0.75 • Günlük/haftalık trend kritik • Makro risklere dikkat.",
            "Dengeli": "Önerilen risk: %0.5 – %1.0 • R/R en az 1:2 hedeflenmeli."
        }

        st.markdown(
            f"""
            <div class="risk-card">
                <b>🎯 Seçilen Trader Modu:</b> {trader_mode}<br>
                <div class="risk-highlight">{mode_recommendations[trader_mode]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        symbol_risk = col1.text_input("Sembol (opsiyonel)", value="BTCUSDT")
        balance = col1.number_input("💰 Toplam Kasa ($)", min_value=0.0, value=1000.0)

        calc_type = col1.radio("Risk Türü", ["Yüzde", "Sabit Tutar"])
        if calc_type == "Yüzde":
            risk_pct = col1.number_input("Risk (%)", min_value=0.0, max_value=100.0, value=1.0)
            risk_amount = balance * (risk_pct / 100) if balance > 0 else 0.0
        else:
            risk_amount = col1.number_input("Risk ($)", min_value=0.0, value=10.0)
            risk_pct = (risk_amount / balance * 100) if balance > 0 else 0.0

        leverage = col2.number_input("🔗 Kaldıraç (x)", min_value=1.0, value=1.0, step=1.0)
        entry = col2.number_input("Giriş Fiyatı", min_value=0.0)
        stop = col2.number_input("Stop Fiyatı", min_value=0.0)

        exchange = col3.selectbox(
            "🏦 Borsa / Ürün",
            options=[
                "Binance Futures (USDT-M)",
                "Bybit USDT Perp",
                "OKX Futures",
                "Bitget Futures",
                "Spot / Diğer"
            ],
            index=0
        )

        direction = col3.radio("Pozisyon Yönü", ["Long", "Short"], horizontal=True)
        tp1 = col3.number_input("🎯 TP1", min_value=0.0)
        tp2 = col3.number_input("TP2", min_value=0.0)
        tp3 = col3.number_input("TP3", min_value=0.0)

        default_mmr_map = {
            "Binance Futures (USDT-M)": 0.004,
            "Bybit USDT Perp": 0.004,
            "OKX Futures": 0.004,
            "Bitget Futures": 0.004,
            "Spot / Diğer": 0.0
        }
        default_mmr = float(default_mmr_map.get(exchange, 0.004))

        mmr = st.slider(
            "Maintenance Margin Oranı (tahmini)",
            min_value=0.0,
            max_value=0.05,
            value=default_mmr,
            step=0.001,
            help="Borsaya göre değişir. Yaklaşık tasfiye fiyatı hesaplamak içindir, %100 doğru olmayabilir."
        )

        st.markdown("---")

        liq_price_val = None

        if entry > 0 and stop > 0 and risk_amount > 0:
            price_risk = abs(entry - stop)
            if price_risk == 0:
                st.error("Giriş ve stop aynı olamaz!")
            else:
                qty = risk_amount / price_risk
                notional = qty * entry
                margin = notional / leverage if leverage > 0 else notional
                margin_pct = (margin / balance * 100) if balance > 0 else 0.0

                colA, colB, colC = st.columns(3)
                colA.metric("📦 Girilecek Adet", f"{qty:.4f}")
                colB.metric("💼 Pozisyon Değeri", f"${notional:,.2f}")
                colC.metric("🔒 Gerekli Marjin", f"${margin:,.2f}")

                st.markdown(
                    f"""
                    <div class="risk-highlight">
                        Kasaya oranla marjin: <b>%{margin_pct:.2f}</b><br>
                        Gerçek risk: <b>${risk_amount:.2f}</b> ({risk_pct:.2f}%)
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if exchange != "Spot / Diğer" and qty > 0 and leverage > 0:
                    notional = entry * qty
                    margin = notional / leverage
                    maint_margin = notional * mmr
                    loss_to_liq = margin - maint_margin
                    if loss_to_liq > 0:
                        price_move = loss_to_liq / qty
                        if direction == "Long":
                            liq_price = entry - price_move
                        else:
                            liq_price = entry + price_move

                        if liq_price > 0:
                            liq_price_val = liq_price
                            st.markdown(
                                f"""
                                <div class="risk-highlight">
                                    Tahmini tasfiye fiyatı ({direction}): 
                                    <b>{liq_price:.6f}</b><br>
                                    <span class="small-muted">
                                    Not: Bu yaklaşık bir hesaplamadır, borsanın gerçek likidasyon fiyatıyla birebir uyuşmayabilir.
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("Bakiyeye göre tasfiye fiyatı hesaplanamadı (maintenance margin > marjin).")
                else:
                    st.info("Spot işlemlerde tasfiye fiyatı yoktur; sadece stop-loss ile risk yönetimi yapılır.")

                st.markdown("#### 📊 R:R ve TP Analizi")
                def compute_rr(tp_price: float):
                    if tp_price <= 0 or tp_price == entry:
                        return None
                    reward = abs(tp_price - entry)
                    rr = reward / price_risk
                    profit = reward * qty
                    return rr, profit

                any_tp = False
                for label, tp_val in [("TP1", tp1), ("TP2", tp2), ("TP3", tp3)]:
                    res = compute_rr(tp_val)
                    if res is None:
                        continue
                    any_tp = True
                    rr, profit = res
                    st.success(f"**{label} = {tp_val}** → Tahmini Kâr: **${profit:.2f}** | R:R = **{rr:.2f}**")

                if not any_tp:
                    st.caption("TP fiyatları girdiğinde burada R:R oranlarını görebilirsin.")

                st.markdown("#### ⚠️ Mod Bazlı Öneriler")
                if trader_mode == "Scalper":
                    st.warning("Scalper modunda geniş stop ve yüksek kaldıraç çok risklidir. Spread ve wick’lere dikkat et.")
                elif trader_mode == "Swing":
                    st.info("Swing işlemlerinde 4H/1D trendi, EMA50/200 birlikteliği ve R/R ≥ 2 çok önemli.")
                elif trader_mode == "Pozisyon":
                    st.warning("Pozisyon işlemlerinde BTC dominansı, makro veri ve uzun vadeli trend kritik öneme sahiptir.")
                else:
                    st.info("Dengeli mod için ATR tabanlı stop ve kademeli TP iyi çalışır.")

                if st.button("💾 Bu Hesabı History'e Kaydet"):
                    record = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "symbol": symbol_risk,
                        "exchange": exchange,
                        "mode": trader_mode,
                        "direction": direction,
                        "balance": balance,
                        "risk_type": calc_type,
                        "risk_pct": risk_pct,
                        "risk_amount": risk_amount,
                        "leverage": leverage,
                        "entry": entry,
                        "stop": stop,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "quantity": qty,
                        "margin": margin,
                        "liq_price": liq_price_val,
                    }
                    st.session_state.risk_history.append(record)
                    save_risk_record_db(record)
                    st.success("✅ Bu hesaplama hem oturum history'e hem veritabanına eklendi.")

    st.markdown("### 🌍 Piyasa Paneli")
    with st.expander("Global Duyarlılık & Makro (Örnek)", expanded=False):
        cm1, cm2 = st.columns([1, 2])

        with cm1:
            st.markdown("##### Crypto Fear & Greed Index")
            if st.button("🔄 F&G Verisini Yenile"):
                get_fear_and_greed_index.clear()
                st.experimental_rerun()
            val, lbl, fetched_at = get_fear_and_greed_index()
            st.plotly_chart(create_gauge_chart(val, lbl), use_container_width=True)
            st.caption(
                f"Index: {val} ({lbl})  \n"
                f"Güncelleme zamanı (UTC): {fetched_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        with cm2:
            tab2 = st.tabs(["Makro Gündem"])[0]
            with tab2:
                df = get_mock_macro_events()
                st.markdown("#### Yaklaşan Makro Veriler (Örnek)")
                for _, r in df.iterrows():
                    st.warning(
                        f"{r['date'].strftime('%d %b %Y')} {r['time']} - "
                        f"{r['currency']} - {r['event']} (Beklenti: {r['forecast']})"
                    )

    # ------------------ MANUEL HABER & SENTIMENT ------------------ #
    st.markdown("### 📰 Haber & Duygu Analizi (Manuel)")

    with st.expander("Kendi Haber Metinlerinle Sentiment Analizi", expanded=False):
        news_text = st.text_area(
            "Başlık / Haber Metinleri (her satıra bir haber veya kısa özet yaz)",
            help="Örn:\nBTC ETF onayına ilişkin haberler...\nBinance hacimlerinde ani artış...",
            height=150
        )

        if st.button("🧠 Sentiment Analizi Yap"):
            if not news_text.strip():
                st.warning("Önce birkaç başlık veya haber özeti yazmalısın.")
            elif not st.session_state.api_status:
                st.error("Gemini API bağlantısı yapmadan sentiment analizi yapılamaz.")
            else:
                model, err, resolved_name = get_gemini_model(
                    st.session_state.api_key,
                    st.session_state.model_name
                )
                if not model:
                    st.error(f"Model hatası: {mask_error(err)}")
                else:
                    sentiment_prompt = f"""
                    Sen deneyimli bir kripto analistisin.
                    Aşağıda kullanıcı tarafından girilmiş haber/başlık özetleri var:

                    {news_text}

                    Görevin:
                    1) Genel piyasa duyarlılığını sınıflandır (boğa, ayı, nötr).
                    2) +2 ile -2 arası bir duygu skoru ver (2: çok boğa, -2: çok ayı).
                    3) 3-5 maddelik kısa özet çıkar.
                    4) Mümkünse coin veya genel piyasa için kısa stratejik yorum ekle (ama yatırım tavsiyesi verme).

                    Cevabını Türkçe markdown olarak ver.
                    """
                    with st.spinner("AI sentiment analizi yapılıyor..."):
                        try:
                            resp = model.generate_content(sentiment_prompt)
                            text = resp.text if hasattr(resp, "text") else str(resp)
                            st.markdown("#### 🧠 AI Sentiment Özeti")
                            st.markdown(text)
                        except Exception as e:
                            st.error(f"Sentiment analizi sırasında hata: {e}")

# ------------------------ TAB 3: CANLI MARKET ANALİZİ ------------------------ #
with tab_live:
    st.markdown("### 📊 Canlı Market Analizi (Binance / OKX / Bybit / Coinbase / Upbit + İndikatörler)")

    with st.expander("📥 Borsa Verisi + RSI / MACD / EMA / Bollinger", expanded=True):
        c1, c2, c3 = st.columns(3)

        exchange_live = c1.selectbox(
            "Borsa",
            options=["Binance", "OKX", "Bybit", "Coinbase", "Upbit"],
            index=0
        )

        coin_choice = c2.selectbox(
            "Coin",
            options=[
                "Bitcoin (BTC)",
                "Ethereum (ETH)",
                "BNB",
                "Solana (SOL)",
                "XRP",
                "Dogecoin (DOGE)",
                "Cardano (ADA)",
                "Toncoin (TON)",
                "Chainlink (LINK)",
                "Pepe (PEPE)",
                "Shiba Inu (SHIB)",
                "Optimism (OP)",
                "Arbitrum (ARB)",
                "Pi Network (PI)",
            ],
            index=0
        )

        interval = c3.selectbox(
            "Zaman Dilimi",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            index=3
        )

        c4, c5 = st.columns(2)
        limit = c4.slider(
            "Mum Sayısı",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Ne kadar çok mum, o kadar uzun tarihsel görünüm."
        )

        refresh_mode = c5.radio(
            "Veri Yenileme Modu",
            ["Manuel", "15 sn Auto-Refresh"],
            index=0,
            horizontal=False,
            key="live_refresh_mode_radio"
        )
        st.session_state.live_refresh_mode = refresh_mode

        if refresh_mode == "15 sn Auto-Refresh":
            _ = st_autorefresh(interval=15_000, key="live_auto_refresh_counter")

        # Sembol eşlemeleri
        binance_symbol_map = {
            "Bitcoin (BTC)": "BTCUSDT",
            "Ethereum (ETH)": "ETHUSDT",
            "BNB": "BNBUSDT",
            "Solana (SOL)": "SOLUSDT",
            "XRP": "XRPUSDT",
            "Dogecoin (DOGE)": "DOGEUSDT",
            "Cardano (ADA)": "ADAUSDT",
            "Toncoin (TON)": "TONUSDT",
            "Chainlink (LINK)": "LINKUSDT",
            "Pepe (PEPE)": "PEPEUSDT",
            "Shiba Inu (SHIB)": "SHIBUSDT",
            "Optimism (OP)": "OPUSDT",
            "Arbitrum (ARB)": "ARBUSDT",
            # Pi Network Binance'te yok; eşlemezsek hata mesajı döner
        }

        okx_inst_map = {
            "Bitcoin (BTC)": "BTC-USDT",
            "Ethereum (ETH)": "ETH-USDT",
            "BNB": "BNB-USDT",
            "Solana (SOL)": "SOL-USDT",
            "XRP": "XRP-USDT",
            "Dogecoin (DOGE)": "DOGE-USDT",
            "Cardano (ADA)": "ADA-USDT",
            "Toncoin (TON)": "TON-USDT",
            "Chainlink (LINK)": "LINK-USDT",
            "Pepe (PEPE)": "PEPE-USDT",
            "Shiba Inu (SHIB)": "SHIB-USDT",
            "Optimism (OP)": "OP-USDT",
            "Arbitrum (ARB)": "ARB-USDT",
            "Pi Network (PI)": "PI-USDT",
        }

        bybit_symbol_map = {
            "Bitcoin (BTC)": "BTCUSDT",
            "Ethereum (ETH)": "ETHUSDT",
            "BNB": "BNBUSDT",
            "Solana (SOL)": "SOLUSDT",
            "XRP": "XRPUSDT",
            "Dogecoin (DOGE)": "DOGEUSDT",
            "Cardano (ADA)": "ADAUSDT",
            "Toncoin (TON)": "TONUSDT",
            "Chainlink (LINK)": "LINKUSDT",
            "Pepe (PEPE)": "PEPEUSDT",
            "Shiba Inu (SHIB)": "SHIBUSDT",
            "Optimism (OP)": "OPUSDT",
            "Arbitrum (ARB)": "ARBUSDT",
            # Pi, Bybit'te olmayabilir
        }

        coinbase_product_map = {
            "Bitcoin (BTC)": "BTC-USD",
            "Ethereum (ETH)": "ETH-USD",
            "Solana (SOL)": "SOL-USD",
            "XRP": "XRP-USD",
            "Dogecoin (DOGE)": "DOGE-USD",
            "Cardano (ADA)": "ADA-USD",
            "Chainlink (LINK)": "LINK-USD",
            "Shiba Inu (SHIB)": "SHIB-USD",
            "Pepe (PEPE)": "PEPE-USD",
            "Optimism (OP)": "OP-USD",
            "Arbitrum (ARB)": "ARB-USD",
            # BNB, TON, PI vb. Coinbase'te yok -> eşleme yok
        }

        upbit_market_map = {
            # Upbit'te USDT marketleri yaygın; yoksa KRW- pair gerekir.
            "Bitcoin (BTC)": "USDT-BTC",
            "Ethereum (ETH)": "USDT-ETH",
            "BNB": "USDT-BNB",
            "Solana (SOL)": "USDT-SOL",
            "XRP": "USDT-XRP",
            "Dogecoin (DOGE)": "USDT-DOGE",
            "Cardano (ADA)": "USDT-ADA",
            "Toncoin (TON)": "USDT-TON",
            "Chainlink (LINK)": "USDT-LINK",
            "Pepe (PEPE)": "USDT-PEPE",
            "Shiba Inu (SHIB)": "USDT-SHIB",
            "Optimism (OP)": "USDT-OP",
            "Arbitrum (ARB)": "USDT-ARB",
            "Pi Network (PI)": "USDT-PI",
        }

        okx_bar_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }

        fetch_trigger = False
        if refresh_mode == "Manuel":
            if st.button("📥 Veriyi Çek ve Hesapla", key="live_fetch_manual"):
                fetch_trigger = True
        else:
            # Auto-refresh modunda her run'da çek
            fetch_trigger = True

        if fetch_trigger:
            with st.spinner("Veriler çekiliyor ve indikatörler hesaplanıyor..."):
                df_ohlc = None
                error_msg = None
                used_source_str = ""

                try:
                    if exchange_live == "Binance":
                        symbol = binance_symbol_map.get(coin_choice)
                        if not symbol:
                            error_msg = "Bu coin için Binance üzerinde sembol eşlemesi yapılmadı."
                        else:
                            df_ohlc, error_msg = get_ohlc_binance(symbol, interval=interval, limit=limit)
                            used_source_str = f"Binance • Sembol: {symbol}"

                    elif exchange_live == "OKX":
                        inst_id = okx_inst_map.get(coin_choice)
                        if not inst_id:
                            error_msg = "Bu coin için OKX üzerinde sembol eşlemesi yapılmadı."
                        else:
                            bar = okx_bar_map.get(interval, "1H")
                            df_ohlc, error_msg = get_ohlc_okx(inst_id, bar=bar, limit=limit)
                            used_source_str = f"OKX • Enstrüman: {inst_id}"

                    elif exchange_live == "Bybit":
                        symbol = bybit_symbol_map.get(coin_choice)
                        if not symbol:
                            error_msg = "Bu coin için Bybit üzerinde sembol eşlemesi yapılmadı."
                        else:
                            df_ohlc, error_msg = get_ohlc_bybit(symbol, interval=interval, limit=limit)
                            used_source_str = f"Bybit • Sembol: {symbol}"

                    elif exchange_live == "Coinbase":
                        product_id = coinbase_product_map.get(coin_choice)
                        if not product_id:
                            error_msg = "Bu coin için Coinbase üzerinde product eşlemesi yapılmadı."
                        else:
                            df_ohlc, error_msg = get_ohlc_coinbase(product_id, interval=interval, limit=limit)
                            used_source_str = f"Coinbase • Product: {product_id}"

                    elif exchange_live == "Upbit":
                        market = upbit_market_map.get(coin_choice)
                        if not market:
                            error_msg = "Bu coin için Upbit üzerinde market eşlemesi yapılmadı."
                        else:
                            df_ohlc, error_msg = get_ohlc_upbit(market, interval=interval, limit=limit)
                            used_source_str = f"Upbit • Market: {market}"
                except Exception as e:
                    error_msg = f"Veri çekilirken beklenmeyen hata: {e}"

                if df_ohlc is None or df_ohlc.empty:
                    msg = "OHLC verisi alınamadı."
                    if error_msg:
                        msg += f" Detay: {error_msg}"
                    st.error(msg)
                else:
                    df_ind = compute_indicators(df_ohlc)
                    fig = create_live_market_figure(df_ind)
                    st.plotly_chart(fig, use_container_width=True)

                    last = df_ind.iloc[-1]
                    colX, colY, colZ = st.columns(3)
                    colX.metric("Son Kapanış", f"{last['close']:.4f}")
                    if not np.isnan(last.get("ema20", np.nan)):
                        colY.metric("EMA 20", f"{last['ema20']:.4f}")
                    if not np.isnan(last.get("rsi14", np.nan)):
                        colZ.metric("RSI 14", f"{last['rsi14']:.2f}")

                    caption_str = f"Veri kaynağı: {exchange_live}"
                    if used_source_str:
                        caption_str += f" ({used_source_str})"
                    st.caption(caption_str + " • Bu bölüm eğitim amaçlıdır; gerçek zamanlı borsa arayüzü değildir.")

# ------------------------ TAB 4: AI TRADE PLANLAYICI ------------------------ #
with tab_planner:
    st.markdown("### 🤖 AI Trade Planlayıcı")

    st.markdown(
        """
        <div class="ai-card">
        Bu bölüm, seçtiğin parametrelere göre örnek bir trade planı oluşturur.  
        Planlar, eğitim ve strateji geliştirme amaçlıdır; yatırım tavsiyesi değildir.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("🧠 Otomatik Trade Planı Oluştur (AI Destekli)", expanded=True):
        c1, c2, c3 = st.columns(3)
        symbol = c1.text_input("Sembol", value="BTCUSDT")
        timeframe = c1.selectbox("Zaman Dilimi", ["1m", "5m", "15m", "1H", "4H", "1D"], index=4)

        plan_direction = c2.radio("Yön Tercihi", ["Long", "Short", "Her İkisi"], index=2)

        plan_balance = c2.number_input("Hesap Büyüklüğü (USD)", min_value=0.0, value=1000.0)
        plan_risk_pct = c2.number_input("Bu trade'de risk (%)", min_value=0.0, max_value=100.0, value=1.0)

        plan_mode = c3.selectbox(
            "Planlama Modu (Trader Tarzı)",
            ["Aynı (Sidebar'daki)", "Scalper", "Swing", "Pozisyon", "Dengeli"],
            index=0
        )

        extra_plan_notes = st.text_area(
            "Ek Notlar (opsiyonel)",
            help="Örn: 'Yalnızca trend yönünde işlemler', 'FED açıklaması sonrası' vb.",
            key="plan_notes"
        )

        if st.button("📋 Trade Planı Oluştur"):
            if not st.session_state.api_status:
                st.error("Önce sol menüden API bağlantısını yapmalısın (Gemini API key).")
            else:
                model, err, resolved_name = get_gemini_model(
                    st.session_state.api_key,
                    st.session_state.model_name
                )
                if not model:
                    safe_err = mask_error(err)
                    st.error(f"Model oluşturulamadı: {safe_err}")
                else:
                    if resolved_name and resolved_name != st.session_state.model_name:
                        st.session_state.model_name = resolved_name
                        st.info(f"Planlama modeli {resolved_name} olarak güncellendi.")

                    risk_amount = plan_balance * (plan_risk_pct / 100.0) if plan_balance > 0 else 0.0
                    if plan_mode == "Aynı (Sidebar'daki)":
                        effective_mode = st.session_state.trader_mode
                    else:
                        effective_mode = plan_mode

                    global_ctx = build_global_market_context()

                    with st.spinner("AI trade planı hazırlanıyor..."):
                        try:
                            plan_text = generate_ai_trade_plan(
                                model=model,
                                symbol=symbol,
                                timeframe=timeframe,
                                balance=plan_balance,
                                risk_amount=risk_amount,
                                direction=plan_direction,
                                trader_mode=effective_mode,
                                extra_notes=extra_plan_notes,
                                global_ctx=global_ctx
                            )

                            parsed = try_parse_json(plan_text)
                            if parsed:
                                st.success("✅ Trade planı JSON (structured) olarak alındı.")
                                render_trade_plan_structured(parsed)
                            else:
                                st.warning("AI çıktısı tam JSON formatında değil, düz metin olarak gösteriliyor.")
                                st.markdown(plan_text)

                            record = {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "direction": plan_direction,
                                "mode": effective_mode,
                                "balance": plan_balance,
                                "risk_pct": plan_risk_pct,
                                "risk_amount": risk_amount,
                                "notes": extra_plan_notes,
                                "plan_text": plan_text,
                            }

                            st.session_state.plan_history.append(record)
                            save_plan_record_db(record)
                            st.success("✅ Trade planı hem oturum history'e hem veritabanına kaydedildi.")
                        except Exception as e:
                            st.error(f"Plan oluşturulurken hata oluştu: {e}")

# ------------------------ TAB 5: HISTORY ------------------------ #
with tab_history:
    st.markdown("### 📚 History")

    if not st.session_state.risk_history and not st.session_state.plan_history:
        st.info("Bu oturumda henüz kayıtlı bir risk hesabı veya trade planı yok.")

    source_choice = st.radio(
        "History Kaynağı",
        ["Bu Oturum", "Veritabanı (Tüm Kayıtlar)"],
        index=0,
        horizontal=True
    )

    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("🧹 Risk History'yi Temizle (Oturum)"):
            st.session_state.risk_history = []
            st.success("Risk history (oturum) temizlendi.")
        if st.button("🧹 Risk History'yi Temizle (DB)"):
            clear_risk_history_db()
            st.success("Risk history veritabanı temizlendi.")

    with col_clear2:
        if st.button("🧹 Plan History'yi Temizle (Oturum)"):
            st.session_state.plan_history = []
            st.success("Plan history (oturum) temizlendi.")
        if st.button("🧹 Plan History'yi Temizle (DB)"):
            clear_plan_history_db()
            st.success("Plan history veritabanı temizlendi.")

    st.markdown("---")

    sub_tab1, sub_tab2 = st.tabs(["🧮 Risk Hesaplamaları", "🤖 AI Trade Planları"])

    with sub_tab1:
        if source_choice == "Bu Oturum":
            if not st.session_state.risk_history:
                st.info("Bu oturumda henüz kaydedilmiş risk hesaplaması yok.")
            else:
                df_risk = pd.DataFrame(st.session_state.risk_history)
                st.dataframe(
                    df_risk,
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("Not: Bu tablo yalnızca mevcut oturum süresince saklanır.")
        else:
            df_risk_db = load_risk_history_db()
            if df_risk_db is None or df_risk_db.empty:
                st.info("Veritabanında kayıtlı risk hesabı bulunmuyor.")
            else:
                st.dataframe(df_risk_db, use_container_width=True)
                st.caption("Veri kaynağı: SQLite veritabanı (ai_kripto_analist.db).")

    with sub_tab2:
        if source_choice == "Bu Oturum":
            plans_source = st.session_state.plan_history
        else:
            df_plans_db = load_plan_history_db()
            if df_plans_db is None or df_plans_db.empty:
                st.info("Veritabanında kayıtlı trade planı bulunmuyor.")
                plans_source = []
            else:
                plans_source = df_plans_db.to_dict("records")

        if not plans_source:
            st.info("Henüz gösterilecek bir trade planı yok.")
        else:
            for i, rec in enumerate(plans_source[::-1], start=1):
                header = (
                    f"#{i} | {rec['timestamp']} • {rec['symbol']} "
                    f"({rec['timeframe']}, {rec['mode']}, {rec['direction']})"
                )
                with st.expander(header, expanded=False):
                    st.markdown(
                        f"""
                        <span class="history-badge">Risk: {float(rec['risk_pct']):.2f}% (~${float(rec['risk_amount']):.2f})</span>
                        <span class="history-badge">Hesap: ${float(rec['balance']):.2f}</span>
                        """,
                        unsafe_allow_html=True
                    )
                    plan_text = rec["plan_text"]
                    parsed = try_parse_json(plan_text)
                    if parsed:
                        render_trade_plan_structured(parsed)
                    else:
                        st.markdown(plan_text)
                    if rec.get("notes"):
                        st.markdown("**Notlar:**")
                        st.markdown(rec["notes"])

st.caption("⚠️ Buradaki tüm analizler ve planlar eğitim amaçlıdır, yatırım tavsiyesi değildir.")
