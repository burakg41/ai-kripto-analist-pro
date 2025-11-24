import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from PIL import Image
from datetime import datetime, timedelta
import numpy as np

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
# 2. YARDIMCI FONKSİYONLAR & GÜVENLİK
# =============================================================================

def mask_error(err) -> str:
    """Hata mesajından API anahtarının bir kısmını maskeler."""
    text = str(err)
    key = st.session_state.api_key
    if key:
        tail = key[-8:]
        text = text.replace(tail, "********")
    return text

def validate_image(file) -> bool:
    """Dosya tipini, boyutunu ve geçerli image olup olmadığını kontrol eder."""
    allowed_types = {"image/png", "image/jpeg", "image/jpg"}
    if file.type not in allowed_types:
        return False

    file_size = getattr(file, "size", None)
    if file_size is not None and file_size > 10 * 1024 * 1024:  # 10 MB
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

@st.cache_data(ttl=300)
def get_crypto_market_overview():
    url = "https://api.coingecko.com/api/v3/global"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", {}) or {}

        mcap_perc = data.get("market_cap_percentage", {}) or {}
        btc_dom = mcap_perc.get("btc")
        eth_dom = mcap_perc.get("eth")

        total_mcap = data.get("total_market_cap", {}).get("usd")
        total_volume = data.get("total_volume", {}).get("usd")
        mcap_change_24h = data.get("market_cap_change_percentage_24h_usd")

        alt_dom = 100.0 - btc_dom if isinstance(btc_dom, (int, float)) else None

        fetched_at = datetime.utcnow()
        return {
            "btc_dom": btc_dom,
            "eth_dom": eth_dom,
            "alt_dom": alt_dom,
            "total_mcap": total_mcap,
            "total_volume": total_volume,
            "mcap_change_24h": mcap_change_24h,
            "fetched_at": fetched_at,
        }
    except Exception:
        return None

# -------------------------------------------------------------------------
# BINANCE & OKX OHLC VERİ ÇEKME (CANLI)
# -------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_ohlc_binance(symbol: str, interval: str = "1h", limit: int = 200):
    """
    Binance spot/futures sembolü için OHLC verisi çeker.
    Örn: symbol='BTCUSDT', interval='1h', limit=200
    """
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
            return None
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
        df = pd.DataFrame(data)
        df = df.sort_values("time")
        return df
    except Exception:
        return None

@st.cache_data(ttl=60)
def get_ohlc_okx(inst_id: str, bar: str = "1H", limit: int = 200):
    """
    OKX sembolü için OHLC verisi çeker.
    Örn: inst_id='BTC-USDT', bar='1H', limit=200
    """
    url = "https://www.okx.com/api/v5/market/candles"
    params = {
        "instId": inst_id,
        "bar": bar,       # '1m','5m','15m','1H','4H','1D' vb.
        "limit": limit
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json().get("data", [])
        if not raw:
            return None
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
        df = pd.DataFrame(data)
        df = df.sort_values("time")
        return df
    except Exception:
        return None

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
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Fiyat & EMA & Bollinger", "RSI (14)", "MACD (12,26,9)")
    )

    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC"
        ),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(x=df["time"], y=df["ema20"], mode="lines", name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["ema50"], mode="lines", name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], mode="lines", name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"], mode="lines", name="BB Mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], mode="lines", name="BB Lower"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi14"], mode="lines", name="RSI 14"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)

    fig.add_trace(go.Bar(x=df["time"], y=df["macd_hist"], name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd"], mode="lines", name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["macd_signal"], mode="lines", name="Signal"), row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig

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

def format_usd_compact(value):
    if value is None:
        return "-"
    try:
        v = float(value)
    except Exception:
        return "-"
    abs_v = abs(v)
    if abs_v >= 1_000_000_000_000:
        return f"${v/1_000_000_000_000:.2f} T"
    elif abs_v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f} B"
    elif abs_v >= 1_000_000:
        return f"${v/1_000_000:.2f} M"
    else:
        return f"${v:,.0f}"

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
    mkt = get_crypto_market_overview()
    lines = [f"Global Crypto Fear & Greed Index şu anda {fg_val} ({fg_lbl})."]
    if mkt:
        if isinstance(mkt.get("btc_dom"), (int, float)):
            lines.append(f"BTC dominansı yaklaşık %{mkt['btc_dom']:.2f} seviyesinde.")
        if isinstance(mkt.get("eth_dom"), (int, float)):
            lines.append(f"ETH dominansı yaklaşık %{mkt['eth_dom']:.2f} seviyesinde.")
        if isinstance(mkt.get("alt_dom"), (int, float)):
            lines.append(f"Altcoin dominansı kabaca %{mkt['alt_dom']:.2f} civarında.")
        if isinstance(mkt.get("mcap_change_24h"), (int, float)):
            lines.append(f"Toplam market cap'in 24 saatlik değişimi %{mkt['mcap_change_24h']:.2f} civarında.")
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

    Lütfen şu yapıda bir plan üret:

    1️⃣ Genel Bakış:
    - Paritenin mevcut durumu (trend, volatilite, BTC ve makro bağlamla ilişki)

    2️⃣ Senaryo:
    - Long senaryosu (giriş bölgesi, stop, TP1, TP2, opsiyonel TP3)
    - Short senaryosu (giriş bölgesi, stop, TP1, TP2, opsiyonel TP3)
    - Eğer sadece tek yön mantıklıysa, diğer yön için "şu anda zayıf" gibi uyarı ekle.

    3️⃣ R/R ve Risk Yönetimi:
    - Örnek pozisyon büyüklüğü (mantıksal açıklama)
    - Tahmini R/R oranları
    - Max riskin neden makul veya aşırı olduğuna dair yorum

    4️⃣ Zamanlama:
    - Scalper ise: daha kısa sürede gerçekleşebilecek senaryolar
    - Swing ise: birkaç gün sürebilecek plan
    - Pozisyon ise: haftalar sürebilecek plan

    5️⃣ Dikkat Edilmesi Gerekenler:
    - Haber akışı, volatilite patlamaları, likidite düşüklüğü
    - Kaldıraç konusunda net uyarılar
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

# ------------------------ TAB 2: ARAÇLAR (RİSK + PİYASA PANELİ) ------------------------ #
with tab_tools:
    st.markdown("### 🧮 Akıllı Risk, Marjin & Likidasyon Hesaplayıcı")

    with st.expander("Akıllı Risk Hesaplayıcı", expanded=True):
        trader_mode = st.session_state.trader_mode

        mode_recommendations = {
            "Scalper": "Önerilen risk: **%0.2 – %0.5** • Çok dar stop • 1–5dk volatilitesine dikkat • Spread ve wick’e karşı tetikte ol.",
            "Swing": "Önerilen risk: **%0.5 – %1.5** • Daha geniş stop • 2–3 TP’li yapı mantıklı.",
            "Pozisyon": "Önerilen risk: **%0.25 – %0.75** • Günlük/haftalık trend kritik • Makro risklere dikkat.",
            "Dengeli": "Önerilen risk: **%0.5 – %1.0** • R/R en az 1:2 hedeflenmeli."
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
            help="Borsaya göre değişir. Bu değer yaklaşık bir tasfiye fiyatı hesaplamak içindir, %100 doğru olmayabilir."
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
                    st.warning("⚡ Scalper modunda geniş stop ve yüksek kaldıraç çok risklidir. Spread ve wick’lere dikkat et.")
                elif trader_mode == "Swing":
                    st.info("📈 Swing işlemlerinde 4H/1D trendi, EMA50/200 birlikteliği ve R/R ≥ 2 çok önemli.")
                elif trader_mode == "Pozisyon":
                    st.warning("📉 Pozisyon işlemlerinde BTC dominansı, makro veri ve uzun vadeli trend kritik öneme sahiptir.")
                else:
                    st.info("⚖️ Dengeli mod için ATR tabanlı stop ve kademeli TP iyi çalışır.")

                # History'e kaydet
                if st.button("💾 Bu Hesabı History'e Kaydet"):
                    st.session_state.risk_history.append({
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
                    })
                    st.success("✅ Bu hesaplama history'e eklendi.")

    st.markdown("### 🌍 Piyasa Paneli")
    with st.expander("Global Duyarlılık & Makro (Örnek)", expanded=False):
        cm1, cm2 = st.columns([1, 2])

        with cm1:
            st.markdown("##### Crypto Fear & Greed Index")
            if st.button("🔄 F&G Verisini Yenile"):
                get_fear_and_greed_index.clear()
                st.rerun()
            val, lbl, fetched_at = get_fear_and_greed_index()
            st.plotly_chart(create_gauge_chart(val, lbl), use_container_width=True)
            st.caption(
                f"Index: **{val}** ({lbl})  \n"
                f"Güncelleme zamanı (UTC): {fetched_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        with cm2:
            tab1, tab2 = st.tabs(["Kripto Piyasa Özeti", "Makro Gündem"])
            with tab1:
                mkt = get_crypto_market_overview()
                if not mkt:
                    st.warning("Piyasa verileri şu anda çekilemedi. Sonra tekrar deneyin.")
                else:
                    cA, cB, cC = st.columns(3)
                    if isinstance(mkt.get("btc_dom"), (int, float)):
                        cA.metric("BTC Dominance", f"{mkt['btc_dom']:.2f}%")
                    else:
                        cA.metric("BTC Dominance", "-")

                    if isinstance(mkt.get("alt_dom"), (int, float)):
                        cB.metric("Altcoin Dominance (≈)", f"{mkt['alt_dom']:.2f}%")
                    else:
                        cB.metric("Altcoin Dominance (≈)", "-")

                    if isinstance(mkt.get("eth_dom"), (int, float)):
                        cC.metric("ETH Dominance", f"{mkt['eth_dom']:.2f}%")
                    else:
                        cC.metric("ETH Dominance", "-")

                    cD, cE, cF = st.columns(3)
                    cD.metric("Toplam Market Cap", format_usd_compact(mkt.get("total_mcap")))
                    cE.metric("24h Hacim", format_usd_compact(mkt.get("total_volume")))
                    if isinstance(mkt.get("mcap_change_24h"), (int, float)):
                        cF.metric("Market Cap 24h %", f"{mkt['mcap_change_24h']:.2f}%")
                    else:
                        cF.metric("Market Cap 24h %", "-")

                    st.caption(
                        "Veri kaynağı: CoinGecko Global API  \n"
                        f"Güncelleme zamanı (UTC): {mkt['fetched_at'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            with tab2:
                df = get_mock_macro_events()
                st.markdown("#### Yaklaşan Makro Veriler (Örnek)")
                for _, r in df.iterrows():
                    st.warning(
                        f"**{r['date'].strftime('%d %b %Y')} {r['time']}** - "
                        f"{r['currency']} - {r['event']} (Beklenti: {r['forecast']})"
                    )

# ------------------------ TAB 3: CANLI MARKET ANALİZİ (BINANCE / OKX) ------------------------ #
with tab_live:
    st.markdown("### 📊 Canlı Market Analizi (Binance / OKX OHLC + İndikatörler)")

    with st.expander("📥 Borsa Verisi + RSI / MACD / EMA / Bollinger", expanded=True):
        c1, c2, c3 = st.columns(3)

        exchange_live = c1.selectbox(
            "Borsa",
            options=["Binance", "OKX"],
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
            # "Pi Network (PI)" bilinçli olarak eklenmedi → API'de genelde yok / IOU
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
            # "Pi Network (PI)" yok
        }

        okx_bar_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }

        if st.button("📥 Veriyi Çek ve Hesapla", key="live_fetch"):
            with st.spinner("Veriler çekiliyor ve indikatörler hesaplanıyor..."):
                df_ohlc = None

                if exchange_live == "Binance":
                    if coin_choice not in binance_symbol_map:
                        st.error("Bu coin için Binance OHLC verisi desteklenmiyor (örn. Pi Network).")
                        df_ohlc = None
                    else:
                        symbol = binance_symbol_map[coin_choice]
                        df_ohlc = get_ohlc_binance(symbol, interval=interval, limit=limit)
                else:
                    if coin_choice not in okx_inst_map:
                        st.error("Bu coin için OKX OHLC verisi desteklenmiyor (örn. Pi Network).")
                        df_ohlc = None
                    else:
                        inst_id = okx_inst_map[coin_choice]
                        bar = okx_bar_map[interval]
                        df_ohlc = get_ohlc_okx(inst_id, bar=bar, limit=limit)

                if df_ohlc is None or df_ohlc.empty:
                    st.error("OHLC verisi alınamadı. Bir süre sonra tekrar deneyin.")
                else:
                    df_ind = compute_indicators(df_ohlc)
                    fig = create_live_market_figure(df_ind)
                    st.plotly_chart(fig, use_container_width=True)

                    last = df_ind.iloc[-1]
                    colX, colY, colZ = st.columns(3)
                    colX.metric(
                        "Son Kapanış",
                        f"{last['close']:.4f} USDT"
                    )
                    if not np.isnan(last.get("ema20", np.nan)):
                        colY.metric("EMA 20", f"{last['ema20']:.4f}")
                    if not np.isnan(last.get("rsi14", np.nan)):
                        colZ.metric("RSI 14", f"{last['rsi14']:.2f}")

                    st.caption(
                        f"Veri kaynağı: {exchange_live} • Bu bölüm eğitim amaçlıdır; gerçek zamanlı borsa arayüzü değildir."
                    )

# ------------------------ TAB 4: AI TRADE PLANLAYICI ------------------------ #
with tab_planner:
    st.markdown("### 🤖 AI Trade Planlayıcı")

    st.markdown(
        """
        <div class="ai-card">
        Bu bölüm, seçtiğin parametrelere göre <b>örnek bir trade planı</b> oluşturur.  
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
                        st.info(f"Planlama modeli **{resolved_name}** olarak güncellendi.")

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
                            st.markdown(plan_text)

                            st.session_state.plan_history.append({
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
                            })
                            st.success("✅ Trade planı history'e kaydedildi.")
                        except Exception as e:
                            st.error(f"Plan oluşturulurken hata oluştu: {e}")

# ------------------------ TAB 5: HISTORY ------------------------ #
with tab_history:
    st.markdown("### 📚 History (Bu Oturum)")

    if not st.session_state.risk_history and not st.session_state.plan_history:
        st.info("Bu oturumda henüz kayıtlı bir risk hesabı veya trade planı yok.")
    else:
        col_clear1, col_clear2 = st.columns(2)
        if col_clear1.button("🧹 Risk History'yi Temizle"):
            st.session_state.risk_history = []
            st.success("Risk history temizlendi.")
        if col_clear2.button("🧹 Plan History'yi Temizle"):
            st.session_state.plan_history = []
            st.success("Plan history temizlendi.")

        st.markdown("---")

        sub_tab1, sub_tab2 = st.tabs(["🧮 Risk Hesaplamaları", "🤖 AI Trade Planları"])

        with sub_tab1:
            if not st.session_state.risk_history:
                st.info("Henüz kaydedilmiş risk hesaplaması yok.")
            else:
                df_risk = pd.DataFrame(st.session_state.risk_history)
                st.dataframe(
                    df_risk,
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("Not: Bu tablo yalnızca mevcut oturum süresince saklanır.")

        with sub_tab2:
            if not st.session_state.plan_history:
                st.info("Henüz kaydedilmiş bir trade planı yok.")
            else:
                for i, rec in enumerate(st.session_state.plan_history[::-1], start=1):
                    header = (
                        f"#{i} | {rec['timestamp']} • {rec['symbol']} "
                        f"({rec['timeframe']}, {rec['mode']}, {rec['direction']})"
                    )
                    with st.expander(header, expanded=False):
                        st.markdown(
                            f"""
                            <span class="history-badge">Risk: {rec['risk_pct']:.2f}% (~${rec['risk_amount']:.2f})</span>
                            <span class="history-badge">Hesap: ${rec['balance']:.2f}</span>
                            """,
                            unsafe_allow_html=True
                        )
                        st.markdown(rec["plan_text"])
                        if rec.get("notes"):
                            st.markdown("**Notlar:**")
                            st.markdown(rec["notes"])

st.caption("⚠️ Buradaki tüm analizler ve planlar eğitim amaçlıdır, yatırım tavsiyesi değildir.")
