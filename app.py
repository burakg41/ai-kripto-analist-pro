import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import google.generativeai as genai
from PIL import Image
from datetime import datetime, timedelta

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
    # Burada başlangıçta pattern tutuyoruz, gerçek model adı list_models'tan gelecek
    st.session_state.model_name = "gemini"
if "api_status" not in st.session_state:
    st.session_state.api_status = False
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

MAX_REQUESTS = 50  # Bir session'da maksimum analiz isteği sayısı

# Tema / CSS
st.markdown(
    """
    <style>
        .stApp { background-color: #0e1117; }
        .stFileUploader { border: 2px dashed #4CAF50; border-radius: 10px; padding: 20px; }
        .event-card { background-color: #262730; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 4px solid #4CAF50; }
        .impact-high { border-left-color: #ff4b4b !important; }
        .risk-card { background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# 2. GÜVENLİK / YARDIMCI FONKSİYONLAR
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
        # Bozuk / exploit içeren dosyaları elemek için verify
        img = Image.open(file)
        img.verify()
    except Exception:
        return False
    finally:
        # verify sonrası pointer'ı başa sar
        file.seek(0)

    return True

@st.cache_data(ttl=900)  # 15 dakika cache
def get_fear_and_greed_index():
    """
    Fear & Greed Index verisini çeker.
    Burada alternative.me API'sini kullanıyoruz (CoinMarketCap de aynı indekse dayanıyor).
    15 dakikada bir otomatik olarak yenilenir (ttl=900).
    """
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
    """Sadece configure eder, validasyon ayrı yapılıyor."""
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
    """
    1) API anahtarı ile list_models() çağır.
    2) preferred_pattern'e uyan modellere öncelik ver.
    3) Hiçbiri olmazsa generateContent destekleyen herhangi bir modeli seç.
    """
    clean_key = configure_gemini(api_key)

    try:
        all_models = list(genai.list_models())
    except Exception as e:
        return None, f"Model listesi alınamadı: {e}", None

    if not all_models:
        return None, "Bu API anahtarıyla erişilebilir model bulunamadı.", None

    # generateContent destekleyenleri filtrele
    generative_models = [m for m in all_models if _supports_generate_content(m)]

    if not generative_models:
        return None, "generateContent destekleyen model bulunamadı.", None

    candidates = []

    # 1) preferred_pattern içerenler
    if preferred_pattern:
        for m in generative_models:
            if preferred_pattern in m.name:
                candidates.append(m.name)

    # 2) Yoksa 'gemini' + 'vision' içerenler
    if not candidates:
        for m in generative_models:
            if "gemini" in m.name and "vision" in m.name:
                candidates.append(m.name)

    # 3) Hâlâ yoksa, herhangi bir 'gemini' modeli
    if not candidates:
        for m in generative_models:
            if "gemini" in m.name:
                candidates.append(m.name)

    # 4) Son çare: tüm generateContent modelleri
    if not candidates:
        candidates = [m.name for m in generative_models]

    last_err = None
    tried = []

    for name in candidates:
        try:
            model = genai.GenerativeModel(name)
            _ = model.generate_content("Test")  # basit test
            return model, None, name
        except Exception as e:
            tried.append(name)
            last_err = e
            continue

    err_msg = (
        f"Şu model isimleri denendi ama çalışmadı: {tried}. "
        f"Son hata: {last_err}"
    )
    return None, err_msg, None

def analyze_chart_with_gemini(model, image: Image.Image, extra_context: str = "") -> str:
    """
    Tradingview / kripto grafiği için Türkçe teknik analiz prompt'u.
    Güvenlik vurgusu eklenmiş hali.
    """
    safety_header = """
    ÇOK ÖNEMLİ TALİMATLAR:
    - Kesin "al" veya "sat" sinyali verme.
    - Kaldıraçlı işlem açmayı doğrudan önermemelisin.
    - Cevaplarının yatırım tavsiyesi değil, eğitim amaçlı bir analiz örneği olduğunu belirt.
    """

    base_prompt = f"""
    {safety_header}

    Sen deneyimli bir Türk teknik analist ve kripto trader'sın.

    Aşağıdaki fiyat grafiğini analiz et ve cevaplarını mümkün olduğunca
    sayısal seviyelerle ve maddeler halinde ver.

    {extra_context}

    Cevap formatı:

    1️⃣ Trend:
    - Genel trend yönü (Boğa / Ayı / Yatay)
    - Kısa, orta ve uzun vade için yorum

    2️⃣ Destek & Direnç:
    - En az 3 ana destek seviyesi (sadece rakam, gerekiyorsa aralıkla)
    - En az 3 ana direnç seviyesi
    - Bu seviyelerin neden önemli olduğuna dair kısa açıklama

    3️⃣ Formasyonlar:
    - Olası formasyon(lar) (ör: üçgen, omuz-baş-omuz, çift dip, takoz vs.)
    - Formasyonun hedef fiyat bölgesi (varsa)
    - Formasyon ne aşamada? (oluşum, kırılım, retest, başarısız vs.)

    4️⃣ İşlem Stratejisi:
    - Olası AL stratejisi (giriş bölgesi, stop, ilk ve ikinci TP)
    - Olası SAT / SHORT stratejisi (varsa)
    - Risk yönetimi önerisi (max risk %, volatilite yorumu)
    - Gereksiz agresif öneriler verme, temkinli ol.

    5️⃣ Risk Uyarıları:
    - Grafikte dikkat çeken anormal hareketler (ani spike, likidite boşluğu vs.)
    - Haber, makro, FED vb. dış faktörlere karşı genel uyarı
    """

    response = model.generate_content([base_prompt, image])
    return response.text if hasattr(response, "text") else str(response)

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
    st.stop()  # Geri kalan kodlar çalışmasın

# =============================================================================
# 4. SIDEBAR: API VE AYARLAR
# =============================================================================

with st.sidebar:
    st.header("🔐 API Bağlantısı")

    # Cloud secrets kontrolü
    cloud_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            cloud_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        cloud_key = None

    # Model tipi seçimi (pattern olarak kullanılacak)
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
        index=3  # varsayılan: "gemini"
    )

    # API anahtarını sıfırlama (logout)
    if st.button("🔓 API Anahtarını Temizle"):
        st.session_state.api_key = ""
        st.session_state.api_status = False
        st.session_state.model_name = "gemini"
        st.session_state.last_error = None
        st.success("API anahtarı hafızadan temizlendi.")

    # Cloud key varsa otomatik kullan
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
                st.info(
                    f"Pattern: `{st.session_state.model_name}` → "
                    f"Gerçek model: **{resolved_name}**"
                )
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
            help="API anahtarını Google AI Studio veya MakerSuite'ten alabilirsin."
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
                            st.info(
                                f"Pattern: `{st.session_state.model_name}` → "
                                f"Gerçek model: **{resolved_name}**"
                            )
                        st.session_state.model_name = resolved_name
                        st.success(f"✅ Bağlantı başarılı! Aktif model: {resolved_name}")
                    else:
                        st.session_state.api_status = False
                        st.session_state.last_error = err
                        safe_err = mask_error(err)
                        st.error(f"❌ Bağlantı hatası: {safe_err}")
            else:
                st.warning("Lütfen API anahtarını giriniz.")

    # Durum bilgisi
    if st.session_state.api_status:
        st.caption(f"🔌 API durumu: **Bağlı** | Model: `{st.session_state.model_name}`")
    else:
        st.caption("🔌 API durumu: **Bağlı değil**")
        if st.session_state.last_error:
            st.caption(f"Son hata: `{mask_error(st.session_state.last_error)}`")

# =============================================================================
# 5. ANA BÖLÜM - TEKNİK ANALİZ
# =============================================================================

st.title("📈 AI Teknik Analiz Merkezi")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📤 Grafik Yükle")
    uploaded_files = st.file_uploader(
        "TradingView veya borsa grafiği ekran görüntüsü (Max 15 görsel)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    extra_notes = st.text_area(
        "İsteğe bağlı not / ek bilgi",
        help="Örn: 'BTCUSDT 4H grafik, son dump sonrası durum' gibi kısa notlar ekleyebilirsin."
    )

with col_right:
    st.markdown("### ℹ️ Kullanım Notları")
    st.markdown(
        """
        - Birden fazla grafiği aynı anda yükleyebilirsin.
        - Her grafik için ayrı ayrı teknik analiz üretir.
        - Analizler **öğretici ve temkinli** olacak şekilde tasarlanmıştır.
        - Çıkan sonuçlar yatırım tavsiyesi değildir, sadece eğitim amaçlıdır.
        """
    )

if uploaded_files:
    if len(uploaded_files) > 15:
        st.error("⚠️ Maksimum 15 dosya yükleyebilirsiniz.")
    else:
        start_analysis = st.button("🔍 Analizi Başlat", type="primary")

        if start_analysis:
            # Basit rate limiting
            if st.session_state.request_count + len(uploaded_files) > MAX_REQUESTS:
                st.error(
                    "⚠️ Maksimum istek limitine ulaştınız. "
                    "Yeni analiz için sayfayı yenileyerek yeni oturum başlatın."
                )
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
                            st.info(f"Analiz modeli otomatik olarak **{resolved_name}** olarak güncellendi.")

                        st.session_state.request_count += len(uploaded_files)

                        st.markdown("---")
                        st.subheader("🧠 Yapay Zeka Analizleri")

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
                                st.image(
                                    image,
                                    caption=f"{uploaded_file.name}",
                                    use_container_width=True
                                )

                            with col_txt:
                                with st.spinner("Grafik analiz ediliyor..."):
                                    try:
                                        text = analyze_chart_with_gemini(
                                            model=model,
                                            image=image,
                                            extra_context=extra_notes
                                        )
                                        st.markdown(text)
                                    except Exception as e:
                                        st.error(f"Analiz sırasında hata oluştu: {e}")

                            st.markdown("---")

                        progress_bar.empty()

# =============================================================================
# 6. YARDIMCI ARAÇLAR
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🛠️ Yardımcı Araçlar")

# ------------------------ RİSK HESAPLAYICI ------------------------ #
with st.expander("🧮 Risk Hesaplayıcı", expanded=False):
    c1, c2, c3 = st.columns(3)

    balance = c1.number_input(
        "Kasa ($)",
        min_value=0.0,
        value=1000.0
    )
    risk_pct = c1.number_input(
        "Risk (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0
    )
    entry = c2.number_input(
        "Giriş Fiyatı",
        min_value=0.0,
        value=0.0
    )
    stop = c2.number_input(
        "Stop Fiyatı",
        min_value=0.0,
        value=0.0
    )
    tp = c3.number_input(
        "Hedef Fiyat (TP)",
        min_value=0.0,
        value=0.0
    )

    if entry > 0 and stop > 0 and balance > 0 and risk_pct > 0:
        risk_val = balance * (risk_pct / 100)
        price_risk = abs(entry - stop)

        if price_risk > 0:
            size = risk_val / price_risk
            st.info(
                f"Girilecek Adet: **{size:.4f}** | "
                f"Risk Tutarı: **${risk_val:.2f}**"
            )

            if tp > 0 and tp != entry:
                potential_profit_per_unit = abs(tp - entry)
                potential_profit = potential_profit_per_unit * size
                rr_ratio = potential_profit_per_unit / price_risk
                st.success(
                    f"📊 Tahmini Kâr: **${potential_profit:.2f}** | "
                    f"R/R Oranı: **{rr_ratio:.2f}**"
                )
        else:
            st.warning("Giriş ve stop fiyatı aynı olamaz.")

# ------------------------ PİYASA PANELİ ------------------------ #
with st.expander("🌍 Piyasa Paneli", expanded=False):
    cm1, cm2 = st.columns([1, 2])

    with cm1:
        st.markdown("##### Crypto Fear & Greed Index")

        # Manuel yenileme butonu (cache temizleyip yeniden çekiyoruz)
        if st.button("🔄 F&G Verisini Yenile"):
            get_fear_and_greed_index.clear()
            st.rerun()  # <--- BURASI GÜNCELLENDİ

        val, lbl, fetched_at = get_fear_and_greed_index()
        st.plotly_chart(create_gauge_chart(val, lbl), use_container_width=True)
        st.caption(
            f"Index: **{val}** ({lbl})  \n"
            f"Güncelleme zamanı (UTC): {fetched_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    with cm2:
        df = get_mock_macro_events()
        st.markdown("#### Yaklaşan Makro Veriler (Örnek)")
        for _, r in df.iterrows():
            st.warning(
                f"**{r['date'].strftime('%d %b %Y')} {r['time']}** - "
                f"{r['currency']} - {r['event']} "
                f"(Beklenti: {r['forecast']})"
            )

st.caption("⚠️ Buradaki tüm analizler eğitim amaçlıdır, yatırım tavsiyesi değildir.")
