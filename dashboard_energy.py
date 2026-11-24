import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw
from wordcloud import STOPWORDS, WordCloud
import subprocess
import sys
from pathlib import Path

from streamlit.runtime.scriptrunner import get_script_run_ctx

# Якщо файл запущено звичайним python, перезапускає його через streamlit
if get_script_run_ctx() is None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", __file__],
        cwd=Path(__file__).parent,
    )
    sys.exit()
st.set_page_config(page_title="Відновлювана енергетика у світі", page_icon="⚡", layout="wide")

DATA_PATH = Path(__file__).parent / "owid-energy-data.csv"

@st.cache_data(show_spinner="Завантаження даних OWID...")
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
is_country = df.iso_code.notna() & ~df.iso_code.astype(str).str.startswith("OWID")
countries = df[is_country].copy()
world = df[df.country == "World"].copy()

coverage = countries.groupby("year").renewables_share_elec.count()
latest_year = int(coverage[coverage >= 100].index.max())
all_countries = sorted(countries.country.unique())

# Шапка
st.title("⚡ Відновлювана енергетика у світі")
st.markdown(
    "Хто, наскільки і як швидко переходить на чисту електроенергію. "
    )

tabs = st.tabs([
    "📋 Дані", "1. Структура генерації", "2. Вітер і сонце", "3. Лідери за часткою",
    "4. ВВП і частка", "5. Темпи зростання", "6. Мапа країн", "7. Електростанції",
    "8. Хмара слів", "Загальні висновки",
])

# Дані
with tabs[0]:
    st.subheader("Увесь датасет")
    c1, c2 = st.columns([2, 1])
    sel = c1.multiselect("Країни (порожньо = усі)", all_countries, key="d_countries")
    yr = c2.slider("Роки", int(df.year.min()), int(df.year.max()),
                   (2000, int(df.year.max())), key="d_years")
    view = df[df.year.between(*yr)]
    if sel:
        view = view[view.country.isin(sel)]
    st.caption(f"Рядків: {len(view):,} | стовпчиків: {view.shape[1]}. Агрегати (World, Europe...) мають порожній iso_code або код OWID_*.")
    st.dataframe(view, use_container_width=True, height=450)
    st.download_button("Завантажити вибірку (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "energy_selection.csv", "text/csv")

# Структура
with tabs[1]:
    st.subheader("Як змінювалась структура світового виробництва електроенергії")
    sources = ["coal_electricity", "oil_electricity", "gas_electricity", "nuclear_electricity",
               "hydro_electricity", "wind_electricity", "solar_electricity", "biofuel_electricity",
               "other_renewable_exc_biofuel_electricity"]
    sources = [c for c in sources if c in world.columns]
    labels = {c: c.replace("_electricity", "").replace("_", " ") for c in sources}
    colors = {"coal": "#4d4d4d", "oil": "#8c6d46", "gas": "#b0a090", "nuclear": "#9467bd",
              "hydro": "#1f77b4", "wind": "#2ca02c", "solar": "#ffbf00", "biofuel": "#7f9f3f",
              "other renewable exc biofuel": "#17becf"}
    mix = world.set_index("year")[sources].rename(columns=labels).dropna(how="all").fillna(0)
    c1, c2, c3 = st.columns([2, 2, 1])
    y1 = c1.slider("Діапазон років", int(mix.index.min()), int(mix.index.max()),
                   (1990, int(mix.index.max())), key="mix_years")
    chosen = c2.multiselect("Джерела", list(mix.columns), default=list(mix.columns), key="mix_src")
    as_pct = c3.radio("Показати в", ["ТВт·год", "%"], key="mix_unit")
    m = mix.loc[y1[0]:y1[1], chosen].reset_index().melt("year", var_name="Джерело", value_name="ТВт·год")
    fig = px.area(m, x="year", y="ТВт·год", color="Джерело", color_discrete_map=colors,
                  groupnorm="percent" if as_pct == "%" else None,
                  labels={"year": "Рік", "ТВт·год": "Частка, %" if as_pct == "%" else "Генерація, ТВт·год"})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Висновки до графіка 1", expanded=True):
        st.markdown("""
1. Світове виробництво електроенергії за 30+ років зросло приблизно в 2,5 раза, зростання майже безперервне.
2. Вугілля - найбільше окреме джерело, а майже весь приріст останнього десятиліття дали вітер і сонце.
3. Вітер і сонце з майже нульової позначки на початку 2000-х вийшли на ~12-15% генерації; сонце росте найшвидше.
4. Гідроенергетика - найбільше «чисте» джерело, але її частка майже не змінюється.
5. Газ поступово збільшує частку, виступаючи «перехідним» джерелом між вугіллям та відновлюваними.
""")

# Вітер і сонце
with tabs[2]:
    st.subheader("Вітер і сонце: порівняння країн у динаміці")
    focus = ["China", "United States", "Germany", "India", "Brazil", "Ukraine"]
    c1, c2, c3 = st.columns([3, 2, 1])
    pick = c1.multiselect("Країни", all_countries, default=[c for c in focus if c in all_countries], key="ws_c")
    ys = c2.slider("Роки", 1985, int(countries.year.max()), (2000, int(countries.year.max())), key="ws_y")
    use_log = c3.checkbox("Логарифмічна шкала", key="ws_log")
    ws = countries[countries.country.isin(pick) & countries.year.between(*ys)].copy()
    ws["wind_solar"] = ws[["wind_electricity", "solar_electricity"]].sum(axis=1, min_count=1)
    if use_log:
        ws = ws[ws.wind_solar > 0]
    if ws.empty:
        st.info("Оберіть хоча б одну країну.")
    else:
        fig = px.line(ws, x="year", y="wind_solar", color="country", log_y=use_log,
                      labels={"year": "Рік", "wind_solar": "Вітер + сонце, ТВт·год", "country": "Країна"})
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Висновки до графіка 2", expanded=True):
        st.markdown("""
1. **Китай** - беззаперечний лідер за абсолютними обсягами; у середині 2010-х він обігнав США та Німеччину.
2. **США** - друге місце зі стабільним зростанням; **Німеччина** була піонером, але її крива вирівнюється.
3. **Індія** та **Бразилія** набирають оберти у другій половині 2010-х.
4. **Україна** на лінійній шкалі майже зливається з нулем; логарифмічна шкала показує бум «зелених» станцій у 2018-2020 роках і зупинку динаміки після 2022 року через війну.
""")

# Лідери
with tabs[3]:
    st.subheader("Лідери за часткою відновлюваних джерел в електроенергії")
    c1, c2, c3 = st.columns(3)
    yr3 = c1.slider("Рік", 2000, latest_year, latest_year, key="lead_y")
    topn = c2.slider("Скільки країн показати", 5, 30, 15, key="lead_n")
    min_gen = c3.number_input("Мінімальна генерація, ТВт·год", 0.0, 500.0, 5.0, key="lead_g",
                              help="Відсікає мікродержави з мізерним виробництвом")
    sub = countries[(countries.year == yr3) & (countries.electricity_generation >= min_gen)]
    sub = sub.dropna(subset=["renewables_share_elec"])
    top = sub.nlargest(topn, "renewables_share_elec").sort_values("renewables_share_elec")
    fig = px.bar(top, x="renewables_share_elec", y="country", orientation="h",
                 text=top.renewables_share_elec.round(0).astype(int).astype(str) + "%",
                 color_discrete_sequence=["seagreen"],
                 labels={"renewables_share_elec": "Частка відновлюваних у генерації, %", "country": ""})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Висновки до графіка 3", expanded=True):
        st.markdown("""
1. Топ очолюють країни з великими річками чи геотермальними ресурсами (Норвегія, Ісландія, Парагвай, Коста-Рика, Ефіопія, Непал) - переважно гідро, а не вітер і сонце.
2. Серед лідерів майже немає великих індустріальних економік: у них складніша структура навантаження.
3. Високою часткою не означає великий обсяг: рейтинг показує структуру, а не масштаб.
""")

# ВВП і частка
with tabs[4]:
    st.subheader("Чи залежить частка чистої електроенергії від рівня розвитку?")
    ok = countries.dropna(subset=["gdp", "population", "renewables_share_elec", "electricity_generation"])
    years4 = sorted(ok.year.unique())
    c1, c2 = st.columns(2)
    y4 = c1.select_slider("Рік", options=[int(y) for y in years4], value=int(max(years4)), key="gdp_y")
    min_g4 = c2.slider("Мінімальна генерація, ТВт·год", 0, 100, 1, key="gdp_g")
    sc = ok[(ok.year == y4) & (ok.electricity_generation >= min_g4)].copy()
    sc["gdp_per_capita"] = sc.gdp / sc.population
    fig = px.scatter(sc, x="gdp_per_capita", y="renewables_share_elec", size="electricity_generation",
                     size_max=45, hover_name="country", log_x=True, opacity=0.6,
                     color_discrete_sequence=["teal"],
                     labels={"gdp_per_capita": "ВВП на душу населення, міжн. $ (лог. шкала)",
                             "renewables_share_elec": "Частка відновлюваних, %",
                             "electricity_generation": "Генерація, ТВт·год"})
    st.plotly_chart(fig, use_container_width=True)
    if len(sc) > 2:
        rho = sc.gdp_per_capita.rank().corr(sc.renewables_share_elec.rank())
        st.metric("Рангова кореляція ВВП на душу і частки відновлюваних", f"{rho:.2f}", f"країн: {len(sc)}", delta_color="off")
    with st.expander("Висновки до графіка 4", expanded=True):
        st.markdown("""
1. Хмара точок широка і майже без чіткого тренду - рівень доходу не визначає частку чистої електроенергії.
2. На всіх рівнях ВВП є і «зелені», і «брудні» країни; структуру визначають природні ресурси та історичні рішення.
3. Обмеження: кореляція не доводить причинності.
""")

# Темпи
with tabs[5]:
    st.subheader("Хто швидше за всіх збільшує частку відновлюваних джерел")
    c1, c2, c3 = st.columns(3)
    ys5 = c1.slider("Початковий рік", 1990, 2015, 2000, key="gr_s")
    ye5 = c2.slider("Кінцевий рік", ys5 + 1, latest_year, latest_year, key="gr_e")
    n5 = c3.slider("Кількість країн у кожній групі", 3, 15, 10, key="gr_n")
    start = countries[countries.year == ys5].set_index("country").renewables_share_elec
    end = countries[countries.year == ye5].set_index("country")
    ch = pd.DataFrame({"s": start, "e": end.renewables_share_elec, "g": end.electricity_generation}).dropna()
    ch = ch[ch.g >= 5]
    ch["delta"] = ch.e - ch.s
    plot = pd.concat([ch.nsmallest(n5 // 2 + 1, "delta"), ch.nlargest(n5, "delta")]).sort_values("delta")
    plot["Напрям"] = np.where(plot.delta >= 0, "Зростання", "Спад")
    fig = px.bar(plot.reset_index(), x="delta", y="country", orientation="h", color="Напрям",
                 color_discrete_map={"Зростання": "seagreen", "Спад": "tomato"},
                 labels={"delta": f"Зміна частки {ys5}-{ye5}, відсоткові пункти (п.п.)", "country": ""})
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Висновки до графіка 5", expanded=True):
        st.markdown("""
1. Найбільший прогрес - у країн, що різко змінили структуру генерації (переважно європейських і кількох тих, що розвиваються): десятки п.п. за 20+ років.
2. Червоні стовпчики - країни, де генерація з викопного палива росла швидше, ніж відновлювана.
3. Темп важливіший за поточну позицію: лідери за темпами не збігаються з лідерами за часткою.
""")

# Мапа
with tabs[6]:
    st.subheader("Частка відновлюваних джерел за країнами")
    y6 = st.slider("Рік", 2000, latest_year, latest_year, key="map_y")
    md = countries[countries.year == y6].dropna(subset=["renewables_share_elec"]).copy()
    md["renewables_share_elec"] = md.renewables_share_elec.clip(0, 100)
    fig = px.choropleth(md, locations="iso_code", color="renewables_share_elec", hover_name="country",
                        range_color=(0, 100), color_continuous_scale="YlGn",
                        labels={"renewables_share_elec": "Частка, %"})
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Висновки до мапи 1", expanded=True):
        st.markdown("""
1. Найтемніші території: Скандинавія, Південна Америка (Бразилія, Парагвай, Уругвай), Центральна Африка, Гімалаї - регіони з великими річками та гірським рельєфом.
2. Найсвітліші: Близький Схід, Північна Африка, Центральна Азія - країни з великими запасами нафти, газу чи вугілля.
3. Європа - «мозаїка»: від Норвегії та Австрії (70-80%+) до Польщі й Балкан. Україна - у середній групі.
4. Сірий колір - країни без даних: картограма чесно показує прогалини в статистиці.
""")

# Електростанції
plants = pd.DataFrame([
    ("Three Gorges Dam", "China", "Hydro", 22500, 30.823, 111.003),
    ("Itaipu Dam", "Brazil/Paraguay", "Hydro", 14000, -25.408, -54.589),
    ("Xiluodu Dam", "China", "Hydro", 13860, 28.257, 103.636),
    ("Belo Monte", "Brazil", "Hydro", 11233, -3.127, -51.775),
    ("Gansu Wind Farm (Jiuquan)", "China", "Wind", 7965, 40.200, 96.000),
    ("Alta Wind Energy Center", "USA", "Wind", 1548, 35.030, -118.330),
    ("Muppandal Wind Farm", "India", "Wind", 1500, 8.260, 77.550),
    ("Hornsea 2 (offshore)", "United Kingdom", "Wind", 1386, 53.885, 1.791),
    ("Bhadla Solar Park", "India", "Solar", 2245, 27.539, 71.915),
    ("Tengger Desert Solar Park", "China", "Solar", 1547, 37.580, 105.050),
    ("Noor Ouarzazate Complex", "Morocco", "Solar", 580, 31.000, -6.860),
    ("Topaz Solar Farm", "USA", "Solar", 550, 35.380, -120.070),
    ("Ivanpah (CSP)", "USA", "Solar", 392, 35.560, -115.470),
    ("Dnipro HPP (Zaporizhzhia)", "Ukraine", "Hydro", 1569, 47.870, 35.090),
    ("Kremenchuk HPP", "Ukraine", "Hydro", 625, 49.070, 33.250),
    ("Chornobyl Solar Plant", "Ukraine", "Solar", 1, 51.390, 30.100),
], columns=["name", "country", "type", "capacity_mw", "lat", "lon"])

with tabs[7]:
    st.subheader("Найбільші відновлювані електростанції світу")
    types = st.multiselect("Тип станції", ["Hydro", "Wind", "Solar"], default=["Hydro", "Wind", "Solar"], key="pl_t")
    pv = plants[plants.type.isin(types)]
    fig = px.scatter_geo(pv, lat="lat", lon="lon", size="capacity_mw", size_max=40, color="type",
                         color_discrete_map={"Hydro": "royalblue", "Wind": "seagreen", "Solar": "orange"},
                         hover_name="name", hover_data={"country": True, "capacity_mw": ":,", "lat": False, "lon": False},
                         labels={"capacity_mw": "Потужність, МВт", "type": "Тип"},
                         projection="natural earth")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pv.rename(columns={"capacity_mw": "capacity, MW"}), use_container_width=True)
    st.caption("Таблиця станцій складена вручну за відкритими довідковими джерелами.")
    with st.expander("Висновки до мапи 2", expanded=True):
        st.markdown("""
1. Найпотужніші об'єкти - ГЕС: одна дорівнює десяткам сонячних парків, але потребує великої річки та гір.
2. Сонячні та вітрові парки розташовані там, де найкращі умови: у пустелях і на вітряних узбережжях.
3. Найбільші станції зосереджені в Китаї, Індії та США - тих самих країнах, що лідирують у графіку 2.
4. Для України показані ГЕС на Дніпрі (значну частину пошкоджено чи втрачено через війну) і Чорнобильська сонячна станція.
""")

# Хмара слів
TEXT = """Renewable energy is energy that comes from natural sources that are constantly replenished, such as sunlight, wind, rain, tides, waves and geothermal heat. Unlike fossil fuels, which are burned and gradually run out, renewable sources can be used again and again without being exhausted. In recent years renewable energy has become one of the fastest growing parts of the global energy system.

Solar power uses photovoltaic panels to turn sunlight directly into electricity. The cost of solar panels has fallen dramatically over the past decade, and solar farms are now being built in deserts, on rooftops and even on water. Wind power uses turbines to convert the movement of air into electricity. Wind farms can be built on land or offshore, where the wind is stronger and more stable. Hydropower, produced by flowing water in rivers and dams, is the oldest and largest source of renewable electricity, and it supplies most of the electricity in countries such as Norway, Brazil and Paraguay. Geothermal plants use heat from the Earth and provide almost all the heating and a large part of the electricity in Iceland. Bioenergy is produced from wood, crops and waste.

Governments support the energy transition with subsidies, auctions and green tariffs, and with climate targets that aim to cut greenhouse gas emissions and slow global warming. Burning coal, oil and gas releases carbon dioxide, which is the main driver of climate change. Replacing fossil fuels with clean electricity also improves air quality and reduces dependence on imported fuel, which is important for energy security.

However, the transition faces serious challenges. Sunshine and wind vary during the day and between seasons, so power grids need storage, flexible demand and strong transmission lines. Batteries, pumped hydro and green hydrogen can store surplus electricity and release it when demand is high. Building new grids and storage requires large investment, long planning and public support. War and conflict can also damage power plants and networks, as seen in Ukraine, where energy infrastructure has been attacked and many power stations have been destroyed or occupied.

Despite these problems, experts expect that wind and solar will provide a growing share of the world electricity in the coming decades, and that countries such as China, the United States, India and members of the European Union will continue to lead the investment in clean energy."""

CUSTOM_STOPWORDS = {
    "also", "many", "much", "one", "two", "used", "use", "using", "uses", "will", "can", "may", "could", "would",
    "new", "however", "despite", "often", "even", "well", "still", "every", "part", "parts", "recent", "years",
    "year", "decade", "decades", "coming", "past", "large", "largest", "oldest", "almost", "most", "among",
    "between", "around", "since", "within", "without", "while", "such", "like", "make", "made", "become",
    "becoming", "including", "include", "includes", "based", "high", "low", "long", "time", "times", "shown",
    "seen", "expect", "experts", "aim", "important", "serious", "strong", "faces", "face", "growing", "share",
}


def bulb_mask(size=900):
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    cx, r, cy = size // 2, int(size * 0.24), int(size * 0.36)
    step = int(size * 0.035)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)
    d.polygon([(cx - int(r * .62), cy + int(r * .8)), (cx + int(r * .62), cy + int(r * .8)),
               (cx + int(r * .4), cy + int(r * 1.75)), (cx - int(r * .4), cy + int(r * 1.75))], fill=0)
    for k in range(3):
        y0 = cy + int(r * 1.8) + k * step
        d.rounded_rectangle([cx - int(r * .42), y0, cx + int(r * .42), y0 + step - 4], radius=12, fill=0)
    y_tip = cy + int(r * 1.8) + 3 * step
    d.ellipse([cx - int(r * .16), y_tip, cx + int(r * .16), y_tip + int(size * .04)], fill=0)
    for ang in (180, 200, 225, 250, 290, 315, 340, 0):
        a = math.radians(ang)
        d.line([cx + math.cos(a) * r * 1.18, cy + math.sin(a) * r * 1.18,
                cx + math.cos(a) * r * 1.45, cy + math.sin(a) * r * 1.45], fill=0, width=int(size * .025))
    return np.array(img)


@st.cache_data
def make_cloud(custom: bool, max_words: int):
    stops = set(STOPWORDS) | (CUSTOM_STOPWORDS if custom else set())
    return WordCloud(background_color="white", max_words=max_words, mask=bulb_mask(), stopwords=stops,
                     collocations=False, colormap="YlGn", contour_width=3, contour_color="darkgreen",
                     random_state=42).generate(TEXT).to_array()


with tabs[8]:
    st.subheader("Про що говорять тексти про відновлювану енергетику")
    c1, c2 = st.columns(2)
    custom = c1.checkbox("Прибрати власні стоп-слова", value=True, key="wc_c")
    mw = c2.slider("Максимум слів", 20, 300, 300, key="wc_m")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(make_cloud(custom, mw), interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    with st.expander("Висновки до хмари слів", expanded=True):
        st.markdown("""
1. Центральні слова - solar, wind, power, electricity, hydropower, fossil, grid, storage, climate: дискурс про джерела та принципи роботи.
2. Поряд із позитивною лексикою є слова-проблеми: тексти говорять і про технічні та політичні обмеження.
3. Власні стоп-слова прибрали загальні слова, і тематична лексика вийшла на передній план - у цьому мета роботи зі стоп-словами.
""")

# Загальні висновки
with tabs[9]:
    st.subheader("Загальні висновки")
    st.markdown("""
1. Світове виробництво електроенергії зростає; вугілля досі найбільше джерело, але приріст останніх років забезпечили вітер і сонце, а гідроенергетика майже не росте.
2. За часткою лідирують країни з рельєфом і водою (Норвегія, Парагвай, Ісландія), за обсягами - Китай, США, Індія, за темпами - країни, що швидко змінили політику.
3. Кореляція ВВП на душу з часткою відновлюваних джерел слабка - вирішують ресурси та політичні рішення.
4. Картограма показує великий розрив між регіонами, а мапа станцій - що найбільші об'єкти розташовані там, де найкращі природні умови.
5. Відновлювані джерела - це і економіка, і безпека: децентралізована генерація стійкіша до ударів, ніж великі станції.
6. Хмара слів поєднує оптимізм щодо технологій з тривогою щодо мереж, зберігання й інвестицій.
""")
