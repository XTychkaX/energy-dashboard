import pandas as pd
import plotly.express as px
import streamlit as st
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


WORLD_PARTS = {
    "France": "Європа", "UK": "Європа", "Italy": "Європа", "Germany": "Європа",
    "Spain": "Європа", "Sweden": "Європа", "Norway": "Європа", "Denmark": "Європа",
    "US": "Північна Америка", "Canada": "Північна Америка",
    "Japan": "Азія", "Singapore": "Азія", "South Korea": "Азія",
    "Australia": "Австралія та Океанія",
}

ENV_LABELS = {"Urban": "Міські (Urban)", "Suburban": "Приміські (Suburban)"}

STATION_RANGES = ["Всі значення", "1 - 10", "11 - 30", "31 - ..."]


def load_rail(path: str = "data_RailTransport.csv") -> pd.DataFrame:
    # utf-8-sig прибирає невидимий BOM на початку файлу
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Перетворює у числа, сміття стає NaN
    df["End Year"] = pd.to_numeric(df["End Year"], errors="coerce").astype("Int64")

    # В Environment один рядок містить назву проєкту замість середовища
    df.loc[~df["Environment"].isin(ENV_LABELS), "Environment"] = "Suburban"

    # Частина світу за країною
    df["World part"] = df["Country"].map(WORLD_PARTS)

    # Вартість у млн GBP та довжина
    df["Length (mi)"] = df["Length (Miles)"]
    df["Cost (mil GBP)"] = df["Converted to mil GBP"]
    df["Country, city"] = df["Country"] + ", " + df["City"]
    return df


def filter_rail(df: pd.DataFrame, stations_range: str, world_part: str) -> pd.DataFrame:
    out = df
    n = out["Number of Stations"]
    if stations_range == "1 - 10":
        out = out[(n >= 1) & (n <= 10)]
    elif stations_range == "11 - 30":
        out = out[(n >= 11) & (n <= 30)]
    elif stations_range == "31 - ...":
        out = out[n >= 31]
    if world_part != "Всі частини світу":
        out = out[out["World part"] == world_part]
    return out

st.set_page_config(page_title="Rail Transport Infrastructure", page_icon="🚆", layout="wide")


@st.cache_data
def get_data():
    return load_rail("data_RailTransport.csv")


df = get_data()

# Заголовок
st.title("Rail Transport Infrastructure")

# Бічна панель
st.sidebar.header("Фільтри")

st.sidebar.write("Оберіть, скільки станцій має мати залізниця:")
stations_range = st.sidebar.selectbox("Кількість станцій", STATION_RANGES)

st.sidebar.write("Оберіть частину світу, де розташована залізниця:")
parts = ["Всі частини світу"] + sorted(df["World part"].dropna().unique())
world_part = st.sidebar.radio("Частина світу", parts)

# Вкладки
tab1, tab2 = st.tabs(["Огляд даних", "Відфільтровані залізниці"])

# Вкладка 1
with tab1:
    st.write("Щоб оглянути весь датафрейм, натисніть на кнопку нижче.")

    
    if "show_df" not in st.session_state:
        st.session_state.show_df = False
    if st.button("Показати датафрейм"):
        st.session_state.show_df = True

    if st.session_state.show_df:
        # Слайдер перед датафреймом: фільтр за роками закінчення будівництва
        known = df["End Year"].dropna().astype(int)
        year_min, year_max = int(known.min()), int(known.max())
        years = st.slider(
            "Роки закінчення будівництва залізниці",
            year_min, year_max, (year_min, year_max),
        )
        filtered = df[df["End Year"].between(years[0], years[1])]
        st.caption(
            f"Показано {len(filtered)} з {len(df)} залізниць "
            "(рядки, де рік закінчення невідомий, у фільтрі не враховуються)."
        )
        st.dataframe(filtered, use_container_width=True)

    st.subheader("Залізниці за середовищем розташування")
    env_counts = (
        df["Environment"].map(ENV_LABELS).value_counts().rename_axis("Середовище").reset_index(name="Кількість")
    )
    fig = px.pie(env_counts, names="Середовище", values="Кількість", hole=0.3,
                 title="Частка залізниць за середовищем розташування, кількість залізниць (шт.)")
    fig.update_traces(textinfo="label+percent+value")
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        f"Міських залізниць: **{env_counts.loc[env_counts['Середовище'].str.startswith('Міські'), 'Кількість'].iloc[0]}** шт., "
        f"приміських: **{env_counts.loc[env_counts['Середовище'].str.startswith('Приміські'), 'Кількість'].iloc[0]}** шт."
    )

# Вкладка 2
with tab2:
    res = filter_rail(df, stations_range, world_part)
    st.write(f"Фільтри: станцій - **{stations_range}**, частина світу - **{world_part}**. "
             f"Знайдено залізниць: **{len(res)}**")

    if res.empty:
        st.info("За обраними фільтрами залізниць немає.")
    else:
        headers = ["Назва залізниці", "Країна та місто", "Протяжність, милі", "Вартість, млн GBP (£)"]
        with st.container(height=550):
            cols = st.columns(4)
            for col, h in zip(cols, headers):
                col.markdown(f"**{h}**")
            for _, r in res.iterrows():
                c1, c2, c3, c4 = st.columns(4)
                c1.write(r["Project title"])
                c2.write(r["Country, city"])
                c3.write(f"{r['Length (mi)']:.2f} mi")
                c4.write(f"{r['Cost (mil GBP)']:,} млн £")
