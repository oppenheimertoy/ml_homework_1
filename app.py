import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# загрузка модели
@st.cache_resource
def load_model():
    with open('model.pickle', 'rb') as f:
        return pickle.load(f)

model_data = load_model()
pipeline = model_data['pipeline']
feature_names = model_data['feature_names']
cat_features = model_data['cat_features']
num_features = model_data['num_features']
train_sample = model_data['train_data']

st.title("Предсказание цены автомобиля")

page = st.sidebar.selectbox(
    "Выберите раздел",
    ["EDA", "Предсказание", "Веса модели"]
)

if page == "EDA":
    st.header("Разведочный анализ данных")
    st.subheader("Пример данных")
    st.dataframe(train_sample.head(20))
    st.subheader("Статистики числовых признаков")
    st.dataframe(train_sample.describe())
    # распределение цены
    st.subheader("Распределение цены (selling_price)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(train_sample['selling_price'], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Цена")
    ax.set_ylabel("Количество")
    st.pyplot(fig)

    # цена vs год
    st.subheader("Цена vs Год выпуска")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=train_sample, x='year', y='selling_price', alpha=0.5, ax=ax)
    ax.set_xlabel("Год")
    ax.set_ylabel("Цена")
    st.pyplot(fig)

    # цена vs пробег
    st.subheader("Цена vs Пробег")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=train_sample, x='km_driven', y='selling_price', alpha=0.5, ax=ax)
    ax.set_xlabel("Пробег (км)")
    ax.set_ylabel("Цена")
    st.pyplot(fig)

    # корреляционная матрица
    st.subheader("Корреляционная матрица")
    num_cols = train_sample.select_dtypes(include=['number']).columns
    corr = train_sample[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # распределение по категориям
    st.subheader("Распределение по типу топлива")
    fig, ax = plt.subplots(figsize=(6, 4))
    train_sample['fuel'].value_counts().plot(kind='bar', ax=ax)
    ax.set_ylabel("Количество")
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif page == "Предсказание":
    st.header("Предсказание цены")
    
    input_mode = st.radio("Способ ввода данных:", ["Ручной ввод", "Загрузка CSV"])

    if input_mode == "Ручной ввод":
        st.subheader("Введите характеристики автомобиля")

        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2018)
            km_driven = st.number_input("Пробег (км)", min_value=0, max_value=1000000, value=50000)
            mileage = st.number_input("Расход (kmpl)", min_value=0.0, max_value=50.0, value=18.0)
            engine = st.number_input("Объём двигателя (CC)", min_value=500, max_value=6000, value=1200)

        with col2:
            max_power = st.number_input("Мощность (bhp)", min_value=30.0, max_value=500.0, value=80.0)
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
            fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
            transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

        if st.button("Предсказать цену"):
            # собираем данные
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'seats': [seats],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner]
            })
            # приводим типы как в train
            for col in ['fuel', 'seller_type', 'transmission', 'owner']:
                input_data[col] = input_data[col].astype('category')

            try:
                prediction = pipeline.predict(input_data)[0]
                st.success(f"Предсказанная цена: {prediction:,.0f} рупий")
                st.info(f"Примерно {prediction / 83:,.0f} USD (курс ~83 INR/USD)")
            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
    else:  # загрузка csv
        st.subheader("Загрузите CSV-файл")
        st.markdown("Файл должен содержать колонки: year, km_driven, mileage, engine, max_power, seats, fuel, seller_type, transmission, owner")
        uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.write("Загруженные данные:")
                st.dataframe(df_input.head())
                
                # приводим типы
                for col in ['fuel', 'seller_type', 'transmission', 'owner']:
                    if col in df_input.columns:
                        df_input[col] = df_input[col].astype('category')
                
                if st.button("Предсказать для всех"):
                    predictions = pipeline.predict(df_input)
                    df_input['predicted_price'] = predictions
                    st.write("Результаты:")
                    st.dataframe(df_input)
                    # скачать результат
                    csv = df_input.to_csv(index=False)
                    st.download_button(
                        label="Скачать результаты CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Ошибка: {e}")
                
elif page == "Веса модели":
    st.header("Веса (коэффициенты) модели")
    
    try:
        # достаём коэффициенты из пайплайна
        model = pipeline.named_steps['model']
        coefs = model.coef_
        # получаем названия признаков после OHE
        preprocessor = pipeline.named_steps['preprocess']
        ohe = preprocessor.named_transformers_['cat']

        # формируем имена
        ohe_names = []
        for feat, cats in zip(cat_features, ohe.categories_):
            for cat in cats:
                ohe_names.append(f"{feat}_{cat}")

        all_names = ohe_names + num_features

        # если длины совпадают
        if len(all_names) == len(coefs):
            coef_df = pd.DataFrame({
                'Признак': all_names,
                'Коэффициент': coefs
            }).sort_values('Коэффициент', key=lambda x: abs(x), ascending=False)

            st.subheader("Топ-15 самых важных признаков (по модулю)")
            top_15 = coef_df.head(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if x > 0 else 'red' for x in top_15['Коэффициент']]
            ax.barh(top_15['Признак'], top_15['Коэффициент'], color=colors)
            ax.set_xlabel("Коэффициент")
            ax.set_title("Важность признаков (Ridge регрессия)")
            plt.gca().invert_yaxis()
            st.pyplot(fig)

            st.subheader("Все коэффициенты")
            st.dataframe(coef_df)
            
            st.markdown("""
            Интерпретация:
            - Зелёный - положительное влияние на цену
            - Красный - отрицательное влияние на цену
            - Чем больше модуль коэффициента, тем сильнее влияние признака
            """)
        else:
            st.warning(f"Несоответствие размерностей: {len(all_names)} признаков vs {len(coefs)} коэффициентов")
            st.write("Коэффициенты модели:", coefs)

    except Exception as e:
        st.error(f"Не удалось извлечь коэффициенты: {e}")
        st.write("Структура пайплайна:", pipeline.named_steps.keys())