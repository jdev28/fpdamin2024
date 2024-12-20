import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Fungsi Scraping Data
def scrape_books():
    import requests
    from bs4 import BeautifulSoup

    base_url = "https://books.toscrape.com/catalogue/"
    start_url = "https://books.toscrape.com/catalogue/page-1.html"

    books_data = []
    max_books = 100

    while start_url and len(books_data) < max_books:
        response = requests.get(start_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        for book in soup.find_all('article', class_='product_pod'):
            if len(books_data) >= max_books:
                break

            title = book.h3.a['title']
            price = book.find('p', class_='price_color').text[1:].replace('Â', '').strip()
            rating = book.p['class'][1]
            availability = book.find('p', class_='instock availability').text.strip()

            books_data.append({
                'Title': title,
                'Price': float(price.replace('£', '')),
                'Rating': rating,
                'Availability': availability
            })

        next_page = soup.find('li', class_='next')
        if next_page:
            next_url = next_page.a['href']
            start_url = base_url + next_url
        else:
            start_url = None

    return pd.DataFrame(books_data)

# Fungsi Preprocessing Data
def clean_and_preprocess_data(df):
    df.drop(columns=['Availability'], inplace=True, errors='ignore')
    df.drop_duplicates(inplace=True)
    rating_mapping = {
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Four': 4,
        'Five': 5
    }
    df['Rating'] = df['Rating'].map(rating_mapping)
    df.dropna(inplace=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(inplace=True)
    return df

# Fungsi Visualisasi
def visualize_data(df):
    st.subheader("Visualisasi Data")
    st.write("Berikut adalah beberapa visualisasi data untuk analisis awal.")

    # Visualisasi Harga Buku
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(df['Price'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Distribusi Harga Buku')
    sns.boxplot(x=df['Price'], ax=ax[1])
    ax[1].set_title('Variasi Harga Buku')
    st.pyplot(fig)

    # Distribusi Rating
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Rating', data=df, palette='viridis', ax=ax)
    ax.set_title('Distribusi Rating Buku')
    st.pyplot(fig)

    # Korelasi Harga dan Rating
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Rating', y='Price', data=df, ax=ax)
    ax.set_title('Korelasi Harga dan Rating')
    st.pyplot(fig)

# Fungsi untuk Menjalankan Skenario
def run_scenario(df):
    scenario_option = st.selectbox("Pilih Skenario:", ["Skenario 1: Prediksi Harga Buku", "Skenario 2: Prediksi Ketersediaan Stok"])

    if scenario_option == "Skenario 1: Prediksi Harga Buku":
        st.subheader("Skenario 1: Prediksi Harga Buku berdasarkan Rating")
        X = df[['Rating']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_option = st.selectbox("Pilih Model:", ["Linear Regression", "Decision Tree Regression"])

        if model_option == "Linear Regression":
            model = LinearRegression()
        else:
            model = DecisionTreeRegressor(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R-squared:** {r2:.2f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_title(f"Hasil Prediksi dengan {model_option}")
        st.pyplot(fig)

    elif scenario_option == "Skenario 2: Prediksi Ketersediaan Stok":
        st.subheader("Skenario 2: Prediksi Ketersediaan Stok")
        df['Availability'] = df['Price'].apply(lambda x: 'In stock' if x > 50 else 'Out of stock')
        X = df[['Price', 'Rating']]
        y = df['Availability'].map({'In stock': 1, 'Out of stock': 0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_option = st.selectbox("Pilih Model Klasifikasi:", ["Random Forest Classifier", "Logistic Regression"])

        if model_option == "Random Forest Classifier":
            model = RandomForestClassifier(random_state=42)
        else:
            model = LogisticRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.write(f"**Akurasi:** {accuracy:.2f}")
        st.write("**Classification Report:**")
        st.dataframe(pd.DataFrame(report).transpose())

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

# Fungsi Utama
def main():
    st.title("Analisis Buku dengan Streamlit")
    st.sidebar.header("Navigasi")
    menu = st.sidebar.radio("Pilih Langkah:", ["Scrape Data", "Visualisasi", "Scenario", "Kesimpulan"])

    if menu == "Scrape Data":
        st.header("Scraping Data")
        df = scrape_books()
        st.dataframe(df.head())
        st.write("Data berhasil di-scrape!")

    elif menu == "Visualisasi":
        st.header("Visualisasi Data")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        visualize_data(df)

    elif menu == "Scenario":
        st.header("Scenario Analisis")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        run_scenario(df)

    elif menu == "Kesimpulan":
        st.header("Kesimpulan")
        st.write("Model yang digunakan memberikan wawasan menarik tentang harga dan ketersediaan buku berdasarkan fitur yang tersedia.")

if __name__ == "__main__":
    main()
