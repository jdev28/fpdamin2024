# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk scraping data
def scrape_books():
    base_url = "https://books.toscrape.com/catalogue/"
    start_url = "https://books.toscrape.com/catalogue/page-1.html"

    books_data = []
    max_books = 400

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

    books_df = pd.DataFrame(books_data)
    return books_df

# Fungsi untuk membersihkan data
def clean_and_preprocess_data(df):
    if 'Availability' in df.columns:
        df.drop(columns=['Availability'], inplace=True)
    df.drop_duplicates(inplace=True)

    rating_mapping = {
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Four': 4,
        'Five': 5
    }
    if 'Rating' in df.columns:
        df['Rating'] = df['Rating'].map(rating_mapping)

    df.dropna(inplace=True)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(inplace=True)

    return df

# Fungsi untuk visualisasi data
def visualize_data(df):
    st.subheader("Visualisasi Data")
    st.write("Berikut adalah distribusi harga dan rating buku.")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.histplot(df['Price'], bins=30, kde=True, ax=axs[0])
    axs[0].set_title('Distribusi Harga Buku')
    axs[0].set_xlabel('Harga (£)')
    axs[0].set_ylabel('Frekuensi')

    sns.boxplot(x=df['Price'], ax=axs[1])
    axs[1].set_title('Variasi Harga Buku')
    axs[1].set_xlabel('Harga (£)')

    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.countplot(x='Rating', data=df, ax=ax)
    ax.set_title('Distribusi Rating Buku')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Jumlah Buku')
    st.pyplot(fig)

# Fungsi clustering
def perform_clustering(df):
    st.subheader("Clustering Buku")
    st.write("Mengelompokkan buku berdasarkan harga dan rating menggunakan algoritma K-Means.")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Price', 'Rating']])

    fig, ax = plt.subplots()
    sns.scatterplot(x='Price', y='Rating', hue='Cluster', palette='viridis', data=df, ax=ax)
    ax.set_title('Hasil Clustering K-Means')
    st.pyplot(fig)

# Fungsi regresi
def perform_regression(df):
    st.subheader("Prediksi Harga dengan Regresi")
    st.write("Menggunakan regresi linear untuk memprediksi harga buku berdasarkan rating.")

    X = df[['Rating']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_title('Prediksi vs Harga Aktual')
    ax.set_xlabel('Harga Aktual')
    ax.set_ylabel('Harga Prediksi')
    st.pyplot(fig)

# Main Program
def main():
    st.set_page_config(page_title="Final Project Data Mining", page_icon="📚", layout="wide")

    st.title("Final Project Data Mining")
    st.subheader("Kelompok 8")

    st.sidebar.header("Navigasi")
    options = st.sidebar.radio(
        "Pilih Langkah:",
        ["Scrape Data", "Visualisasi", "Clustering", "Regresi", "Kesimpulan"]
    )

    if options == "Scrape Data":
        st.header("Scraping Data")
        st.write("Menyediakan data buku dari website Books to Scrape.")
        df = scrape_books()
        st.dataframe(df.head())
        st.success("Data berhasil di-scrape!")

    elif options == "Visualisasi":
        st.header("Visualisasi Data")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        visualize_data(df)

    elif options == "Clustering":
        st.header("Clustering Data")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        perform_clustering(df)

    elif options == "Regresi":
        st.header("Prediksi Harga dengan Regresi")
        df = scrape_books()
        df = clean_and_preprocess_data(df)
        perform_regression(df)

    elif options == "Kesimpulan":
        st.header("Kesimpulan")
        st.write("Aplikasi ini menyediakan analisis data buku, termasuk visualisasi, clustering, dan prediksi harga.")
        st.success("Terima kasih telah menggunakan aplikasi ini!")

if __name__ == "__main__":
    main()
