import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
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
    scenario_option = st.selectbox("Pilih Skenario:", [
        "Skenario 1: Prediksi Harga Buku",
        "Skenario 2: Prediksi Ketersediaan Stok",
        "Skenario 3: Segmentasi Buku Berdasarkan Harga dan Rating",
        "Skenario 4: Prediksi Harga Buku dengan KNN"
    ])

    if scenario_option == "Skenario 1: Prediksi Harga Buku":
        st.subheader("Skenario 1: Prediksi Harga Buku berdasarkan Rating")
        st.write("**Penjelasan:**")
        st.write("Skenario ini bertujuan untuk memprediksi harga buku berdasarkan rating yang diberikan.")
        st.write("Model yang digunakan:")
        st.write("1. **Linear Regression**: Untuk hubungan linear antara rating dan harga.")
        st.write("2. **Decision Tree Regression**: Untuk menangkap pola non-linear.")
    
        # Splitting the data
        X = df[['Rating']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train Linear Regression
        model_reg = LinearRegression()
        model_reg.fit(X_train, y_train)
        y_pred_reg = model_reg.predict(X_test)
    
        # Train Decision Tree Regression
        model_tree = DecisionTreeRegressor(random_state=42)
        model_tree.fit(X_train, y_train)
        y_pred_tree = model_tree.predict(X_test)
    
        # Calculate metrics
        mse_reg = mean_squared_error(y_test, y_pred_reg)
        mse_tree = mean_squared_error(y_test, y_pred_tree)
        r2_reg = r2_score(y_test, y_pred_reg)
        r2_tree = r2_score(y_test, y_pred_tree)
    
        # Evaluation Metrics
        st.write("**Hasil:**")
        st.write("- **Mean Squared Error (Linear Regression):** {:.2f}".format(mse_reg))
        st.write("- **Mean Squared Error (Decision Tree Regression):** {:.2f}".format(mse_tree))
        st.write("- **R-squared (Linear Regression):** {:.2f}".format(r2_reg))
        st.write("- **R-squared (Decision Tree Regression):** {:.2f}".format(r2_tree))
    
        # Visualization: MSE and R-squared Comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Visualisasi Perbandingan MSE
        mse_scores = [mse_reg, mse_tree]
        axes[0].bar(['Linear Regression', 'Decision Tree'], mse_scores, color=['blue', 'orange'])
        axes[0].set_title('Perbandingan Mean Squared Error (MSE)')
        axes[0].set_ylabel('MSE')
        for i, v in enumerate(mse_scores):
            axes[0].text(i, v + 0.05, f'{v:.2f}', ha='center')
    
        # Visualisasi Perbandingan R-squared
        r2_scores = [r2_reg, r2_tree]
        axes[1].bar(['Linear Regression', 'Decision Tree'], r2_scores, color=['blue', 'orange'])
        axes[1].set_title('Perbandingan R-squared')
        axes[1].set_ylabel('R-squared')
        for i, v in enumerate(r2_scores):
            axes[1].text(i, v + 0.02, f'{v:.2f}', ha='center')
    
        st.pyplot(fig)
    
        # Determine optimal model
        st.write("**Model yang Lebih Optimal:**")
        if mse_reg < mse_tree:
            st.write("- Linear Regression lebih optimal berdasarkan MSE.")
        else:
            st.write("- Decision Tree Regression lebih optimal berdasarkan MSE.")
    
        if r2_reg > r2_tree:
            st.write("- Linear Regression lebih optimal berdasarkan R-squared.")
        else:
            st.write("- Decision Tree Regression lebih optimal berdasarkan R-squared.")


    elif scenario_option == "Skenario 2: Prediksi Ketersediaan Stok":
        st.subheader("Skenario 2: Prediksi Ketersediaan Stok")
        st.write("**Penjelasan:**")
        st.write("Skenario ini bertujuan untuk memprediksi apakah suatu buku tersedia ('In stock') atau tidak ('Out of stock') berdasarkan harga dan rating buku.")
        st.write("Model yang digunakan:")
        st.write("1. **Random Forest Classifier**: Menggunakan ensemble decision trees untuk meningkatkan akurasi.")
        st.write("2. **Logistic Regression**: Memprediksi probabilitas ketersediaan stok berdasarkan fitur input.")
    
        # Preprocessing for Availability
        df['Availability'] = df['Price'].apply(lambda x: 'In stock' if x > 50 else 'Out of stock')
        X = df[['Price', 'Rating']]
        y = df['Availability'].map({'In stock': 1, 'Out of stock': 0})
    
        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train Logistic Regression
        model_lr = LogisticRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        cm_lr = confusion_matrix(y_test, y_pred_lr)
    
        # Train Random Forest Classifier
        model_rf = RandomForestClassifier(random_state=42)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
    
        # Evaluation Metrics
        st.write("**Hasil:**")
        st.write("- **Akurasi Logistic Regression:** {:.2f}".format(accuracy_lr))
        st.write("- **Akurasi Random Forest Classifier:** {:.2f}".format(accuracy_rf))
    
        # Visualization: Accuracy Comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(['Logistic Regression', 'Random Forest'], [accuracy_lr, accuracy_rf], color=['blue', 'green'])
        ax.set_title('Perbandingan Accuracy')
        ax.set_ylabel('Accuracy')
        for i, v in enumerate([accuracy_lr, accuracy_rf]):
            ax.text(i, v + 0.02, f'{v:.2f}', ha='center')
        st.pyplot(fig)
    
        # Visualization: Confusion Matrix Comparison
        st.write("**Perbandingan Confusion Matrix:**")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Logistic Regression Confusion Matrix
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title("Logistic Regression")
        axes[0].set_xlabel("Predicted Label")
        axes[0].set_ylabel("True Label")
    
        # Random Forest Confusion Matrix
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title("Random Forest Classifier")
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")
    
        st.pyplot(fig)


    elif scenario_option == "Skenario 3: Segmentasi Buku Berdasarkan Harga dan Rating":
        st.subheader("Skenario 3: Segmentasi Buku Berdasarkan Harga dan Rating")
        st.write("**Penjelasan:**")
        st.write("Segmentasi buku berdasarkan harga dan rating menggunakan K-Means atau DBSCAN.")
        X_clust = df[['Price', 'Rating']]
        model_option = st.selectbox("Pilih Model Clustering:", ["K-Means", "DBSCAN"])

        if model_option == "K-Means":
            model = KMeans(n_clusters=3, random_state=42)
        else:
            model = DBSCAN(eps=0.5, min_samples=5)

        labels = model.fit_predict(X_clust)
        df['Cluster'] = labels

        st.write("**Hasil:**")
        st.write("- Scatter plot menunjukkan distribusi cluster berdasarkan harga dan rating.")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='Price', y='Rating', hue='Cluster', data=df, palette='viridis', ax=ax)
        ax.set_title(f"Hasil Clustering dengan {model_option}")
        st.pyplot(fig)

    elif scenario_option == "Skenario 4: Prediksi Harga Buku dengan KNN":
        st.subheader("Skenario 4: Prediksi Harga Buku dengan KNN")
        st.write("**Penjelasan:**")
        st.write("Pada skenario ini, kami memprediksi harga buku berdasarkan rating menggunakan algoritma K-Nearest Neighbors (KNN) atau Support Vector Machine (SVM).")
    
        # Split data
        X = df[['Rating']]
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Train and predict with KNN
        model_knn = KNeighborsRegressor(n_neighbors=5)
        model_knn.fit(X_train, y_train)
        y_pred_knn = model_knn.predict(X_test)
    
        # Train and predict with SVM
        model_svm = SVR(kernel='linear')
        model_svm.fit(X_train, y_train)
        y_pred_svm = model_svm.predict(X_test)
    
        # Calculate metrics
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        mse_svm = mean_squared_error(y_test, y_pred_svm)
        r2_knn = r2_score(y_test, y_pred_knn)
        r2_svm = r2_score(y_test, y_pred_svm)
    
        st.write("**Hasil:**")
        st.write("- **Mean Squared Error (MSE):** MSE lebih kecil menunjukkan prediksi yang lebih akurat.")
        st.write("- **R-squared (R²):** R² mendekati 1 menunjukkan performa model yang lebih baik.")
    
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
        # Visualisasi MSE
        mse_values = [mse_knn, mse_svm]
        axes[0].bar(['KNN', 'SVM'], mse_values, color=['blue', 'green'])
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Mean Squared Error')
        axes[0].set_title('Perbandingan Mean Squared Error')
    
        # Visualisasi R2 Score
        r2_values = [r2_knn, r2_svm]
        axes[1].bar(['KNN', 'SVM'], r2_values, color=['blue', 'green'])
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('R2 Score')
        axes[1].set_title('Perbandingan R2 Score')
    
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
