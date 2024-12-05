import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu từ URL
url = 'https://github.com/Hunggi123/sentiment_sample/raw/refs/heads/main/test.csv'
df = pd.read_csv(url, encoding='unicode_escape')

# Loại bỏ các giá trị NaN trong cột 'text'
df = df.dropna(subset=['text'])

# Tách dữ liệu thành các tập train và test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Biểu diễn văn bản dưới dạng ma trận TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Xây dựng mô hình MLP (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42)
mlp.fit(X_train_tfidf, y_train)

# Dự đoán trên tập kiểm tra
y_pred = mlp.predict(X_test_tfidf)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Hiển thị kết quả đánh giá mô hình
st.title('Dự đoán cảm xúc từ văn bản')
st.write(f'Accuracy: {accuracy:.4f}')
st.write('Classification Report:')
st.text(report)

# Hàm dự đoán cảm xúc cho văn bản mới
def predict_sentiment(new_text):
    new_text_tfidf = vectorizer.transform([new_text])
    prediction = mlp.predict(new_text_tfidf)
    return prediction[0]

# Giao diện người dùng để nhập văn bản và nhận dự đoán cảm xúc
st.header('Nhập văn bản để dự đoán cảm xúc')
user_input = st.text_area('Văn bản:')
if st.button('Dự đoán'):
    if user_input:
        predicted_sentiment = predict_sentiment(user_input)
        st.write(f'Predicted Sentiment: {predicted_sentiment}')
    else:
        st.write('Vui lòng nhập văn bản để dự đoán.')