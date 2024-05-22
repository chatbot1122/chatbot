import re

import nltk
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download dataset stopwords dari NLTK
nltk.download('stopwords')

app = Flask(__name__)

# Fungsi untuk membersihkan teks dari karakter non-alfabet dan spasi
def data_cleaning(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

# Fungsi untuk menghapus simbol dari teks
def remove_symbols(text):
    symbols_removed = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return symbols_removed

# Fungsi untuk tokenisasi teks
def tokenization(text):
    tokens = word_tokenize(text)
    return tokens

# Fungsi untuk stemming kata-kata
def stemming(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

# Fungsi untuk menghapus stopwords dari teks
def stopword_removal(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

@app.route('/', methods=['GET', 'POST'])
def sentiment_analysis():
    result = None
    sentence = None  # Variabel untuk menyimpan kalimat dari formulir

    if request.method == 'POST':
        sentence = request.form['sentence']

        # Langkah-langkah analisis sentimen
        cleaned_sentence = data_cleaning(sentence)
        symbols_removed_sentence = remove_symbols(cleaned_sentence)
        tokens = tokenization(symbols_removed_sentence)
        stemmed_tokens = stemming(tokens)
        filtered_tokens = stopword_removal(stemmed_tokens)

        #hasil positif jika jumlah kata lebih dari 3
        result = "Positive" if len(filtered_tokens) > 3 else "Negative"

    return render_template('index.html', sentence=sentence, result=result)

if __name__ == '__main__':
    app.run(debug=True)
