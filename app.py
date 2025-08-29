from flask import Flask, render_template, request
from recommender import (
    hybrid_recommend,
    recommend_by_genre,
    recommend_by_keyword,
    surprise_me,
    load_data,
)

app = Flask(__name__)

# ======================
# Load dataset once at startup
# ======================
DATA_URL = "https://strmovies.blob.core.windows.net/datasets-movies/movies_metadata.csv"
load_data(DATA_URL)   # ensure df and cosine_sim are initialized


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    option = request.form.get("option")

    if option == "movie":
        title = request.form.get("movie_title")
        result = hybrid_recommend(title, top_n=10)
    elif option == "genre":
        genre = request.form.get("genre")
        result = recommend_by_genre(genre, top_n=10)
    elif option == "keyword":
        keyword = request.form.get("keyword")
        result = recommend_by_keyword(keyword, top_n=10)
    elif option == "surprise":
        genre = request.form.get("genre", None)
        result = surprise_me(genre)
    else:
        result = "⚠️ Invalid option selected."

    return render_template("result.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
