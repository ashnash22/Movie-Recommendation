from flask import Flask, render_template, request
import pandas as pd
from recommender import hybrid_recommend, recommend_by_genre, recommend_by_keyword, surprise_me, load_data

app = Flask(__name__)
# ======================
# Load dataset once at startup from Azure Blob Storage
# ======================
def load_data(path):
    import pandas as pd
    df = pd.read_csv("https://strmovies.blob.core.windows.net/datasets-movies/movies_metadata.csv")
    return df


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    option = request.form["option"]

    if option == "movie":
        title = request.form["movie_title"]
        result = hybrid_recommend(title, top_n=10)
    elif option == "genre":
        genre = request.form["genre"]
        result = recommend_by_genre(genre, top_n=10)
    elif option == "keyword":
        keyword = request.form["keyword"]
        result = recommend_by_keyword(keyword, top_n=10)
    elif option == "surprise":
        genre = request.form.get("genre", None)
        result = surprise_me(genre)
    else:
        result = "Invalid option selected."

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)


