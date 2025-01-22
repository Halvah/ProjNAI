from flask import Flask, request, jsonify, render_template
from predict import process_news

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def index():
    return render_template("app.html")
@app.route('/analyze', methods=['POST'])
def analyze_news():
    data = request.json
    news_text = data.get("news_text", "")
    if not news_text:
        return jsonify({"error": "No news text provided"}), 400

    results = process_news(news_text)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)