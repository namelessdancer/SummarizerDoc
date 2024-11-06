from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Khởi tạo các mô hình BART và LED
bart_summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
led_summarizer = pipeline('summarization', model='allenai/led-large-16384-arxiv')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    text = data.get('text', '')
    model_type = data.get('model_type', 'bart')  # Mặc định là BART nếu không chỉ định

    # Kiểm tra nếu người dùng không nhập nội dung
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    # Chọn mô hình dựa trên tham số model_type
    summarizer = bart_summarizer if model_type == 'bart' else led_summarizer

    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        if summary and isinstance(summary, list) and len(summary) > 0:
            return jsonify({'summary_text': summary[0]['summary_text']})
        else:
            return jsonify({'error': 'No summary generated'}), 500
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
