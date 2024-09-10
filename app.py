from flask import Flask, render_template, request, jsonify
from query_processor import Generator

app = Flask(__name__)
model = Generator()

def generate_response(query):
    return model.process(query)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('query')
    response = generate_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
