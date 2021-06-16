from flask import Flask, request, jsonify
import traceback
import joblib
import sys
import pandas as pd

app = Flask(__name__)

@app.route('/is_strong', methods=['POST'])
def is_strong():
    if model:
        try:
            json_ = request.json
            query = pd.DataFrame(json_)
            vectorized_query = vectorizer.transform(query['text'].values)
            is_strong = model.predict(vectorized_query)
            return jsonify({ 'is_strong': is_strong.tolist() })

        except:

            return jsonify({ 'trace': traceback.format_exc() })

    else:
        return('No model available')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 5000
    
    model = joblib.load('../models/model.pkl')
    print('Model loaded')

    vectorizer = joblib.load('../models/vectorizer.pkl')
    print('Vectorizer loaded')
    app.run(port=port, debug=True)