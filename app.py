from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn
import linear_model

app = Flask(__name__)

model_file = open('model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', insurance_cost=0)

@app.route('/predict', methods=['POST'])
def predict():
    Tahun = request.args.get('Tahun', -1, type=int)
    if tahun == -1:
        return """<p>Input Salah</p>"""
    df = pd.read_csv("mhs-2011-2021.csv")
    data = df.loc[:, ['tahundaftar', 'jumlah']]
    sum_data = data.groupby('tahundaftar').sum()

    # Modifikasi untuk prediksi
    sum_data.insert(0, 'tahundaftar', sum_data.index.values, True)

    # Pembuatan data latih dan pengujian
    # msk = np.random.rand(len(sum_data)) < 0.8
    # train = sum_data[msk]
    # test = sum_data[~msk]

    # Bikin Model
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(sum_data[['tahundaftar']])
    train_y = np.asanyarray(sum_data[['jumlah']])
    regr.fit (train_x, train_y)
    hasil = regr.predict([[Tahun]])
    return '''<h1>Prediksi Mahasiswa Tahun {} adalah {}</h1><p><a href="/">Kembali</a></p>'''.format(str(Tahun), str(int(hasil[0][0])))
    return render_template('index.html', insurance_cost=hasil)
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page

    age, sex, smoker = [x for x in request.form.values()]

    data = []

    data.append(int(age))
    if sex == 'Laki-laki':
        data.extend([0, 1])
    else:
        data.extend([1, 0])

    if smoker == 'Ya':
        data.extend([0, 1])
    else:
        data.extend([1, 0])
    
    prediction = model.predict([data])
    output = round(prediction[0], 2)

    return render_template('index.html', insurance_cost=output, age=age, sex=sex, smoker=smoker)
    '''

if __name__ == '__main__':
    app.run(debug=True)
