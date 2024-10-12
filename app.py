import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

app = Flask(__name__)

# 设置上传文件夹和允许的扩展名
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'xlsx'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('train_model', filename=file.filename))


@app.route('/train/<filename>')
def train_model(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # 读取Excel文件中的数据，指定sheet名称
    data = pd.read_excel(filepath, sheet_name='Sheet1')

    # 假设数据格式：第一列为水质参数 "shuju"，第二列和第三列分别为经纬度 "X" 和 "Y"
    X = data[['X', 'Y']]  # 经纬度作为自变量
    y = data['shuju']  # 水质参数作为目标变量

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用随机森林模型进行训练
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # 进行预测
    predictions = model.predict(X_test)

    # 将预测结果与经纬度结合
    result_df = pd.DataFrame({
        '经度': X_test['X'],
        '纬度': X_test['Y'],
        '预测水质参数': predictions
    })

    # 保存预测结果到文件
    result_path = os.path.join('visualizations', 'predictions.csv')
    result_df.to_csv(result_path, index=False)

    return redirect(url_for('visualize', filepath=result_path))


@app.route('/visualize/<filepath>')
def visualize(filepath):
    data = pd.read_csv(filepath)

    # 3D可视化预测结果
    fig = px.scatter_3d(data, x='经度', y='纬度', z='预测水质参数', hover_data=['经度', '纬度', '预测水质参数'])
    graph_html = fig.to_html(full_html=False)

    return render_template('result.html', graph_html=graph_html)


if __name__ == '__main__':
    app.run(debug=True)
