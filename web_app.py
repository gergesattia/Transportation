"""
تطبيق ويب Flask للتنبؤ برحلات التأخير
Web Application for Transport Delay Prediction
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

app = Flask(__name__)
app.config['JSON_SUPPORT_360_NANS'] = False

# تحميل النموذج
MODEL_PATH = r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\best_delay_model.pkl'
DATA_PATH = r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\dataset_with_features.csv'

def load_model():
    """تحميل النموذج المدرب"""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def load_data():
    """تحميل البيانات للإحصائيات"""
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return None

# تحميل النموذج والبيانات عند بدء التطبيق
model = load_model()
data = load_data()

# ==================== الصفحات ====================

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """صفحة التنبؤ"""
    return render_template('predict.html')

@app.route('/analysis')
def analysis_page():
    """صفحة تحليل البيانات"""
    return render_template('analysis.html')

@app.route('/about')
def about_page():
    """صفحة حول التطبيق"""
    return render_template('about.html')

# ==================== API Endpoints ====================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """التنبؤ برحلة التأخير"""
    if model is None:
        return jsonify({'error': 'النموذج غير متاح'}), 500
    
    try:
        data_json = request.json
        
        # تحويل البيانات
        features_dict = {
            'hour': int(data_json.get('hour', 9)),
            'day_of_week': int(data_json.get('day_of_week', 2)),
            'month': int(data_json.get('month', 1)),
            'is_weekend': int(data_json.get('is_weekend', 0)),
            'day_period_encoded': int(data_json.get('day_period_encoded', 0)),
            'time_category_en_encoded': int(data_json.get('time_category_en_encoded', 0)),
            'passenger_count_final': float(data_json.get('passenger_count_final', 50)),
            'passenger_load_index': float(data_json.get('passenger_load_index', 1.0)),
            'prev_delay': float(data_json.get('prev_delay', 0)),
            'rush_period': int(data_json.get('rush_period', 0)),
            'route_frequency': float(data_json.get('route_frequency', 69)),
            'speed_proxy': float(data_json.get('speed_proxy', 0.5)),
            'weather_severity': float(data_json.get('weather_severity', 0.0)),
            'distance_change': float(data_json.get('distance_change', 0.0)),
            'latitude_clean': float(data_json.get('latitude_clean', 25.5)),
            'longitude_clean': float(data_json.get('longitude_clean', 32.0)),
            'weather_en_encoded': int(data_json.get('weather_en_encoded', 0)),
            'passenger_level_encoded': int(data_json.get('passenger_level_encoded', 0)),
        }
        
        # التنبؤ
        df = pd.DataFrame([features_dict])
        prediction = max(0, float(model.predict(df)[0]))
        
        # تحديد درجة الخطورة
        if prediction < 5:
            severity = "ممتاز"
            severity_icon = "✅"
            color = "success"
        elif prediction < 15:
            severity = "جيد"
            severity_icon = "⚠️"
            color = "warning"
        elif prediction < 30:
            severity = "متوسط"
            severity_icon = "⚠️⚠️"
            color = "info"
        elif prediction < 60:
            severity = "كبير"
            severity_icon = "⛔"
            color = "danger"
        else:
            severity = "خطير جداً"
            severity_icon = "🔴"
            color = "dark"
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'prediction_hours': round(prediction / 60, 2),
            'severity': severity,
            'severity_icon': severity_icon,
            'color': color
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics')
def api_statistics():
    """إحصائيات البيانات"""
    if data is None:
        return jsonify({'error': 'البيانات غير متاحة'}), 500
    
    try:
        # حساب الإحصائيات
        delay_col = 'delay_minutes_corrected'
        valid_delays = data[delay_col].dropna()
        
        stats = {
            'total_records': len(data),
            'valid_records': len(valid_delays),
            'avg_delay': float(valid_delays.mean()),
            'max_delay': float(valid_delays.max()),
            'min_delay': float(valid_delays.min()),
            'std_delay': float(valid_delays.std()),
            'median_delay': float(valid_delays.median()),
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/scenarios')
def api_scenarios():
    """تحليل السيناريوهات"""
    if model is None:
        return jsonify({'error': 'النموذج غير متاح'}), 500
    
    try:
        scenarios = {}
        
        # السيناريو 1: تأثير الركاب
        passenger_scenario = []
        for count in [20, 50, 100, 150, 200]:
            features = {
                'hour': 9, 'day_of_week': 2, 'month': 1, 'is_weekend': 0,
                'day_period_encoded': 0, 'time_category_en_encoded': 0,
                'passenger_count_final': count, 'passenger_load_index': 1.0,
                'prev_delay': 0, 'rush_period': 0, 'route_frequency': 69,
                'speed_proxy': 0.5, 'weather_severity': 0.0, 'distance_change': 0.0,
                'latitude_clean': 25.5, 'longitude_clean': 32.0,
                'weather_en_encoded': 0, 'passenger_level_encoded': 0,
            }
            delay = float(model.predict(pd.DataFrame([features]))[0])
            passenger_scenario.append({'passengers': count, 'delay': round(max(0, delay), 2)})
        scenarios['passengers'] = passenger_scenario
        
        # السيناريو 2: تأثير الطقس
        weather_scenario = []
        weathers = [
            {'name': 'صافي', 'severity': 0.0, 'encoded': 0},
            {'name': 'غائم', 'severity': 0.5, 'encoded': 1},
            {'name': 'ممطر', 'severity': 1.0, 'encoded': 2},
            {'name': 'عاصفة', 'severity': 1.5, 'encoded': 3},
        ]
        for weather in weathers:
            features = {
                'hour': 9, 'day_of_week': 2, 'month': 1, 'is_weekend': 0,
                'day_period_encoded': 0, 'time_category_en_encoded': 0,
                'passenger_count_final': 50, 'passenger_load_index': 1.0,
                'prev_delay': 0, 'rush_period': 0, 'route_frequency': 69,
                'speed_proxy': 0.5, 'weather_severity': weather['severity'],
                'distance_change': 0.0, 'latitude_clean': 25.5, 'longitude_clean': 32.0,
                'weather_en_encoded': weather['encoded'], 'passenger_level_encoded': 0,
            }
            delay = float(model.predict(pd.DataFrame([features]))[0])
            weather_scenario.append({'weather': weather['name'], 'delay': round(max(0, delay), 2)})
        scenarios['weather'] = weather_scenario
        
        # السيناريو 3: تأثير الساعة
        hour_scenario = []
        for hour in [6, 9, 12, 15, 18, 21, 23]:
            features = {
                'hour': hour, 'day_of_week': 2, 'month': 1, 'is_weekend': 0,
                'day_period_encoded': 0 if hour < 12 else (1 if hour < 18 else 2),
                'time_category_en_encoded': 0, 'passenger_count_final': 50,
                'passenger_load_index': 1.0, 'prev_delay': 0, 'rush_period': 0,
                'route_frequency': 69, 'speed_proxy': 0.5, 'weather_severity': 0.0,
                'distance_change': 0.0, 'latitude_clean': 25.5, 'longitude_clean': 32.0,
                'weather_en_encoded': 0, 'passenger_level_encoded': 0,
            }
            delay = float(model.predict(pd.DataFrame([features]))[0])
            hour_scenario.append({'hour': f'{hour:02d}:00', 'delay': round(max(0, delay), 2)})
        scenarios['hours'] = hour_scenario
        
        return jsonify(scenarios)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model-info')
def api_model_info():
    """معلومات عن النموذج"""
    return jsonify({
        'model_type': 'Machine Learning Regression',
        'features_count': 18,
        'models_tested': ['Random Forest', 'Gradient Boosting', 'Ridge Regression', 'Linear Regression'],
        'best_model': 'Gradient Boosting (Typically)',
        'status': 'مثبت وجاهز للاستخدام' if model else 'غير متاح',
    })

if __name__ == '__main__':
    import socket
    import os
    # Get local machine IP
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # For production (Render)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print("=" * 70)
    print("🚀 Transport Delay Prediction System is Running!")
    print("=" * 70)
    print(f"💻 Local Computer Access: http://127.0.0.1:{port}")
    print(f"🏠 Home Network Access: http://{local_ip}:{port}")
    print(f"🌐 Public Access: https://your-app-name.onrender.com")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(debug=debug, host='0.0.0.0', port=port)
