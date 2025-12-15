"""
سكريبت للتنبؤ برحلات التأخير باستخدام النموذج المدرب
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def load_model(model_path=r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\best_delay_model.pkl'):
    """تحميل النموذج المدرب"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_delay(model, features_dict):
    """
    التنبؤ برحلة التأخير
    
    Parameters:
    -----------
    model: النموذج المدرب
    features_dict: قاموس يحتوي على الميزات المطلوبة
    
    Returns:
    --------
    predicted_delay: التأخير المتوقع بالدقائق
    """
    # تحويل القاموس إلى DataFrame
    df = pd.DataFrame([features_dict])
    
    # التنبؤ
    prediction = model.predict(df)[0]
    
    return max(0, prediction)  # التأخير لا يمكن أن يكون سالباً

def main():
    """مثال على الاستخدام"""
    print("🚌 نموذج التنبؤ برحلات التأخير")
    print("=" * 50)
    
    # تحميل النموذج
    model = load_model()
    print("✓ تم تحميل النموذج بنجاح")
    
    # مثال على الميزات المطلوبة
    example_features = {
        'hour': 18,
        'day_of_week': 2,  # الأربعاء
        'month': 1,
        'is_weekend': 0,
        'day_period_encoded': 2,  # مساء
        'time_category_en_encoded': 2,  # مساء
        'passenger_count_final': 50,
        'passenger_load_index': 1.5,
        'prev_delay': 15,
        'rush_period': 1,
        'route_frequency': 69,
        'speed_proxy': 0.5,
        'weather_severity': 0.0,
        'distance_change': 0.5,
        'latitude_clean': 25.5,
        'longitude_clean': 32.0,
        'weather_en_encoded': 0,  # صافي
        'passenger_level_encoded': 1,  # منخفض
    }
    
    # التنبؤ
    predicted_delay = predict_delay(model, example_features)
    
    print(f"\n📊 النتائج:")
    print(f"  التأخير المتوقع: {predicted_delay:.2f} دقيقة")
    print(f"  التأخير المتوقع: {predicted_delay / 60:.2f} ساعة")
    
    # تقسيم التأخير حسب المستويات
    if predicted_delay < 5:
        severity = "✅ تأخير بسيط جداً"
    elif predicted_delay < 15:
        severity = "⚠️ تأخير بسيط"
    elif predicted_delay < 30:
        severity = "⚠️⚠️ تأخير متوسط"
    elif predicted_delay < 60:
        severity = "⛔ تأخير كبير"
    else:
        severity = "🔴 تأخير خطير جداً"
    
    print(f"  درجة الخطورة: {severity}")

if __name__ == "__main__":
    main()
