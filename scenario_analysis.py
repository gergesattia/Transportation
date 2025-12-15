"""
مثال متقدم: استخدام النموذج للتنبؤ بسيناريوهات مختلفة
"""

import pickle
import pandas as pd
import numpy as np
from predict_delay import load_model, predict_delay

def scenario_analysis():
    """تحليل سيناريوهات مختلفة"""
    
    print("=" * 70)
    print("🎯 تحليل السيناريوهات المختلفة للتنبؤ برحلات التأخير")
    print("=" * 70)
    
    # تحميل النموذج
    model = load_model()
    
    # الميزات الأساسية (قيم افتراضية)
    base_features = {
        'hour': 9,
        'day_of_week': 2,
        'month': 1,
        'is_weekend': 0,
        'day_period_encoded': 0,
        'time_category_en_encoded': 0,
        'passenger_count_final': 50,
        'passenger_load_index': 1.0,
        'prev_delay': 0,
        'rush_period': 0,
        'route_frequency': 69,
        'speed_proxy': 0.5,
        'weather_severity': 0.0,
        'distance_change': 0.0,
        'latitude_clean': 25.5,
        'longitude_clean': 32.0,
        'weather_en_encoded': 0,
        'passenger_level_encoded': 0,
    }
    
    # ============ السيناريو الأول: تأثير الركاب ============
    print("\n📊 السيناريو 1️⃣: تأثير عدد الركاب على التأخير")
    print("-" * 70)
    
    passenger_counts = [20, 50, 100, 150, 200]
    results = []
    
    for count in passenger_counts:
        features = base_features.copy()
        features['passenger_count_final'] = count
        delay = predict_delay(model, features)
        results.append({'الركاب': count, 'التأخير (دقيقة)': delay})
    
    df_scenario1 = pd.DataFrame(results)
    print(df_scenario1.to_string(index=False))
    
    # ============ السيناريو الثاني: تأثير الطقس ============
    print("\n\n🌤️ السيناريو 2️⃣: تأثير حالة الطقس على التأخير")
    print("-" * 70)
    
    weathers = [
        {'name': 'صافي', 'severity': 0.0, 'encoded': 0},
        {'name': 'غائم', 'severity': 0.5, 'encoded': 1},
        {'name': 'ممطر', 'severity': 1.0, 'encoded': 2},
        {'name': 'عاصفة', 'severity': 1.5, 'encoded': 3},
    ]
    
    results = []
    for weather in weathers:
        features = base_features.copy()
        features['weather_severity'] = weather['severity']
        features['weather_en_encoded'] = weather['encoded']
        delay = predict_delay(model, features)
        results.append({'حالة الطقس': weather['name'], 'التأخير (دقيقة)': delay})
    
    df_scenario2 = pd.DataFrame(results)
    print(df_scenario2.to_string(index=False))
    
    # ============ السيناريو الثالث: تأثير الساعة ============
    print("\n\n⏰ السيناريو 3️⃣: تأثير الساعة على التأخير")
    print("-" * 70)
    
    hours = [6, 9, 12, 15, 18, 21, 23]
    results = []
    
    for hour in hours:
        features = base_features.copy()
        features['hour'] = hour
        
        # تحديد فترة اليوم
        if 6 <= hour < 12:
            features['day_period_encoded'] = 0  # صباح
            period = "صباح"
        elif 12 <= hour < 18:
            features['day_period_encoded'] = 1  # عصر
            period = "عصر"
        elif 18 <= hour < 24:
            features['day_period_encoded'] = 2  # مساء
            period = "مساء"
        else:
            features['day_period_encoded'] = 3  # ليل
            period = "ليل"
        
        delay = predict_delay(model, features)
        results.append({'الساعة': f'{hour:02d}:00', 'الفترة': period, 'التأخير (دقيقة)': delay})
    
    df_scenario3 = pd.DataFrame(results)
    print(df_scenario3.to_string(index=False))
    
    # ============ السيناريو الرابع: مقارنة أيام الأسبوع ============
    print("\n\n📅 السيناريو 4️⃣: تأثير يوم الأسبوع على التأخير")
    print("-" * 70)
    
    days = [
        {'id': 0, 'name': 'الاثنين', 'weekend': 0},
        {'id': 1, 'name': 'الثلاثاء', 'weekend': 0},
        {'id': 2, 'name': 'الأربعاء', 'weekend': 0},
        {'id': 3, 'name': 'الخميس', 'weekend': 0},
        {'id': 4, 'name': 'الجمعة', 'weekend': 1},
        {'id': 5, 'name': 'السبت', 'weekend': 1},
        {'id': 6, 'name': 'الأحد', 'weekend': 0},
    ]
    
    results = []
    for day in days:
        features = base_features.copy()
        features['day_of_week'] = day['id']
        features['is_weekend'] = day['weekend']
        delay = predict_delay(model, features)
        day_type = "🎉 عطلة" if day['weekend'] else "📅 عمل"
        results.append({'اليوم': day['name'], 'النوع': day_type, 'التأخير (دقيقة)': delay})
    
    df_scenario4 = pd.DataFrame(results)
    print(df_scenario4.to_string(index=False))
    
    # ============ السيناريو الخامس: أسوأ الظروف ============
    print("\n\n🔴 السيناريو 5️⃣: أسوأ الظروف الممكنة")
    print("-" * 70)
    
    worst_case = base_features.copy()
    worst_case.update({
        'hour': 18,                          # ساعة الذروة المسائية
        'day_of_week': 4,                   # الجمعة
        'is_weekend': 1,                    # عطلة نهاية أسبوع
        'day_period_encoded': 2,            # مساء
        'passenger_count_final': 200,       # حد أقصى من الركاب
        'passenger_load_index': 3.0,        # تحميل عالي جداً
        'prev_delay': 30,                   # تأخير سابق كبير
        'rush_period': 1,                   # فترة ذروة
        'weather_severity': 1.5,            # طقس سيء
        'weather_en_encoded': 3,            # عاصفة
        'distance_change': 2.0,             # تغيير كبير في المسافة
    })
    
    worst_delay = predict_delay(model, worst_case)
    
    best_case = base_features.copy()
    best_case.update({
        'hour': 10,                         # ساعة مريحة
        'day_of_week': 2,                  # يوم عادي
        'is_weekend': 0,                   # يوم عمل
        'passenger_count_final': 20,       # عدد قليل من الركاب
        'passenger_load_index': 0.5,       # تحميل منخفض
        'prev_delay': 0,                   # لا توجد تأخيرات سابقة
        'rush_period': 0,                  # ليس فترة ذروة
        'weather_severity': 0.0,           # طقس صافي
        'weather_en_encoded': 0,           # صافي
        'distance_change': 0.0,            # لا يوجد تغيير
    })
    
    best_delay = predict_delay(model, best_case)
    
    print(f"\n❌ أسوأ الظروف:")
    print(f"   التأخير المتوقع: {worst_delay:.2f} دقيقة")
    print(f"   ≈ {worst_delay / 60:.2f} ساعة")
    
    print(f"\n✅ أفضل الظروف:")
    print(f"   التأخير المتوقع: {best_delay:.2f} دقيقة")
    
    print(f"\n📊 الفرق: {worst_delay - best_delay:.2f} دقيقة")
    print(f"   النسبة: {worst_delay / best_delay:.2f}x")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    scenario_analysis()
