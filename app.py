import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ==================== تحميل البيانات ====================
def load_and_prepare_data(file_path):
    """تحميل ومعالجة البيانات"""
    print("📊 تحميل البيانات...")
    df = pd.read_csv(file_path)
    
    print(f"عدد الصفوف الأصلية: {len(df)}")
    print(f"\n الأعمدة المتاحة:\n{df.columns.tolist()}")
    
    # إزالة الصفوف التي تحتوي على قيم فارغة في المتغير التابع (delay_minutes_corrected)
    df_clean = df.dropna(subset=['delay_minutes_corrected']).copy()
    print(f"عدد الصفوف بعد إزالة البيانات الفارغة: {len(df_clean)}")
    
    # ملء القيم الفارغة في الأعمدة الأخرى
    df_clean['prev_delay'] = df_clean['prev_delay'].fillna(0)
    
    # إزالة الصفوف التي تحتوي على infinity أو قيم كبيرة جداً
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.dropna()
    
    # حساب الميزات الزمنية الإضافية
    df_clean['scheduled_datetime'] = pd.to_datetime(df_clean['scheduled_date'])
    df_clean['day_of_week'] = df_clean['scheduled_datetime'].dt.dayofweek
    df_clean['month'] = df_clean['scheduled_datetime'].dt.month
    df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)
    
    # معالجة الأعمدة الفئوية
    categorical_cols = ['time_category_en', 'day_period', 'weather_en', 
                       'passenger_level', 'rush_period', 'delay_status_en']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].fillna('Unknown'))
            label_encoders[col] = le
    
    return df_clean, label_encoders

# ==================== تحضير الميزات ====================
def prepare_features(df, label_encoders):
    """تحضير الميزات المناسبة للنموذج"""
    print("\n🔧 تحضير الميزات...")
    
    # اختيار الميزات ذات الصلةy
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend', 'day_period_encoded',
        'time_category_en_encoded', 'passenger_count_final', 'passenger_load_index',
        'prev_delay', 'rush_period', 'route_frequency', 'speed_proxy',
        'weather_severity', 'distance_change', 'latitude_clean', 'longitude_clean',
        'weather_en_encoded', 'passenger_level_encoded'
    ]
    
    # إزالة الأعمدة التي لا توجد في البيانات
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['delay_minutes_corrected']
    
    print(f"عدد الميزات: {len(feature_cols)}")
    print(f"الميزات المستخدمة: {feature_cols}")
    
    return X, y, feature_cols

# ==================== بناء النماذج ====================
def build_models(X_train, X_test, y_train, y_test):
    """بناء وتدريب عدة نماذج مختلفة"""
    print("\n🤖 بناء النماذج...")
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, 
                                               random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, 
                                                       learning_rate=0.1, 
                                                       max_depth=5, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n  📍 تدريب {name}...")
        
        # إنشاء Pipeline مع تطبيع البيانات
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # تدريب النموذج
        pipeline.fit(X_train, y_train)
        
        # التنبؤ
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        # حساب المقاييس
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        
        results[name] = {
            'model': pipeline,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'predictions': y_pred_test
        }
        
        print(f"    ✓ Train R²: {train_r2:.4f}")
        print(f"    ✓ Test R²: {test_r2:.4f}")
        print(f"    ✓ Test RMSE: {test_rmse:.4f}")
        print(f"    ✓ Test MAE: {test_mae:.4f}")
    
    return results

# ==================== عرض النتائج ====================
def print_results(results):
    """عرض مقارنة النتائج"""
    print("\n" + "="*70)
    print("📊 مقارنة النماذج")
    print("="*70)
    
    results_df = pd.DataFrame({
        model_name: {
            'Train R²': result['train_r2'],
            'Test R²': result['test_r2'],
            'Test RMSE': result['test_rmse'],
            'Test MAE': result['test_mae']
        }
        for model_name, result in results.items()
    }).T
    
    print(results_df.to_string())
    print("\n✅ أفضل نموذج:", results_df['Test R²'].idxmax())
    
    return results_df

# ==================== تصور النتائج ====================
def visualize_results(results, y_test):
    """رسم النتائج"""
    print("\n📈 رسم النتائج...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('مقارنة نماذج التنبؤ برحلات التأخير', fontsize=16, fontweight='bold')
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        
        predictions = result['predictions']
        ax.scatter(y_test, predictions, alpha=0.5, s=20)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('القيمة الفعلية للتأخير (دقائق)')
        ax.set_ylabel('القيمة المتنبأ بها (دقائق)')
        ax.set_title(f'{name}\nR² = {result["test_r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ تم حفظ الرسم في 'model_comparison.png'")
    plt.show()

# ==================== تحليل أهمية الميزات ====================
def feature_importance(results, feature_cols):
    """عرض أهمية الميزات"""
    print("\n🎯 أهمية الميزات (للنموذج الأفضل):")
    
    # استخراج أفضل نموذج
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    # التحقق من وجود feature_importances_
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance_df = pd.DataFrame({
            'الميزة': feature_cols,
            'الأهمية': importances
        }).sort_values('الأهمية', ascending=False)
        
        print(feature_importance_df.head(10).to_string(index=False))
        
        # رسم أهمية الميزات
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance_df.head(10)
        ax.barh(range(len(top_features)), top_features['الأهمية'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['الميزة'].values)
        ax.set_xlabel('درجة الأهمية')
        ax.set_title(f'أهمية الميزات - {best_model_name}')
        plt.tight_layout()
        plt.savefig(r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ تم حفظ رسم الميزات في 'feature_importance.png'")
        plt.show()

# ==================== الدالة الرئيسية ====================
def main():
    # تحميل البيانات
    file_path = r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\dataset_with_features.csv'
    df, label_encoders = load_and_prepare_data(file_path)
    
    # تحضير الميزات
    X, y, feature_cols = prepare_features(df, label_encoders)
    
    # تقسيم البيانات
    print("\n📂 تقسيم البيانات...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"بيانات التدريب: {len(X_train)}")
    print(f"بيانات الاختبار: {len(X_test)}")
    
    # بناء النماذج
    results = build_models(X_train, X_test, y_train, y_test)
    
    # عرض النتائج
    print_results(results)
    
    # تصور النتائج
    visualize_results(results, y_test)
    
    # تحليل أهمية الميزات
    feature_importance(results, feature_cols)
    
    print("\n✅ اكتمل التدريب بنجاح!")
    
    # حفظ أفضل نموذج
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    import pickle
    with open(r'c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI\best_delay_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\n💾 تم حفظ أفضل نموذج ({best_model_name}) في 'best_delay_model.pkl'")

if __name__ == "__main__":
    main()
