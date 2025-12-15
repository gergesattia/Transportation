#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
دليل الإعداد الأولي
Initial Setup Guide

هذا السكريبت يساعد في التحقق من أن كل شيء معد بشكل صحيح
"""

import os
import sys

def check_python():
    """التحقق من إصدار Python"""
    print("✓ Python Version:", sys.version)
    if sys.version_info >= (3, 7):
        print("  ✅ Python 3.7 أو أحدث (مقبول)")
        return True
    else:
        print("  ❌ Python قديم جداً")
        return False

def check_packages():
    """التحقق من المكتبات المثبتة"""
    print("\n✓ التحقق من المكتبات:")
    
    required = {
        'pandas': 'معالجة البيانات',
        'numpy': 'الحسابات العددية',
        'sklearn': 'التعلم الآلي',
        'matplotlib': 'الرسوم البيانية',
        'seaborn': 'الرسوم المتقدمة',
        'flask': 'تطبيق الويب',
    }
    
    missing = []
    for package, description in required.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ❌ {package}: {description} - غير مثبت")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_files():
    """التحقق من الملفات المهمة"""
    print("\n✓ التحقق من الملفات:")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    files = {
        'app.py': 'ملف تدريب النموذج',
        'web_app.py': 'تطبيق الويب',
        'predict_delay.py': 'سكريبت التنبؤ',
        'dataset_with_features.csv': 'ملف البيانات',
        'requirements.txt': 'المكتبات المطلوبة',
        'templates': 'مجلد صفحات HTML',
        'README.md': 'التوثيق',
    }
    
    missing = []
    for file, description in files.items():
        full_path = os.path.join(base_path, file)
        if os.path.exists(full_path):
            print(f"  ✅ {file}: {description}")
        else:
            print(f"  ❌ {file}: {description} - غير موجود")
            missing.append(file)
    
    return len(missing) == 0, missing

def check_model():
    """التحقق من وجود النموذج المدرب"""
    print("\n✓ التحقق من النموذج:")
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'best_delay_model.pkl')
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"  ✅ النموذج موجود ({size:.2f} MB)")
        return True
    else:
        print(f"  ⚠️  النموذج غير موجود")
        print(f"     تشغيل 'python app.py' سيقوم بتدريب النموذج")
        return False

def print_instructions():
    """طباعة التعليمات"""
    print("\n" + "="*70)
    print("📋 التعليمات:")
    print("="*70)
    
    print("\n1️⃣  تدريب النموذج (أولى مرة فقط):")
    print("   python app.py")
    
    print("\n2️⃣  بدء تطبيق الويب:")
    print("   python web_app.py")
    
    print("\n3️⃣  فتح المتصفح:")
    print("   http://127.0.0.1:5000")
    
    print("\n4️⃣  التنبؤ البسيط (بدون ويب):")
    print("   python predict_delay.py")
    
    print("\n5️⃣  تحليل السيناريوهات:")
    print("   python scenario_analysis.py")
    
    print("\n" + "="*70)

def main():
    print("\n" + "="*70)
    print("🚌 نظام التنبؤ برحلات التأخير")
    print("   Delay Prediction System - Setup Check")
    print("="*70 + "\n")
    
    # التحقق من Python
    python_ok = check_python()
    
    # التحقق من المكتبات
    packages_ok, missing_packages = check_packages()
    
    # التحقق من الملفات
    files_ok, missing_files = check_files()
    
    # التحقق من النموذج
    model_ok = check_model()
    
    # طباعة النتائج
    print("\n" + "="*70)
    print("📊 ملخص الفحص:")
    print("="*70)
    
    status = []
    status.append(("Python", "✅" if python_ok else "❌"))
    status.append(("المكتبات", "✅" if packages_ok else "❌"))
    status.append(("الملفات", "✅" if files_ok else "❌"))
    status.append(("النموذج", "✅" if model_ok else "⚠️ (اختياري)"))
    
    for name, result in status:
        print(f"  {result} {name}")
    
    # طباعة التعليمات
    print_instructions()
    
    # التوصيات
    if not packages_ok:
        print("⚠️  التنبيهات:")
        print("   المكتبات المفقودة:")
        for pkg in missing_packages:
            print(f"     - {pkg}")
        print("\n   قم بتشغيل:")
        print("   pip install -r requirements.txt")
    
    if not model_ok:
        print("\n⚠️  تنبيه:")
        print("   النموذج المدرب غير موجود")
        print("   قم بتشغيل: python app.py")
        print("   هذا قد يستغرق بضع دقائق")
    
    print("\n✅ انتهى الفحص!")
    print("="*70 + "\n")
    
    return python_ok and packages_ok and files_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
