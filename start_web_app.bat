@echo off
REM تطبيق التنبؤ برحلات التأخير - نسخة الويب
REM Delay Prediction Web Application Launcher

cd /d "c:\Users\gerge\OneDrive\سطح المكتب\VSCODE\c++\AI"

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║   🚌 نظام التنبؤ برحلات التأخير - تطبيق الويب               ║
echo ║     Delay Prediction Web Application                          ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM التحقق من وجود النموذج
if not exist "best_delay_model.pkl" (
    echo ⚠️  تحذير: النموذج المدرب (best_delay_model.pkl) غير موجود!
    echo    يرجى تشغيل app.py أولاً لتدريب النموذج
    echo.
    pause
    exit /b 1
)

echo ✅ تم العثور على النموذج المدرب
echo ✅ جاري بدء التطبيق...
echo.

REM بدء التطبيق
C:\Users\gerge\AppData\Local\Programs\Python\Python314\python.exe web_app.py

pause
