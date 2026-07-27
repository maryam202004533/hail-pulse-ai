from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from google import genai

app = Flask(__name__)

# إعداد عميل جيميناي الجديد (استبدل مفتاح الـ API بمفتاحك الخاص أو اتركه ليأخذه من متغيرات البيئة)
# يمكنك وضع مفتاحك هنا مباشرة بين العلامتين: api_key="AIzaSy..."
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY"))

# 1. تحميل النموذج والمحول
MODEL_PATH = "pulse_decision_tree.joblib"
ENCODER_PATH = "encoder.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("⚠️ ملفات الـ Model أو Encoder غير موجودة! يرجى تشغيل train.py أولاً.")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

print("✅ تم تحميل النموذج والمحول بنجاح!")


def parse_score(val, default=5):
    if isinstance(val, (int, float)):
        return float(val)
    mapping = {'قوي جداً': 10, 'متوسط': 6, 'ضعيف': 3, 'لا يوجد': 1, 'عالية': 10, 'متوسطة': 6, 'مبتدئة': 3, 'نعم': 10, 'لا': 1}
    return mapping.get(str(val).strip(), default)


def generate_ai_recommendations(project_name, activity, status, pulse_score, spending_pct, days_renewal, identity_score, tourist_suitable):
    """توليد توصيات استباقية ذكية باستخدام مكتبة جوجل الحديثة google.genai"""
    try:
        prompt = f"""
        أنت مستشار استثماري واقتصادي خبير في سوق منطقة حائل بالمملكة العربية السعودية.
        قم بتحليل بيانات المشروع التالي واكتب 3 إلى 4 توصيات استباقية واحترافية باللغة العربية لمساعدة صاحب المشروع على تحسين أدائه:
        - اسم المشروع: {project_name}
        - النشاط: {activity}
        - حالة المشروع المقدرة: {status}
        - درجة النبض: {pulse_score} من 10
        - نسبة الإنفاق على المظهر: {spending_pct}%
        - الأيام منذ آخر تجديد/فعالية: {days_renewal} يوم
        - مستوى الربط بالهوية الحائلية: {identity_score}/10
        - الملاءمة السياحية: {tourist_suitable}/10

        اجعل التوصيات مختصرة، عملية، ومباشرة في شكل نقاط تبدأ بعبارات تحفيزية أو إرشادية. لا تضع أي مقدمات أو خواتيم.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        text = response.text.strip()
        recs = [line.strip().lstrip("-*• 123456789.") for line in text.split("\n") if line.strip()]
        return recs if recs else ["تؤكد المؤشرات الحالية على ضرورة مراجعة الخطة التشغيلية وفق معايير السوق."]
    except Exception as e:
        print(f"⚠️ تنبيه أثناء توليد التوصيات بالذكاء الاصطناعي: {e}")
        # توصيات احتياطية في حال تعذر اتصال الـ API أو عدم ضبط المفتاح
        return [
            f"نسبة الإنفاق الحالية ({spending_pct}%) تتطلب إعادة تقييم موازنة التأسيس.",
            f"مراعاة دورة التجديد الدورية (مرت {days_renewal} يوم) لتفادي ركود النشاط.",
            "تعزيز الهوية المحلية والأنشطة السياحية المرتبطة بطبيعة منطقة حائل لجذب زوار أكثر."
        ]


@app.route('/')
@app.route('/pulse')
def pulse_page():
    return render_template('pulse.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    
    project_name = data.get('project_name', 'مشروع حائل')
    activity = data.get("activity") or data.get("النشاط") or "مقهى / كوفي"
    location = data.get("location") or data.get("الموقع") or "شارع تجاري حيوي"
    capital = float(data.get("capital") or data.get("رأس المال") or 200000)
    min_capital = float(data.get("min_capital") or data.get("الحد الأدنى التشغيلي") or 150000)
    spending_pct = float(data.get("spending_pct") or data.get("نسبة الإنفاق على المظهر") or 25)
    days_renewal = int(data.get("days_since_renewal") or data.get("الأيام منذ آخر تجديد") or 90)
    
    identity_score = parse_score(data.get("identity_link") or data.get("الربط بالهوية الحائلية") or 10, 10)
    team_score = parse_score(data.get("team_experience") or data.get("خبرة الفريق") or 8, 8)
    
    tourist_suitable = parse_score(data.get("tourist_suitable") or data.get("مناسب للمناطق السياحية") or 8, 8)
    tourist_attract = parse_score(data.get("tourist_attract") or data.get("يجذب السياح") or 8, 8)
    is_seasonal = parse_score(data.get("is_seasonal") or data.get("النشاط موسمي") or 5, 5)
    location_tourism = parse_score(data.get("location_tourism") or data.get("الموقع يخدم السياحة") or 8, 8)

    df_input = pd.DataFrame([{
        "نوع_النشاط": activity,
        "الموقع_الجغرافي": location,
        "رأس_المال_ريال": capital,
        "الحد_الأدنى_التشغيلي_ريال": min_capital,
        "نسبة_الإنفاق_على_المظهر_pct": spending_pct,
        "الأيام_منذ_آخر_تجديد": days_renewal,
        "مستوى_الربط_بالهوية_10": identity_score,
        "خبرة_الفريق_التشغيلي_10": team_score
    }])

    try:
        encoded_cats = encoder.transform(df_input[["نوع_النشاط", "الموقع_الجغرافي"]])
        df_input["نوع_النشاط"] = encoded_cats[:, 0]
        df_input["الموقع_الجغرافي"] = encoded_cats[:, 1]
    except Exception as e:
        print(f"⚠️ تنبيه أثناء الترميز: {e}")

    feature_cols = [
        "نوع_النشاط",
        "الموقع_الجغرافي",
        "رأس_المال_ريال",
        "الحد_الأدنى_التشغيلي_ريال",
        "نسبة_الإنفاق_على_المظهر_pct",
        "الأيام_منذ_آخر_تجديد",
        "مستوى_الربط_بالهوية_10",
        "خبرة_الفريق_التشغيلي_10"
    ]

    predicted_status = model.predict(df_input[feature_cols])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df_input[feature_cols])[0]
        pulse_score = round(max(probabilities) * 10, 1)
    else:
        pulse_score = 7.5

    capital_score = round(min((capital / max(min_capital, 1)) * 5, 10), 1)

    # توليد التوصيات عبر الذكاء الاصطناعي مع ضمان عدم حدوث صفحة بيضاء
    recommendations = generate_ai_recommendations(
        project_name, activity, str(predicted_status), pulse_score, spending_pct, days_renewal, identity_score, tourist_suitable
    )

    return jsonify({
        "project_name": project_name,
        "activity": activity,
        "location": location,
        "status": str(predicted_status),
        "pulse_score": pulse_score,
        "capital": capital,
        "min_capital": min_capital,
        "spending_pct": spending_pct,
        "days_since_renewal": days_renewal,
        "capital_sufficiency": capital_score,
        "identity_link": identity_score,
        "team_experience": team_score,
        "tourist_suitable": tourist_suitable,
        "tourist_attract": tourist_attract,
        "is_seasonal": is_seasonal,
        "location_tourism": location_tourism,
        "recommendations": recommendations
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)






