
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__)

# 1. تحميل وتجهيز البيانات والنموذج مباشرة من ملف البيانات النظيف
EXCEL_PATH = 'hail_cafes_dataset_clean.xlsx'

if not os.path.exists(EXCEL_PATH):
    if os.path.exists('dataset_hail_cafes_training_ai_1000.xlsx'):
        EXCEL_PATH = 'dataset_hail_cafes_training_ai_1000.xlsx'
    elif os.path.exists('dataset_hail_cafes_ready_for_ai.xlsx'):
        EXCEL_PATH = 'dataset_hail_cafes_ready_for_ai.xlsx'

df = pd.read_excel(EXCEL_PATH)
df.columns = df.columns.astype(str).str.strip()

feature_cols = [
    'النشاط', 'الموقع', 'رأس المال', 'الحد الأدنى التشغيلي', 'نسبة الإنفاق على المظهر', 
    'الأيام منذ آخر تجديد', 'الربط بالهوية الحائلية', 'خبرة الفريق',
    'مناسب للمناطق السياحية', 'يجذب السياح', 'النشاط موسمي', 'الموقع يخدم السياحة'
]

X = df[feature_cols]
y = df['حالة المشروع']

categorical_cols = [
    'النشاط', 'الموقع', 'الربط بالهوية الحائلية', 'خبرة الفريق', 
    'مناسب للمناطق السياحية', 'يجذب السياح', 'النشاط موسمي', 'الموقع يخدم السياحة'
]

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

# تدريب النموذج داخل الـ Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=6, random_state=42))
])

model_pipeline.fit(X, y)
print("✅ تم تدريب نموذج DecisionTreeClassifier بنجاح على بيانات حائل الميدانية.")


# 2. المسارات (Routes)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/pulse')
def pulse_page():
    return render_template('pulse.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}

    project_name = data.get('project_name', 'مشروع حائل')
    activity = data.get('activity', 'مقهى / كوفي')
    location_type = data.get('location', 'شارع تجاري حيوي')
    location_text = data.get('location_text', 'وسط المدينة، حائل')
    
    capital = float(data.get('capital', 200000))
    min_capital = float(data.get('min_capital', 150000))
    spending_pct = float(data.get('spending_pct', data.get('spending', 25)))
    days_since_renewal = int(data.get('days_since_renewal', 90))
    
    identity_link = data.get('identity_link', 'قوي جداً')
    team_experience = data.get('team_experience', 'عالية')
    tourism_suitable = data.get('tourism_suitable', 'نعم')
    tourism_attract = data.get('tourism_attract', 'نعم')
    is_seasonal = data.get('is_seasonal', 'لا')
    location_tourism = data.get('location_tourism', 'نعم')
    
    latitude = float(data.get('latitude', 27.5219))
    longitude = float(data.get('longitude', 41.6961))

    # تجهيز المدخلات للنموذج بنفس ترتيب أعمدة التدريب بالضبط (بالأسماء العربية)
    input_df = pd.DataFrame([{
        'النشاط': activity,
        'الموقع': location_type,
        'رأس المال': capital,
        'الحد الأدنى التشغيلي': min_capital,
        'نسبة الإنفاق على المظهر': spending_pct,
        'الأيام منذ آخر تجديد': days_since_renewal,
        'الربط بالهوية الحائلية': identity_link,
        'خبرة الفريق': team_experience,
        'مناسب للمناطق السياحية': tourism_suitable,
        'يجذب السياح': tourism_attract,
        'النشاط موسمي': is_seasonal,
        'الموقع يخدم السياحة': location_tourism
    }])

    # التنبؤ بالحالة بواسطة نموذج الذكاء الاصطناعي
    predicted_status = model_pipeline.predict(input_df)[0]

    # حساب درجة النبض المباشرة من احتمالية نجاح شجرة القرار
    probabilities = model_pipeline.predict_proba(input_df)[0]
    pulse_score = round(max(probabilities) * 10, 1)

    # التوصيات تعتمد على النتيجة التي أخرجها النموذج
    recommendations = []
    if str(predicted_status) != "مستقر / واعد":
        recommendations.append("أظهر تحليل نموذج الذكاء الاصطناعي وجود نقاط ضغط تشغيلية، يُنصح بمراجعة رأس المال التشغيلي ونسبة الإنفاق.")
    else:
        recommendations.append("مؤشرات المشروع تتطابق تماماً مع نمط المشاريع الناجحة والمستقرة في حائل بناءً على بيانات التدريب.")

    return jsonify({
        "project_name": project_name,
        "activity": activity,
        "location": location_text,
        "location_type": location_type,
        "status": str(predicted_status),
        "pulse_score": pulse_score,
        "recommendations": recommendations,
        "latitude": latitude,
        "longitude": longitude
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)



