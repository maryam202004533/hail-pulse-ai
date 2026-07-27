

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
import joblib
import os

EXCEL_FILE = "hail_cafes_dataset_clean.xlsx"
if not os.path.exists(EXCEL_FILE):
    EXCEL_FILE = "dataset_hail_cafes_training_ai_1000.xlsx"

print(f"📥 جاري قراءة الملف: {EXCEL_FILE}")
df = pd.read_excel(EXCEL_FILE)

# إزالة أي مسافات زائدة من أسماء الأعمدة
df.columns = df.columns.astype(str).str.strip()

# قاموس مطابقة أسماء الأعمدة الفعلية في ملفك إلى الأسماء البرمجية المطلوبة للنموذج
column_mapping = {
    'النشاط': 'نوع_النشاط',
    'الموقع': 'الموقع_الجغرافي',
    'رأس المال': 'رأس_المال_ريال',
    'الحد الأدنى التشغيلي': 'الحد_الأدنى_التشغيلي_ريال',
    'نسبة الإنفاق على المظهر': 'نسبة_الإنفاق_على_المظهر_pct',
    'الأيام منذ آخر تجديد': 'الأيام_منذ_آخر_تجديد',
    'درجة الهوية': 'مستوى_الربط_بالهوية_10',
    'درجة الخبرة': 'خبرة_الفريق_التشغيلي_10',
    'حالة المشروع': 'حالة_المشروع_التشغيلية'
}

# إعادة تسمية الأعمدة بناءً على القاموس
df = df.rename(columns=column_mapping)

print("📋 الأعمدة بعد المطابقة:", df.columns.tolist())

# قائمة الأعمدة المطلوبة للتدريب
feature_categorical = ["نوع_النشاط", "الموقع_الجغرافي"]
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
target_col = "حالة_المشروع_التشغيلية"

# التحقق من وجود الأعمدة بالكامل بعد التعديل
missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise KeyError(f"⚠️ الأعمدة التالية غير موجودة بعد المطابقة: {missing_cols}")

# تنظيف البيانات المفقودة إن وجدت
df = df.dropna(subset=feature_cols + [target_col])

# تجهيز المحول للمتغيرات النصية (OrdinalEncoder)
encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df[feature_categorical] = encoder.fit_transform(df[feature_categorical])

# تحديد المدخلات والمخرجات
X = df[feature_cols]
y = df[target_col]

# تدريب نموذج أشجار القرار
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X, y)

# حفظ النموذج والمحول
joblib.dump(model, "pulse_decision_tree.joblib")
joblib.dump(encoder, "encoder.joblib")

print("\n✅ تم تدريب النموذج وحفظ الملفات (pulse_decision_tree.joblib & encoder.joblib) بنجاح!")

