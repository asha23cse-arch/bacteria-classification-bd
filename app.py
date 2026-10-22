# ==========================================
# Streamlit App: Bacterial Image Classifier
# ResNet50 features -> MI selection -> Scaler -> SVM + Logistic Regression (Combined)
# ==========================================
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="Bacteria Classifier", page_icon="🦠", layout="centered")

ARTIFACT = "bacteria_models.joblib"


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    art = joblib.load(ARTIFACT)  # selector, scaler, svm, logreg, ensemble, class_names
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    extractor = torch.nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval()
    return art, extractor


tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(img, art, extractor, model_key):
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        f = torch.flatten(extractor(x), 1).numpy()
    f = art["scaler"].transform(art["selector"].transform(f))
    return art[model_key].predict_proba(f)[0]


st.title("🦠 Bacterial Image Classification")
st.caption("ResNet50 features + Mutual Information + SVM & Logistic Regression")

try:
    art, extractor = load_all()
except FileNotFoundError:
    st.error(f"`{ARTIFACT}` file পাওয়া যায়নি। Kaggle notebook চালিয়ে ফাইলটি এই folder এ রাখুন।")
    st.stop()

names = art["class_names"]
choice = st.radio(
    "Model বাছাই করুন",
    ["Combined (SVM + Logistic Regression)", "SVM", "Logistic Regression"],
    horizontal=True,
)
key = {"Combined (SVM + Logistic Regression)": "ensemble", "SVM": "svm", "Logistic Regression": "logreg"}[choice]

if "accuracy" in art:
    with st.expander("Test accuracy (notebook থেকে)"):
        st.dataframe(pd.DataFrame(art["accuracy"]), hide_index=True, use_container_width=True)

file = st.file_uploader("ব্যাকটেরিয়ার ছবি আপলোড করুন", type=["png", "jpg", "jpeg", "tif", "tiff"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    with st.spinner("Predicting..."):
        probs = predict(img, art, extractor, key)
    top = np.argsort(probs)[::-1][:5]
    st.success(f"Prediction: **{names[top[0]]}**  ({probs[top[0]] * 100:.1f}% confidence)")
    st.bar_chart(pd.DataFrame({"Probability (%)": probs[top] * 100}, index=[names[i] for i in top]))
