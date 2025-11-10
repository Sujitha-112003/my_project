import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

from auth import login_page, register_page, logout



# Handle Authentication
query_params = st.experimental_get_query_params()
current_page = query_params.get("page", ["login"])[0]
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if not st.session_state.logged_in:
    if current_page == "register":
        register_page()
    else:
        login_page()
    st.stop()

# Logout button
st.sidebar.button("Logout", on_click=logout)

# Set background image function
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background for the main page
set_background("https://img.freepik.com/free-photo/veggie-quinoa-bowl-cooking-recipe_53876-110662.jpg?semt=ais_hybrid")

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="AUriIUOQuEbHt8npqPyt"
)

# Define recommendations based on detected class
RECOMMENDATIONS = {
    "Apple Scab": "🟢 **Apple Scab Treatment & Prevention:**\n- Apply fungicides like Mancozeb or Captan.\n- Prune trees to improve airflow.\n- Remove infected leaves and fruits.",
    "Black Rot": "🔴 **Black Rot Treatment & Prevention:**\n- Remove infected fruit and branches.\n- Use copper-based fungicides.\n- Ensure proper tree spacing for ventilation.",
    "Flyspeck": "🟡 **Flyspeck Treatment & Prevention:**\n- Wash apples thoroughly before consumption.\n- Improve orchard ventilation.\n- Use fungicides if severe.",
    "Fresh Apple": "✅ **No Disease Detected!**\n- Store apples in a cool, dry place.\n- Wash before eating to remove any surface bacteria."
}

st.title("🍏 Apple Disease Detection & Recommendation System")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an Apple Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load image using PIL
    image = Image.open(temp_path)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Perform inference using Roboflow API
    try:
        result = CLIENT.infer(temp_path, model_id="apple-disease-classification-ppc5r/1")

        # Show results
        if "predictions" in result and result["predictions"]:
            for pred in result["predictions"]:
                disease = pred["class"]
                confidence = pred["confidence"]

                # Display results
                st.subheader(f"🟢 Detected: **{disease}**")
                st.write(f"**Confidence Score:** {confidence:.2%}")

                # Show recommendations
                if disease in RECOMMENDATIONS:
                    st.markdown(RECOMMENDATIONS[disease])

        else:
            st.warning("⚠ No disease detected or low confidence.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

    # Clean up temp file
    os.remove(temp_path)
