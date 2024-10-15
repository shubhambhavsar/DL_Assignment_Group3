# Importing Libraries
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model  # For DNN and CNN models
from PIL import Image
import numpy as np
import lime.lime_tabular
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import img_to_array
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load models and preprocessors
try:
    preprocessor = joblib.load(r'is_validated/preprocessor_pipeline.pkl')
    dnn_model = load_model(r"parking_prediction_models/dnn_model.h5")  # Keras DNN model
    ada_model = joblib.load(r"parking_prediction_models/adaboost_model.pkl")  # AdaBoost model
    xgb_model = joblib.load(r"parking_prediction_models/xgboost_model.pkl")  # XGBoost model
    ensemble_model = joblib.load(r"parking_prediction_models/ensemble_model.pkl")  # Ensemble model combining Ada, XGBoost, RandomForest
    cnn_model = load_model(r"parking_prediction_models/CNN_Model.h5")  # CNN model for image classification
    # dnn_image_model = joblib.load(r"parking_prediction_models/dnn_image_model.pkl")  # DNN model for image classification
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Image Labels
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

# Load X_train from a pickle file
try:
    X_train = joblib.load(r"parking_prediction_data/X_train.pkl")
except Exception as e:
    st.error(f"❌ Error loading training data: {e}")
    st.stop()

# Preprocess X_train
try:
    X_train_encoded = preprocessor.transform(X_train)
except Exception as e:
    st.error(f"❌ Error in preprocessing training data: {e}")
    st.stop()

# Extract feature names from the preprocessor after transformation
feature_names = preprocessor.get_feature_names_out()

# Identify categorical features (if they were one-hot encoded in preprocessor)
categorical_features = [i for i, col in enumerate(feature_names) if 'onehot' in col]

# LIME Tabular explainer for the first tab
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_encoded,
    mode='classification',
    feature_names=feature_names,
    class_names=['No Parking', 'Parking'],
    categorical_features=categorical_features,
    discretize_continuous=True
)

# Prediction function for CSV-based models
def predict_parking(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        st.error(f"❌ Error in predicting parking: {e}")
        return None

# Output interpretation function for DNN parking prediction
def interpret_dnn_output(predictions):
    result = []
    for i, pred in enumerate(predictions):
        if pred >= 0.5:
            result.append(("Business has parking and parking is validated.", "success"))
        else:
            result.append(("Business does not have parking or parking is not validated.", "error"))
    return result

# LIME prediction function for interpretability
def predict_fn(x):
    try:
        proba_class_1 = dnn_model.predict(x)
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0, proba_class_1))
    except Exception as e:
        st.error(f"❌ Error in LIME prediction function: {e}")
        return None

# Image processing and prediction
def preprocess_image(img, model_type):
    try:
        img = img.resize((128, 128))
        img_array = img_to_array(img)  # Convert image to array
        img_array /= 255.0  # Normalize to [0, 1]

        if model_type == "DNN":
            img_array = img_array.flatten()  # Flatten for DNN input
            img_array = np.expand_dims(img_array, axis=0)
        elif model_type == "CNN":
            img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"❌ Error in preprocessing image: {e}")
        return None

def get_predictions(model, image, model_type):
    try:
        if model_type == "DNN":
            image = image.flatten()  # Flatten the image for DNN
            image = np.expand_dims(image, axis=0)
        return model.predict(image)
    except Exception as e:
        st.error(f"❌ Error in getting predictions: {e}")
        return None

# Streamlit app with two tabs
st.title("🚗 Business Parking and Image Labeling with Interpretability")

# Create tabs
# tab1, tab2 = st.tabs(["CSV Prediction (Parking)", "Image Prediction (Label)"])
# Navigation sidebar to switch between pages
page = st.sidebar.selectbox("Select a Page", ["CSV Prediction (Parking)", "Image Prediction (CNN)", "Image Prediction (DNN)"])

# ---- Page 1: CSV Prediction for Business Parking ----
if page == "CSV Prediction (Parking)":
    st.subheader("Predict Parking Availability and Validation from CSV Data")

    # File uploader for CSV data
    st.write("#### Upload your input file in CSV format:")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        # Read and display CSV file
        try:
            input_data = pd.read_csv(uploaded_file)
            st.write("### 📋 Preview of the uploaded data:")
            st.write(input_data.head())
        except Exception as e:
            st.error(f"❌ Error reading the CSV file: {e}")
            st.stop()

        # Preprocess the data
        try:
            input_data_encoded = preprocessor.transform(input_data)
        except Exception as e:
            st.error(f"❌ Error in preprocessing the input data: {e}")
            st.stop()

        # Ensure input shape matches the model
        if input_data_encoded.shape[1] != dnn_model.input_shape[1]:
            st.error(f"❌ Input shape mismatch. Expected {dnn_model.input_shape[1]} features, but got {input_data_encoded.shape[1]}.")
            st.stop()

        # Model selection
        model_choice = st.selectbox("Select the Model", ["DNN", "Adaboost", "XGBoost", "Ensemble"])

        # Prediction
        if st.button("🔍 Predict (CSV)", key="predict_csv"):
            predictions = None
            if model_choice == "DNN":
                predictions = predict_parking(dnn_model, input_data_encoded)
            elif model_choice == "Adaboost":
                predictions = predict_parking(ada_model, input_data_encoded)
            elif model_choice == "XGBoost":
                predictions = predict_parking(xgb_model, input_data_encoded)
            elif model_choice == "Ensemble":
                predictions = predict_parking(ensemble_model, input_data_encoded)

            if predictions is not None:
                # Binary conversion for DNN/XGBoost/Ensemble
                if model_choice in ["DNN", "XGBoost", "Ensemble"]:
                    predictions = (predictions > 0.5).astype(int)

                # Beautify predictions
                st.write("### 🔍 Predictions and Interpretations:")
                interpretation = interpret_dnn_output(predictions)
                for i, (text, status) in enumerate(interpretation):
                    if status == "success":
                        st.success(f"Sample {i + 1}: {text}")
                    else:
                        st.error(f"Sample {i + 1}: {text}")

                # Download results
                result_df = pd.DataFrame({"Prediction": predictions.flatten(), "Interpretation": [text for text, _ in interpretation]})
                csv = result_df.to_csv(index=False)
                st.download_button(label="📥 Download Prediction Results", data=csv, file_name="predictions.csv", mime="text/csv")

                # LIME interpretability for CSV predictions (for all samples)
                st.write("### 🧠 Lime Interpretability for All Samples")

                # Loop through each sample and show the LIME explanation graph
                for i in range(min(10, len(input_data_encoded))):  # Limit to first 10 samples for display purposes
                    st.write(f"#### Sample {i + 1} Explanation:")
                    exp = explainer.explain_instance(input_data_encoded[i], predict_fn, num_features=10)

                    # Display the LIME explanation as a graph
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)

# ---- Page 2: CNN Image Prediction ----
elif page == "Image Prediction (CNN)":
    
    st.subheader("Predict Label of an Image using CNN or DNN")

    # File uploader for image data
    st.write("#### Upload an image file:")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Load and display the image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Model selection
        model_choice_i = "CNN" #st.selectbox("Select the Model", ["CNN", "DNN"])
        st.write(f"Selected model: {model_choice_i}")

        # Preprocess the image based on the selected model
        preprocessed_image = preprocess_image(image, model_choice_i)

        # Prediction
        if st.button("🔍 Predict"):
            predictions = get_predictions(cnn_model, preprocessed_image, model_choice_i)
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]

            # Display predicted label
            st.write(f"### 🎯 Predicted Label: **{predicted_label}**")

            # Generate LIME explanation
            explainer = lime_image.LimeImageExplainer()

            # LIME explanation function
            def lime_predict(images):
                # Preprocess the images for predictions
                processed_images = []
                for img in images:
                    img_resized = Image.fromarray((img * 255).astype(np.uint8)).resize((128, 128))  # Resize to match input size
                    processed_img = preprocess_image(img_resized, model_choice_i)
                    processed_images.append(processed_img)
                processed_images = np.vstack(processed_images)  # Stack the images for batch prediction
                
                return cnn_model.predict(processed_images)

            # Use LIME to explain the instance
            explanation = explainer.explain_instance(
                np.array(image.resize((128, 128))) / 255.0,  # Resize and normalize
                lime_predict,  # Pass the LIME prediction function
                top_labels=5,
                hide_color=0,
                num_samples=1000
            )

            # Get the image and mask for the predicted class
            temp, mask = explanation.get_image_and_mask(
                predicted_class,
                positive_only=True,
                num_features=5,
                hide_rest=True
            )

            # Visualize the results
            st.write("### 🧠 LIME Explanation")
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            ax[0].imshow(image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # LIME explanation
            ax[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
            ax[1].set_title(f'LIME Explanation\nPredicted Class: {predicted_label}')
            ax[1].axis('off')

            # Heatmap
            ax[2].imshow(mask, cmap='hot', interpolation='nearest')
            ax[2].set_title('Heatmap')
            ax[2].axis('off')

            # Display the LIME explanations as a plot
            st.pyplot(fig)

            # Show the top 5 features contributing to the prediction
            st.write("### 🔍 Top 5 Features Contributing to the Prediction:")
            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            sorted_features = sorted(dict_heatmap.items(), key=lambda x: x[1], reverse=True)

            # Create a grid layout with 2 columns
            columns = st.columns(2)

            # Iterate through the top 5 features and display them in the grid
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                # Split into columns for a grid layout
                col = columns[i % 2]  # Alternate between the two columns

                with col:
                    # Display the importance score and description
                    st.write(f"**Region {feature}: Importance {importance:.4f}**")

                    # Get the image and mask for the current feature
                    temp, mask = explanation.get_image_and_mask(ind, positive_only=True, num_features=feature, hide_rest=False)

                    # Plot the image with boundaries of the important feature
                    fig, ax = plt.subplots()
                    ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
                    ax.axis('off')  # Hide axis
                    st.pyplot(fig)

# ---- Page 3: DNN Image Prediction ----
elif page == "Image Prediction (DNN)":
    st.markdown("[Go to the DNN Image Prediction App](https://dlassignmentgroup3-dnn.streamlit.app/)")

    # # File uploader for image data
    # st.write("#### Upload an image file:")
    # uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    # if uploaded_image is not None:
    #     # Load and display the image
    #     image = Image.open(uploaded_image)
    #     st.image(image, caption="Uploaded Image", use_column_width=True)

    #     # Model selection
    #     model_choice_i = "DNN"  # st.selectbox("Select the Model", ["CNN", "DNN"])
    #     st.write(f"Selected model: {model_choice_i}")

    #     # Preprocess the image based on the selected model
    #     preprocessed_image = preprocess_image(image, model_choice_i)

    #     # Prediction
    #     if st.button("🔍 Predict"):
    #         st.write(f"Model is taking too long to generate the output, please use another model.")

            # try:
            #     predictions = get_predictions(dnn_image_model, preprocessed_image, model_choice_i)
            #     predicted_class = np.argmax(predictions[0])
            #     predicted_label = class_labels[predicted_class]

            #     # Display predicted label
            #     st.write(f"### 🎯 Predicted Label: **{predicted_label}**")

            #     # Generate LIME explanation
            #     explainer = lime_image.LimeImageExplainer()

            #     # LIME explanation function
            #     def lime_predict(images):
            #         # Preprocess the images for predictions
            #         processed_images = []
            #         for img in images:
            #             img_resized = Image.fromarray((img * 255).astype(np.uint8)).resize((128, 128))  # Resize to match input size
            #             processed_img = preprocess_image(img_resized, model_choice_i)
            #             processed_images.append(processed_img)
            #         processed_images = np.vstack(processed_images)  # Stack the images for batch prediction
            #         return dnn_image_model.predict(processed_images)

            #     # Use LIME to explain the instance
            #     explanation = explainer.explain_instance(
            #         np.array(image.resize((128, 128))) / 255.0,  # Resize and normalize
            #         lime_predict,  # Pass the LIME prediction function
            #         top_labels=5,
            #         hide_color=0,
            #         num_samples=1000
            #     )

            #     # Get the image and mask for the predicted class
            #     temp, mask = explanation.get_image_and_mask(
            #         predicted_class,
            #         positive_only=True,
            #         num_features=5,
            #         hide_rest=True
            #     )

            #     # Visualize the results
            #     st.write("### 🧠 LIME Explanation")
            #     fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            #     # Original image
            #     ax[0].imshow(image)
            #     ax[0].set_title('Original Image')
            #     ax[0].axis('off')

            #     # LIME explanation
            #     ax[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
            #     ax[1].set_title(f'LIME Explanation\nPredicted Class: {predicted_label}')
            #     ax[1].axis('off')

            #     # Heatmap
            #     ax[2].imshow(mask, cmap='hot', interpolation='nearest')
            #     ax[2].set_title('Heatmap')
            #     ax[2].axis('off')

            #     # Display the LIME explanations as a plot
            #     st.pyplot(fig)

            #     # Show the top 5 features contributing to the prediction
            #     st.write("### 🔍 Top 5 Features Contributing to the Prediction:")
            #     ind = explanation.top_labels[0]
            #     dict_heatmap = dict(explanation.local_exp[ind])
            #     sorted_features = sorted(dict_heatmap.items(), key=lambda x: x[1], reverse=True)

            #     # Create a grid layout with 2 columns
            #     columns = st.columns(2)

            #     # Iterate through the top 5 features and display them in the grid
            #     for i, (feature, importance) in enumerate(sorted_features[:5]):
            #         # Split into columns for a grid layout
            #         col = columns[i % 2]  # Alternate between the two columns

            #         with col:
            #             # Display the importance score and description
            #             st.write(f"**Region {feature}: Importance {importance:.4f}**")

            #             # Get the image and mask for the current feature
            #             temp, mask = explanation.get_image_and_mask(ind, positive_only=True, num_features=feature, hide_rest=False)

            #             # Plot the image with boundaries of the important feature
            #             fig, ax = plt.subplots()
            #             ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
            #             ax.axis('off')  # Hide axis
            #             st.pyplot(fig)

            # except Exception as e:
            #     # Resetting the app if an error occurs
            #     st.error(f"An error occurred: {e}. Resetting the app...")
            #     st.session_state.reset = True
            
st.markdown("---")
st.write("Deep Learning Case Study 1 - Group 3")