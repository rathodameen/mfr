import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, AutoImageProcessor

# Cached model and processor to save memory
@st.cache_resource
def load_model_and_processor():
    model_name = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)
    return model.eval(), processor

model, processor = load_model_and_processor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def convert_to_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def apply_attention_visualization(image, model, prediction_idx):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, output_attentions=True)

    attentions = outputs.attentions
    last_attention = attentions[-1].squeeze(0).mean(dim=0)
    attention_resized = cv2.resize(last_attention.mean(dim=0).cpu().numpy(), (image.size[0], image.size[1]))

    attention_min = attention_resized.min()
    attention_max = attention_resized.max()
    normalized_attention = (attention_resized - attention_min) / (attention_max - attention_min + 1e-8)

    image_np = np.array(image.convert("RGB"), dtype=np.uint8).copy()

    if prediction_idx == 1:
        threshold = np.percentile(normalized_attention, 98)
        mask = np.uint8(normalized_attention >= threshold)
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        red_overlay = np.zeros_like(image_np, dtype=np.uint8)
        red_overlay[:, :, 0] = 255
        masked_image = cv2.bitwise_and(red_overlay, red_overlay, mask=dilated_mask)
        emphasized_image = cv2.addWeighted(image_np, 0.7, masked_image, 0.3, 0)
        return emphasized_image
    else:
        heatmap = np.uint8(255 * normalized_attention)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
        overlay = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)
        return blended


def predict(image):
    try:
        image_rgb = image.convert("RGB")
        inputs = processor(images=image_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = torch.argmax(outputs.logits).item()

        return predicted_class_idx
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def main():
    st.title("Pneumonia Detection from Chest X-ray")
    st.write("Upload a chest X-ray image to detect pneumonia.")

    uploaded_image = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image (Converted to RGB)", use_column_width=True)

        if st.button("Detect Pneumonia"):
            prediction_idx = predict(image)
            if prediction_idx is not None:
                if prediction_idx == 1:
                    st.error("Pneumonia Detected")
                    attention_image = apply_attention_visualization(image, model, prediction_idx)
                    st.image(attention_image, caption="Attention (Focused Red Dots - Pneumonia)", use_column_width=True)
                else:
                    st.success("No Pneumonia Detected")
                    attention_image = apply_attention_visualization(image, model, prediction_idx)
                    st.image(attention_image, caption="Attention (Subtle Green Overlay - No Pneumonia)", use_column_width=True)


if __name__ == "__main__":
    main()
