# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:50:08 2025

@author: milos
"""

import streamlit as st         # Import Streamlit for building the web interface
import ollama                  # Import Ollama for interacting with the AI model
import tempfile                # Import tempfile for handling temporary files

# Set the title of the Streamlit app
st.title("Llama vision: Image Describer")
st.write("Upload an image and let the model describe it!")

# --- Image Upload Section ---
# Create a file uploader widget that accepts jpg, jpeg, or png images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If the user has uploaded a file, proceed
if uploaded_file is not None:
    # Show a preview of the uploaded image in the app
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save the uploaded image to a temporary file (model requires a file path) 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        # Write the contents of the uploaded file to the temp file
        tmp_file.write(uploaded_file.read())
        # Store the path to the temp image for later use
        temp_image_path = tmp_file.name

    # Create a button that, when clicked, will describe the image
    if st.button("Describe Image"):
        # Show a spinner while the AI is processing the image
        with st.spinner("Thinking..."):
            # Send the image to the Ollama API for description
            response = ollama.chat(
                model="llama3.2-vision:latest",  # Specify the vision-capable model
                messages=[
                    {
                        "role": "user",
                        "content": "Describe the image.",  # Prompt for the model
                        "images": [temp_image_path],      # Attach the image file
                    }
                ]
            )
            # Display the model's description in the app
            st.markdown("### Description:")
            st.markdown(response['message']['content'])
