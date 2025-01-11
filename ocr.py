import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
from deskew import determine_skew
from gtts import gTTS
from typing import Tuple, Union
import math
import cv2
import io
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import textwrap
import datetime
from firebase_admin import firestore

@st.cache_resource
def load_model():
    model_path = "Ho-ot/pegasus-fyp"
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

def summarize_text(text, max_length=512):
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, max_length=max_length, min_length=30)
    paragraphs = textwrap.wrap(text, width=max_length, break_long_words=False)
    summaries = []

    total_paragraphs = len(paragraphs)
    progress_bar = st.progress(0)

    for i, paragraph in enumerate(paragraphs):
        summary = summarizer(paragraph)
        summaries.append(summary[0]['summary_text'])
        progress_bar.progress((i + 1) / total_paragraphs)

    final_summary = ' '.join(summaries)
    return final_summary


# pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR/tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def pre_process_image(img):
    """This function will pre-process a image with: cv2 & deskew
    so it can be process by tesseract"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change color format from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #format image to gray scale
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 11) #to remove background
    return img


def detect(img):
    detected_text = pytesseract.image_to_string(img)
    return detected_text

def app():
    timestamp = datetime.datetime.now().isoformat()
    db = firestore.client()
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('OCR - Optical Character Recognition')
    st.subheader('Extract text from images')
    st.text('Note: OCR only works for English text at the moment.')

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'JPG'])

    if st.button("Convert"):
        if image_file is not None:
            img= Image.open(image_file)
            image = np.array(img)
            #image = cv2.imread('imgp')
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            angle = determine_skew(grayscale)
            rotated = rotate(image, angle, (0, 0, 0))
            cv2.imwrite('output.png', rotated)
            img = pre_process_image(rotated)

            st.subheader('Uploaded Image')
            st.image(img, width=450)

            with st.spinner('Extracting Text from Image...'):
                detected_text = pytesseract.image_to_string(img, lang='eng')
                st.subheader('Extracted text:')
                st.write(detected_text)

            st.subheader('Generated Audio')
            with st.spinner('Generating Audio...'):
                tts = gTTS(text=detected_text, lang='en')
                tts.save('output.mp3')
            st.audio('output.mp3', format='audio/mp3')


            st.subheader("Text Summary")
            with st.spinner('Generating Summary...'):
                summary = summarize_text(detected_text)
                    #st.write(summary[0]['summary_text'])
                st.write(summary)

            with st.spinner('Saving Result...'):
                try:
                    if 'db' not in st.session_state:
                        st.session_state.db = ''
                    st.session_state.db = db
                    info = db.collection('History').document(st.session_state.useremail).get()
                    
                    if info.exists:
                        info = info.to_dict()
                        if 'OCR' in info.keys():
                            pos = db.collection('History').document(st.session_state.useremail)
                            pos.update({u'OCR': firestore.ArrayUnion([u'{}'.format(detected_text)])})
                            pos.update({u'Summary': firestore.ArrayUnion([u'{}'.format(summary)])})
                            pos.update({u'Timestamp': firestore.ArrayUnion([u'{}'.format(timestamp)])})
                            st.write('Post uploaded!!')
                        else:
                            data = {"OCR": [detected_text], "Summary": [summary], "Timestamp": [timestamp], 'email': st.session_state.useremail}
                            db.collection('History').document(st.session_state.useremail).set(data)
                            st.success('Post uploaded!!')
                    else:
                        data = {"OCR": [detected_text], "Summary": [summary], "Timestamp": [timestamp], 'email': st.session_state.useremail}
                        db.collection('History').document(st.session_state.useremail).set(data)
                        st.success('Post uploaded!!')
                except:
                    if st.session_state.username == '' and st.session_state.useremail == '':
                        st.subheader('Please Login first...')

        else:
            st.subheader('Image not found! Please Upload an Image.')

   
    