import tempfile
from app_modules.nlp_radio_processing.N06_model_merging import transcribe_audio, analyze_radio_message
import streamlit as st
import sys
import os

# Add the NLP module path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
    __file__), '../../NLP_radio_processing/NLP_utils')))


def render_radio_analysis_view():
    """
    Streamlit component for uploading an MP3, transcribing it, and analyzing the radio message.
    """

    st.markdown("---")
    st.header("Analyze a Team Radio Audio File")
    st.write("Upload a team radio MP3 file to transcribe and analyze its content using the NLP pipeline.")

    uploaded_file = st.file_uploader("Upload MP3 file", type=["mp3"])
    if uploaded_file is not None:
        # Guardar archivo temporalmente en una carpeta específica
        os.makedirs("outputs/temp", exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir="outputs/temp") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name
        st.audio(temp_audio_path)

        with st.spinner("Transcribing audio..."):
            try:
                transcribed_text = transcribe_audio(temp_audio_path)
                st.success("Transcription complete!")
                st.markdown("**Transcribed Text:**")
                st.code(transcribed_text)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                return

        with st.spinner("Analyzing radio message..."):
            try:
                json_filename = analyze_radio_message(transcribed_text)
                st.success("Analysis complete!")
                st.markdown("**Analysis JSON file:**")
                st.code(json_filename)
                # Optionally, display the analysis content
                import json
                with open(json_filename, "r", encoding="utf-8") as f:
                    analysis = json.load(f)
                st.json(analysis)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass
