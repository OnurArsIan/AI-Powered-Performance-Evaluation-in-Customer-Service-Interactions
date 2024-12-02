# AI-Powered Performance Evaluation in Customer Service Interactions

This project analyzes customer service interaction audio recordings using machine learning to assess the performance and overall quality of the interaction. The system evaluates the emotional state conveyed in the conversation and provides a star rating (1-5) based on the analysis.

## Features
- **Audio Analysis**: Extracts features such as spectral centroid, MFCCs, and more from audio recordings using `librosa`.
- **Emotion Classification**: Predicts emotions like happiness, anger, and sadness using a trained XGBoost model.
- **Performance Rating**: Calculates a star rating based on the predominant emotions in the interaction.
- **Streamlit Interface**: A user-friendly web interface for uploading audio files and viewing results.
- **Real-Time Analysis**: Processes audio in segments and provides detailed results for each segment.

## How It Works
1. **Feature Extraction**: Extracts key audio features using `librosa`.
2. **Emotion Prediction**: Uses a trained XGBoost model to classify the emotions in each segment.
3. **Rating Calculation**: Combines emotion predictions to generate an overall star rating.
4. **Visualization**: Displays emotion percentages and segment-wise predictions in an interactive dashboard.

## Dataset
This project utilizes the **Turkish Emotion-Voice Database (TurEV-DB)**. The dataset includes sound files and metadata for emotion recognition tasks. It is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### Citation
If you use this project or the dataset, please cite TurEV-DB as follows:

#### BibTeX
```bibtex
@inproceedings{canpolat2020turkish,
  title={Turkish Emotion Voice Database (TurEV-DB)},
  author={Canpolat, Salih Firat and Ormano{\u{g}}lu, Zuhal and Zeyrek, Deniz},
  booktitle={Proceedings of the 1st Joint Workshop on Spoken Language
  Technologies for Under-resourced languages (SLTU) and Collaboration
  and Computing for Under-Resourced Languages (CCURL)},
  pages={368--375},
  year={2020}
}
```

#### APA
Canpolat, S. F., Ormano\u{g}lu, Z., & Zeyrek, D. (2020, May). Turkish Emotion Voice Database (TurEV-DB). In Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL) (pp. 368-375).

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit application**:
   ```bash
   streamlit run uygulama3.py
   ```

## Usage
1. **Upload a customer service interaction audio file** (preferably 4 minutes long).
2. **View the analysis**, including:
   - Emotion percentages.
   - Segment-wise emotion predictions.
   - Overall star rating.
3. **Play the uploaded audio file** within the interface.

## File Structure
- **`import os.py`**: Script for creating datasets and feature extraction.
- **`uygulama3.py`**: Streamlit application for real-time analysis.
- **`xgb_model.pkl`**, **`scaler.pkl`**, **`label_encoder.pkl`**: Pre-trained model and necessary encoders/scalers.

## Future Work
- **Extend the model for multilingual emotion detection**.
- **Enhance the Streamlit interface** with more visualization options.
- **Develop an API** for integration with mobile applications.

## Acknowledgments
This project uses the **`librosa`** library for audio processing, **`XGBoost`** for machine learning, and the **Turkish Emotion-Voice Database (TurEV-DB)** for training and evaluation. Special thanks to the open-source community and TurEV-DB contributors for providing resources that made this project possible.

## License
This project uses the **Turkish Emotion-Voice Database (TurEV-DB)**, which is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

