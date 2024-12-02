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


![photo1](https://github.com/user-attachments/assets/ae9cb252-869a-4947-bb7b-e3e4732d71d3)
![photo2](https://github.com/user-attachments/assets/0323fff5-3385-4ee4-aded-a11cb7e92b41)
