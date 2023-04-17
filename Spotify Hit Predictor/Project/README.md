# READ ME

## Problem Statement
Artists, the  music industry is a highly competitive and constantly evolving market, and the success of a song can have a significant impact on your career and the industry as a whole. However, predicting the popularity of a song is a challenging task, and there is a need for an accurate and reliable method to do so.

This project aims to address this problem by building a machine learning model to predict the popularity of songs based on various musical features. The model was trained using a dataset of over 40,000 songs pulled from Spotify, and was evaluated using cross-validation and various performance metrics.

The successful development of this machine learning model will provide a valuable tool for music industry professionals to make informed decisions around song writing, music production, mixing, mastering, and everything in between. It will also contribute to our understanding of the underlying factors that drive song popularity and help identify patterns and trends in the music industry over time. Ultimately, this project has the potential to benefit the music industry by providing an understanding of what makes a song popular. 

## Data

### Main Dataset: The Spotify Hit Predictor Dataset from 1960 - 2019. Data obtained from Kaggle (Farooq Ansari, 2019). 

|Feature Name    | Description|
|----------------|-----------------------------------------------------------------------------------------------------------------|
| track          | The name of the track.|
| artist         | The name of the artist.|
| uri            | The resource identifier for the track.|
| danceability   | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.|
| energy| Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.|
| key| The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C?/D?, 2 = D, and so on. If no key was detected, the value is -1.|
| loudness| The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.|
| mode| Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.|
| speechiness| Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.|
| acousticness| A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.|
| instrumentalness| Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.|
| liveness| Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.|
| valence| A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
| tempo| The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
| duration_ms| The duration of the track in milliseconds.|
| time_signature| An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). |
| chorus_hit| The timestamp of the start of the third section of the track, which is the author's best estimate of when the chorus would start for the track. This feature was extracted from the data received by the API call for Audio Analysis of that particular track. |
| sections| The number of sections the particular track has. This feature was extracted from the data received by the API call for Audio Analysis of that particular track. |
| target | The target variable for the track. It can be either '0' or '1'. '1' implies that this song has featured in the weekly list (Issued by Billboards) of Hot-100 tracks in that decade at least once and is therefore a 'hit'. '0' implies that the track is a 'flop'. |

## Secondary Dataset: 
|Feature Name    | Description|
|----------------|-----------------------------------------------------------------------------------------------------------------|
| title          | The name of the track.                                                                          |
| artist         | The name of the artist.                                                                         |
| top genre      | The primary genre of the track.                                                                 |
| year           | The year the track was released.                                                                |
| bpm            | The tempo of the track in beats per minute (BPM).                                               |
| nrgy           | Energy is a measure from 0 to 100 and represents a perceptual measure of intensity and activity.|
| dnce           | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.|
| dB             | The overall loudness of a track in decibels (dB).                                               |
| live           | Detects the presence of an audience in the recording.                                           |
| val            | A measure from 0 to 100 describing the musical positiveness conveyed by a track.               |
| dur            | The duration of the track in milliseconds.                                                      |
| acous          | A confidence measure from 0 to 100 of whether the track is acoustic.                            |
| spch           | Speechiness detects the presence of spoken words in a track.                                    |
| pop            | The target variable for the track. It can be either '0' or '1'. '1' implies that this song has featured in the weekly list of Hot-100 tracks in that decade at least once and is therefore a 'hit'. '0' implies that the track is a 'flop'.|

## Methods

I combined the seprate datasets into one dataset. I also feature engineered a year column and saved the final dataset as a csv to develop machine learning models. 

## Models Trained 

- Logistic Regression
    - Simple Logistic Regression: 74.19% Accuracy
    - Gridsearched Hyperparameters: 74.17 Accuracy
        - penalty: l1, l2
        - C: 0.1, 1, 10
        - solver: liblinear, saga 
        
- KNN
    - Simple KNN: 74.7% Accuracy
        - n_neighbors = 5
        
- Decision Tree
    - Simple Decision Tree: 69.92% Accuracy
    - Simple Decision Tree + Standard Scaling: 70.03% Accuracy
    - Gridsearched Hyperparameters: 69.89% Accuracy and 69.9% Accuracy
        - max depth: None, 1,2,3,4,5,7
        - min samples split: 2, 3, 4, 5, 10
        - min samples leaf: 1,2,3,4,5 
        - maxfeatures: None, sqrt, log2
        
- AdaBoost
    - Ada Boost: 77.5% Accuracy
        - n estimators: 100
        - learning rate: 1 
        - algorithm: SAMME.R
        - DecisionTreeClassifier: max depth: 1
        
- Gradient Boost
    - Simple Gradient Boost: 79.34% Accuracy
    
- Support Vector Machine
    - Gridsearched Hyperparameters: 73.98% Accuracy
        - C: np.linspace(0.0001, 2, 10)
        - max_iter: 20000
        - n_splits: 5
        - shuffle: True
        
- Random Forest
    - Gridsearched Hyperparameters: 80.8% Accuracy
        - max depth: None, 1, 2, 3, 4, 5, 6, 7
        - n estimators: 100, 150, 200
    - Gridsearched Hyperparameters + Standard Scaling: 81.03% Accuracy
        - max depth: None, 1, 2, 3, 4, 5, 6, 7
        - n estimators: 100, 150, 200
        
## Best Performing Model

Random Forest with Gridsearched Hyperparameters & Standard Scaling: 81.03% Accuracy
        - max depth: None, 1, 2, 3, 4, 5, 6, 7
        - n estimators: 100, 150, 200
        
## Recommendations

- Incorporate a more comprehensive set of musical features: While the analysis revealed strong correlations between certain musical features and song popularity, it is likely that other important features were not included in our model, such as genre, lyrics, and melody. Future research should explore the use of additional features to further enhance the accuracy of the model. For instance, an NLP model that analyzes lyrics to predict song popularity would be an interesting next step in this analysis.
- Expand the dataset: Future research should focus on including a wider range of songs from different genres and time periods. This will increase the diversity of the data and help to ensure that the model is not biased towards certain types of music. Additionally, incorporating more data from independent and emerging artists could provide valuable insights into what factors contribute to the success of less-established musicians.
- Test the model in real-world scenarios: While the model shows promising results, it is essential to test the model in real-world scenarios to validate its effectiveness. Future research should explore how the model performs in predicting the popularity of new songs and how it can be used by independent artists to grow their audiences. Additionally, evaluating the model's performance across different regions and cultures could provide valuable insights into how music preferences vary across different demographics.
- Compare the performance of different models: Although the study compared the performance of several machine learning models, there are many other models that could be used for predicting song popularity. Future research should compare the performance of different models and identify the most effective approach. This could include exploring hybrid models that combine multiple machine learning techniques to improve prediction accuracy.

## Conslusions

I’ve successfully developed a machine learning model to predict song popularity using various musical features. Through data cleaning, feature engineering, and 
model selection, I achieved an accuracy of 81.3% using a Random Forest algorithm.

Analysis also revealed interesting insights into the relationship between certain musical features and song popularity. Specifically, instrumentalness, acousticness, danceability, and energy were strongly correlated with song popularity.

These findings can be valuable for artists and music producers in guiding their creative processes and increasing their chances of producing popular songs. Additionally, we recommend that future research explores the use of additional features, such as genre and lyrics, to further enhance the model's accuracy.

Overall, this machine learning model presents a promising approach to predicting song popularity and has the potential to be a useful tool for independent artists and the music industry as a whole.
