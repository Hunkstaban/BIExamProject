# Business Intelligence Exam Project

## Project
### Movie Recommendation System
- Challenge: Helping users discover movies they are likely to enjoy based on other movies they like.

- Why?: Choice fatigue is often a factor when sitting down and being presented with the thousands of options available on streaming platforms. Personalized recommendations could both improve user satisfaction and could increase engagement and retention for streaming services.

- Solution: The system will be able to find movies based on other movies that a user chooses, showing its title, rating, and other important information.

- Impact: An interactive recommendation system helps users quickly find movies they will enjoy, making their movie nights more efficient and enjoyable. Streaming platforms and movie websites benefit from increased user engagement and satisfaction, while users save time and discover new content tailored to their preferences. This idea could also help highlight lesser-known films.

## Implementation
- The dataset used is 'The Movies Dataset' from Kaggle: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset. Due to the datasets large files it's important to download the csv files included in it yourself to run the notebooks. 
- The csv files used from the dataset is credits.csv, keywords.csv, and movies_metadata.csv.
- The initial cleaning and exploration appears in exploration-wrangling-engineering.ipynb, followed by exploration-continued.ipynb. Some of the notebooks should be able to run without the other csv files as the /data folder includes merged_dataset.csv, movies_dataset.csv, and movies_with_nlp.csv that has been created during the project.
- Running the streamlit have, make sure you're in the ./app folder and then run 'streamlit run app.py'

Packages needed:
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- ast
- sklearn
- joblib
- streamlit
- plotly
