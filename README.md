# ML-Project
Traffic Prediction App
Overview
The Traffic Prediction App is a Streamlit-based web application designed to predict and analyze traffic congestion in different zones of Berlin. The app uses a pre-trained machine learning model to provide traffic volume predictions based on user inputs such as date, time, and zone. Additionally, the app offers visualizations and reports to understand traffic patterns and model performance.

Features
Home: Welcome page with an overview of the app.
Predict: Allows users to predict traffic volume based on selected date, time, and zone.
Report: Displays the model's performance metrics and a comparison between actual and predicted traffic volumes.
Dashboard: Provides various visualizations to analyze traffic patterns over time, by zone, and other factors.
Installation
Clone the repository:

git clone https://github.com/yourusername/traffic-prediction-app.git
cd traffic-prediction-app
Install the required packages:

pip install -r requirements.txt
Ensure you have the model and data files in the correct directories:

./model/newtraffic_model.pkl - Pre-trained machine learning model.
./csv/traffic.csv - Traffic data CSV file.
Make sure the up.sh script is executable:

chmod +x up.sh
Usage
Start the Streamlit app:

streamlit run app.py
Open your browser and navigate to http://localhost:8501 to access the app.

App Structure
Home
Displays a welcome message and an image related to traffic congestion.
Predict
Users can select a zone, date, and time to get a traffic volume prediction.
The predicted number of vehicles is displayed along with a relevant image.
Report
Shows model performance metrics such as Mean Squared Error (MSE) and R-squared (R²).
Displays a scatter plot comparing actual vs. predicted traffic volumes.
Dashboard
Provides multiple visualizations to analyze traffic data:
Traffic volume over time
Traffic volume by zone
Average traffic volume by hour of the day
Average traffic volume by day of the week
Data Preparation
The data is loaded from ./csv/traffic.csv.
The DateTime column is converted to datetime format.
Additional columns for Year, Month, Date, Hour, Day, DayOfWeek, and HourOfDay are created for easier analysis and prediction.
Zone Mapping
A dictionary is used to map numeric zone values to their corresponding names:

zone_mapping = {
    1: 'North Berlin',
    2: 'South Berlin',
    3: 'East Berlin',
    4: 'West Berlin'
}
Error Handling
The app checks if the model is loaded correctly and displays an error message if there's an issue.
Validates the time input format and shows an error message for invalid formats.
Contributions
Feel free to fork the repository and make contributions. Pull requests are welcome!

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Thanks to the contributors of the various open-source libraries used in this project.
Image credits: AI-in-transportation.webp, header-mobility.jpg, 3.jpeg, 4.jpg.
This README provides an overview of the Traffic Prediction App, its features, installation instructions, and usage guidelines. Adjust the content as needed to match your project's specific details and structure.
