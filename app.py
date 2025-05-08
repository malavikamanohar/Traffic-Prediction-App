import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import subprocess  # Import subprocess module

# To Set page configuration
st.set_page_config(
    page_title="Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Load the model
try:
    model = joblib.load('./model/newtraffic_model.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# To Load the data
@st.cache_data
def load_data():
    ##url = "https://drive.google.com/file/d/1WA4SUMm30vSNkENpuiOyTESt64yPdH0_/view?usp=sharing"
    ##path = "https://drive.google.com/uc?export=download&id=" + url.split("/")[-2]
    path = './csv/traffic.csv'
    data = pd.read_csv(path)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['Year'] = data['DateTime'].dt.year
    data['Month'] = data['DateTime'].dt.month
    data['Date'] = data['DateTime'].dt.day
    data['Hour'] = data['DateTime'].dt.hour
    data['Day'] = data['DateTime'].dt.strftime('%A')
    data['DayOfWeek'] = data['DateTime'].dt.dayofweek
    data['HourOfDay'] = data['DateTime'].dt.hour
    return data

Traffic_prediction = load_data()

# To Zone mapping dictionary
zone_mapping = {
    1: 'North Berlin',
    2: 'South Berlin',
    3: 'East Berlin',
    4: 'West Berlin'
}

# Map numeric zone values to their corresponding names
Traffic_prediction['ZoneName'] = Traffic_prediction['Zone'].map(zone_mapping)


# Main function
def main():
    st.title('Traffic Congestion Prediction')

    st.sidebar.title('Menu')
    menu = st.sidebar.radio("Menu", ['Home', 'Predict', 'Report', 'Dashboard'])

    if menu == 'Home':
        st.subheader('Home')
        st.write('Welcome to the Traffic Congestion Prediction App.')
        st.image('./img/AI-in-transportation.webp', caption='Traffic Congestion')

    elif menu == 'Predict':
        st.subheader('Predict Traffic Congestion')
        # Dictionary to map zone names to their corresponding values
        zone_mapping = {
            'North Berlin': 1,
            'South Berlin': 2,
            'East Berlin': 3,
            'West Berlin': 4
        }
         # Zone = st.selectbox('Zone', Traffic_prediction['Zone'].unique())
        selected_zone_name = st.selectbox('Zone', list(zone_mapping.keys()))
        Zone = zone_mapping[selected_zone_name]
        selected_date = st.date_input('Select a date')
        selected_time = st.time_input('Select a time')

        # Extracting year, month, day, hour, day of the week from the selected date and time
        year = selected_date.year
        month = selected_date.month
        date = selected_date.day
        hour = selected_time.hour
        minute = selected_time.minute
        day_of_week = selected_date.weekday()  # Monday is 0 and Sunday is 6
        day = selected_date.strftime('%A')
        hour_of_day = hour

        input_data = pd.DataFrame({
            'Zone': [Zone],
            'Year': [year],
            'Month': [month],
            'Date': [date],
            'Hour': [hour],
            'DayOfWeek': [day_of_week],
            'HourOfDay': [hour_of_day],
            'Day': [day]
        })

        if st.button('Predict'):
            prediction = model.predict(input_data)
            st.write(f'The predicted number of vehicles is: {int(prediction[0])}')
            st.image('./img/header-mobility.jpg', caption='Predict')

    elif menu == 'Report':
        st.subheader('Model Report')
        st.image('./img/4.jpg', caption='Report')
        y_test = Traffic_prediction['Vehicles']
        y_pred = model.predict(Traffic_prediction.drop(columns=['Vehicles', 'ID', 'DateTime']))
        report_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        st.write(report_df.head(10))
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse:.4f}')
        st.write(f'R-squared: {r2:.4f}')

        # Plotting actual vs predicted
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Vehicles")
        plt.ylabel("Predicted Vehicles")
        plt.title("Actual vs Predicted Vehicles")
        st.pyplot(plt)

    elif menu == 'Dashboard':
        st.subheader('Dashboard')
        st.image('./img/3.jpeg', caption='Dashboard')

        # Traffic volume over time
        st.write('### Traffic Volume Over Time')
        
        zone_mapping = {
            'North Berlin': 1,
            'South Berlin': 2,
            'East Berlin': 3,
            'West Berlin': 4
        }
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='DateTime', y='Vehicles', hue='ZoneName', data=Traffic_prediction, palette='tab10')
        plt.title("Traffic Volume Over Time")
        plt.ylabel("Number of Vehicles")
        plt.xlabel("DateTime")
        plt.xticks(rotation=45)
        plt.legend(title='Zone')
        st.pyplot(plt)

        # Traffic volume by Zone using violin plot
        st.write('### Traffic Volume by Zone')
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Zone', y='Vehicles', data=Traffic_prediction, palette='muted')
        plt.title('Traffic Volume by Zone')
        plt.xlabel('Zone')
        plt.ylabel('Number of Vehicles')
        st.pyplot(plt)

        # Average traffic volume by hour of the day
        Traffic_prediction['HourOfDay'] = Traffic_prediction['DateTime'].dt.hour
        hourly_traffic = Traffic_prediction.groupby('HourOfDay')['Vehicles'].mean().reset_index()
        st.write('### Average Traffic Volume by Hour of Day')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='HourOfDay', y='Vehicles', data=hourly_traffic, palette='viridis')
        plt.title('Average Traffic Volume by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Traffic Volume')
        plt.xticks(range(0, 24))
        st.pyplot(plt)

        # Average traffic volume by day of the week
        Traffic_prediction['DayOfWeek'] = Traffic_prediction['DateTime'].dt.day_name()
        weekly_traffic = Traffic_prediction.groupby('DayOfWeek')['Vehicles'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).reset_index()
        st.write('### Average Traffic Volume by Day of Week')
        plt.figure(figsize=(12, 6))
        sns.barplot(x='DayOfWeek', y='Vehicles', data=weekly_traffic, palette='cubehelix')
        plt.title('Average Traffic Volume by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Traffic Volume')
        st.pyplot(plt)

if __name__ == '__main__':
    main()
