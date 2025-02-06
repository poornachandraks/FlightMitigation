import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import pickle
import xgboost as xgb

def load_model():
    """Load the trained XGBoost model"""
    try:
        with open('./models/xgb_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def initialize_prediction_state():
    """Initialize session state variables for prediction page"""
    if 'pred_model' not in st.session_state:
        model = load_model()
        if model:
            st.session_state.pred_model = model
        else:
            st.error("Failed to load prediction model")
            st.session_state.pred_model = dummy_predict
    
    if 'flight_data' not in st.session_state:
        try:
            # Load flight data for dropdowns
            df = pd.read_csv('data/delay_data.csv')
            
            # Store min/max values for validation
            st.session_state.min_distance = df['DISTANCE'].min()
            st.session_state.max_distance = df['DISTANCE'].max()
            st.session_state.min_elapsed = df['CRS_ELAPSED_TIME'].min()
            st.session_state.max_elapsed = df['CRS_ELAPSED_TIME'].max()
            
            # Store the dataframe for reference
            st.session_state.flight_data = df
            
        except Exception as e:
            st.error(f"Error loading flight data: {e}")
            # Set default values if data loading fails
            st.session_state.min_distance = 100
            st.session_state.max_distance = 3000
            st.session_state.min_elapsed = 60
            st.session_state.max_elapsed = 360

def dummy_predict(features):
    """Temporary dummy prediction function"""
    base_delay = 15
    
    # Time of day factor
    hour = features['CRS_DEP_TIME'] // 100
    time_factors = {
        'morning': 1.2 if 5 <= hour < 12 else 1.0,
        'peak': 1.5 if (12 <= hour < 14) or (16 <= hour < 19) else 1.0,
        'night': 1.3 if hour >= 22 or hour < 5 else 1.0
    }
    
    # Distance and elapsed time factors
    distance_factor = features['DISTANCE'] / 1000  # Normalize distance
    elapsed_factor = features['CRS_ELAPSED_TIME'] / 60  # Convert to hours
    
    # Calculate delay
    delay = (base_delay * 
             sum(time_factors.values()) * 
             (1 + distance_factor * 0.1) *
             (1 + elapsed_factor * 0.1))
    
    return round(delay, 1)

def format_time(time_val):
    """Format HHMM time to HH:MM"""
    time_str = str(time_val).zfill(4)
    return f"{time_str[:2]}:{time_str[2:]}"

def predict_delay(features):
    """Make prediction using the loaded model"""
    try:
        # Convert features to DataFrame with proper column names
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = st.session_state.pred_model.predict(input_df)[0]
        return round(float(prediction), 1)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return dummy_predict(features)

def delay_prediction_page():
    """Main function for delay prediction page"""
    st.title("Flight Delay Prediction")
    
    # Initialize session state
    initialize_prediction_state()
    
    # Create form for input
    with st.form("delay_prediction_form"):
        st.subheader("Flight Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Basic flight schedule
            dep_time = st.number_input(
                "Scheduled Departure (HHMM)",
                min_value=0,
                max_value=2359,
                value=1200,
                help="Enter scheduled departure time (e.g., 1430 for 2:30 PM)"
            )
            
            arr_time = st.number_input(
                "Scheduled Arrival (HHMM)",
                min_value=0,
                max_value=2359,
                value=1400,
                help="Enter scheduled arrival time"
            )
            
            elapsed_time = st.number_input(
                "Scheduled Flight Time (minutes)",
                min_value=int(st.session_state.min_elapsed),
                max_value=int(st.session_state.max_elapsed),
                value=int(st.session_state.min_elapsed),
                help="Scheduled flight duration"
            )
            
            distance = st.number_input(
                "Flight Distance (miles)",
                min_value=int(st.session_state.min_distance),
                max_value=int(st.session_state.max_distance),
                value=int(st.session_state.min_distance)
            )
        
        with col2:
            # Additional flight times
            dep_delay = st.number_input(
                "Departure Delay (minutes)",
                min_value=-60,
                max_value=180,
                value=0,
                help="Known departure delay (can be negative for early departures)"
            )
            
            taxi_out = st.number_input(
                "Taxi Out Time (minutes)",
                min_value=1,
                max_value=60,
                value=15,
                help="Expected taxi time from gate to takeoff"
            )
            
            wheels_off = st.number_input(
                "Wheels Off Time (HHMM)",
                min_value=0,
                max_value=2359,
                value=dep_time,
                help="Actual takeoff time"
            )
            
            wheels_on = st.number_input(
                "Wheels On Time (HHMM)",
                min_value=0,
                max_value=2359,
                value=arr_time,
                help="Actual landing time"
            )
            
            taxi_in = st.number_input(
                "Taxi In Time (minutes)",
                min_value=1,
                max_value=60,
                value=10,
                help="Expected taxi time from landing to gate"
            )
        
        with col3:
            # Date and status
            month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: datetime(2024, x, 1).strftime('%B')
            )
            
            day = st.selectbox(
                "Day",
                options=range(1, 32)
            )
            
            year = st.selectbox(
                "Year",
                options=[2023, 2024],
                index=1
            )
            
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week = st.selectbox(
                "Day of Week",
                options=range(1, 8),
                format_func=lambda x: f"{day_names[x-1]} ({x})"
            )
            
            diverted = st.selectbox(
                "Flight Diverted",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                help="Whether the flight is expected to be diverted"
            )
        
        submitted = st.form_submit_button("Predict Delay")
        
        if submitted:
            # Create feature dictionary with all required features
            features = {
                'CRS_DEP_TIME': dep_time,
                'DEP_DELAY': dep_delay,
                'TAXI_OUT': taxi_out,
                'WHEELS_OFF': wheels_off,
                'WHEELS_ON': wheels_on,
                'TAXI_IN': taxi_in,
                'CRS_ARR_TIME': arr_time,
                'DIVERTED': diverted,
                'CRS_ELAPSED_TIME': elapsed_time,
                'DISTANCE': distance,
                'DAY': day,
                'MONTH': month,
                'YEAR': year,
                'DAY_OF_WEEK': day_of_week
            }
            
            # Make prediction using the model
            predicted_delay = predict_delay(features)
            
            # Display results
            st.success("Prediction Complete!")
            
            # Show prediction results in columns
            results_col1, results_col2, results_col3 = st.columns(3)
            
            with results_col1:
                st.metric(
                    "Predicted Arrival Delay",
                    f"{predicted_delay} minutes",
                    help="Estimated arrival delay"
                )
            
            with results_col2:
                risk_level = "Low" if predicted_delay < 15 else "Medium" if predicted_delay < 30 else "High"
                risk_color = {
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red"
                }[risk_level]
                st.markdown(f"""
                    <div style='text-align: center'>
                        <p>Risk Level</p>
                        <h2 style='color: {risk_color}'>{risk_level}</h2>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show detailed flight information
            st.subheader("Flight Details")
            details_col1, details_col2, details_col3 = st.columns(3)
            
            with details_col1:
                st.write("Schedule Information:")
                st.write(f"• Scheduled Departure: {format_time(dep_time)}")
                st.write(f"• Scheduled Arrival: {format_time(arr_time)}")
                st.write(f"• Flight Duration: {elapsed_time} minutes")
                st.write(f"• Distance: {distance} miles")
            
            with details_col2:
                st.write("Actual Times:")
                st.write(f"• Departure Delay: {dep_delay} minutes")
                st.write(f"• Wheels Off: {format_time(wheels_off)}")
                st.write(f"• Wheels On: {format_time(wheels_on)}")
                st.write(f"• Taxi Times: {taxi_out}/{taxi_in} min")
            
            with details_col3:
                st.write("Flight Status:")
                st.write(f"• Date: {datetime(year, month, day).strftime('%B %d, %Y')}")
                st.write(f"• Day: {day_names[day_of_week-1]} ({day_of_week})")
                st.write(f"• Diverted: {'Yes' if diverted else 'No'}")

if __name__ == "__main__":
    delay_prediction_page() 