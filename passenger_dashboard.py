import streamlit as st
import pandas as pd
import datetime

def normalize_time(time_value):
    if pd.isnull(time_value):
        return None
        
    hours = time_value // 100
    minutes = time_value % 100
    
    if minutes >= 60:
        hours += minutes // 60
        minutes = minutes % 60
    elif minutes < 0:
        hours_reduction = (abs(minutes) + 59) // 60
        hours -= hours_reduction
        minutes = 60 - (abs(minutes) % 60)
    
    hours = hours % 24
    return hours * 100 + minutes

def format_time(time_value):
    if pd.isnull(time_value):
        return "Not available"
    
    time_value = normalize_time(time_value)
    hours = time_value // 100
    minutes = time_value % 100
    period = "AM" if hours < 12 else "PM"
    if hours > 12:
        hours -= 12
    elif hours == 0:
        hours = 12
    return f"{hours:02d}:{minutes:02d} {period}"

def format_delay_status(delay):
    if pd.isnull(delay):
        return "No Information", None
    elif delay > 0:
        return "Delayed", f"+{delay:.0f} min"
    elif delay < 0:
        return "Early", f"{delay:.0f} min"
    else:
        return "On Time", None

def initialize_data():
    df = pd.read_csv('./data/preprocessed_flights.csv')
    return df

def search_flight(df, flight_time):
    time_in_minutes = flight_time.hour * 100 + flight_time.minute
    flight = df[df['CRS_DEP_TIME'] == time_in_minutes]
    return flight

def main():
    st.set_page_config(page_title="Flight Dashboard", layout="wide")
    
    st.title("✈️ Flight Status Dashboard")
    
    df = initialize_data()
    
    st.subheader("🔍 Search Your Flight")
    
    col1, col2 = st.columns(2)
    with col1:
        hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, value=12)
    with col2:
        minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
    
    flight_time = datetime.time(hour, minute)
    
    if st.button("Search Flight"):
        flight = search_flight(df, flight_time)
        
        if not flight.empty:
            st.success("Flight Found!")
            
            # Departure Information
            st.subheader("📥 Departure Information")
            scheduled_dep = flight['CRS_DEP_TIME'].iloc[0]
            dep_delay = flight['DEP_DELAY'].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Scheduled Departure", 
                    format_time(scheduled_dep)
                )
            with col2:
                actual_dep = normalize_time(scheduled_dep + (dep_delay if pd.notnull(dep_delay) else 0))
                st.metric(
                    "Actual Departure",
                    format_time(actual_dep)
                )
            with col3:
                status, delta = format_delay_status(dep_delay)
                st.metric(
                    "Departure Status", 
                    status,
                    delta,
                    delta_color="inverse"
                )
            
            # Arrival Information
            st.subheader("📤 Arrival Information")
            scheduled_arr = flight['CRS_ARR_TIME'].iloc[0]
            pred_delay = flight['pred_delay'].iloc[0]
            actual_arr = flight['ACTUAL_ARR_TIME'].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Scheduled Arrival", 
                    format_time(scheduled_arr)
                )
            with col2:
                if pd.notnull(actual_arr):
                    st.metric(
                        "Actual Arrival", 
                        format_time(actual_arr)
                    )
                else:
                    predicted_arr = normalize_time(scheduled_arr + (pred_delay if pd.notnull(pred_delay) else 0))
                    st.metric(
                        "Predicted Arrival", 
                        format_time(predicted_arr)
                    )
            with col3:
                if pd.notnull(pred_delay):
                    arr_status, arr_delta = format_delay_status(pred_delay)
                    st.metric(
                        "Arrival Status",
                        arr_status,
                        arr_delta,
                        delta_color="inverse"
                    )
            
            # Flight Details
            st.subheader("✈️ Additional Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Flight Distance",
                    f"{flight['DISTANCE'].iloc[0]:.0f} miles"
                )
            with col2:
                st.metric(
                    "Scheduled Duration",
                    f"{flight['CRS_ELAPSED_TIME'].iloc[0]:.0f} minutes"
                )
        else:
            st.error("No flight found at this time")

if __name__ == "__main__":
    main()