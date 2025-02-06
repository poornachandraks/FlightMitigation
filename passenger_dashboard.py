import streamlit as st
import pandas as pd
import datetime
import json
from gate_storage import get_gate_assignments

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

def format_time_from_minutes(minutes):
    """Convert minutes since midnight to HH:MM format"""
    if pd.isnull(minutes):
        return "Not available"
    hours = int(minutes) // 60
    mins = int(minutes) % 60
    return f"{hours:02d}:{mins:02d}"

def initialize_data():
    df = pd.read_csv('./data/preprocessed_flights.csv')
    return df

def format_remaining_time(gate_info, current_time):
    """Calculate and format the remaining time at gate"""
    if gate_info['status'] == 'occupied':
        remaining_time = gate_info['occupied_until'] - current_time
        if remaining_time > 0:
            return f"{remaining_time:.0f} min"
    return "Available"

def get_status_display(gate_info, current_time):
    """Get the status display information for a gate"""
    if gate_info['status'] == 'departed':
        return "⚫ Departed", "off"
    elif gate_info['status'] == 'occupied':
        remaining_time = gate_info['occupied_until'] - current_time
        if remaining_time > 0:
            return "�� At Gate", "normal"
        else:
            return "🟡 Departing", "normal"
    else:
        return "⚪ Available", "off"

def search_flight(df, flight_time=None, flight_number=None):
    """Search for a flight by either time or flight number"""
    if flight_time:
        time_in_minutes = flight_time.hour * 100 + flight_time.minute
        flight = df[df['CRS_DEP_TIME'] == time_in_minutes]
        search_key = time_in_minutes
    elif flight_number:
        flight = df[df['CRS_DEP_TIME'] == flight_number]
        search_key = flight_number
    else:
        return pd.DataFrame()

    # Get gate assignments
    gate_assignments = get_gate_assignments()
    
    # Get current time from simulation if available
    try:
        with open('current_time.txt', 'r') as f:
            current_time = int(f.read().strip())
    except:
        current_time = 0
    
    # Check if there's gate assignment information
    if not flight.empty:
        gate_info = gate_assignments.get(str(search_key))
        if gate_info:
            flight = flight.copy()
            flight['assigned_gate'] = gate_info['gate_number']
            flight['actual_gate_arrival'] = gate_info['actual_arrival_time']
            flight['gate_wait_time'] = gate_info['wait_time']
            
            # Add gate information section
            st.subheader("🚪 Gate Information")
            col1, col2, col3, col4 = st.columns(4)
            
            # Gate number and status
            with col1:
                status_icon, status_color = get_status_display(gate_info, current_time)
                st.metric(
                    "Gate Status",
                    f"Gate {gate_info['gate_number']}",
                    status_icon,
                    delta_color=status_color
                )
            
            # Assignment time
            with col2:
                st.metric(
                    "Gate Assignment Time",
                    format_time_from_minutes(gate_info['actual_arrival_time'])
                )
            
            # Wait time
            with col3:
                st.metric(
                    "Gate Wait Time",
                    f"{gate_info['wait_time']:.0f} min"
                )
            
            # Remaining time or departure info
            with col4:
                if gate_info['status'] == 'departed':
                    departure_time = format_time_from_minutes(gate_info['occupied_until'])
                    st.metric(
                        "Departure Time",
                        departure_time
                    )
                elif gate_info['status'] == 'occupied':
                    remaining_time = gate_info['occupied_until'] - current_time
                    if remaining_time > 0:
                        st.metric(
                            "Time Remaining",
                            f"{remaining_time:.0f} min"
                        )
                    else:
                        st.metric(
                            "Status",
                            "Preparing for departure"
                        )
                else:
                    st.metric(
                        "Status",
                        "Gate Available"
                    )

    return flight

def display_flight_info(flight_number):
    # Check if gate assignments exist in session state
    gate_info = None
    if 'gate_assignments' in st.session_state:
        gate_info = st.session_state['gate_assignments'].get(flight_number)
    
    # Display flight information
    st.write(f"Flight Number: {flight_number}")
    
    if gate_info:
        st.write(f"Assigned Gate: {gate_info['gate']}")
        st.write(f"Gate Assignment Time: {gate_info['assignment_time'].strftime('%H:%M:%S')}")
        if gate_info['wait_time'] > 0:
            st.write(f"Wait Time for Gate: {gate_info['wait_time']} minutes")
    else:
        st.write("Gate assignment information not available")

def main():
    st.set_page_config(page_title="Flight Status Dashboard", layout="wide")
    
    st.title("✈️ Flight Status Dashboard")
    
    df = initialize_data()
    
    st.subheader("🔍 Search Your Flight")
    
    # Add search method selection
    search_method = st.radio(
        "Search by:",
        ["Flight Number", "Departure Time"],
        horizontal=True
    )
    
    if search_method == "Flight Number":
        flight_number = st.number_input(
            "Enter Flight Number",
            min_value=0,
            max_value=2359,
            help="Enter the flight number (e.g., 1430)"
        )
        search_params = {'flight_number': flight_number}
    else:
        col1, col2 = st.columns(2)
        with col1:
            hour = st.number_input("Hour (24-hour format)", min_value=0, max_value=23, value=12)
        with col2:
            minute = st.number_input("Minute", min_value=0, max_value=59, value=0)
        flight_time = datetime.time(hour, minute)
        search_params = {'flight_time': flight_time}
    
    if st.button("Search Flight"):
        flight = search_flight(df, **search_params)
        
        if not flight.empty:
            st.success("Flight Found!")
            
            # Add Flight Number display
            st.subheader("✈️ Flight Details")
            st.metric("Flight Number", f"#{flight['CRS_DEP_TIME'].iloc[0]}")
            
            # Departure Information
            st.subheader("📥 Previous Departure Information")
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
            st.subheader("📤 Arrival Status")
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
            st.error("No flight found with the provided criteria")

if __name__ == "__main__":
    main()