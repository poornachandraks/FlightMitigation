import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from gate_assignment import GateAssignment
import time
from queue import PriorityQueue
import numpy as np
import delay_prediction
import json
from gate_storage import init_db, save_gate_assignment, get_gate_assignments, clear_db

def format_time(minutes):
    """Convert minutes since midnight to HH:MM format"""
    if minutes is None:
        return "00:00"
    
    # Handle minutes greater than 24 hours
    minutes = int(minutes)
    day_minutes = minutes % (24 * 60)  # Keep within 24 hours
    hours = day_minutes // 60
    mins = day_minutes % 60
    return f"{hours:02d}:{mins:02d}"

def convert_hhmm_to_minutes(hhmm):
    """Convert HHMM format (e.g., 1430 for 14:30) to minutes since midnight"""
    hours = hhmm // 100
    minutes = hhmm % 100
    return hours * 60 + minutes

def initialize_session_state():
    if 'initialized' not in st.session_state:
        # Read the raw flight data and store it
        st.session_state.df = pd.read_csv('data/preprocessed_flights.csv')
        
        # Convert HHMM times to minutes since midnight
        st.session_state.df['CRS_ARR_TIME_MINS'] = st.session_state.df['CRS_ARR_TIME'].apply(convert_hhmm_to_minutes)
        st.session_state.df['ACTUAL_ARR_TIME_MINS'] = st.session_state.df['CRS_ARR_TIME_MINS'] + st.session_state.df['ARR_DELAY']
        
        # Create initial assignments DataFrame
        assignments = st.session_state.df.copy()
        assignments['actual_arrival_time'] = pd.NA
        assignments['gate_number'] = pd.NA
        assignments['gate_distance'] = pd.NA
        assignments['wait_time'] = pd.NA
        
        # Store in session state
        st.session_state.assignments = assignments
        st.session_state.priority_queue = create_priority_queue(assignments)
        st.session_state.gates_df = pd.read_csv('data/gate_distances.csv')
        st.session_state.queued_flights = set()  # Initialize empty set for queued flights
        
        # Initialize time tracking using actual arrival times in minutes
        min_time = st.session_state.df['ACTUAL_ARR_TIME_MINS'].min()
        max_time = st.session_state.df['ACTUAL_ARR_TIME_MINS'].max()
        
        st.session_state.current_time = int(min_time) if pd.notnull(min_time) else 0
        st.session_state.max_time = int(max_time)
        st.session_state.initialized = True
        st.session_state.running = False
        st.session_state.time_step = 1
        st.session_state.auto_advance = False

def create_gate_visualization(assignments, current_time, gates_df):
    gate_status = []
    
    for _, gate in gates_df.iterrows():
        gate_num = gate['gate_number']
        current_flight = assignments[
            (assignments['gate_number'] == gate_num) & 
            (assignments['actual_arrival_time'] <= current_time) & 
            (assignments['actual_arrival_time'] + 60 > current_time)
        ]
        
        if len(current_flight) > 0:
            flight = current_flight.iloc[0]
            remaining_time = 60 - (current_time - flight['actual_arrival_time'])
            
            # Calculate delays ensuring no negative values
            original_delay = max(0, flight['ARR_DELAY'])
            wait_time = max(0, flight['wait_time']) if pd.notnull(flight['wait_time']) else 0
            total_delay = original_delay + wait_time
            
            # Format scheduled and actual arrival times
            scheduled_arr = convert_hhmm_to_minutes(flight['CRS_ARR_TIME'])
            actual_arr = scheduled_arr + original_delay
            
            flight_info = (
                f"Flight #{flight['CRS_DEP_TIME']}<br>"
                f"Scheduled Arrival: {format_time(scheduled_arr)}<br>"
                f"Actual Arrival: {format_time(actual_arr)}<br>"
                f"Original Delay: {original_delay:.0f} min<br>"
                f"Wait Time: {wait_time:.0f} min<br>"
                f"Total Delay: {total_delay:.0f} min<br>"
                f"Remaining Time: {remaining_time:.0f} min"
            )
        else:
            flight_info = 'Available'
            
        status = {
            'gate_number': gate_num,
            'distance': gate['distance_from_runway'],
            'status': 'Occupied' if len(current_flight) > 0 else 'Available',
            'flight_info': flight_info
        }
        gate_status.append(status)
    
    return pd.DataFrame(gate_status)

def create_airport_layout(gate_status):
    fig = go.Figure()
    
    # Improved runway design
    runway_length = max(gate_status['distance']) * 0.4  # 40% of max gate distance
    runway_width = 0.2
    
    # Add runway with improved design
    fig.add_trace(go.Scatter(
        x=[-runway_length/2, runway_length/2],
        y=[0, 0],
        mode='lines',
        line=dict(
            color='gray',
            width=10
        ),
        name='Runway'
    ))
    
    # Add runway markings
    for x in np.linspace(-runway_length/2, runway_length/2, 8):
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[-runway_width/4, runway_width/4],
            mode='lines',
            line=dict(
                color='white',
                width=2
            ),
            showlegend=False
        ))
    
    # Add taxiway
    fig.add_trace(go.Scatter(
        x=[0, max(gate_status['distance']) * 1.1],
        y=[0, 0],
        mode='lines',
        line=dict(
            color='lightgray',
            width=5,
            dash='dot'
        ),
        name='Taxiway'
    ))
    
    # Calculate y-positions for gates to prevent overlap
    y_positions = []
    for i in range(len(gate_status)):
        if i % 2 == 0:
            y_positions.append(0.5)
        else:
            y_positions.append(-0.5)
    
    # Add gates with improved styling and spacing
    for idx, gate in gate_status.iterrows():
        color = 'rgb(239, 85, 59)' if gate['status'] == 'Occupied' else 'rgb(99, 110, 250)'
        y_pos = y_positions[idx]
        
        # Create hover text
        hover_text = (f"Gate {gate['gate_number']}<br>"
                     f"Distance: {gate['distance']}m<br>"
                     f"{gate['flight_info']}")
        
        fig.add_trace(go.Scatter(
            x=[gate['distance']],
            y=[y_pos],
            mode='markers+text',
            marker=dict(
                size=25,
                color=color,
                line=dict(color='black', width=1)
            ),
            text=[f"Gate {gate['gate_number']}"],  # Only show gate number
            textposition="middle center",
            hovertext=[hover_text],
            hoverinfo='text',
            name=f"Gate {gate['gate_number']}"
        ))
    
    fig.update_layout(
        title={
            'text': 'Airport Gate Layout',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Distance from Runway (meters)',
        yaxis=dict(
            range=[-1, 1],
            showgrid=False,
            zeroline=True,
            showticklabels=False
        ),
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(240,240,240,0.5)',
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='closest'
    )
    
    return fig

def calculate_statistics(assignments, current_time):
    """Calculate accurate statistics for the simulation"""
    # Get waiting flights (those that have arrived but not yet assigned gates)
    waiting_flights = assignments[
        (assignments['actual_arrival_time'].isna()) & 
        (assignments['ACTUAL_ARR_TIME_MINS'] <= current_time)
    ]
    
    # Get completed flights (those that have been assigned gates)
    completed_flights = assignments[
        assignments['actual_arrival_time'].notna()
    ]
    
    stats = {
        'completed_count': len(completed_flights),
        'waiting_count': len(st.session_state.queued_flights),  # Use queue size for accuracy
        'avg_wait_time': 0,
        'avg_total_delay': 0
    }
    
    if len(completed_flights) > 0:
        valid_waits = completed_flights[
            completed_flights['wait_time'].notna() & 
            (completed_flights['wait_time'] >= 0)
        ]
        
        if len(valid_waits) > 0:
            stats['avg_wait_time'] = valid_waits['wait_time'].mean()
            valid_waits['total_delay'] = valid_waits.apply(
                lambda x: max(0, x['ARR_DELAY']) + x['wait_time'], 
                axis=1
            )
            stats['avg_total_delay'] = valid_waits['total_delay'].mean()
    
    return stats

def create_priority_queue(assignments):
    """Create a priority queue of flights based on predicted delay"""
    # Initialize an empty queue - we'll add flights as they arrive
    return PriorityQueue()

def update_priority_queue(assignments, current_time):
    """Update priority queue with newly arrived flights"""
    # No need to check if queued_flights exists anymore since it's initialized in session_state
    
    # Get flights that have just arrived
    new_arrivals = assignments[
        (assignments['actual_arrival_time'].isna()) &  # Not yet assigned a gate
        (assignments['ACTUAL_ARR_TIME_MINS'] <= current_time) &  # Has actually arrived
        (~assignments['CRS_DEP_TIME'].isin(st.session_state.queued_flights))  # Not already in queue
    ]
    
    for _, flight in new_arrivals.iterrows():
        priority = -float(flight['pred_delay'])
        flight_dict = flight.to_dict()
        st.session_state.priority_queue.put((priority, flight_dict))
        st.session_state.queued_flights.add(flight_dict['CRS_DEP_TIME'])

def assign_available_gates(assignments, current_time, gates_df):
    """Assign available gates to waiting flights based on priority"""
    # Get available gates
    available_gates = []
    for _, gate in gates_df.iterrows():
        gate_occupied = len(assignments[
            (assignments['gate_number'] == gate['gate_number']) & 
            (assignments['actual_arrival_time'] <= current_time) & 
            (assignments['actual_arrival_time'] + 60 > current_time)
        ]) > 0
        
        if not gate_occupied:
            available_gates.append(gate)
    
    # Try to assign gates to waiting flights
    assignments_made = []
    
    while available_gates and not st.session_state.priority_queue.empty():
        priority, flight_dict = st.session_state.priority_queue.get()
        gate = available_gates.pop(0)
        
        try:
            # Calculate actual arrival time in minutes
            scheduled_arrival_mins = convert_hhmm_to_minutes(flight_dict['CRS_ARR_TIME'])
            actual_arrival_mins = scheduled_arrival_mins + max(0, flight_dict['ARR_DELAY'])
            
            # Calculate wait time
            wait_time = max(0, current_time - actual_arrival_mins)
            
            # Initialize gate_assignments if it doesn't exist
            if 'gate_assignments' not in st.session_state:
                st.session_state.gate_assignments = {}
            
            # Store gate assignment
            flight_key = flight_dict['CRS_DEP_TIME']
            gate_info = {
                'gate_number': gate['gate_number'],
                'actual_arrival_time': current_time,
                'wait_time': wait_time,
                'status': 'occupied',
                'occupied_until': current_time + 60
            }
            
            save_gate_assignment(flight_key, gate_info)
            
            # Update flight assignment
            flight_update = {
                'actual_arrival_time': current_time,
                'gate_number': gate['gate_number'],
                'gate_distance': gate['distance_from_runway'],
                'wait_time': wait_time
            }
            
            # Remove from queued flights set
            st.session_state.queued_flights.remove(flight_dict['CRS_DEP_TIME'])
            assignments_made.append((flight_dict['CRS_DEP_TIME'], flight_update))
            
        except KeyError as e:
            st.error(f"Missing data in flight record: {e}")
            continue
    
    return assignments_made

def display_waiting_queue():
    """Display active flight queue with priority information"""
    st.subheader("Active Flight Queue")
    
    # Add custom CSS for better table styling and readability
    st.markdown("""
        <style>
        .queue-table {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .queue-header {
            background-color: #f0f2f6;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            font-size: 14px;
            font-weight: 500;
        }
        .dataframe {
            font-size: 14px !important;
            color: black !important;
            background-color: #f0f2f6 !important;
        }
        .dataframe td {
            padding: 8px !important;
            color: black !important;
        }
        .stDataFrame {
            font-size: 14px !important;
            color: black !important;
            background-color: #f0f2f6 !important;
        }
        div[data-testid="stDataFrame"] div[data-testid="stTable"] {
            color: black !important;
            background-color: #f0f2f6 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    total_waiting = len(st.session_state.queued_flights)
    
    if total_waiting > 0:
        # Convert times to HH:MM format
        queue_df = get_top_waiting_flights()
        queue_df['scheduled_arrival'] = queue_df['arrival_time'].apply(
            lambda x: format_time(convert_hhmm_to_minutes(x))
        )
        
        # Create a more informative display
        display_df = pd.DataFrame({
            'Queue Position': queue_df['priority_score'],
            'Flight': queue_df['flight_time'].apply(lambda x: f"#{x}"),
            'Scheduled': queue_df['scheduled_arrival'],
            'Pred. Delay': queue_df['pred_delay'].round(1)
        })
        
        # Style the dataframe
        def style_rows(row):
            color_scale = ['#ffedea', '#fff4f2', '#fff9f8', '#fffafa', '#ffffff']
            if row.name < len(color_scale):
                return [
                    f'background-color: {color_scale[row.name]}; '
                    f'color: black; '
                    f'font-weight: {"bold" if row.name == 0 else "normal"}'
                ] * len(row)
            return ['color: black'] * len(row)
        
        styled_df = display_df.style.apply(style_rows, axis=1)
        
        # Display queue header with count
        st.markdown(f"""
            <div class='queue-header'>
                🛩️ {total_waiting} aircraft in queue ({len(queue_df)} shown)
            </div>
        """, unsafe_allow_html=True)
        
        # Display the styled dataframe
        st.dataframe(
            styled_df,
            column_config={
                'Queue Position': st.column_config.Column(
                    'Position',
                    help='Position in queue based on priority',
                    width='small'
                ),
                'Flight': st.column_config.Column(
                    'Flight ID',
                    help='Flight identifier',
                    width='small'
                ),
                'Scheduled': st.column_config.Column(
                    'Arrival',
                    help='Scheduled arrival time',
                    width='small'
                ),
                'Pred. Delay': st.column_config.NumberColumn(
                    'Delay (min)',
                    help='Predicted delay in minutes',
                    format="%.0f",
                    width='small'
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        if total_waiting > 3:
            st.warning(f"⚠️ High queue volume: {total_waiting} flights waiting")
    else:
        st.markdown("""
            <div class='queue-header' style='text-align: center; color: #666;'>
                ✈️ No flights currently waiting
            </div>
        """, unsafe_allow_html=True)

def get_top_waiting_flights(n=5):
    """Get top n waiting flights from priority queue without removing them"""
    if not hasattr(st.session_state, 'priority_queue'):
        return pd.DataFrame()
    
    # Create a temporary list to store and restore items
    temp_items = []
    top_flights = []
    
    # Get up to n items
    while not st.session_state.priority_queue.empty() and len(top_flights) < n:
        priority, flight = st.session_state.priority_queue.get()
        temp_items.append((priority, flight))
        top_flights.append({
            'flight_time': flight['CRS_DEP_TIME'],
            'pred_delay': -priority,  # Convert back to positive
            'arrival_time': flight['CRS_ARR_TIME'],
            'priority_score': f"#{len(top_flights) + 1}"
        })
    
    # Restore items to queue
    for item in temp_items:
        st.session_state.priority_queue.put(item)
    
    return pd.DataFrame(top_flights)

def main():
    st.set_page_config(page_title="Airport Gate Assignment Simulation", layout="wide")
    
    # Initialize the database
    init_db()
    
    # Add page selection
    page = st.sidebar.radio("Select Page", ["Gate Simulation", "Delay Prediction"])
    
    if page == "Gate Simulation":
        run_simulation()
    else:
        delay_prediction.delay_prediction_page()

def run_simulation():
    # Initialize session state first
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .big-font {
            font-size:24px !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Airport Gate Assignment Simulation</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Controls")
        
        # Time controls
        st.subheader("Time Settings")
        simulation_speed = st.slider("Simulation Speed", 0.1, 5.0, 1.0, 
                                   help="Higher values = faster simulation")
        st.session_state.time_step = st.select_slider(
            "Time Step (minutes)",
            options=[1, 5, 10, 15, 30, 60],
            value=st.session_state.time_step
        )
        
        # Auto-advance toggle
        st.session_state.auto_advance = st.checkbox("Auto Advance", 
                                                  value=st.session_state.auto_advance,
                                                  help="Automatically advance time")
        
        # Time input with proper formatting
        try:
            min_time = int(st.session_state.df['ACTUAL_ARR_TIME_MINS'].min())
            max_time = int(st.session_state.max_time)
            curr_time = min(max(int(st.session_state.current_time), min_time), max_time)
            
            # Show current time in HH:MM format
            st.write(f"Current Time: {format_time(curr_time)}")
            st.write(f"Time Range: {format_time(min_time)} - {format_time(max_time)}")
            
            new_time = st.number_input(
                "Jump to time (minutes from midnight)", 
                min_value=min_time,
                max_value=max_time,
                value=curr_time
            )
            if st.button("Jump"):
                st.session_state.current_time = new_time
        except Exception as e:
            st.error(f"Error with time input. Using default values.")
            st.session_state.current_time = st.session_state.df['ACTUAL_ARR_TIME_MINS'].min()

    # Main content
    # Time display and controls
    col1, col2, col3, col4 = st.columns([2,1,1,1])
    with col1:
        current_time_str = format_time(st.session_state.current_time)
        st.metric(
            "Current Time", 
            current_time_str,
            help="Current simulation time (HH:MM)"
        )
    with col2:
        if st.button('▶️ Start' if not st.session_state.running else '⏸️ Pause'):
            st.session_state.running = not st.session_state.running
    with col3:
        if st.button('⏭️ Step'):
            st.session_state.current_time += st.session_state.time_step
    with col4:
        if st.button('🔄 Reset'):
            # Clear database
            clear_db()
            # Reset simulation state
            st.session_state.current_time = st.session_state.assignments['actual_arrival_time'].min()
            st.session_state.running = False
            # Clear queue
            st.session_state.queued_flights = set()
            st.session_state.priority_queue = create_priority_queue(st.session_state.assignments)
            # Reset assignments
            st.session_state.assignments['actual_arrival_time'] = pd.NA
            st.session_state.assignments['gate_number'] = pd.NA
            st.session_state.assignments['gate_distance'] = pd.NA
            st.session_state.assignments['wait_time'] = pd.NA
    
    # Create gate status visualization
    gate_status = create_gate_visualization(
        st.session_state.assignments,
        st.session_state.current_time,
        st.session_state.gates_df
    )
    
    # Display airport layout
    st.plotly_chart(create_airport_layout(gate_status), use_container_width=True)
    
    # Statistics section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Statistics")
        stat_col1, stat_col2, stat_col3 = st.columns(3)  # Changed from 4 to 3 columns
        
        stats = calculate_statistics(st.session_state.assignments, st.session_state.current_time)
        
        with stat_col1:
            st.metric("Completed Flights", stats['completed_count'])
        with stat_col2:
            st.metric("Avg Wait Time", f"{stats['avg_wait_time']:.1f} min")
        with stat_col3:
            st.metric("Avg Total Delay", f"{stats['avg_total_delay']:.1f} min")
    
    with col2:
        # Use new priority queue display
        display_waiting_queue()
    
    # Update simulation state
    if (st.session_state.running or st.session_state.auto_advance) and \
       st.session_state.current_time < st.session_state.max_time:
        
        # Update current time file
        update_current_time(st.session_state.current_time)
        
        # Update gate statuses
        update_gate_statuses(st.session_state.current_time)
        
        # Update priority queue with newly arrived flights
        update_priority_queue(st.session_state.assignments, st.session_state.current_time)
        
        # Try to assign available gates
        new_assignments = assign_available_gates(
            st.session_state.assignments,
            st.session_state.current_time,
            st.session_state.gates_df
        )
        
        # Update assignments with new gate assignments
        for flight_time, updates in new_assignments:
            try:
                mask = st.session_state.assignments['CRS_DEP_TIME'] == flight_time
                if mask.any():
                    for key, value in updates.items():
                        st.session_state.assignments.loc[mask, key] = value
                else:
                    st.warning(f"Could not find flight with departure time {flight_time}")
            except Exception as e:
                st.error(f"Error updating assignments: {e}")
        
        # Advance time
        time.sleep(1 / simulation_speed)
        st.session_state.current_time += st.session_state.time_step
        st.rerun()

def update_gate_statuses(current_time):
    """Update status and remaining time for all gate assignments"""
    try:
        # Get all current assignments
        assignments = get_gate_assignments()
        
        # Update each assignment
        for flight_key, gate_info in assignments.items():
            # Update status based on current time
            if gate_info['occupied_until'] <= current_time:
                gate_info['status'] = 'departed'  # Changed from 'available' to 'departed'
            elif gate_info['actual_arrival_time'] <= current_time:
                gate_info['status'] = 'occupied'
            
            # Save the updated info back to storage
            save_gate_assignment(flight_key, gate_info)
            
    except Exception as e:
        st.error(f"Error updating gate statuses: {e}")

def assign_gates(flights_df):
    # Create a dictionary to store gate assignment details
    gate_assignments = {}
    current_time = pd.Timestamp.now()
    
    for idx, flight in flights_df.iterrows():
        wait_time = 0
        assigned_gate = None
        
        # Your existing gate assignment logic here
        # ... existing code ...
        
        # Store the assignment details
        gate_assignments[flight['flight_number']] = {
            'gate': assigned_gate,
            'assignment_time': current_time,
            'wait_time': wait_time
        }
    
    # Store in session state
    st.session_state['gate_assignments'] = gate_assignments
    return gate_assignments

def update_current_time(current_time):
    """Save current simulation time to file"""
    try:
        with open('current_time.txt', 'w') as f:
            f.write(str(current_time))
    except Exception as e:
        st.error(f"Error saving current time: {e}")

if __name__ == "__main__":
    main() 