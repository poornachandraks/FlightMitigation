# Flight Gate Management System

A real-time flight gate management system with gate assignment simulation and passenger information dashboard.

## Features

- **Gate Assignment Simulation**
  - Real-time gate allocation
  - Visual airport layout
  - Queue management
  - Performance statistics

- **Passenger Information Dashboard**
  - Flight status lookup
  - Real-time gate information
  - Departure and arrival details
  - Delay predictions

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/FlightMitigation.git
   cd FlightMitigation
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Unix/MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

The system requires running two applications simultaneously:

1. **Start Gate Assignment Simulation**
   ```bash
   streamlit run simulation_ui.py
   ```
   This will open the simulation interface in your browser (typically http://localhost:8501)

2. **Start Passenger Dashboard**
   ```bash
   # Open a new terminal and run:
   streamlit run passenger_dashboard.py
   ```
   This will open the passenger dashboard in your browser (typically http://localhost:8502)

## Usage Guide

### Gate Assignment Simulation
- Use the control panel to:
  - Start/Pause simulation
  - Step through time
  - Reset simulation
- Monitor:
  - Current gate assignments
  - Waiting queue
  - Performance metrics
  - Airport layout visualization

### Passenger Dashboard
- Search flights by:
  - Flight number
  - Departure time
- View:
  - Gate assignments
  - Real-time gate status
  - Departure/Arrival times
  - Delay information

## Project Structure

- `simulation_ui.py`: Main simulation interface
- `passenger_dashboard.py`: Passenger information interface
- `gate_assignment.py`: Core assignment logic
- `requirements.txt`: Project dependencies
- `data/`: Contains flight and gate data

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- NumPy
- SQLite3

## Note

Both applications must run simultaneously for full functionality. The passenger dashboard relies on the simulation for real-time gate assignment data.
