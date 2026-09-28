import ee

def logging_in():
    """
    Function to handle user login and set up Earth Engine.
    """
    ee.Authenticate()
    ee.Initialize(project='aqi-prediction-508009')