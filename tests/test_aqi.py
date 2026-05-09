from src.aqi import (
    pm25_to_aqi,
    classify_aqi
)

def test_pm25_to_aqi():

    result = pm25_to_aqi(10)

    assert result > 0

def test_classify_aqi():

    result = classify_aqi(40)

    assert result == "Good"