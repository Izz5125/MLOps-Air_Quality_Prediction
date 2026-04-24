def pm25_to_aqi(pm25):
    if pm25 <= 12:
        return (50 / 12) * pm25

    elif pm25 <= 35.4:
        return ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51

    elif pm25 <= 55.4:
        return ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101

    elif pm25 <= 150.4:
        return ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151

    else:
        return 300
    
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"
    
def get_recommendation(aqi_status):
    recommendations = {
        "Good": "Udara sedang baik. Cocok untuk jalan-jalan, olahraga, atau beraktivitas di luar ruangan.",
        
        "Moderate": "Kualitas udara cukup aman, tetapi kelompok sensitif disarankan mengurangi aktivitas luar ruangan yang terlalu lama.",
        
        "Unhealthy for Sensitive Groups": "Kelompok sensitif seperti anak-anak, lansia, dan penderita gangguan pernapasan disarankan memakai masker dan membatasi aktivitas di luar.",
        
        "Unhealthy": "Udara kurang sehat. Disarankan memakai masker saat keluar rumah dan mengurangi aktivitas luar ruangan.",
        
        "Very Unhealthy": "Udara sangat buruk. Sebaiknya hindari aktivitas di luar ruangan kecuali benar-benar diperlukan.",
        
        "Hazardous": "Udara berbahaya. Sangat tidak disarankan keluar rumah. Gunakan perlindungan tambahan jika harus keluar."
    }

    return recommendations.get(
        aqi_status,
        "Tidak ada rekomendasi tersedia."
    )