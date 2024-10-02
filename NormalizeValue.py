def convert_height_to_cm(height):
    try:
        feet, inches = map(int, height.split("'"))
        total_inches = feet * 12 + inches
        return round(total_inches * 2.54, 2)
    except Exception:
        return None

def convert_weight_to_kg(weight):
    try:
        return round(float(weight.replace('lbs', '').strip()) * 0.453592, 2)
    except Exception:
        return None

def convert_value_wage(value):
    if value[-1] == 'M':
        return float(value[1:-1]) * 1e6
    elif value[-1] == 'K':
        return float(value[1:-1]) * 1e3
    return float(value[1:])
