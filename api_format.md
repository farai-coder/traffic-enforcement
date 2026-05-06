# Violation POST API Format

## Endpoint
```
POST /api/violations
```

## Headers
```
Content-Type: multipart/form-data
Authorization: Bearer {token}
```

## Request Body (multipart/form-data)

| Field            | Type     | Required | Description                                      |
|------------------|----------|----------|--------------------------------------------------|
| violation_type   | string   | Yes      | One of: `red_light`, `stop_line`, `lane_change`  |
| plate_number     | string   | No       | Extracted plate text, e.g. `ABC1234`. Null if unreadable |
| timestamp        | string   | Yes      | ISO 8601 format: `2026-03-30T10:15:30`           |
| vehicle_id       | integer  | Yes      | Tracking ID assigned by the detection system     |
| confidence       | float    | No       | Detection confidence score 0.0 - 1.0             |
| image            | file     | Yes      | JPG evidence screenshot of the violation frame   |
| light_state      | string   | Yes      | Traffic light state: `red`, `yellow`, `green`    |

## Example Request (Python)
```python
import requests

url = "http://your-server.com/api/violations"
headers = {"Authorization": "Bearer your-token-here"}

data = {
    "violation_type": "red_light",
    "plate_number": "AEG4521",
    "timestamp": "2026-03-30T10:15:30",
    "vehicle_id": 3,
    "confidence": 0.87,
    "light_state": "red",
}

files = {
    "image": ("violation.jpg", open("data/violations/red_light_3_20260330_101530.jpg", "rb"), "image/jpeg"),
}

response = requests.post(url, headers=headers, data=data, files=files)
print(response.json())
```

## Expected Response (201 Created)
```json
{
    "success": true,
    "data": {
        "id": 42,
        "violation_type": "red_light",
        "plate_number": "AEG4521",
        "timestamp": "2026-03-30T10:15:30",
        "vehicle_id": 3,
        "confidence": 0.87,
        "light_state": "red",
        "image_url": "/storage/violations/red_light_3_20260330_101530.jpg",
        "fine_amount": 30.00,
        "status": "pending"
    }
}
```

## Error Response (422)
```json
{
    "success": false,
    "message": "Validation error",
    "errors": {
        "violation_type": ["The violation type field is required."],
        "image": ["The image must be a file of type: jpg, jpeg, png."]
    }
}
```
