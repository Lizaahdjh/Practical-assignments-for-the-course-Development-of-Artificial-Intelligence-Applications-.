# API-контракти

Базовий URL: `http://127.0.0.1:8000`

## Єдиний формат помилки

```json
{
  "error": {
    "kind": "invalid_input | not_found | provider_unavailable | timeout | rate_limit | internal",
    "message": "текст для людини",
    "retryable": false,
    "request_id": "uuid"
  }
}