"""Модуль інтеграції із зовнішнім API погоди Open-Meteo з підтримкою локалізації."""

from functools import lru_cache
from typing import Tuple
import httpx
from pydantic import BaseModel, Field

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 5.0


class WeatherError(Exception):
    def __init__(self, message_uk: str, message_en: str, status_code: int = 500):
        self.message_uk = message_uk
        self.message_en = message_en
        self.status_code = status_code
        super().__init__(message_uk)


class CityNotFoundError(WeatherError):
    def __init__(self, city_name: str):
        super().__init__(
            f"Місто '{city_name}' не знайдено",
            f"City '{city_name}' not found",
            status_code=404
        )


class ExternalAPIError(WeatherError):
    pass


class WeatherResponse(BaseModel):
    city: str
    country: str
    temperature: float
    temperature_unit: str
    windspeed: float
    windspeed_unit: str
    weathercode: int
    theme_class: str = Field(description="day, night, sunset, cloudy або rainy")
    icon: str
    description: str
    time: str


def _get_theme_and_icon(code: int, is_day: int, lang: str = "uk") -> Tuple[str, str, str]:
    """Повертає тему, іконку та локалізований опис."""
    is_uk = (lang == "uk")
    
    if code in (51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99):
        desc = "Дощ / Злива" if is_uk else "Rain / Showers"
        return "rainy", "🌧️", desc
    elif code in (71, 73, 75, 77, 85, 86):
        desc = "Снігопад" if is_uk else "Snowfall"
        return "rainy", "❄️", desc
    elif code in (2, 3):
        desc = "Хмаро з проясненнями" if is_uk else "Partly Cloudy"
        return "cloudy", "☁️", desc
    elif code in (0, 1):
        if is_day == 1:
            desc = "Ясно" if is_uk else "Clear Sky"
            return "day", "☀️", desc
        else:
            desc = "Ясна ніч" if is_uk else "Clear Night"
            return "night", "🌙", desc
            
    desc = "Туман" if is_uk else "Foggy"
    return "cloudy", "🌫️", desc


@lru_cache(maxsize=128)
def _getCachedCoordinates(name: str, lang: str = "uk") -> Tuple[str, str, float, float]:
    params = {"name": name, "count": 1, "language": lang, "format": "json"}
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        res = client.get(GEOCODING_URL, params=params)
        res.raise_for_status()
        data = res.json()

    results = data.get("results")
    if not results:
        raise CityNotFoundError(name)

    loc = results[0]
    return (loc.get("name", name), loc.get("country", ""), loc["latitude"], loc["longitude"])


async def find_city(name: str, lang: str = "uk") -> dict:
    if not name or not name.strip():
        raise WeatherError(
            "Назва міста не може бути порожньою",
            "City name cannot be empty",
            status_code=400
        )

    clean_name = name.strip().lower()
    try:
        res_name, country, lat, lon = _getCachedCoordinates(clean_name, lang)
        return {"name": res_name, "country": country, "latitude": lat, "longitude": lon}
    except CityNotFoundError:
        raise
    except httpx.TimeoutException:
        raise ExternalAPIError(
            "Перевищено час очікування геокодування",
            "Geocoding service timeout",
            status_code=504
        )
    except Exception as exc:
        raise ExternalAPIError(
            f"Помилка геокодування: {exc}",
            f"Geocoding error: {exc}",
            status_code=502
        )


async def get_current_weather(city: str, lang: str = "uk") -> WeatherResponse:
    location = await find_city(city, lang)

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": ["temperature_2m", "wind_speed_10m", "weather_code", "is_day"]
    }

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.get(FORECAST_URL, params=params)
            if response.status_code >= 500:
                raise ExternalAPIError(
                    "Погодний сервіс недоступний",
                    "Weather service unavailable",
                    status_code=502
                )
            elif response.status_code >= 400:
                raise WeatherError(
                    "Помилка запиту",
                    "Bad request error",
                    status_code=400
                )

            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            raise ExternalAPIError(
                "Таймаут погодного сервісу",
                "Weather service timeout",
                status_code=504
            )
        except httpx.RequestError as exc:
            raise ExternalAPIError(
                f"Мережева помилка: {exc}",
                f"Network error: {exc}",
                status_code=502
            )

    current = data.get("current", {})
    units = data.get("current_units", {})

    code = current.get("weather_code", 0)
    is_day = current.get("is_day", 1)

    theme, icon, desc = _get_theme_and_icon(code, is_day, lang)

    return WeatherResponse(
        city=location["name"],
        country=location["country"],
        temperature=current.get("temperature_2m", 0.0),
        temperature_unit=units.get("temperature_2m", "°C"),
        windspeed=current.get("wind_speed_10m", 0.0),
        windspeed_unit=units.get("wind_speed_10m", "km/h"),
        weathercode=code,
        theme_class=theme,
        icon=icon,
        description=desc,
        time=current.get("time", "")
    )