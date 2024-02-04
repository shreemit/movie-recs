import requests
import json

def fetch_poster(movie_id, api_key = "a510a1b10dac2c213e3ff479255b52b9"):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}')
    data = response.json()
    # print(data)
    poster = data.get("poster_path")
    if poster:
        return f"https://image.tmdb.org/t/p/w500/{poster}"
    return None