import fastf1
import pandas as pd

# Enable FastF1 cache (downloads will be stored locally so you don't redownload)
fastf1.Cache.enable_cache("C:/Users/Harpita Bakshi/Downloads/f1_cache")  # creates folder 'f1_cache' if not exists

all_quali_data = []

# Loop through all 2024 races (there were 24 scheduled)
for rnd in range(1, 25):
    try:
        print(f"Fetching Qualifying data for 2024 Round {rnd}...")
        quali = fastf1.get_session(2024, rnd, 'Q')
        quali.load()  # download and parse session data

        # Results dataframe
        results = quali.results[['DriverNumber','FullName','TeamName','Q1','Q2','Q3']].copy()
        results = results.rename(columns={
            'FullName': 'Driver',
            'TeamName': 'Team'
        })

        # Add race metadata
        results['Year'] = 2024
        results['Round'] = rnd
        results['GP_Name'] = quali.event['EventName']

        # Convert lap times (timedelta → seconds)
        for col in ['Q1','Q2','Q3']:
            results[col] = results[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else None)

        # Add first available weather sample
        weather = quali.weather_data
        if weather is not None and not weather.empty:
            first_weather = weather.iloc[0]
            results['AirTemp'] = first_weather.get('AirTemp', None)
            results['TrackTemp'] = first_weather.get('TrackTemp', None)
            results['Humidity'] = first_weather.get('Humidity', None)
            results['WindSpeed'] = first_weather.get('WindSpeed', None)
            results['Weather'] = first_weather.get('Weather', None)

        all_quali_data.append(results)

    except Exception as e:
        print(f"⚠️ Could not fetch Round {rnd}: {e}")
        continue

# Combine all into one dataframe
final_df = pd.concat(all_quali_data, ignore_index=True)

# Save to CSV
final_df.to_csv("F1_2024_Qualifying_with_Weather.csv", index=False)
print("✅ Saved all 2024 qualifying data to F1_2024_Qualifying_with_Weather.csv")
import pandas as pd

df = pd.read_csv("F1_2024_Qualifying_with_Weather.csv")

print(df.shape)   # how many rows and columns
print(df.head())  # first 5 rows
