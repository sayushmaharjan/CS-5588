package com.climatehealth.service;

import java.util.Map;

import com.climatehealth.api.WeatherAPI;

public class SuggestionService {
    private GroqService groqService = new GroqService();

    public String getSuggestions(String location) {
        try {
            // Fetch weather information
            String weatherInfo = fetchWeatherInfo(location);
            if (weatherInfo == null || weatherInfo.contains("Failed")) {
                return "Weather data unavailable.";
            }
            // Generate a prompt for Groq
            String prompt = String.format("Provide health and activity suggestions for someone in %s based on the following weather: %s", location, weatherInfo);

            // Get suggestions from Groq
            String groqResponse = groqService.getGroqResponse(prompt);
            
            if (groqResponse == null || groqResponse.isEmpty()) {
                return "AI suggestion service is down.";
            }


         // Remove unnecessary markdown formatting (**)
            String cleanedResponse = groqResponse.replace("**", "");
            return weatherInfo + "\n\n" + cleanedResponse;

        } catch (Exception e) {
            return "Failed to fetch suggestions: " + e.getMessage();
        }
    }

    private String fetchWeatherInfo(String location) {
        WeatherAPI weatherAPI = new WeatherAPI();
        try {
            Map<String, Object> weatherDetails = weatherAPI.getWeather(location);
            String description = (String) weatherDetails.get("description");
            Double temperature = (Double) weatherDetails.get("temperature");
            Integer humidity = (Integer) weatherDetails.get("humidity");

            return String.format("Weather in %s:\n- Description: %s\n- Temperature: %.1f°C\n- Humidity: %d%%",
                    location, description, temperature, humidity);
        } catch (Exception e) {
            return "Failed to fetch weather information: " + e.getMessage();
        }
    }
}