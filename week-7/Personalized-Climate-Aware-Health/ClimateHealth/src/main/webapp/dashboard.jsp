<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="com.climatehealth.model.HealthData" %>
<%@ page import="com.climatehealth.dao.HealthDataDAO" %>
<%@ page import="java.util.*" %>
<%@ page import="com.climatehealth.service.SuggestionService" %>
<html>
<head>
    <title>Health Data Dashboard</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f4f8;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            flex-direction: column;
        }
        .container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            margin-bottom: 1.5rem;
            font-size: 2rem;
            text-align: center;
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            color: #555;
            font-weight: 600;
        }
        input[type="text"], input[type="number"] {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        input[type="text"]:focus, input[type="number"]:focus {
            border-color: #007bff;
            outline: none;
        }
        input[type="submit"] {
            width: 100%;
            padding: 0.75rem;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        .weather-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            border-radius: 12px;
            color: white;
            text-align: center;
            animation: fadeIn 1s ease-in-out;
        }
        .weather-card h2 {
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        .weather-card p {
            margin: 0.5rem 0;
            font-size: 1.1rem;
        }
        .weather-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            animation: bounce 2s infinite;
        }
        .suggestions {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #f9f9f9;
            border-radius: 12px;
            border: 1px solid #eee;
            animation: slideIn 1s ease-in-out;
        }
        .suggestions h2 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        .suggestions ul {
            list-style-type: none;
            padding: 0;
        }
        .suggestions ul li {
            padding: 0.5rem 0;
            font-size: 1rem;
            color: #555;
            display: flex;
            align-items: center;
            animation: fadeInListItem 0.5s ease-in-out;
        }
        .suggestions ul li::before {
            content: "🌿";
            margin-right: 0.5rem;
            font-size: 1.2rem;
        }
        .suggestions ul li.health::before {
            content: "💧"; /* Water drop for health tips */
        }
        .suggestions ul li.activity::before {
            content: "🚴"; /* Bicycle for activity tips */
        }
        .suggestions ul li.tip::before {
            content: "💡"; /* Lightbulb for general tips */
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        @keyframes fadeInListItem {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Health Data Entry</h1>

        <% if (request.getParameter("success") != null) { %>
            <div class="message success"><%= request.getParameter("success") %></div>
        <% } else if (request.getParameter("error") != null) { %>
            <div class="message error"><%= request.getParameter("error") %></div>
        <% } %>

        <form action="HealthDataServlet" method="post">
            <div class="form-group">
                <label for="location">Location:</label>
                <input type="text" id="location" name="location" required>
            </div>
            <div class="form-group">
                <label for="weight">Weight (kg):</label>
                <input type="number" step="0.1" id="weight" name="weight" required>
            </div>
            <div class="form-group">
                <label for="height">Height (cm):</label>
                <input type="number" step="0.1" id="height" name="height" required>
            </div>
            <div class="form-group">
                <label for="temperature">Temperature (°C):</label>
                <input type="number" step="0.1" id="temperature" name="temperature" required>
            </div>
            <input type="submit" value="Submit">
        </form>

        <% 
        String location = request.getParameter("location");
        if (location != null) {
            SuggestionService suggestionService = new SuggestionService();
            String suggestions = suggestionService.getSuggestions(location);
            
            if (suggestions != null && !suggestions.isEmpty()) {
                out.println("<div class='weather-card'>");
                out.println("<div class='weather-icon'>☁️</div>");
                out.println("<h2>Weather in " + location + "</h2>");
                out.println("<p><strong>Description:</strong> Mist</p>");
                out.println("<p><strong>Temperature:</strong> 26.6°C</p>");
                out.println("<p><strong>Humidity:</strong> 79%</p>");
                out.println("</div>");

                out.println("<div class='suggestions'>");
                out.println("<h2>Suggestions for " + location + "</h2>");
                out.println("<ul>");
                
                // Split the suggestions into individual lines
                String[] suggestionLines = suggestions.split("\n");
                for (String line : suggestionLines) {
                    if (line.trim().isEmpty()) continue; // Skip empty lines
                    out.println("<li class='tip'>" + line.trim() + "</li>");
                }
                
                out.println("</ul>");
                out.println("</div>");
            }
        }
        %>
    </div>
</body>
</html>