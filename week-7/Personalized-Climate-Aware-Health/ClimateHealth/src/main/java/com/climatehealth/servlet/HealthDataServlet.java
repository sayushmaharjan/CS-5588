package com.climatehealth.servlet;

import com.climatehealth.dao.HealthDataDAO;
import com.climatehealth.model.HealthData;

import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;

public class HealthDataServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String location = request.getParameter("location");
        double weight = Double.parseDouble(request.getParameter("weight"));
        double height = Double.parseDouble(request.getParameter("height"));
        double temperature = Double.parseDouble(request.getParameter("temperature"));

        HealthData healthData = new HealthData();
        healthData.setLocation(location);
        healthData.setWeight((float) weight);
        healthData.setHeight((float) height);
        healthData.setTemperature((float) temperature);

        HealthDataDAO healthDataDAO = new HealthDataDAO();
        boolean isSaved = healthDataDAO.saveHealthData(healthData);

        // Redirecting to dashboard.jsp with success message and location
        if (isSaved) {
            response.sendRedirect("dashboard.jsp?success=Data saved successfully&location=" + location);
        } else {
            response.sendRedirect("dashboard.jsp?error=Failed to save data&location=" + location);
        }
    }
}
