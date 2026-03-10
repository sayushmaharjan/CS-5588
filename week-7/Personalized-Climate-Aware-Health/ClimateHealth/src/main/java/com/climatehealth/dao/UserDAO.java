package com.climatehealth.dao;

import com.climatehealth.model.User;
import org.mindrot.jbcrypt.BCrypt;

import java.io.FileInputStream;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Properties;

public class UserDAO {
	private String dbUrl;
    private String dbUser;
    private String dbPassword;

    public UserDAO() {
        try {
            dbUrl = System.getenv("MYSQL_URL");  
            dbUser = System.getenv("MYSQL_USER");  
            dbPassword = System.getenv("MYSQL_PASSWORD");  

            if (dbUrl == null || dbUser == null || dbPassword == null) {
                throw new RuntimeException("Missing environment variables for database connection.");
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load database configuration", e);
        }
    }



    private Connection getConnection() throws SQLException {
        try {
            Class.forName("com.mysql.cj.jdbc.Driver");
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            throw new SQLException("MySQL JDBC Driver not found.", e);
        }
        return DriverManager.getConnection(dbUrl, dbUser, dbPassword);
    }

    // Method to register a new user
    public void registerUser(User user) {
        String sql = "INSERT INTO users (username, password, email) VALUES (?, ?, ?)";
        try (Connection connection = getConnection();
             PreparedStatement statement = connection.prepareStatement(sql)) {
            statement.setString(1, user.getUsername());
            statement.setString(2, user.getPassword());
            statement.setString(3, user.getEmail());
            statement.executeUpdate();
        } catch (SQLException e) {
            System.err.println("SQL error occurred while registering user: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // Method to log in a user
    public User loginUser(String username, String password) {
        String sql = "SELECT * FROM users WHERE username = ?";
        try (Connection connection = getConnection();
             PreparedStatement statement = connection.prepareStatement(sql)) {
            statement.setString(1, username);
            ResultSet resultSet = statement.executeQuery();
            if (resultSet.next()) {
                String hashedPassword = resultSet.getString("password");
                if (BCrypt.checkpw(password, hashedPassword)) {
                    User user = new User();
                    user.setId(resultSet.getInt("id"));
                    user.setUsername(resultSet.getString("username"));
                    user.setEmail(resultSet.getString("email"));
                    return user;
                }
            }
        } catch (SQLException e) {
            System.err.println("SQL error occurred during login: " + e.getMessage());
            e.printStackTrace();
        }
        return null; // Invalid credentials
    }
}
