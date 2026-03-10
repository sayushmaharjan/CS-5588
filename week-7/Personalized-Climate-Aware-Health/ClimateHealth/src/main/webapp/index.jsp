<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Welcome to ClimateHealth</title>
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
        h2 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        .buttons {
            display: flex;
            gap: 1rem;
        }
        a {
            display: inline-block;
            padding: 1rem 2rem;
            text-decoration: none;
            color: white;
            background-color: #28a745;
            border-radius: 8px;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
        a:hover {
            background-color: #218838;
        }
    </style>
</head>
<body>
    <h2>Welcome to the Personalized Climate-Driven Health Management Platform</h2>
    <div class="buttons">
        <a href="register.jsp">Register</a>
        <a href="login.jsp">Login</a>
    </div>
</body>
</html>