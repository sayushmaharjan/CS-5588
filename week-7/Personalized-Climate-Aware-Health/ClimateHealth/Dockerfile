# ✅ First Stage: Build the WAR file using Maven
FROM maven:3.8.8-eclipse-temurin-17 AS build
WORKDIR /app

# ✅ Copy the POM file and install dependencies
COPY pom.xml .
RUN mvn dependency:go-offline

# ✅ Copy the source code
COPY . .

# ✅ Build the WAR file
RUN mvn clean package

# ✅ Second Stage: Deploy in Tomcat
FROM tomcat:10.1.18-jdk17
WORKDIR /usr/local/tomcat

# ✅ Copy the built WAR file
COPY --from=build /app/target/climate-health.war /usr/local/tomcat/webapps/ROOT.war

# ✅ Expose Port 8080 for Web Traffic
EXPOSE 8080

# ✅ Start Tomcat
CMD ["catalina.sh", "run"]
