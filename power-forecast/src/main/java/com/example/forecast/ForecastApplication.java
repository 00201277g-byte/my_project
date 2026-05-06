package com.example.forecast;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan("com.example.forecast.mapper")   // ⭐ 必须加
public class ForecastApplication {
    public static void main(String[] args) {
        SpringApplication.run(ForecastApplication.class, args);
    }
}