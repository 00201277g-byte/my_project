package com.example.forecast.controller;

import com.example.forecast.common.R;
import com.example.forecast.service.ForecastService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class ForecastController {

    private final ForecastService forecastService;

    @PostMapping("/upload")
    public R<Map<String, Object>> upload(@RequestParam("file") MultipartFile file) {
        try {
            return R.ok(forecastService.uploadFile(file));
        } catch (IllegalArgumentException e) {
            return R.fail(400, e.getMessage());
        } catch (Exception e) {
            log.error("文件上传失败", e);
            return R.fail("文件上传失败: " + e.getMessage());
        }
    }

    @PostMapping("/predict")
    public R<Map<String, Object>> predict(
            @RequestParam String fileName,
            @RequestParam(defaultValue = "cnn_transformer") String model) {
        try {
            return R.ok(forecastService.predict(fileName, model));
        } catch (IllegalArgumentException e) {
            return R.fail(400, e.getMessage());
        } catch (RuntimeException e) {
            log.error("预测失败", e);
            return R.fail(e.getMessage());
        }
    }

    @GetMapping("/history")
    public R<Map<String, Object>> history(
            @RequestParam(defaultValue = "1") int page,
            @RequestParam(defaultValue = "10") int size) {
        return R.ok(forecastService.getHistory(page, size));
    }

    /** 查询单条详情 GET /api/history/{id} */
    @GetMapping("/history/{id}")
    public R<Map<String, Object>> historyDetail(@PathVariable Long id) {
        try {
            return R.ok(forecastService.getHistoryDetail(id));
        } catch (RuntimeException e) {
            return R.fail(404, e.getMessage());
        }
    }

    /** 删除记录 DELETE /api/history/{id} */
    @DeleteMapping("/history/{id}")
    public R<Void> deleteHistory(@PathVariable Long id) {
        try {
            forecastService.deleteHistory(id);
            return R.ok(null);
        } catch (RuntimeException e) {
            return R.fail(e.getMessage());
        }
    }

    @GetMapping("/status")
    public R<Map<String, Object>> status() {
        return R.ok(forecastService.checkPythonServer());
    }
}