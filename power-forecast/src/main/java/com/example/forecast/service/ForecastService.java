package com.example.forecast.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.forecast.entity.PredictionRecord;
import com.example.forecast.mapper.PredictionRecordMapper;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.util.*;

@Slf4j
@Service
@RequiredArgsConstructor
public class ForecastService {

    private final WebClient webClient;
    private final PredictionRecordMapper recordMapper;

    @Value("${upload.path}")
    private String uploadPath;

    // ---- 模型路由 ----
    private static final Map<String, String> MODEL_ENDPOINTS = Map.of(
            "cnn_transformer", "/predict",
            "lgbm",            "/predict/lgbm",
            "lstm",            "/predict/lstm"
    );

    private static final Map<String, String> MODEL_DISPLAY = Map.of(
            "cnn_transformer", "CNN+Transformer",
            "lgbm",            "LightGBM",
            "lstm",            "LSTM"
    );

    /**
     * 上传 CSV 文件，返回文件路径和预览数据
     */
    public Map<String, Object> uploadFile(MultipartFile file) throws IOException, com.opencsv.exceptions.CsvValidationException {
        String originalName = file.getOriginalFilename();
        if (originalName == null || !originalName.endsWith(".csv")) {
            throw new IllegalArgumentException("只支持 CSV 格式文件");
        }

        // 保存文件
        Path dir = Paths.get(uploadPath);
        Files.createDirectories(dir);
        String savedName = System.currentTimeMillis() + "_" + originalName;
        Path savedPath = dir.resolve(savedName);
        file.transferTo(savedPath);
        log.info("文件已保存: {}", savedPath);

        // 解析CSV预览（前5行）
        List<Map<String, String>> preview = new ArrayList<>();
        List<String> headers = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(savedPath.toFile()))) {
            String[] header = reader.readNext();
            if (header != null) {
                headers.addAll(Arrays.asList(header));
            }
            String[] row;
            int count = 0;
            while ((row = reader.readNext()) != null && count < 5) {
                Map<String, String> rowMap = new LinkedHashMap<>();
                for (int i = 0; i < headers.size() && i < row.length; i++) {
                    rowMap.put(headers.get(i), row[i]);
                }
                preview.add(rowMap);
                count++;
            }
        }

        // 统计行数
        long totalRows;
        try (var lines = Files.lines(savedPath)) {
            totalRows = lines.count() - 1; // 减去header行
        }

        Map<String, Object> result = new HashMap<>();
        result.put("fileName", savedName);
        result.put("originalName", originalName);
        result.put("filePath", savedPath.toString());
        result.put("totalRows", totalRows);
        result.put("headers", headers);
        result.put("preview", preview);
        return result;
    }

    /**
     * 读取 CSV 文件，转为 JSON 数组供 Flask 服务使用
     */
    public List<Map<String, Object>> readCsvAsJson(String fileName) throws IOException, com.opencsv.exceptions.CsvValidationException {
        Path filePath = Paths.get(uploadPath, fileName);
        if (!Files.exists(filePath)) {
            throw new FileNotFoundException("文件不存在: " + fileName);
        }

        List<Map<String, Object>> rows = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(filePath.toFile()))) {
            String[] headers = reader.readNext();
            if (headers == null) throw new IOException("CSV文件为空");

            String[] row;
            while ((row = reader.readNext()) != null) {
                Map<String, Object> rowMap = new LinkedHashMap<>();
                for (int i = 0; i < headers.length && i < row.length; i++) {
                    String val = row[i].trim();
                    // 尝试解析为数字
                    try {
                        rowMap.put(headers[i], Double.parseDouble(val));
                    } catch (NumberFormatException e) {
                        rowMap.put(headers[i], val);
                    }
                }
                rows.add(rowMap);
            }
        }
        return rows;
    }

    /**
     * 调用 Flask 服务执行预测
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> predict(String fileName, String modelKey) {
        if (!MODEL_ENDPOINTS.containsKey(modelKey)) {
            throw new IllegalArgumentException("不支持的模型: " + modelKey
                    + "，可选: " + MODEL_ENDPOINTS.keySet());
        }

        // 构建请求体
        List<Map<String, Object>> data;
        try {
            data = readCsvAsJson(fileName);
        } catch (IOException | CsvValidationException e) {
            throw new RuntimeException("读取文件失败: " + e.getMessage(), e);
        }

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("data", data);

        // 调用 Flask
        String endpoint = MODEL_ENDPOINTS.get(modelKey);
        log.info("调用 Flask 服务: {} 文件: {} 模型: {}", endpoint, fileName, modelKey);

        Map<String, Object> flaskResponse;
        try {
            flaskResponse = webClient.post()
                    .uri(endpoint)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(requestBody)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();
        } catch (Exception e) {
            log.error("Flask 服务调用失败", e);
            // 保存失败记录
            saveRecord(fileName, modelKey, null, "failed", e.getMessage());
            throw new RuntimeException("预测服务不可用，请确认 Python 服务已启动: " + e.getMessage());
        }

        // 保存成功记录
        saveRecord(fileName, modelKey, flaskResponse, "success", null);
        return flaskResponse;
    }

    /**
     * 保存预测记录到 MySQL
     */
    private void saveRecord(String fileName, String modelKey,
                            Map<String, Object> response, String status, String errorMsg) {
        try {
            PredictionRecord record = new PredictionRecord();
            record.setFileName(fileName);
            record.setModelName(MODEL_DISPLAY.getOrDefault(modelKey, modelKey));
            record.setPredictTime(LocalDateTime.now());
            record.setForecastHorizon(24);
            record.setStatus(status);
            record.setErrorMsg(errorMsg);
            record.setCreateTime(LocalDateTime.now());

            if (response != null) {
                Object predictions = response.get("predictions");
                if (predictions != null) {
                    try {
                        record.setPredictionsJson(new ObjectMapper().writeValueAsString(predictions));
                    } catch (Exception ex) {
                        record.setPredictionsJson(predictions.toString());
                    }
                }
                Object mapeRef = response.get("mape_ref");
                if (mapeRef instanceof Number) {
                    record.setMapeRef(((Number) mapeRef).doubleValue());
                }
            }

            recordMapper.insert(record);
        } catch (Exception e) {
            log.warn("保存预测记录失败（不影响预测结果）: {}", e.getMessage());
        }
    }

    /**
     * 查询单条历史记录详情，解析 predictionsJson 返回结构化数据
     */
    public Map<String, Object> getHistoryDetail(Long id) {
        PredictionRecord record = recordMapper.selectById(id);
        if (record == null) {
            throw new RuntimeException("记录不存在: id=" + id);
        }
        Map<String, Object> result = new java.util.HashMap<>();
        result.put("id",              record.getId());
        result.put("modelName",       record.getModelName());
        result.put("fileName",        record.getFileName());
        result.put("predictTime",     record.getPredictTime());
        result.put("forecastHorizon", record.getForecastHorizon());
        result.put("mapeRef",         record.getMapeRef());
        result.put("status",          record.getStatus());
        result.put("errorMsg",        record.getErrorMsg());
        result.put("predictionsJson", record.getPredictionsJson());
        return result;
    }

    /**
     * 删除历史记录
     */
    public void deleteHistory(Long id) {
        PredictionRecord record = recordMapper.selectById(id);
        if (record == null) {
            throw new RuntimeException("记录不存在: id=" + id);
        }
        recordMapper.deleteById(id);
        log.info("已删除预测记录: id={}", id);
    }

    /**
     * 查询历史预测记录（分页）
     */
    public Map<String, Object> getHistory(int page, int size) {
        Page<PredictionRecord> pageReq = new Page<>(page, size);
        LambdaQueryWrapper<PredictionRecord> wrapper = new LambdaQueryWrapper<PredictionRecord>()
                .orderByDesc(PredictionRecord::getCreateTime);
        Page<PredictionRecord> result = recordMapper.selectPage(pageReq, wrapper);

        Map<String, Object> resp = new HashMap<>();
        resp.put("total", result.getTotal());
        resp.put("pages", result.getPages());
        resp.put("current", result.getCurrent());
        resp.put("records", result.getRecords());
        return resp;
    }

    /**
     * 查询 Flask 服务健康状态
     */
    public Map<String, Object> checkPythonServer() {
        try {
            Map result = webClient.get()
                    .uri("/health")
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();
            return result != null ? result : Map.of("status", "unknown");
        } catch (Exception e) {
            return Map.of("status", "unreachable", "error", e.getMessage());
        }
    }
}