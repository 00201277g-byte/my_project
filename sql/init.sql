-- 创建数据库
CREATE DATABASE IF NOT EXISTS power_forecast
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE power_forecast;

-- 预测记录表
CREATE TABLE IF NOT EXISTS prediction_record (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键',
    model_name      VARCHAR(50)     NOT NULL COMMENT '模型名称',
    file_name       VARCHAR(255)    NOT NULL COMMENT '上传文件名',
    predict_time    DATETIME        NOT NULL COMMENT '预测时间',
    forecast_horizon INT            DEFAULT 24 COMMENT '预测步数',
    mape_ref        DOUBLE          COMMENT 'MAPE参考值(%)',
    predictions_json TEXT           COMMENT '预测结果JSON',
    status          VARCHAR(20)     DEFAULT 'success' COMMENT '状态',
    error_msg       TEXT            COMMENT '错误信息',
    create_time     DATETIME        DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_create_time (create_time),
    INDEX idx_model_name  (model_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='预测记录表';
