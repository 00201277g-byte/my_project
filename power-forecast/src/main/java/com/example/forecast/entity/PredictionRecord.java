package com.example.forecast.entity;

import com.baomidou.mybatisplus.annotation.*;
import lombok.Data;
import java.time.LocalDateTime;

@Data
@TableName("prediction_record")
public class PredictionRecord {

    @TableId(type = IdType.AUTO)
    private Long id;

    /** 预测使用的模型名称 */
    private String modelName;

    /** 上传的文件名 */
    private String fileName;

    /** 预测触发时间 */
    private LocalDateTime predictTime;

    /** 预测步数（固定24） */
    private Integer forecastHorizon;

    /** 整体 MAPE 参考值 */
    private Double mapeRef;

    /** 预测结果 JSON 字符串（存储24步预测值） */
    private String predictionsJson;

    /** 状态：success / failed */
    private String status;

    /** 错误信息（失败时记录） */
    private String errorMsg;

    @TableField(fill = FieldFill.INSERT)
    private LocalDateTime createTime;
}
