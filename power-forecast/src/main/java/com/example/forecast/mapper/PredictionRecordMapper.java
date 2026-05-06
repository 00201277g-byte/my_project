package com.example.forecast.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.forecast.entity.PredictionRecord;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface PredictionRecordMapper extends BaseMapper<PredictionRecord> {
}