package com.p_project.diary;

import com.p_project.writing.WritingSessionEntity;
import org.springframework.data.jpa.repository.JpaRepository;

public interface DiaryRepository extends JpaRepository<WritingSessionEntity, Long> {

    int countByUserIdAndTypeAndStatusAndDeletedAtIsNull(Long userId, WritingSessionEntity.Type type, String status); // TODO: status도 조건문 추가 필요

}
